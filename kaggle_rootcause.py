"""Root-cause completion for SVHalluc TEMPORAL (Qwen2.5-Omni-3B). Usage:
  python kaggle_rootcause.py ROOT OUT PER_LABEL SHARD_ID NUM_SHARDS
Per item (TA/TF/CMTB), all scored with rotated option orders and summed per option content:
  base                  native model                                                        -> reference answer
  para1, para2          two paraphrases of the question                                    -> #12 question wording
  prior                 black video + silent audio of the same length                     -> #13 content-free prior
  vision_only           original video, audio removed                                     -> modality reliance
  audio_only            black video + original audio                                      -> modality reliance
  (CMTB) gate, audio_block, gate_audio, crop                                               -> #9/#10 per-error decomposition
  hear (text) + whisper (analysis-only reference transcript of the speech window)          -> #4 speech content
  hidden states (last token, mean of question rows) at several layers from the base run    -> #14 representation-action gap
  (CMTB, base wrong and gate_audio right) activation patching sweep                        -> where the error enters
Shard 0 additionally runs:
  synthetic text-only relation questions (no media)                                        -> #11 relation reasoning
  eager attention of the question rows for the first 24 items                              -> #8 front-loading at scale
Labels are used only for scoring. Whisper is an analysis tool only; it never feeds the model."""
import os, sys, json, re, subprocess, time, random
from pathlib import Path
import numpy as np
import torch

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PER_LABEL, SHARD, NSHARD = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
MAX_DUR = float(os.environ.get('LAB_MAX_DUR', 42.0 if torch.cuda.device_count() >= 2 else 28.0))
sys.path.insert(0, str(ROOT))
import av, librosa
from huggingface_hub import hf_hub_download

DATASET, REVISION = 'zcs15/SVHalluc', '211b4f14ab66ae9557b69e302d1f090e0e610839'
MODEL, MREV = 'Qwen/Qwen2.5-Omni-3B', 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
BS = chr(92); FF = os.environ.get('FFMPEG_BIN', 'ffmpeg')
LAYERS_KEEP = [6, 12, 15, 18, 21, 24, 27, 30, 33, 36]


def decode_audio(path):
    chunks = []
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        for fr in c.decode(audio=0):
            for p in rs.resample(fr): chunks.append(p.to_ndarray().reshape(-1))
        for p in rs.resample(None): chunks.append(p.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype('float32', copy=False)


_orig = librosa.load
def _load(path, sr=None, offset=0.0, duration=None, **k):
    if str(path).lower().endswith('.mp4'):
        a = decode_audio(path)
        return a[int(sr * offset): None if duration is None else int(sr * (offset + duration))], sr
    return _orig(path, sr=sr, offset=offset, duration=duration, **k)
librosa.load = _load

from lab_native_video import prepare_video, qwen_inputs, move_inputs, SYSTEM
from speech_gate import media_layout, gated_mask
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

CLIPS = OUT / 'clips'; CLIPS.mkdir(parents=True, exist_ok=True)
HID = OUT / 'hidden'; HID.mkdir(parents=True, exist_ok=True)
ENC = ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac']


def fetch(name):
    local = os.environ.get('LAB_LOCAL_DATA')
    if local and (Path(local) / name).exists(): return Path(local) / name
    return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def ffmpeg(*args):
    r = subprocess.run([FF, '-hide_banner', '-y', *args], capture_output=True)
    if r.returncode != 0 and b'fps_mode' in r.stderr:      # older ffmpeg builds only know -vsync
        r = subprocess.run([FF, '-hide_banner', '-y', *['-vsync' if a == '-fps_mode' else a for a in args]], capture_output=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.decode(errors='replace')[-400:])


def cut(src, start, end, tag):
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists():
        ffmpeg('-ss', f'{start:.2f}', '-to', f'{end:.2f}', '-i', str(src), '-map', '0:v:0', '-map', '0:a:0', '-vf', 'fps=30', '-fps_mode', 'cfr', *ENC,
               '-af', 'aresample=async=1:first_pts=0', str(dst))
    return dst


def video_size(src):
    with av.open(str(src)) as c:
        s = c.streams.video[0]; return s.width, s.height


def blank_video(src, dur, keep_audio):
    """black frames of the same size/length; either the original audio or digital silence."""
    w, h = video_size(src)
    dst = CLIPS / f'{Path(src).stem}_{"blackA" if keep_audio else "prior"}.mp4'
    if not dst.exists():
        if keep_audio:
            ffmpeg('-f', 'lavfi', '-i', f'color=black:s={w}x{h}:r=30:d={dur:.2f}', '-i', str(src), '-map', '0:v:0', '-map', '1:a:0', '-shortest', *ENC, str(dst))
        else:
            ffmpeg('-f', 'lavfi', '-i', f'color=black:s={w}x{h}:r=30:d={dur:.2f}', '-f', 'lavfi', '-i', f'anullsrc=r=16000:cl=mono',
                   '-t', f'{dur:.2f}', '-map', '0:v:0', '-map', '1:a:0', *ENC, str(dst))
    return dst


def speech_window(a):
    hop = 800; n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    act = db > lo + 0.5 * (hi - lo)
    if act.sum() == 0: return None
    t = (np.arange(n) + 0.5) * 0.05; idx = np.where(act)[0]
    return float(t[idx[0]]), float(t[idx[-1]])


def parse_question(q):
    parts = re.split(r'\s*\(([ABC])\)\s*', q)
    return parts[0].strip(), [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


def rotations(k): return [tuple((s + j) % k for j in range(k)) for s in range(k)]
def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))


PARA = {'TA': ['Does the action that the speaker talks about happen in the video at the same moment the speaker is talking?',
               'Look at the video only while the speech is playing. Is the narrated action visible during that time?'],
        'TF': ['The speaker describes an action. Compared with the time the speaker is talking, does that action appear in the video earlier, at the same time, or later?',
               'When does the narrated action happen in the video relative to the speech?'],
        'CMTB': ['Which action is visible in the video while the person is speaking?', 'During the speech, what is being done on screen?']}

MODEL_ = PROC = LETTERS = None


def load_model():
    cache = os.environ.get('LAB_CACHE')
    kw = dict(revision=MREV, torch_dtype=torch.float16, attn_implementation='sdpa', cache_dir=cache)
    if os.environ.get('LAB_NF4'):
        from transformers import BitsAndBytesConfig
        kw.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                  bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    elif torch.cuda.device_count() >= 2: kw.update(device_map='balanced', max_memory={0: '10GiB', 1: '10GiB'})
    else: kw.update(device_map={'': 0})
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    return m, Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=MREV, cache_dir=cache)


def masked_forward(inp, mask_kind, window, record_layers=None, patch=None):
    """mask_kind: plain | gate | audio_block | gate_audio. record_layers -> dict(layer -> (last_token, mean_question_rows, question_rows))."""
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]
    spg = inp.get('video_second_per_grid'); spg = float(spg.reshape(-1)[0]) if spg is not None else None
    tg = mask_kind in ('gate', 'gate_audio'); ab = mask_kind in ('audio_block', 'gate_audio')
    mask = gated_mask(L, is_v, sec, last_media, window, 'hard' if tg else 'none', 0.0, emb.dtype, emb.device, spg, 'all', is_a, ab)
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    rows_q = (torch.arange(L) > last_media)
    store, hooks = {}, []
    layers = MODEL_.model.layers
    if record_layers or patch is not None:
        for li in range(len(layers)):
            def hook(mod, args, out, li=li):
                h = out[0]
                rq = rows_q.to(h.device)
                if patch is not None and patch[0] == li:
                    h = h.clone(); h[0, rq] = patch[1].to(h.device, h.dtype); out = (h,) + tuple(out[1:])
                if record_layers and (li + 1) in record_layers:
                    store[li + 1] = (h[0, -1].float().cpu(), h[0, rq].float().mean(0).cpu(), h[0, rq].detach().float().cpu())
                return out
            hooks.append(layers[li].register_forward_hook(hook))
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()
    return lg, store


def scores(media, stem, texts, mode='av', mask_kind=None, window=None):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, mode); inp = move_inputs(inp, MODEL_)
        if mask_kind is None:
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=(mode == 'av')).logits[0, -1].float()
        else:
            lg, _ = masked_forward(inp, mask_kind, window)
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
        del inp
    torch.cuda.empty_cache()
    return tot.round(3).tolist()


HEAR_Q = 'What action does the speaker say is being done? Answer with a short phrase (verb and object) only.'
WHISPER = None


def hear(media):
    inp, _ = qwen_inputs(PROC, HEAR_Q, media, 'audio'); inp = move_inputs(inp, MODEL_)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=16, do_sample=False)
    return PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().split('\n')[0][:120]


def whisper_text(a, s0, s1):
    global WHISPER
    if os.environ.get('LAB_NO_WHISPER'): return None
    if WHISPER is None:
        from transformers import pipeline
        WHISPER = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=0, torch_dtype=torch.float16)
    seg = a[int(s0 * 16000): int(s1 * 16000)]
    return WHISPER({'raw': seg, 'sampling_rate': 16000}, generate_kwargs={'language': 'en'})['text'].strip()


def patch_sweep(media, stem, texts, window):
    """Original option order only. Returns plain/clean letter logits and forward/reverse patching per layer."""
    inp, _ = qwen_inputs(PROC, render(stem, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
    k = len(texts); allL = list(range(1, len(MODEL_.model.layers) + 1))
    lg_p, Sp = masked_forward(inp, 'plain', window, record_layers=allL)
    lg_c, Sc = masked_forward(inp, 'gate_audio', window, record_layers=allL)
    let = lambda lg: [round(float(lg[t]), 3) for t in LETTERS[:k]]
    res = dict(plain=let(lg_p), clean=let(lg_c), fwd={}, rev={})
    for li in list(range(0, len(MODEL_.model.layers), 3)) + [len(MODEL_.model.layers) - 1]:
        res['fwd'][li] = let(masked_forward(inp, 'plain', window, patch=(li, Sc[li + 1][2]))[0])
        res['rev'][li] = let(masked_forward(inp, 'gate_audio', window, patch=(li, Sp[li + 1][2]))[0])
    del inp; torch.cuda.empty_cache()
    return res


def eager_attention(media, stem, texts, window, dur):
    """per layer: attention mass (summed over question rows, mean over heads) to video first/second half, inside/outside window, audio."""
    inp, _ = qwen_inputs(PROC, render(stem, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]; rows_q = torch.arange(L) > last_media
    sec_c, vi, ai = sec.cpu(), is_v.cpu(), is_a.cpu()
    spg = float(inp['video_second_per_grid'].reshape(-1)[0])
    inside = (sec_c + spg > window[0]) & (sec_c < window[1])
    groups = {'v_first_half': vi & (sec_c < dur / 2), 'v_second_half': vi & (sec_c >= dur / 2), 'v_in': vi & inside, 'v_out': vi & ~inside, 'audio': ai}
    counts = {g: int(m.sum()) for g, m in groups.items()}
    stats = {}
    def mk(li):
        def hook(mod, args, out):
            w = out[1]
            if w is None: return
            dv = w.device; q = w[0][:, rows_q.to(dv)].float().sum(1).mean(0)
            stats[li] = {g: round(float(q[m.to(dv)].sum()), 5) for g, m in groups.items()}
            return (out[0], None)
        return hook
    MODEL_.set_attn_implementation('eager')
    hooks = [MODEL_.model.layers[li].self_attn.register_forward_hook(mk(li)) for li in range(len(MODEL_.model.layers))]
    try:
        with torch.no_grad(): MODEL_(**inp, use_audio_in_video=True)
    finally:
        for h in hooks: h.remove()
        MODEL_.set_attn_implementation('sdpa')
    del inp; torch.cuda.empty_cache()
    return dict(counts=counts, per_layer=[stats[li] for li in sorted(stats)])


def text_only_scores(stem, texts):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': render(stem, texts, perm)}]}]
        text = PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = PROC(text=text, return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
        with torch.no_grad(): lg = MODEL_(**inp).logits[0, -1].float()
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
    return tot.round(3).tolist()


ACTIONS = ['pour the oil into the pan', 'chop the onions', 'add salt to the soup', 'stir the sauce', 'slice the tomatoes',
           'crack the eggs into the bowl', 'mash the potatoes', 'spread butter on the bread', 'fry the bacon', 'mix the flour and water']


TENSES = {'future': 'I will {a}', 'present': 'Now I am going to {a} right now', 'past': 'I have already finished: {a}', 'neutral': '{a}'}


def synthetic_relation():
    """relation (Past/Present/Future) x speech tense; text-only facts, no media. Tests reasoning and the tense shortcut."""
    rng = random.Random(0); out = []
    for tense, tmpl in TENSES.items():
        for n in range(30):
            act = rng.choice(ACTIONS); s0 = round(rng.uniform(10, 20), 1); sd = round(rng.uniform(2, 5), 1)
            rel = n % 3                      # 0 Past, 1 Present, 2 Future
            if rel == 1: e0 = round(s0 + rng.uniform(-0.5, 0.5), 1)
            elif rel == 0: e0 = round(s0 - rng.uniform(6, 9), 1)
            else: e0 = round(s0 + sd + rng.uniform(3, 9), 1)
            ed = round(rng.uniform(2, 5), 1); said = tmpl.format(a=act)
            facts = (f'In a cooking video, a person says "{said}" from {s0} to {round(s0 + sd, 1)} seconds. '
                     f'The action "{act}" is actually shown in the video from {e0} to {round(e0 + ed, 1)} seconds.')
            tf = text_only_scores(facts + ' Relative to the moment when the speech is heard, when does the narrated event occur in the video?', ['Past', 'Present', 'Future'])
            tf_plain = text_only_scores(facts + ' Is the action shown before, during, or after the speech?', ['Before', 'During', 'After'])
            ta = text_only_scores(facts + ' When the speech is heard, is the narrated event happening at the same time in the video?', ['Yes', 'No'])
            out.append(dict(tense=tense, rel=rel, tf=tf, tf_plain=tf_plain, ta=ta, ta_gold=0 if rel == 1 else 1))
    return out


def run_item(task, r, shard_attention):
    orig = fetch(r['video_path'].replace(BS, '/'))
    full = cut(orig, 0, 10 ** 4, 'full')
    a = decode_audio(full); dur = len(a) / 16000
    if dur > MAX_DUR: raise RuntimeError('skipped: longer than MAX_DUR')
    w = speech_window(a)
    if w is None: raise RuntimeError('no speech activity')
    s0, s1 = w; win = (s0, s1)
    media = prepare_video(full)
    stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)])
    # base + hidden states from the first option order
    tot = np.zeros(k)
    for n_perm, perm in enumerate(rotations(k)):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_)
        lg, store = masked_forward(inp, 'plain', win, record_layers=LAYERS_KEEP if n_perm == 0 else None)
        if n_perm == 0:
            np.savez_compressed(HID / f'h_{r["_idx"]}.npz', last=np.stack([store[l][0].numpy() for l in LAYERS_KEEP]).astype('float16'),
                                qmean=np.stack([store[l][1].numpy() for l in LAYERS_KEEP]).astype('float16'), layers=np.array(LAYERS_KEEP))
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
        del inp
    rec['base'] = tot.round(3).tolist()
    rec['para1'] = scores(media, PARA[task][0], texts)
    rec['para2'] = scores(media, PARA[task][1], texts)
    rec['vision_only'] = scores(media, stem, texts, mode='vision')
    rec['audio_only'] = scores(prepare_video(blank_video(full, dur, True)), stem, texts)
    rec['prior'] = scores(prepare_video(blank_video(full, dur, False)), stem, texts)
    rec['hear'] = hear(media)
    try: rec['whisper'] = whisper_text(a, s0, s1)
    except Exception as e: rec['whisper_error'] = repr(e)[:200]
    if task == 'CMTB':
        for mk in ('gate', 'audio_block', 'gate_audio'): rec[mk] = scores(media, stem, texts, mask_kind=mk, window=win)
        rec['crop'] = scores(prepare_video(cut(full, s0, s1, 'speech')), stem, texts, mode='vision')
        g = 'ABC'.index(r['answer'])
        if int(np.argmax(rec['base'])) != g and int(np.argmax(rec['gate_audio'])) == g:
            rec['patching'] = patch_sweep(media, stem, texts, win)
    if shard_attention:
        rec['attention'] = eager_attention(media, stem, texts, win, dur)
    return rec


def main():
    global MODEL_, PROC, LETTERS
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cells = {}
    for i, r in enumerate(rows):
        if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    items = []
    for key in sorted(cells):
        pool = cells[key]; random.Random(1).shuffle(pool); items += pool[:PER_LABEL]
    items.sort(key=lambda x: x[0])
    mine = [x for k, x in enumerate(items) if k % NSHARD == SHARD]
    if os.environ.get('LAB_ONLY_IDS'):
        want = {int(v) for v in os.environ['LAB_ONLY_IDS'].split(',')}; mine = [x for x in enumerate(rows) if x[0] in want]
    res = OUT / f'rootcause_shard{SHARD}of{NSHARD}.jsonl'
    done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
    MODEL_, PROC = load_model(); LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    print('model ready; items', len(mine), 'done', len(done), flush=True)
    syn = OUT / 'synthetic_relation.json'
    if SHARD == 0 and not syn.exists() and not os.environ.get('LAB_NO_SYNTH'):
        syn.write_text(json.dumps(synthetic_relation()), encoding='utf-8'); print('synthetic relation done', flush=True)
    n_att = 0
    for n, (i, r) in enumerate(mine):
        if i in done: continue
        t0 = time.time(); task = r['task_type_abbr']
        rec = dict(idx=i, task=task, gold='ABC'.index(r['answer']))
        try:
            r = dict(r); r['_idx'] = i
            att = SHARD == 0 and n_att < 24
            rec.update(run_item(task, r, att)); n_att += int(att); rec['sec'] = round(time.time() - t0)
        except Exception as e:
            rec['error'] = repr(e)[:300]
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        if 'error' in rec: print(n + 1, '/', len(mine), task, i, rec['error'][:60], flush=True); continue
        g = rec['gold']; ok = lambda c: 'Y' if c in rec and int(np.argmax(rec[c])) == g else '-'
        print(n + 1, '/', len(mine), task, i, 'base', ok('base'), 'p1', ok('para1'), 'p2', ok('para2'), 'vis', ok('vision_only'), 'aud', ok('audio_only'),
              'GA', ok('gate_audio'), 'patch' if 'patching' in rec else '', rec['sec'], 's', flush=True)


if __name__ == '__main__':
    main()

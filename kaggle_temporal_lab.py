"""Temporal method lab (sharded). Usage:
  python kaggle_temporal_lab.py ROOT OUT SHARD_ID NUM_SHARDS [PER_LABEL]
Every condition is scored under rotated option orders and summed per option CONTENT (letter position cancels).

Conditions (Qwen2.5-Omni-3B, fp16, one forward per option order, no training):
  base         whole clip, audio+video, original question
  ts05, ts02   TMRoPE temporal ids of media tokens compressed by 0.5 / 0.2 (text after the media shifted to stay contiguous)
               -> tests whether the sparse absolute-time positions cause a position bias
  gate_all     no position may read video tokens outside the audio-active (speech) window, all layers
  gate_late    same mask only in layers >= LATE_FROM (default layers are untouched)
  audio_block  (CMTB only) no non-audio position may read audio tokens
  gate_audio   (CMTB only) gate_all + audio_block
  crop         (CMTB only) only the speech-window frames, video only (upper-bound reference from the earlier run)
  swap_base / swap_gate   the speech segment and the other segment exchanged in time (each keeps its own audio).
               Gold: TA and CMTB unchanged; TF Past <-> Future, Present unchanged.
Speech window = span of 50 ms frames above an RMS threshold of the clip itself. Labels are used only for scoring."""
import os, sys, json, re, subprocess, time
from pathlib import Path
import numpy as np
import torch

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SHARD, NSHARD = int(sys.argv[3]), int(sys.argv[4])
PER_LABEL = int(sys.argv[5]) if len(sys.argv) > 5 else 0     # items per (task, answer label); 0 = the whole benchmark
MAX_DUR = float(os.environ.get('LAB_MAX_DUR', 42.0 if torch.cuda.device_count() >= 2 else 28.0))
LATE_FROM = 18
sys.path.insert(0, str(ROOT))
import av, librosa
from huggingface_hub import hf_hub_download

DATASET, REVISION = 'zcs15/SVHalluc', '211b4f14ab66ae9557b69e302d1f090e0e610839'
MODEL, MREV = 'Qwen/Qwen2.5-Omni-3B', 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
BS = chr(92)


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

from lab_native_video import prepare_video, qwen_inputs, move_inputs
from speech_gate import media_layout, gated_mask
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

CLIPS = OUT / 'clips'; CLIPS.mkdir(parents=True, exist_ok=True)
FF = os.environ.get('FFMPEG_BIN', 'ffmpeg')


def fetch(name):
    local = os.environ.get('LAB_LOCAL_DATA')    # local smoke test only: an existing copy of the dataset
    if local and (Path(local) / name).exists(): return Path(local) / name
    return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def ffmpeg(*args): subprocess.run([FF, '-hide_banner', '-y', *args], check=True, capture_output=True)


def speech_window(a):
    hop = 800; n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    act = db > lo + 0.5 * (hi - lo)
    if act.sum() == 0: return None
    t = (np.arange(n) + 0.5) * 0.05; idx = np.where(act)[0]
    return float(t[idx[0]]), float(t[idx[-1]])


ENC = ['-vf', 'fps=30', '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac']


def cut(src, start, end, tag):
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists():
        ffmpeg('-ss', f'{start:.2f}', '-to', f'{end:.2f}', '-i', str(src), '-map', '0:v:0', '-map', '0:a:0', *ENC, '-af', 'aresample=async=1:first_pts=0', str(dst))
    return dst


def concat(a, b, tag, stem):
    dst = CLIPS / f'{stem}_{tag}.mp4'
    if not dst.exists():
        ffmpeg('-i', str(a), '-i', str(b), '-filter_complex', '[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]',
               '-map', '[v]', '-map', '[a]', '-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(dst))
    return dst


def parse_question(q):
    parts = re.split(r'\s*\(([ABC])\)\s*', q)
    return parts[0].strip(), [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


def rotations(k): return [tuple((s + j) % k for j in range(k)) for s in range(k)]


def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))


MODEL_ = PROC = LETTERS = None


def load_model():
    cache = os.environ.get('LAB_CACHE')       # local smoke test only
    kw = dict(revision=MREV, torch_dtype=torch.float16, attn_implementation='sdpa', cache_dir=cache)
    if os.environ.get('LAB_NF4'):              # local smoke test on a 4 GB card
        from transformers import BitsAndBytesConfig
        kw.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                  bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    elif torch.cuda.device_count() >= 2: kw.update(device_map='balanced', max_memory={0: '10GiB', 1: '10GiB'})
    else: kw.update(device_map={'': 0})
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    return m, Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=MREV, cache_dir=cache)


def causal4d(L, dtype, device):
    m = torch.full((L, L), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
    return m.view(1, 1, L, L)


def compressed_positions(pos, is_v, is_a, last_media, kappa):
    pos = pos.clone(); media = (is_v | is_a).to(pos.device)
    t = pos[0, 0]
    base = t[media].min()
    old_max = int(t[media].max())
    t_new = t.clone()
    t_new[media] = base + torch.round((t[media] - base).float() * kappa).long()
    shift = old_max - int(t_new[media].max())
    after = torch.arange(pos.shape[-1], device=pos.device) > last_media
    pos[0, 0] = t_new
    pos[:, :, after] -= shift
    return pos


def forward_last(inp, kind, window=None, kappa=1.0):
    emb = MODEL_.get_input_embeddings().weight
    L = inp['input_ids'].shape[1]
    if kind == 'plain':
        with torch.no_grad():
            return MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    spg = inp.get('video_second_per_grid'); spg = float(spg.reshape(-1)[0]) if spg is not None else None
    if kind == 'ts': pos = compressed_positions(pos, is_v, is_a, last_media, kappa)
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    hooks = []
    if kind == 'ts':
        mask = causal4d(L, emb.dtype, emb.device)
    elif kind in ('gate_all', 'audio_block', 'gate_audio'):
        mode = 'none' if kind == 'audio_block' else 'hard'
        mask = gated_mask(L, is_v, sec, last_media, window, mode, 0.0, emb.dtype, emb.device, spg, 'all', is_a, kind != 'gate_all')
    elif kind == 'gate_late':
        mask = causal4d(L, emb.dtype, emb.device)
        gm = gated_mask(L, is_v, sec, last_media, window, 'hard', 0.0, emb.dtype, emb.device, spg, 'all')
        def pre(mod, args, kwargs):
            am = kwargs.get('attention_mask')
            kwargs['attention_mask'] = gm.to(am.device if am is not None else gm.device)
            return args, kwargs
        for li in range(LATE_FROM, len(MODEL_.model.layers)):
            hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    else:
        raise ValueError(kind)
    try:
        with torch.no_grad():
            out = MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True)
    finally:
        for h in hooks: h.remove()
    return out.logits[0, -1].float()


def scores(media, stem, texts, kind='plain', window=None, kappa=1.0, media_mode='av'):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, media_mode)
        inp = move_inputs(inp, MODEL_)
        if media_mode != 'av':
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=False).logits[0, -1].float()
        else:
            lg = forward_last(inp, kind, window, kappa)
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
        del inp
    torch.cuda.empty_cache()
    return tot.round(3).tolist()


def run_item(task, r):
    orig = fetch(r['video_path'].replace(BS, '/'))
    full = cut(orig, 0, 10 ** 4, 'full')
    a = decode_audio(full); dur = len(a) / 16000
    if dur > MAX_DUR: raise RuntimeError('skipped: longer than MAX_DUR')
    w = speech_window(a)
    if w is None: raise RuntimeError('no speech activity')
    s0, s1 = w
    stem, texts = parse_question(r['question'])
    rec = dict(task=task, gold='ABC'.index(r['answer']), dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)])
    media = prepare_video(full)
    rec['base'] = scores(media, stem, texts)
    rec['ts05'] = scores(media, stem, texts, 'ts', kappa=0.5)
    rec['ts02'] = scores(media, stem, texts, 'ts', kappa=0.2)
    rec['gate_all'] = scores(media, stem, texts, 'gate_all', (s0, s1))
    rec['gate_late'] = scores(media, stem, texts, 'gate_late', (s0, s1))
    if task == 'CMTB':
        rec['audio_block'] = scores(media, stem, texts, 'audio_block', (s0, s1))
        rec['gate_audio'] = scores(media, stem, texts, 'gate_audio', (s0, s1))
        rec['crop'] = scores(prepare_video(cut(orig, s0, s1, 'speech')), stem, texts, media_mode='vision')
    # segment swap: speech segment <-> the rest (only when the rest lies on one side)
    before, after = s0, dur - s1
    if min(before, after) < 1.0 and max(before, after) >= 1.0:
        seg_s = cut(orig, s0, s1, 'segS')
        if before >= 1.0:
            seg_o = cut(orig, 0, s0, 'segO'); swapped = concat(seg_s, seg_o, 'swap', Path(orig).stem); rec['speech_first_in_orig'] = False
        else:
            seg_o = cut(orig, s1, dur, 'segO'); swapped = concat(seg_o, seg_s, 'swap', Path(orig).stem); rec['speech_first_in_orig'] = True
        # window of the swapped clip from the known segment lengths (re-detecting it by energy is unreliable at the clip start)
        seg_len = s1 - s0
        sw = (0.0, seg_len) if before >= 1.0 else (dur - s1, dur - s1 + seg_len)
        if True:
            m2 = prepare_video(swapped)
            rec['swap_win'] = [round(sw[0], 2), round(sw[1], 2)]
            rec['swap_base'] = scores(m2, stem, texts)
            rec['swap_gate'] = scores(m2, stem, texts, 'gate_all', sw)
            rec['swap_gold'] = {0: 2, 2: 0}.get(rec['gold'], rec['gold']) if task == 'TF' else rec['gold']
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# Dry-run trace: input processing -> tokens -> positions -> logits per layer -> attention, for a few items.
# Everything is written to trace/trace_<idx>.json. Failures here never stop the main run.
# ----------------------------------------------------------------------------------------------------------------------
def runs_of_types(n, is_v, is_a, sec, pos_t):
    """Run-length encode the token sequence into text / video / audio runs with their time and TMRoPE ranges."""
    kind = lambda j: 'video' if bool(is_v[j]) else ('audio' if bool(is_a[j]) else 'text')
    out = []; start = 0
    for j in range(1, n + 1):
        if j == n or kind(j) != kind(start):
            seg = dict(kind=kind(start), start=start, end=j - 1, n=j - start,
                       rope_t_min=int(pos_t[start:j].min()), rope_t_max=int(pos_t[start:j].max()))
            if seg['kind'] != 'text':
                seg.update(sec_min=round(float(sec[start:j].min()), 3), sec_max=round(float(sec[start:j].max()), 3))
            out.append(seg); start = j
    return out


def topk_tokens(lg, k=8):
    v, i = torch.topk(torch.log_softmax(lg.float(), -1), k)
    return [dict(token=PROC.tokenizer.decode([int(t)]), id=int(t), logprob=round(float(x), 3)) for x, t in zip(v, i)]


def trace_item(task, r):
    orig = fetch(r['video_path'].replace(BS, '/'))
    full = cut(orig, 0, 10 ** 4, 'full')
    a = decode_audio(full); dur = len(a) / 16000
    if dur > MAX_DUR: return None
    w = speech_window(a)
    if w is None: return None
    s0, s1 = w
    media = prepare_video(full)
    stem, texts = parse_question(r['question']); k = len(texts)
    prompt = render(stem, texts, tuple(range(k)))
    inp, rendered = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    L = inp['input_ids'].shape[1]
    tr = dict(task=task, question=r['question'], gold_letter=r['answer'], prompt=prompt)
    # ---- 1. input processing ---------------------------------------------------------------------------------------
    vshape = list(media['videos'][0].shape)
    tr['preprocess'] = dict(
        file_duration_s=round(dur, 3), speech_window_s=[round(s0, 3), round(s1, 3)], sampled_fps=media['fps'],
        video_tensor_shape_TCHW=vshape, n_frames=vshape[0], frame_times_s=[round(i / media['fps'], 3) for i in range(vshape[0])],
        audio_samples=int(len(media['audio'][0])), audio_seconds=round(len(media['audio'][0]) / 16000, 3),
        max_pixels_per_frame=media['max_pixels_per_frame'], video_grid_thw=inp['video_grid_thw'].tolist(),
        second_per_grid=inp['video_second_per_grid'].reshape(-1).tolist(),
        audio_feature_frames=int(inp['feature_attention_mask'].sum()), input_features_shape=list(inp['input_features'].shape),
        n_input_tokens=L, chat_template_head=rendered[:700], chat_template_tail=rendered[-400:])
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    ids = inp['input_ids'][0].cpu(); pos_t = pos[0, 0].cpu(); sec_c = sec.cpu(); vi = is_v.cpu(); ai = is_a.cpu()
    tr['tokens'] = dict(n_video=int(vi.sum()), n_audio=int(ai.sum()), n_text=int(L - vi.sum() - ai.sum()), last_media_index=last_media,
                        runs=runs_of_types(L, vi, ai, sec_c, pos_t),
                        text_tokens_after_media=[PROC.tokenizer.decode([int(t)]) for t in ids[last_media + 1:]][:300])
    spg = float(inp['video_second_per_grid'].reshape(-1)[0])
    inside = ((sec_c + spg > s0) & (sec_c < s1))
    tr['window_vs_tokens'] = dict(video_tokens_inside=int((vi & inside).sum()), video_tokens_outside=int((vi & ~inside).sum()),
                                  video_tokens_per_second_bin=[int((vi & (sec_c >= b) & (sec_c < b + 1)).sum()) for b in range(int(dur) + 1)])
    # ---- 2. logits for several conditions ---------------------------------------------------------------------------
    with torch.no_grad():
        plain = MODEL_(**inp, use_audio_in_video=True, output_hidden_states=True)
    lg_plain = plain.logits[0, -1].float()
    conds = {'plain': lg_plain,
             'explicit_ids_and_mask_kappa1 (must equal plain)': forward_last(inp, 'ts', kappa=1.0),
             'ts05': forward_last(inp, 'ts', kappa=0.5), 'ts02': forward_last(inp, 'ts', kappa=0.2),
             'gate_all': forward_last(inp, 'gate_all', (s0, s1)), 'gate_late': forward_last(inp, 'gate_late', (s0, s1))}
    if task == 'CMTB':
        conds['audio_block'] = forward_last(inp, 'audio_block', (s0, s1))
        conds['gate_audio'] = forward_last(inp, 'gate_audio', (s0, s1))
    tr['logits'] = {name: dict(letters={L_: round(float(lg[LETTERS[j]]), 3) for j, L_ in enumerate('ABC'[:k])}, top8=topk_tokens(lg),
                               max_abs_diff_vs_plain=round(float((lg.cpu() - lg_plain.cpu()).abs().max()), 4)) for name, lg in conds.items()}
    # ---- 3. logit lens at the last position (plain forward) ----------------------------------------------------------
    hs = plain.hidden_states; head = MODEL_.lm_head; norm = MODEL_.model.norm; lens = []
    with torch.no_grad():
        for li, h in enumerate(hs):
            x = h[0, -1].to(head.weight.device, head.weight.dtype)
            if li < len(hs) - 1: x = norm(x.to(norm.weight.device)).to(head.weight.device)
            lg = head(x).float()
            lens.append(dict(layer=li, letters=[round(float(lg[LETTERS[j]]), 3) for j in range(k)],
                             top1=PROC.tokenizer.decode([int(lg.argmax())]), hidden_norm=round(float(h[0, -1].float().norm()), 2)))
    tr['logit_lens_plain'] = lens
    del plain, hs
    torch.cuda.empty_cache()
    # ---- 4. attention (eager) -----------------------------------------------------------------------------------------
    try:
        MODEL_.set_attn_implementation('eager')
        rows_q = torch.arange(L) > last_media
        vin, vout = (vi & inside), (vi & ~inside)
        bins = [(vi & (sec_c >= b) & (sec_c < b + 1)) for b in range(int(dur) + 1)]
        stats = {}
        def mk(li):
            def hook(mod, args, out):
                wts = out[1]
                if wts is None: return
                dv = wts.device
                qa = wts[0][:, rows_q.to(dv)].float().sum(1); ql = wts[0][:, -1].float()
                rec = dict(video_in=qa[:, vin.to(dv)].sum(-1).cpu().tolist(), video_out=qa[:, vout.to(dv)].sum(-1).cpu().tolist(),
                           audio=qa[:, ai.to(dv)].sum(-1).cpu().tolist(), total=qa.sum(-1).cpu().tolist(),
                           last_video_in=ql[:, vin.to(dv)].sum(-1).cpu().tolist(), last_video_out=ql[:, vout.to(dv)].sum(-1).cpu().tolist(),
                           last_audio=ql[:, ai.to(dv)].sum(-1).cpu().tolist())
                if li in (8, 17, 26, 35):
                    rec['last_row_mass_per_second_bin_mean_over_heads'] = [round(float(ql[:, m.to(dv)].sum(-1).mean()), 5) for m in bins]
                stats[li] = rec
                return (out[0], None)
            return hook
        hooks = [MODEL_.model.layers[li].self_attn.register_forward_hook(mk(li)) for li in range(len(MODEL_.model.layers))]
        try:
            with torch.no_grad(): MODEL_(**inp, use_audio_in_video=True)
        finally:
            for h in hooks: h.remove()
            MODEL_.set_attn_implementation('sdpa')
        tr['attention'] = dict(n_video_in=int(vin.sum()), n_video_out=int(vout.sum()), n_audio=int(ai.sum()),
                               note='per layer and head: attention mass summed over the question rows (video_in/out, audio, total) and from the last position only (last_*)',
                               per_layer=[stats[li] for li in sorted(stats)])
    except Exception as e:
        try: MODEL_.set_attn_implementation('sdpa')
        except Exception: pass
        tr['attention'] = dict(error=repr(e)[:300])
    return tr


def run_traces(mine):
    out_dir = OUT / 'trace'; out_dir.mkdir(exist_ok=True)
    per_task = {}
    for i, r in mine:
        t = r['task_type_abbr']
        if per_task.get(t, 0) >= 2: continue
        f = out_dir / f'trace_{i}.json'
        if f.exists():
            per_task[t] = per_task.get(t, 0) + 1; continue
        try:
            tr = trace_item(t, r)
            if tr is None: continue
            tr['idx'] = i; f.write_text(json.dumps(tr), encoding='utf-8')
            per_task[t] = per_task.get(t, 0) + 1
            print('trace written', f.name, flush=True)
        except Exception as e:
            print('trace failed', i, repr(e)[:200], flush=True)
        torch.cuda.empty_cache()


def main():
    global MODEL_, PROC, LETTERS
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    import random
    items = [(i, r) for i, r in enumerate(rows) if r['data_type'] == 'temporal']
    if PER_LABEL:                       # stratified pilot; the same seeded order per cell, so a bigger PER_LABEL extends the smaller set
        cells = {}
        for x in items: cells.setdefault((x[1]['task_type_abbr'], x[1]['answer']), []).append(x)
        items = []
        for key in sorted(cells):
            pool = cells[key]; random.Random(0).shuffle(pool); items += pool[:PER_LABEL]
        items.sort(key=lambda x: x[0])
    mine = [x for k, x in enumerate(items) if k % NSHARD == SHARD]
    res = OUT / f'lab_shard{SHARD}of{NSHARD}.jsonl'
    done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
    MODEL_, PROC = load_model()
    LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    print('model ready; shard', SHARD, 'of', NSHARD, 'items', len(mine), 'done', len(done), flush=True)
    if SHARD == 0 and not os.environ.get('LAB_NO_TRACE'): run_traces(mine)
    for k, (i, r) in enumerate(mine):
        if i in done: continue
        t0 = time.time(); task = r['task_type_abbr']
        try:
            rec = run_item(task, r); rec['sec'] = round(time.time() - t0)
        except Exception as e:
            rec = dict(task=task, gold='ABC'.index(r['answer']), error=repr(e)[:300])
        rec['idx'] = i
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        if 'error' in rec: print(k + 1, '/', len(mine), task, i, rec['error'][:70], flush=True)
        else:
            g = rec['gold']; ok = lambda c: int(np.argmax(rec[c])) == g
            print(k + 1, '/', len(mine), task, i, 'base', ok('base'), 'ts02', ok('ts02'), 'gate', ok('gate_all'), 'late', ok('gate_late'), rec['sec'], 's', flush=True)


if __name__ == '__main__':
    main()

"""Stage 2: does the time x modality binding fix survive without the digital-silence artifact?
Usage: python kaggle_temporal_stage2.py ROOT OUT PER_LABEL [SHARD_ID NUM_SHARDS]

Every item is evaluated on two versions of the same clip:
  orig  : the benchmark clip (the non-speech part is digitally silent)
  noisy : digital silence replaced by low-level room-like noise (about 25 dB below speech); speech window re-estimated
Conditions (all option orders rotated, scores summed per option CONTENT; single model, no training, no tools):
  base          whole clip, audio+video, original question
  gate          video tokens outside the speech window hidden from every position
  audio_block   audio tokens hidden from every non-audio position (no window gate)
  gate_audio    both                                   -> the 2x2 factorial (time gate x audio block)
  crop          (CMTB) only the speech-window frames, video only (reference)
  TA/TF only:
  hear          the same model, audio only: "what action does the speaker describe?" (short text, greedy)
  look_*        for each window W (speech / before / after): video outside W hidden AND audio hidden, question
                "Is the following action happening in this video: <hear text>?" -> Yes-minus-No margin (2 orders)
  htl           hear-then-look decision: Present / Yes iff the speech window has the largest margin; otherwise the window
                with the largest margin gives Past (before) or Future (after)
  lac           the same comparison but WITHOUT hiding audio and with the generic "narrated event" question (previous study)
Labels are used only for scoring."""
import os, sys, json, re, subprocess, time, random
from pathlib import Path
import numpy as np
import torch

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PER_LABEL = int(sys.argv[3])
SHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NSHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 1
MAX_DUR = float(os.environ.get('LAB_MAX_DUR', 42.0 if torch.cuda.device_count() >= 2 else 28.0))
sys.path.insert(0, str(ROOT))
import av, librosa, soundfile as sf
from huggingface_hub import hf_hub_download

DATASET, REVISION = 'zcs15/SVHalluc', '211b4f14ab66ae9557b69e302d1f090e0e610839'
MODEL, MREV = 'Qwen/Qwen2.5-Omni-3B', 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
BS = chr(92)
FF = os.environ.get('FFMPEG_BIN', 'ffmpeg')


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
ENC = ['-vf', 'fps=30', '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac']


def fetch(name):
    local = os.environ.get('LAB_LOCAL_DATA')
    if local and (Path(local) / name).exists(): return Path(local) / name
    return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def ffmpeg(*args): subprocess.run([FF, '-hide_banner', '-y', *args], check=True, capture_output=True)


def cut(src, start, end, tag):
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists():
        ffmpeg('-ss', f'{start:.2f}', '-to', f'{end:.2f}', '-i', str(src), '-map', '0:v:0', '-map', '0:a:0', *ENC, '-af', 'aresample=async=1:first_pts=0', str(dst))
    return dst


def noisy(src):
    dst = CLIPS / f'{Path(src).stem}_noisy.mp4'
    if not dst.exists():
        a = decode_audio(src).copy()
        zero = np.abs(a) < 1e-6
        speech_rms = float(np.sqrt((a[~zero] ** 2).mean())) if (~zero).any() else 0.05
        rng = np.random.default_rng(0)
        white = rng.standard_normal(len(a)).astype('float32')
        brown = np.cumsum(white); brown -= np.convolve(brown, np.ones(801) / 801, 'same')
        brown = brown / (brown.std() + 1e-9) * speech_rms * 10 ** (-25 / 20)
        a[zero] = brown[zero]
        wav = CLIPS / f'{Path(src).stem}_noisy.wav'; sf.write(wav, a, 16000)
        ffmpeg('-i', str(src), '-i', str(wav), '-map', '0:v:0', '-map', '1:a:0', *ENC, '-shortest', str(dst))
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


def last_logits(inp, window=None, time_gate=False, audio_block=False):
    if not time_gate and not audio_block:
        with torch.no_grad(): return MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    spg = inp.get('video_second_per_grid'); spg = float(spg.reshape(-1)[0]) if spg is not None else None
    L = inp['input_ids'].shape[1]
    mask = gated_mask(L, is_v, sec, last_media, window, 'hard' if time_gate else 'none', 0.0, emb.dtype, emb.device, spg, 'all', is_a, audio_block)
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    with torch.no_grad():
        return MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()


def scores(media, stem, texts, window=None, time_gate=False, audio_block=False, media_mode='av'):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, media_mode); inp = move_inputs(inp, MODEL_)
        if media_mode != 'av':
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=False).logits[0, -1].float()
        else:
            lg = last_logits(inp, window, time_gate, audio_block)
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
        del inp
    torch.cuda.empty_cache()
    return tot.round(3).tolist()


HEAR_Q = 'What action does the speaker say is being done? Answer with a short phrase (verb and object) only.'


def hear(media):
    inp, _ = qwen_inputs(PROC, HEAR_Q, media, 'audio'); inp = move_inputs(inp, MODEL_)
    with torch.no_grad():
        o = MODEL_.generate(**inp, max_new_tokens=16, do_sample=False)
    t = PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    t = t.strip().split('\n')[0].strip().strip('."\'')
    return t[:120] if t else 'the narrated action'


def windows_of(s0, s1, dur):
    w = {'speech': (s0, s1)}
    if s0 >= 1.0: w['before'] = (0.0, s0)
    if dur - s1 >= 1.0: w['after'] = (s1, dur)
    return w


def compare_decision(task, margins):
    others = {k: v for k, v in margins.items() if k != 'speech'}
    present = margins['speech'] >= (max(others.values()) if others else 0.0)
    if task == 'TA': return 0 if present else 1
    if present: return 1
    return 0 if max(others, key=others.get) == 'before' else 2


def run_version(task, r, src, tag):
    a = decode_audio(src); dur = len(a) / 16000
    w = speech_window(a)
    if w is None: raise RuntimeError('no speech activity')
    s0, s1 = w
    media = prepare_video(src)
    stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)])
    rec['base'] = scores(media, stem, texts)
    rec['gate'] = scores(media, stem, texts, (s0, s1), time_gate=True)
    rec['audio_block'] = scores(media, stem, texts, (s0, s1), audio_block=True)
    rec['gate_audio'] = scores(media, stem, texts, (s0, s1), time_gate=True, audio_block=True)
    if task == 'CMTB':
        rec['crop'] = scores(prepare_video(cut(src, s0, s1, tag + '_speech')), stem, texts, media_mode='vision')
    else:
        wins = windows_of(s0, s1, dur)
        e = hear(media); rec['hear'] = e
        look_q = f'Is the following action happening in this video: {e}?'
        look, lac = {}, {}
        for name, ww in wins.items():
            s = scores(media, look_q, ['Yes', 'No'], ww, time_gate=True, audio_block=True); look[name] = float(s[0] - s[1])
            s = scores(media, 'Is the narrated event shown in the video?', ['Yes', 'No'], ww, time_gate=True); lac[name] = float(s[0] - s[1])
        rec['look_margins'] = look; rec['htl_pred'] = compare_decision(task, look)
        rec['lac_margins'] = lac; rec['lac_pred'] = compare_decision(task, lac)
    return rec


def main():
    global MODEL_, PROC, LETTERS
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cells = {}
    for i, r in enumerate(rows):
        if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    items = []
    for key in sorted(cells):
        pool = cells[key]; random.Random(0).shuffle(pool); items += pool[:PER_LABEL]
    items.sort(key=lambda x: x[0])
    mine = [x for k, x in enumerate(items) if k % NSHARD == SHARD]
    if os.environ.get('LAB_ONLY_IDS'):                       # tests only
        want = {int(v) for v in os.environ['LAB_ONLY_IDS'].split(',')}
        mine = [x for x in enumerate(rows) if x[0] in want]
    res = OUT / f'stage2_shard{SHARD}of{NSHARD}.jsonl'
    done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
    MODEL_, PROC = load_model()
    LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    print('model ready; items', len(mine), 'done', len(done), flush=True)
    for k, (i, r) in enumerate(mine):
        if i in done: continue
        t0 = time.time(); task = r['task_type_abbr']
        rec = dict(idx=i, task=task, gold='ABC'.index(r['answer']))
        try:
            orig = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
            if len(decode_audio(orig)) / 16000 > MAX_DUR: raise RuntimeError('skipped: longer than MAX_DUR')
            rec['orig'] = run_version(task, r, orig, 'orig')
            rec['noisy'] = run_version(task, r, noisy(orig), 'noisy')
            rec['sec'] = round(time.time() - t0)
        except Exception as e:
            rec['error'] = repr(e)[:300]
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        if 'error' in rec: print(k + 1, '/', len(mine), task, i, rec['error'][:70], flush=True); continue
        g = rec['gold']; am = lambda v: int(np.argmax(v))
        msg = ' '.join(f"{v}:{'base' if am(rec[v]['base']) == g else '-'}/{'GA' if am(rec[v]['gate_audio']) == g else '-'}" + (f"/{'HTL' if rec[v].get('htl_pred') == g else '-'}" if 'htl_pred' in rec[v] else '') for v in ('orig', 'noisy'))
        print(k + 1, '/', len(mine), task, i, msg, rec['sec'], 's', flush=True)


if __name__ == '__main__':
    main()

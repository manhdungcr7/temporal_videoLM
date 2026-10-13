"""SVHalluc temporal study, run in the Kaggle venv by the notebook.
Usage: python kaggle_temporal_study.py ROOT OUT PARTS N_PER_LABEL N_NOISE_PER_LABEL
PARTS: comma list of loc, main, noise.

loc   : does the model know WHEN the speech is? (paper's timestamp prompt; first/second-half forced choice)
main  : TA / TF / CMTB on the original clips. Every condition is scored under rotated option orders and summed per
        option CONTENT (letter position cancels).
          base      : whole clip, audio+video, original question, plain forward
          gate_hard : same input; NO position (text, audio, in-window video) may read video tokens outside the speech
                      window (attention-level crop; positions, audio and sequence unchanged)
          gate_soft : same, with a -BETA bias instead of -inf
          gate_text : only the question/answer positions are blocked (media tokens may still mix in the other frames)
          lac       : (TA/TF) look-and-compare. Hard-gate to each window (speech / before / after), ask
                      "Is the narrated event shown in the video?", compare Yes-margins across windows.
noise : same items with the digitally silent part replaced by low-level noise (removes the 'perfect silence'
        artifact); window re-estimated from the noisy audio. base, gate_hard, lac.
Speech window = span of 50 ms frames above an RMS threshold computed from the clip itself (no ASR, no labels).
Labels are used only for scoring."""
import os, sys, json, re, subprocess, time, random
from pathlib import Path
import numpy as np
import torch

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PARTS = set(sys.argv[3].split(','))
N, N_NOISE = int(sys.argv[4]), int(sys.argv[5])
MAX_DUR = 42.0 if torch.cuda.device_count() >= 2 else 28.0   # T4 x2 ran out of memory above ~47 s; a single 16 GB GPU gets a lower cap
BETA = 3.0
sys.path.insert(0, str(ROOT))
import av, librosa, soundfile as sf
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
from speech_gate import gated_last_logits
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

CLIPS = OUT / 'clips'; CLIPS.mkdir(parents=True, exist_ok=True)


def fetch(name): return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def speech_window(a):
    hop = 800; n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    act = db > lo + 0.5 * (hi - lo)
    if act.sum() == 0: return None
    t = (np.arange(n) + 0.5) * 0.05; idx = np.where(act)[0]
    return float(t[idx[0]]), float(t[idx[-1]])


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-y', *args], check=True, capture_output=True)


def normalised(src, tag='full'):
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists():
        ffmpeg('-i', str(src), '-map', '0:v:0', '-map', '0:a:0', '-vf', 'fps=30', '-fps_mode', 'cfr', '-c:v', 'libx264',
               '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-af', 'aresample=async=1:first_pts=0', str(dst))
    return dst


def noisy(src):
    """Replace digital silence by low-level noise (about 25 dB below the speech level)."""
    dst = CLIPS / f'{Path(src).stem}_noisy.mp4'
    if not dst.exists():
        a = decode_audio(src).copy()
        zero = np.abs(a) < 1e-6
        speech_rms = float(np.sqrt((a[~zero] ** 2).mean())) if (~zero).any() else 0.05
        rng = np.random.default_rng(0)
        white = rng.standard_normal(len(a)).astype('float32')
        brown = np.cumsum(white); brown -= np.convolve(brown, np.ones(801) / 801, 'same')   # low-passed, room-like
        brown = brown / (brown.std() + 1e-9) * speech_rms * 10 ** (-25 / 20)
        a[zero] = brown[zero]
        wav = CLIPS / f'{Path(src).stem}_noisy.wav'; sf.write(wav, a, 16000)
        ffmpeg('-i', str(src), '-i', str(wav), '-map', '0:v:0', '-map', '1:a:0', '-vf', 'fps=30', '-fps_mode', 'cfr',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(dst))
    return dst


def parse_question(q):
    parts = re.split(r'\s*\(([ABC])\)\s*', q)
    stem = parts[0].strip(); opts = [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]
    return stem, opts


def rotations(k): return [tuple((s + j) % k for j in range(k)) for s in range(k)]


def render(stem, texts, perm):
    return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))


def load_model():
    kw = dict(revision=MREV, torch_dtype=torch.float16, attn_implementation='sdpa')
    if torch.cuda.device_count() >= 2: kw.update(device_map='balanced', max_memory={0: '10GiB', 1: '10GiB'})
    else: kw.update(device_map={'': 0})        # single 16 GB GPU (P100): ~9 GB of fp16 weights
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    return m, Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=MREV)


MODEL_, PROC = None, None
LETTERS = None


def content_scores(media, stem, texts, mode='plain', window=None, beta=0.0, media_mode='av', scope='all'):
    """Sum of per-option log-probs over all rotations, indexed by option content."""
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, media_mode)
        inp = move_inputs(inp, MODEL_)
        if mode == 'plain':
            with torch.no_grad():
                lg = MODEL_(**inp, use_audio_in_video=(media_mode == 'av')).logits[0, -1].float()
        else:
            lg, _ = gated_last_logits(MODEL_, PROC, inp, window, mode, beta, scope=scope)
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
        del inp
    torch.cuda.empty_cache()
    return tot


EVENT_Q = 'Is the narrated event shown in the video?'


def lac(media, windows):
    """Yes-minus-No margin (both orders) for the narrated event, hard-gated to each window."""
    out = {}
    for name, w in windows.items():
        s = content_scores(media, EVENT_Q, ['Yes', 'No'], 'hard', w)
        out[name] = float(s[0] - s[1])
    return out


def lac_decision(task, margins):
    others = {k: v for k, v in margins.items() if k != 'speech'}
    present = margins['speech'] >= (max(others.values()) if others else 0.0)
    if task == 'TA': return 0 if present else 1                       # A=Yes, B=No
    if present: return 1                                              # TF: A=Past, B=Present, C=Future
    best = max(others, key=others.get)
    return 0 if best == 'before' else 2


def run_item(task, r, src, tag):
    a = decode_audio(src); dur = len(a) / 16000
    w = speech_window(a)
    if w is None: raise RuntimeError('no speech activity')
    s0, s1 = w
    media = prepare_video(src)
    stem, texts = parse_question(r['question'])
    rec = dict(task=task, gold='ABC'.index(r['answer']), dur=round(dur, 1), win=[round(s0, 2), round(s1, 2)], variant=tag)
    rec['base'] = content_scores(media, stem, texts).tolist()
    rec['gate_hard'] = content_scores(media, stem, texts, 'hard', (s0, s1)).tolist()
    if tag == 'orig':
        rec['gate_soft'] = content_scores(media, stem, texts, 'soft', (s0, s1), BETA).tolist()
        rec['gate_text'] = content_scores(media, stem, texts, 'hard', (s0, s1), scope='text').tolist()
    if task in ('TA', 'TF'):
        wins = {'speech': (s0, s1)}
        if s0 >= 1.0: wins['before'] = (0.0, s0)
        if dur - s1 >= 1.0: wins['after'] = (s1, dur)
        m = lac(media, wins); rec['lac_margins'] = m; rec['lac_pred'] = lac_decision(task, m)
    return rec


def part_loc(rows):
    path_rows = sorted({r['video_path'].replace(BS, '/') for r in rows if r['data_type'] == 'temporal'})
    random.Random(0).shuffle(path_rows)
    res = OUT / 'loc.jsonl'
    done = {json.loads(l)['path'] for l in res.read_text().splitlines()} if res.exists() else set()
    Q_TS = 'When is the speech heard in the audio? Provide the start and end timestamps (in seconds).'
    Q_HALF = 'Is the speech heard in the first half or in the second half of the video?'
    n_ok = 0
    for p in path_rows:
        if n_ok >= 40: break
        if p in done: n_ok += 1; continue
        try:
            src = normalised(fetch(p)); a = decode_audio(src); dur = len(a) / 16000
            if dur > MAX_DUR: continue
            w = speech_window(a); media = prepare_video(src)
            inp, _ = qwen_inputs(PROC, Q_TS, media, 'av'); inp = move_inputs(inp, MODEL_)
            with torch.no_grad():
                o = MODEL_.generate(**inp, max_new_tokens=40, do_sample=False, use_audio_in_video=True)
            text = PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            v = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', text)]
            half = content_scores(media, Q_HALF, ['first half', 'second half'])
            rec = dict(path=p, dur=round(dur, 1), ref=[round(w[0], 2), round(w[1], 2)], ts_text=text,
                       ts=v[:2] if len(v) >= 2 else None, half_scores=half.tolist(),
                       ref_half=0 if (w[0] + w[1]) / 2 < dur / 2 else 1)
        except Exception as e:
            rec = dict(path=p, error=repr(e)[:300])
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        n_ok += 'error' not in rec
        print('loc', n_ok, p[-28:], rec.get('ref'), rec.get('ts'), flush=True)


def select(rows, n):
    rng = random.Random(0); sel = []
    for task, labels in (('TA', 'AB'), ('TF', 'ABC'), ('CMTB', 'ABC')):
        for g in labels:
            pool = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == task and r['answer'] == g]
            rng.shuffle(pool); sel += [(task, i, r) for i, r in pool[:n]]
    rng.shuffle(sel); return sel


def part_items(rows, n, variant):
    res = OUT / f'{variant}.jsonl'
    done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
    sel = select(rows, n)
    for k, (task, i, r) in enumerate(sel):
        if i in done: continue
        t0 = time.time()
        try:
            orig = fetch(r['video_path'].replace(BS, '/'))
            src = normalised(orig) if variant == 'orig' else noisy(orig)
            if len(decode_audio(src)) / 16000 > MAX_DUR: raise RuntimeError('skipped: longer than MAX_DUR')
            rec = run_item(task, r, src, variant); rec['sec'] = round(time.time() - t0)
        except Exception as e:
            rec = dict(task=task, gold='ABC'.index(r['answer']), error=repr(e)[:300])
        rec['idx'] = i
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        if 'error' in rec: print(variant, k + 1, '/', len(sel), task, i, rec['error'][:80], flush=True)
        else:
            g = rec['gold']
            print(variant, k + 1, '/', len(sel), task, i, 'base', int(np.argmax(rec['base'])) == g, 'gate', int(np.argmax(rec['gate_hard'])) == g,
                  'lac', rec.get('lac_pred', '-') == g if 'lac_pred' in rec else '-', rec['sec'], 's', flush=True)


def main():
    global MODEL_, PROC, LETTERS
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    MODEL_, PROC = load_model()
    LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    print('model ready', flush=True)
    if 'main' in PARTS: part_items(rows, N, 'orig')
    if 'loc' in PARTS: part_loc(rows)
    if 'noise' in PARTS: part_items(rows, N_NOISE, 'noisy')


if __name__ == '__main__':
    main()

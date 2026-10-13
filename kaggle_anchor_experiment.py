"""CMTB anchor-then-look experiment, run inside the Kaggle venv by the notebook.
Usage: python kaggle_anchor_experiment.py ROOT OUT N_PER_LABEL
Conditions per item, each scored under 3 cyclic orders of the options (so letter position cancels):
  base_av     : whole clip, audio+video, original question
  vision_full : whole clip, video only (audio removed), neutral visual question
  speech_win  : only the audio-active window, video only, neutral visual question
  silent_win  : only the muted window, video only (control: the right action is not on screen here)
Labels are used only for scoring. Speech window = span of 50 ms frames above an RMS threshold (analysis tool)."""
import os, sys, json, re, subprocess, time, random, itertools
from pathlib import Path
import numpy as np
import torch

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT, OUT, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, str(ROOT))
import av, librosa
from huggingface_hub import hf_hub_download

DATASET = 'zcs15/SVHalluc'
REVISION = '211b4f14ab66ae9557b69e302d1f090e0e610839'
BS = chr(92)


def decode_audio(path):
    chunks = []
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        for fr in c.decode(audio=0):
            for p in rs.resample(fr): chunks.append(p.to_ndarray().reshape(-1))
        for p in rs.resample(None): chunks.append(p.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype('float32', copy=False)


_orig_load = librosa.load
def _load(path, sr=None, offset=0.0, duration=None, **k):  # Qwen utils' librosa path fails on some of these MP4s
    if str(path).lower().endswith('.mp4'):
        a = decode_audio(path)
        return a[int(sr * offset): None if duration is None else int(sr * (offset + duration))], sr
    return _orig_load(path, sr=sr, offset=offset, duration=duration, **k)
librosa.load = _load

from lab_native_video import prepare_video, qwen_inputs, move_inputs
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

MODEL = 'Qwen/Qwen2.5-Omni-3B'
MREV = 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
CLIPS = OUT / 'clips'; CLIPS.mkdir(parents=True, exist_ok=True)
RES = OUT / 'anchor_results.jsonl'
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


def fetch(name): return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def speech_window(path):
    a = decode_audio(path); hop = 800; n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    act = db > lo + 0.5 * (hi - lo)
    if act.sum() == 0: return None
    t = (np.arange(n) + 0.5) * 0.05; idx = np.where(act)[0]; dur = len(a) / 16000
    return dict(dur=dur, s0=float(t[idx[0]]), s1=float(t[idx[-1]]), centroid=float(t[act].mean() / dur))


def cut(src, start, end, tag):
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists():
        cmd = ['ffmpeg', '-hide_banner', '-y', '-ss', f'{start:.2f}', '-to', f'{end:.2f}', '-i', str(src),
               '-map', '0:v:0', '-map', '0:a:0', '-vf', 'fps=30', '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'ultrafast',
               '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-af', 'aresample=async=1:first_pts=0', str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
    return dst


def parse_question(q):
    """Return (stem, [option texts in original A,B,C order])."""
    parts = re.split(r'\s*\(([ABC])\)\s*', q)
    stem = parts[0].strip(); opts = {}
    for i in range(1, len(parts) - 1, 2): opts[parts[i]] = parts[i + 1].strip()
    assert list(opts) == ['A', 'B', 'C'], q
    return stem, [opts['A'], opts['B'], opts['C']]


def render(stem, texts, perm, suffix=''):
    body = ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))
    return f'{stem} {body}{suffix}'


def load_model():
    kw = dict(revision=MREV, torch_dtype=torch.float16, attn_implementation='sdpa')
    if torch.cuda.device_count() >= 2:
        kw.update(device_map='balanced', max_memory={0: '10GiB', 1: '10GiB'})
    else:  # local smoke test on a small card
        from transformers import BitsAndBytesConfig
        kw.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    return m, Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=MREV)


def letter_logprobs(model, proc, prompt, media, mode):
    inp, _ = qwen_inputs(proc, prompt, media, mode)
    inp = move_inputs(inp, model)
    with torch.no_grad():
        out = model(**inp, use_audio_in_video=(mode == 'av'))
    lg = out.logits[0, -1].float()
    ids = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    return torch.log_softmax(lg[ids], -1).tolist()


def score_condition(model, proc, media, mode, stem, texts):
    """Sum log-probabilities per option CONTENT over the 3 cyclic orders."""
    content = np.zeros(3); first_order = None
    for perm in PERMS:
        lp = letter_logprobs(model, proc, render(stem, texts, perm), media, mode)
        for pos, j in enumerate(perm): content[j] += lp[pos]
        if first_order is None: first_order = lp
    return dict(content=content.tolist(), pred_rot='ABC'[int(np.argmax(content))],
                pred_single='ABC'[int(np.argmax(first_order))], single_logprobs=first_order)


def main():
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cm = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == 'CMTB']
    rng = random.Random(0); sel = []
    for g in 'ABC':
        pool = [x for x in cm if x[1]['answer'] == g]; rng.shuffle(pool); sel += pool[:N]
    rng.shuffle(sel)
    done = {json.loads(l)['idx'] for l in RES.read_text().splitlines()} if RES.exists() else set()
    model, proc = load_model()
    print('model ready; items', len(sel), 'already done', len(done), flush=True)
    NEUTRAL = 'Which visual action is occurring in this video?'
    for k, (i, r) in enumerate(sel):
        if i in done: continue
        t0 = time.time()
        try:
            src = fetch(r['video_path'].replace(BS, '/'))
            w = speech_window(src)
            if w is None: raise RuntimeError('no speech activity')
            dur, s0, s1 = w['dur'], w['s0'], w['s1']
            stem, texts = parse_question(r['question'])
            gold_content = 'ABC'.index(r['answer'])
            full = cut(src, 0, dur, 'full'); sw = cut(src, s0, min(s1, dur), 'speech')
            silent = (0.0, s0 - 0.05) if w['centroid'] >= 0.5 else (s1 + 0.05, dur)
            rec = dict(idx=i, gold=gold_content, dur=round(dur, 1), speech=[round(s0, 1), round(s1, 1)], silent=[round(x, 1) for x in silent])
            media_full = prepare_video(full)
            rec['base_av'] = score_condition(model, proc, media_full, 'av', stem, texts)
            rec['vision_full'] = score_condition(model, proc, media_full, 'vision', NEUTRAL, texts)
            rec['speech_win'] = score_condition(model, proc, prepare_video(sw), 'vision', NEUTRAL, texts)
            if silent[1] - silent[0] >= 1.0:
                rec['silent_win'] = score_condition(model, proc, prepare_video(cut(src, *silent, 'silent')), 'vision', NEUTRAL, texts)
            rec['sec'] = round(time.time() - t0)
        except Exception as e:
            rec = dict(idx=i, gold='ABC'.index(r['answer']), error=repr(e)[:300])
        torch.cuda.empty_cache()
        with RES.open('a', encoding='utf-8') as f: f.write(json.dumps(rec) + '\n')
        print(k + 1, '/', len(sel), 'idx', i, rec.get('error') or {c: 'ABC'[int(np.argmax(rec[c]['content']))] == 'ABC'[rec['gold']] for c in ('base_av', 'vision_full', 'speech_win')}, rec.get('sec'), flush=True)


if __name__ == '__main__':
    main()

"""Daily-Omni baseline on the remote GPU (Qwen2.5-Omni-3B FP16), 30-second videos only (60-s clips exceed T4x2 memory in our
setup). 1197 four-choice questions in 6 types (Event Sequence, AV Event Alignment, Context understanding, Reasoning,
Inference, Comparative). Scored with 4 cyclic option orders summed per option content, the same protocol as SVHalluc.
Purpose: a general audio-visual benchmark to show later that the method helps temporal/cross-modal types and does not hurt
the others. Writes OUT/dailyomni_base.jsonl."""
import json, tarfile, time, re
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

# 4 options: letters and rendering must cover A-D (the namespace defaults are A-C)
LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABCD']
def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABCD', perm))
DO = Path('/kaggle/working/daily_omni'); DO.mkdir(parents=True, exist_ok=True)
qa = json.loads(Path(hf_hub_download('liarliar/Daily-Omni', 'qa.json', repo_type='dataset')).read_text(encoding='utf-8'))
vid_dir = DO / 'videos'
if not vid_dir.exists():
    tar = hf_hub_download('liarliar/Daily-Omni', 'Videos.tar', repo_type='dataset')
    with tarfile.open(tar) as t:
        names = t.getnames(); print('tar members', len(names), names[:5], flush=True)
        t.extractall(vid_dir)
paths = {}
for p in vid_dir.rglob('*'):
    if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov'): paths.setdefault(p.stem, p)
print('videos found', len(paths), 'example', next(iter(paths.items())), flush=True)
LIMIT = int(globals().get('DO_LIMIT', 10 ** 9))
items = [(i, q) for i, q in enumerate(qa) if q.get('video_duration') == '30s'][:LIMIT]
res = OUT / 'dailyomni_base.jsonl'
if res.exists() and globals().get('DO_RESET', True):
    keep = [l for l in res.read_text().splitlines() if l.strip() and 'error' not in l]; res.write_text(''.join(k + chr(10) for k in keep))
done = {json.loads(l)['i'] for l in res.read_text().splitlines()} if res.exists() else set()
am = lambda v: int(np.argmax(v))
n_ok = 0
for n, (i, q) in enumerate(items):
    if i in done: continue
    t0 = time.time(); rec = dict(i=i, type=q['Type'], gold='ABCD'.index(q['Answer']), video_id=q['video_id'])
    try:
        vp = None
        for k, p in paths.items():
            if q['video_id'] in k: vp = p; break
        if vp is None: raise RuntimeError('video not found')
        src = cut(vp, 0, 10 ** 4, 'full')
        media = prepare_video(src)
        texts = [re.sub(r'^[A-D]\.\s*', '', c).strip() for c in q['Choice']]
        rec['base'] = scores(media, q['Question'], texts)
        rec['vision_only'] = scores(media, q['Question'], texts, mode='vision')
        rec['sec'] = round(time.time() - t0); n_ok += 1
    except Exception as e:
        rec['error'] = repr(e)[:200]
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    if n % 10 == 0 or 'error' in rec:
        print(n + 1, len(items), rec['type'], 'ok' if 'base' in rec and am(rec['base']) == rec['gold'] else ('ERR ' + rec.get('error', '')[:60] if 'error' in rec else 'wrong'), rec.get('sec'), flush=True)
R = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]
R = [r for r in R if 'base' in r]
by = {}
for r in R: by.setdefault(r['type'], []).append(am(r['base']) == r['gold'])
RESULT = {'n': len(R), 'acc': round(float(np.mean([am(r['base']) == r['gold'] for r in R])) * 100, 1) if R else None,
          'by_type': {k: (round(float(np.mean(v)) * 100, 1), len(v)) for k, v in by.items()}}
print(RESULT, flush=True)
LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))

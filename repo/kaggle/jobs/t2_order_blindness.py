"""T2 — Is the temporal-order deficit general? Daily-Omni (natural videos, 30 s), Event Sequence and AV Event Alignment questions.
Conditions (4 rotated option orders): base | video_reversed (frames played backwards, audio unchanged) |
video_halves_swapped (second half first, audio unchanged).
If the model used temporal order, the answer of an Event-Sequence question should usually CHANGE when the video is
reversed or reordered (the correct sequence is no longer the shown one). Measured: base accuracy and the share of items
whose answer changes. Writes OUT/t2_order_blindness.jsonl."""
import json, re, tarfile, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download

LIMIT = int(globals().get('T2_LIMIT', 120))
DO = Path('/kaggle/working/daily_omni'); DO.mkdir(parents=True, exist_ok=True)
qa = json.loads(Path(hf_hub_download('liarliar/Daily-Omni', 'qa.json', repo_type='dataset')).read_text(encoding='utf-8'))
vid_dir = DO / 'videos'
if not vid_dir.exists():
    with tarfile.open(hf_hub_download('liarliar/Daily-Omni', 'Videos.tar', repo_type='dataset')) as t: t.extractall(vid_dir)
paths = {p.stem: p for p in vid_dir.rglob('*') if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')}
items = [(i, q) for i, q in enumerate(qa) if q.get('video_duration') == '30s' and q['Type'] in ('Event Sequence', 'AV Event Alignment')]
items = items[::2][:LIMIT]


def reversed_video(src):
    dst = CLIPS / f'{Path(src).stem}_vrev.mp4'
    if not dst.exists(): ffmpeg('-i', str(src), '-map', '0:v:0', '-map', '0:a:0', '-vf', 'reverse', *ENC[:8], '-c:a', 'copy', str(dst))
    return dst


res = OUT / 't2_order_blindness.jsonl'
done = {json.loads(l)['i'] for l in res.read_text().splitlines() if l.strip()} if res.exists() else set()
am = lambda v: int(np.argmax(v))
for n, (i, q) in enumerate(items):
    if i in done: continue
    t0 = time.time(); rec = dict(i=i, type=q['Type'], gold='ABCD'.index(q['Answer']))
    try:
        vp = next((p for k_, p in paths.items() if q['video_id'] in k_), None)
        if vp is None: raise RuntimeError('video not found')
        src = cut(vp, 0, 10 ** 4, 'full'); dur = len(decode_audio(src)) / SR
        texts = [re.sub(r'^[A-D]\.\s*', '', c).strip() for c in q['Choice']]; stem = q['Question']
        rec['base'] = score(prepare_video(src), stem, texts)
        rec['video_reversed'] = score(prepare_video(reversed_video(src)), stem, texts)
        h1, h2 = halves(src, dur)
        rec['video_halves_swapped'] = score(prepare_video(concat_clips([h2, h1], 'swapv', audio_from=src)), stem, texts)
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:200]; torch.cuda.empty_cache()
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(n + 1, len(items), rec['type'][:14], 'ERR ' + rec['error'][:60] if 'error' in rec else
          {c: am(rec[c]) for c in ('base', 'video_reversed', 'video_halves_swapped')}, 'gold', rec['gold'], rec.get('sec'), flush=True)
R = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]; OK = [r for r in R if 'error' not in r]
RESULT = {}
for t in ('Event Sequence', 'AV Event Alignment'):
    S = [r for r in OK if r['type'] == t]
    if S:
        RESULT[t] = {'n': len(S), 'base_acc': round(100 * float(np.mean([am(r['base']) == r['gold'] for r in S])), 1)}
        for c in ('video_reversed', 'video_halves_swapped'):
            RESULT[t][c + '_acc'] = round(100 * float(np.mean([am(r[c]) == r['gold'] for r in S])), 1)
            RESULT[t][c + '_answer_changed%'] = round(100 * float(np.mean([am(r[c]) != am(r['base']) for r in S])), 1)
print(RESULT, flush=True)

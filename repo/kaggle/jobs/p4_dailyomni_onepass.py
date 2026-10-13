"""P4 — GENERALITY / NO-HARM of the one-pass method on Daily-Omni (natural videos, all 6 question types, 4 options).
Conditions (4 rotated option orders): base | onepass (relevance at layer 12 -> re-allocation + self-decided source gate in
layers 13-27, neutral prior cached per video) | onepass_nogate (time re-allocation only).
30-s videos only (eager attention at one layer). Needs the J8 helpers and inpass.py. Writes OUT/p4_dailyomni.jsonl."""
import json, re, tarfile, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('\nitems = sample_items')[0], globals())
exec((ROOT / 'inpass.py').read_text(encoding='utf-8'), globals())

LIMIT = int(globals().get('DO_LIMIT', 240)); MEAS1, APPLY1 = [12], list(range(13, 28))
DO = Path('/kaggle/working/daily_omni'); DO.mkdir(parents=True, exist_ok=True)
qa = json.loads(Path(hf_hub_download('liarliar/Daily-Omni', 'qa.json', repo_type='dataset')).read_text(encoding='utf-8'))
vid_dir = DO / 'videos'
if not vid_dir.exists():
    with tarfile.open(hf_hub_download('liarliar/Daily-Omni', 'Videos.tar', repo_type='dataset')) as t: t.extractall(vid_dir)
paths = {p.stem: p for p in vid_dir.rglob('*') if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')}
by = {}
for i, q in enumerate(qa):
    if q.get('video_duration') == '30s': by.setdefault(q['Type'], []).append((i, q))
items = []
for k in range(max(map(len, by.values()))):                 # round-robin over the 6 types
    for t in sorted(by):
        if k < len(by[t]): items.append(by[t][k])
items = items[:LIMIT]
res = OUT / 'p4_dailyomni.jsonl'
done = {json.loads(l)['i'] for l in res.read_text().splitlines() if l.strip()} if res.exists() else set()
am = lambda v: int(np.argmax(v)); prior_cache = {}
for n, (i, q) in enumerate(items):
    if i in done: continue
    t0 = time.time(); rec = dict(i=i, type=q['Type'], gold='ABCD'.index(q['Answer']))
    try:
        vp = next((p for k_, p in paths.items() if q['video_id'] in k_), None)
        if vp is None: raise RuntimeError('video not found')
        src = cut(vp, 0, 10 ** 4, 'full'); media = prepare_video(src)
        texts = [re.sub(r'^[A-D]\.\s*', '', c).strip() for c in q['Choice']]; stem = q['Question']
        target = self_target(stem); rec['target'] = target
        if q['video_id'] not in prior_cache: prior_cache[q['video_id']] = neutral_prior(media, MEAS1)
        prior = prior_cache[q['video_id']]
        rec['base'] = score(media, stem, texts)
        rec['onepass'] = score(media, stem, texts, fwd=lambda inp: inpass_forward(inp, prior, target, MEAS1, APPLY1)[0])
        rec['onepass_nogate'] = score(media, stem, texts, fwd=lambda inp: inpass_forward(inp, prior, 'both', MEAS1, APPLY1)[0])
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:200]; torch.cuda.empty_cache()
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(n + 1, len(items), rec['type'][:12], 'ERR ' + rec['error'][:50] if 'error' in rec else
          {c: am(rec[c]) == rec['gold'] for c in ('base', 'onepass', 'onepass_nogate')}, rec.get('target'), rec.get('sec'), flush=True)
R = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]; OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: round(100 * float(np.mean([am(r[c]) == r['gold'] for r in OK if r['type'] == t])), 1) for c in ('base', 'onepass', 'onepass_nogate')}
          | {'n': sum(r['type'] == t for r in OK)} for t in sorted({r['type'] for r in OK})}
RESULT['ALL'] = {c: round(100 * float(np.mean([am(r[c]) == r['gold'] for r in OK])), 1) for c in ('base', 'onepass', 'onepass_nogate')} if OK else None
print(RESULT, flush=True)

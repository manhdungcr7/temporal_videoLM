"""Report for the binding lab with pre-registered criteria. Usage: python binding_lab_report.py DIR..."""
import json, sys, glob, os, random
from pathlib import Path
import numpy as np
paths = []
for a in (sys.argv[1:] or ['.']):
    paths += glob.glob(os.path.join(a, '**', 'binding_shard*.jsonl'), recursive=True) if os.path.isdir(a) else [a]
recs = {}
for p in paths:
    for l in Path(p).read_text(encoding='utf-8').splitlines():
        if l.strip(): r = json.loads(l); recs[r['idx']] = r
R = [r for r in recs.values() if 'error' not in r]
print(f'items {len(recs)} | valid {len(R)} | skipped/errors {len(recs) - len(R)}')
C = ['base', 'bind0', 'bind0_bi', 'bind1_bi', 'bi_only', 'balance']
am = lambda v: int(np.argmax(v))
def ci(d, n=4000):
    g = random.Random(0); b = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(n)); return b[int(.025 * n)], b[int(.975 * n)]
delta = {}
for task in ('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CMSB'):
    rs = [r for r in R if r['task'] == task]
    if not rs: continue
    k = len(rs[0]['base'])
    print(f'\n== {task} n={len(rs)} ==')
    for c in C:
        a = np.mean([am(r[c]) == r['gold'] for r in rs]) * 100
        rec = [np.mean([am(r[c]) == j for r in rs if r['gold'] == j]) * 100 for j in range(k)]
        line = f'  {c:9s} {a:5.1f}  recall/label {np.round(rec, 1).tolist()}'
        if c != 'base':
            d = [int(am(r[c]) == r['gold']) - int(am(r['base']) == r['gold']) for r in rs]; lo, hi = ci(d)
            delta[(task, c)] = np.mean(d) * 100
            line += f'  vs base {np.mean(d) * 100:+5.1f} [{lo * 100:+.1f},{hi * 100:+.1f}] w2c {d.count(1)} c2w {d.count(-1)}'
        print(line)
print('\n== PRE-REGISTERED CRITERION ==')
for c in C[1:]:
    temporal = [delta.get((t, c), np.nan) for t in ('TA', 'TF', 'CMTB')]
    semantic = [delta.get((t, c), np.nan) for t in ('GSA', 'FGSA', 'CMSB')]
    ok = sum(x >= 5 for x in temporal) >= 2 and all(x > -3 for x in semantic if not np.isnan(x))
    print(f'  {c:9s} temporal deltas {np.round(temporal, 1).tolist()} semantic deltas {np.round(semantic, 1).tolist()} -> {"PROMISING" if ok else "no"}')
print('PROMISING = +5 points or more on at least 2 temporal tasks AND no semantic task below -3. Small n: only a screening signal.')

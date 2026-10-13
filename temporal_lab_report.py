"""Merge lab shards and report. Usage: python temporal_lab_report.py DIR_OR_FILES...  (reads every lab_shard*.jsonl found)"""
import json, sys, random, glob, os
from pathlib import Path
import numpy as np

paths = []
for a in (sys.argv[1:] or ['.']):
    paths += glob.glob(os.path.join(a, '**', 'lab_shard*.jsonl'), recursive=True) if os.path.isdir(a) else [a]
recs = {}
for p in paths:
    for l in Path(p).read_text(encoding='utf-8').splitlines():
        if l.strip():
            r = json.loads(l); recs[r['idx']] = r
R = [r for r in recs.values() if 'error' not in r]
E = [r for r in recs.values() if 'error' in r]
print(f'files {len(paths)} | items {len(recs)} | valid {len(R)} | errors {len(E)}')
if E:
    from collections import Counter
    print('  error kinds:', Counter(e['error'][:40] for e in E).most_common(4))

CONDS = ['base', 'ts05', 'ts02', 'gate_all', 'gate_late', 'audio_block', 'gate_audio', 'crop']
pred = lambda r, c: int(np.argmax(r[c]))


def ci(d, n=4000):
    g = random.Random(0); b = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(n)); return b[int(.025 * n)], b[int(.975 * n)]


for task, k in (('TA', 2), ('TF', 3), ('CMTB', 3)):
    rs = [r for r in R if r['task'] == task]
    if not rs: continue
    print(f'\n===== {task}  n={len(rs)} =====')
    for c in [c for c in CONDS if c in rs[0]]:
        acc = np.mean([pred(r, c) == r['gold'] for r in rs]) * 100
        rec = [np.mean([pred(r, c) == j for r in rs if r['gold'] == j]) * 100 for j in range(k)]
        share = [np.mean([pred(r, c) == j for r in rs]) for j in range(k)]
        line = f'  {c:11s} acc {acc:5.1f}  recall/label {np.round(rec, 1).tolist()}  predicted share {np.round(share, 2).tolist()}'
        if c != 'base':
            d = [int(pred(r, c) == r['gold']) - int(pred(r, 'base') == r['gold']) for r in rs]
            lo, hi = ci(d)
            line += f'  | vs base {np.mean(d) * 100:+5.1f} [{lo * 100:+.1f},{hi * 100:+.1f}]  w2c {sum(x == 1 for x in d)} c2w {sum(x == -1 for x in d)}'
        print(line)
    if task == 'TF':
        rule = np.mean([(2 if (r['win'][0] + r['win'][1]) / 2 < r['dur'] / 2 else 0) == r['gold'] for r in rs]) * 100
        print(f'  reference: speech-position-only rule (no video) {rule:.1f}%')
    # ---- causal position test: same items, segment order swapped ----
    sw = [r for r in rs if 'swap_base' in r]
    if sw:
        print(f'  -- segment swap (n={len(sw)}) --')
        for c_o, c_s in (('base', 'swap_base'), ('gate_all', 'swap_gate')):
            a_o = np.mean([pred(r, c_o) == r['gold'] for r in sw]) * 100
            a_s = np.mean([pred(r, c_s) == r['swap_gold'] for r in sw]) * 100
            # accuracy by where the SPEECH segment sits, pooled over original and swapped versions of the same items
            first = [pred(r, c_o) == r['gold'] for r in sw if r['speech_first_in_orig']] + [pred(r, c_s) == r['swap_gold'] for r in sw if not r['speech_first_in_orig']]
            second = [pred(r, c_o) == r['gold'] for r in sw if not r['speech_first_in_orig']] + [pred(r, c_s) == r['swap_gold'] for r in sw if r['speech_first_in_orig']]
            print(f'  {c_o:9s} orig {a_o:5.1f} / swapped {a_s:5.1f}  |  speech segment FIRST {np.mean(first) * 100:5.1f} (n={len(first)})  vs SECOND {np.mean(second) * 100:5.1f} (n={len(second)})')
            if task == 'TF':
                flip = {0: 2, 2: 0, 1: 1}
                cons = np.mean([flip[pred(r, c_o)] == pred(r, c_s) for r in sw]) * 100
                print(f'            TF answers that flip consistently with the swap (Past<->Future): {cons:.1f}%')

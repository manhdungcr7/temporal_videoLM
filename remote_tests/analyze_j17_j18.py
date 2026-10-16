"""Offline analysis of J17 (question first) and J18 (TMRoPE time ablation).
    python analyze_j17_j18.py j17 file1.jsonl [file2.jsonl ...]      (shards merged by idx)
    python analyze_j17_j18.py j18 file1.jsonl [...]
Always reported per (task, gold) + balanced accuracy + predicted-answer distribution (to catch a collapse onto one answer)."""
import collections, json, sys
import numpy as np

kind, files = sys.argv[1], sys.argv[2:]
rows = {}
for f in files:
    for l in open(f, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            if 'error' not in r: rows[r['idx']] = r
R = list(rows.values()); am = lambda v: int(np.argmax(v))
C = ('base', 'qpre', 'qpre_blind', 'plan_after', 'qpre_plan') if kind == 'j17' else ('base', 'v_tconst', 'v_tshuffle', 'a_tconst', 'av_tconst')
print('items', len(R), dict(collections.Counter(r['task'] for r in R)))


def table(R, C):
    print('%-5s %-4s %3s ' % ('task', 'gold', 'n') + ' '.join('%-11s' % c for c in C))
    for t in ('TA', 'TF', 'CMTB'):
        for g in sorted({r['gold'] for r in R if r['task'] == t}):
            S = [r for r in R if r['task'] == t and r['gold'] == g]
            print('%-5s %-4s %3d ' % (t, 'ABCD'[g], len(S)) + ' '.join('%-11s' % ('%d/%d' % (sum(am(r[c]) == g for r in S), len(S))) for c in C))
    print('\nbalanced accuracy (mean over gold labels) and predicted-answer distribution:')
    for t in ('TA', 'TF', 'CMTB'):
        S = [r for r in R if r['task'] == t]
        if not S: continue
        k = len(S[0]['base']); golds = sorted({r['gold'] for r in S})
        for c in C:
            bal = np.mean([np.mean([am(r[c]) == g for r in S if r['gold'] == g]) for g in golds])
            cnt = collections.Counter('ABCD'[am(r[c])] for r in S)
            ch = np.mean([am(r[c]) != am(r['base']) for r in S]) if c != 'base' else 0
            print(f'  {t:5s} {c:11s} balanced {100 * bal:5.1f}  answers ' + ' '.join(f'{L}:{cnt.get(L, 0):2d}' for L in 'ABCD'[:k]) + f'  changed vs base {100 * ch:4.0f}%')


table(R, C)

if kind == 'j17':
    d = [r['check_4d_maxdiff'] for r in R if 'check_4d_maxdiff' in r]
    if d: print(f'\ncheck_4d_maxdiff (letter logits): median {np.median(d):.4f}  max {max(d):.4f}  (must be ~0 for qpre_blind to be valid)')
    # contrastive readout (1+a)*qpre - a*base, offline
    for a in (0.5, 1.0):
        for src in ('qpre', 'qpre_plan'):
            for r in R: r[f'cr_{src}_{a}'] = list((1 + a) * np.array(r[src]) - a * np.array(r['base']))
    print('\ncontrastive readout (1+a)*x - a*base:')
    table(R, ('base',) + tuple(f'cr_{s}_{a}' for s in ('qpre', 'qpre_plan') for a in (0.5, 1.0)))
    # mechanism: attention mass from media tokens to the prefix
    for key in ('mass_qpre', 'mass_qpre_plan'):
        V = [np.array([[x if x is not None else np.nan for x in row] for row in r[key]['video']]) for r in R if key in r]
        A = [np.array([[x if x is not None else np.nan for x in row] for row in r[key]['audio']]) for r in R if key in r]
        if not V: continue
        vl = np.nanmean([np.nanmean(v, 1) for v in V], 0); al = np.nanmean([np.nanmean(a, 1) for a in A], 0)
        print(f'\n[{key}] mean attention mass on the prefix ({np.mean([r[key]["prefix_tokens"] for r in R]):.0f} tokens), by layer (video | audio):')
        for l in range(0, len(vl), 3): print(f'   L{l:2d}  {vl[l]:.4f} | {al[l]:.4f}')
        # trend over time: correlation of the per-chunk mass (layer-averaged) with the chunk index
        tv = [np.corrcoef(np.arange(v.shape[1]), np.nanmean(v, 0))[0, 1] for v in V if v.shape[1] > 2]
        ta = [np.corrcoef(np.arange(a.shape[1]), np.nanmean(a, 0))[0, 1] for a in A if a.shape[1] > 2]
        print(f'   correlation (mass vs chunk index): video {np.nanmean(tv):+.2f}  audio {np.nanmean(ta):+.2f}   (negative = later chunks read the prefix less)')
    plans = [r.get('plan', '') for r in R][:5]
    print('\nexample plans:'); [print('  -', p) for p in plans]

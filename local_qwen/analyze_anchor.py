import json, sys, random, collections
import numpy as np
R0 = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
bad = [r for r in R0 if 'error' in r]; R = [r for r in R0 if 'error' not in r]
print('valid', len(R), 'errors', len(bad), [b['error'][:100] for b in bad[:3]])
C = ['base_av', 'vision_full', 'speech_win', 'silent_win']
ok = lambda r, c, k: r[c][k] == 'ABC'[r['gold']]
for c in C:
    rs = [r for r in R if c in r]
    print(f'{c:12s} n={len(rs)} rot {np.mean([ok(r,c,"pred_rot") for r in rs])*100:5.1f}  single {np.mean([ok(r,c,"pred_single") for r in rs])*100:5.1f}',
          'letter dist', {L: round(sum(r[c]['pred_single'] == L for r in rs) / len(rs), 2) for L in 'ABC'},
          'rot-content dist', {L: round(sum(r[c]['pred_rot'] == L for r in rs) / len(rs), 2) for L in 'ABC'})
def paired(a, b):
    rs = [r for r in R if a in r and b in r]
    w = sum((not ok(r, a, 'pred_rot')) and ok(r, b, 'pred_rot') for r in rs); l = sum(ok(r, a, 'pred_rot') and not ok(r, b, 'pred_rot') for r in rs)
    d = [int(ok(r, b, 'pred_rot')) - int(ok(r, a, 'pred_rot')) for r in rs]; g = random.Random(0)
    bs = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(5000))
    print(f'{a}->{b}: w2c {w} c2w {l} delta {np.mean(d)*100:+.1f} CI [{bs[125]*100:+.1f},{bs[4874]*100:+.1f}] n={len(rs)}')
paired('base_av', 'vision_full'); paired('vision_full', 'speech_win'); paired('base_av', 'speech_win')
# by gold content position and by duration
print('speech_win acc by gold letter (original order):', {L: round(np.mean([ok(r, 'speech_win', 'pred_rot') for r in R if 'ABC'[r['gold']] == L]) * 100, 1) for L in 'ABC'})
for lo, hi in ((0, 20), (20, 30), (30, 99)):
    rs = [r for r in R if lo <= r['dur'] < hi]
    print(f'dur {lo}-{hi} n={len(rs)}', {c: round(np.mean([ok(r, c, "pred_rot") for r in rs if c in r]) * 100) for c in C})
# margin: how decisive is each condition (gap between gold content score and best other)
for c in C:
    m = [r[c]['content'][r['gold']] - max(v for j, v in enumerate(r[c]['content']) if j != r['gold']) for r in R if c in r]
    print(c, 'median gold-minus-best-other (sum of 3 log-probs):', round(float(np.median(m)), 2))
# consistency of the model across rotations: fraction of items where the 3 single-order picks agree on content

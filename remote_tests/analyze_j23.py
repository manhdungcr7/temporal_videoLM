"""Analysis of J23 (pre-registered confirmation of gated_anchor).   python analyze_j23.py s1.jsonl s2.jsonl
Primary: rescued/damaged of gated_anchor vs base over all items, two-sided exact sign test (success: p < 0.05 AND rescued > damaged).
Secondary (no-harm): semantic balanced accuracy drop <= 2 points; per (task, gold) table; gate-open rate per task; parity guard."""
import json, math, sys
import numpy as np

R = [json.loads(l) for f in sys.argv[1:] for l in open(f, encoding='utf-8') if l.strip()]
err = [r for r in R if 'error' in r]; R = [r for r in R if 'error' not in r]
am = lambda v: int(np.argmax(v)); TASKS = ('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CSMB'); C = ('base', 'anchor_video', 'gated_anchor')
print('items ok', len(R), '| errors', len(err), '| parity null==base', sum(am(r['null']) == am(r['base']) for r in R), 'of', len(R))
print('gate open:', {t: f"{sum(r['gate_open'] for r in R if r['task'] == t)}/{sum(r['task'] == t for r in R)}" for t in TASKS})


def signp(b, c):
    n = b + c; return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


cells = [(t, g) for t in TASKS for g in sorted({r['gold'] for r in R if r['task'] == t})]
def bal(c, fam): return 100 * np.mean([np.mean([am(r[c]) == g for r in R if r['task'] == t and r['gold'] == g]) for t, g in cells if t in fam])
print('   method        ' + ' '.join(f'{t}/{"ABC"[g]:<4s}' for t, g in cells) + '  bal-temp bal-sem  rescued/damaged (p)')
for c in C:
    row = []
    for t, g in cells:
        cc = [r for r in R if r['task'] == t and r['gold'] == g]; k = sum(am(r[c]) == g for r in cc); row.append(f'{k}/{len(cc) - k}')
    b = sum(am(r[c]) == r['gold'] != am(r['base']) for r in R); d = sum(am(r['base']) == r['gold'] != am(r[c]) for r in R)
    print(f'   {c:13s} ' + ' '.join(f'{x:<8s}' for x in row) + f'  {bal(c, TASKS[:3]):7.1f} {bal(c, TASKS[3:]):7.1f}  {b}/{d} (p={signp(b, d):.3f})')
b = sum(am(r['gated_anchor']) == r['gold'] != am(r['base']) for r in R); d = sum(am(r['base']) == r['gold'] != am(r['gated_anchor']) for r in R)
drop = bal('base', TASKS[3:]) - bal('gated_anchor', TASKS[3:])
print(f'\nPRIMARY: gated_anchor rescued {b} / damaged {d}, p = {signp(b, d):.4f} ->', 'PASS' if signp(b, d) < 0.05 and b > d else 'FAIL',
      f'| semantic drop {drop:.1f} ->', 'PASS' if drop <= 2 else 'FAIL')

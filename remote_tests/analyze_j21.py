"""Analysis of J21 (internal temporal routing).   python analyze_j21.py s1.jsonl s2.jsonl [--j20 j20_s1.jsonl j20_s2.jsonl]
Per (task, gold) correct/wrong, balanced accuracy, prediction histogram, rescued/damaged vs base with an exact sign test, separately for
the development items (= the J20 items, seed 2028) and the FRESH items (seed 2029).  Also: null == base sanity, anchor-on-speech rate
(VAD used for analysis only), and an offline entropy gate with the threshold fixed in advance (0.8)."""
import collections, json, math, sys
import numpy as np

a = sys.argv[1:]; j20 = []
if '--j20' in a: k = a.index('--j20'); j20 = a[k + 1:]; a = a[:k]
R = [json.loads(l) for f in a for l in open(f, encoding='utf-8') if l.strip()]
R = [r for r in R if 'error' not in r]
dev_ids = {json.loads(l)['idx'] for f in j20 for l in open(f, encoding='utf-8') if l.strip()}
am = lambda v: int(np.argmax(v)); GATE = 0.8
C = ['base', 'null', 'block_audio', 'anchor_video', 'split', 'split_random']
for r in R: r['gated_split'] = r['split'] if r['anchor_entropy'] < GATE else r['base']
C.append('gated_split')
print('items', len(R), '| dev (J20 items)', sum(r['idx'] in dev_ids for r in R), '| fresh', sum(r['idx'] not in dev_ids for r in R))
print('sanity null == base answers:', sum(am(r['null']) == am(r['base']) for r in R), 'of', len(R),
      '| max |diff|', round(max(float(np.abs(np.array(r['null']) - np.array(r['base'])).max()) for r in R), 3))
sp = []
for r in R:
    if r['vad_speech']:
        lo, hi = r['anchor_win']; s0, s1 = r['vad_speech']
        sp.append((2 * lo) < s1 and (2 * hi + 2) > s0)
print(f'anchor window overlaps the VAD speech: {100 * np.mean(sp):.0f}% of {len(sp)} | gate open (entropy < {GATE}):',
      {t: f"{sum(r['anchor_entropy'] < GATE for r in R if r['task'] == t)}/{sum(r['task'] == t for r in R)}" for t in ('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CSMB')})


def signp(b, c):
    n = b + c; return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def report(S, title):
    print(f'\n===== {title}: {len(S)} items')
    tasks = [t for t in ('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CSMB') if any(r['task'] == t for r in S)]
    cells = [(t, g) for t in tasks for g in sorted({r['gold'] for r in S if r['task'] == t})]
    print('   method        ' + ' '.join(f'{t}/{"ABC"[g]:<3s}' for t, g in cells) + '  bal-temporal bal-semantic  rescued/damaged (p)')
    for c in C:
        row = []
        for t, g in cells:
            cc = [r for r in S if r['task'] == t and r['gold'] == g]; k = sum(am(r[c]) == g for r in cc); row.append(f'{k}/{len(cc) - k}')
        def bal(fam):
            v = [np.mean([am(r[c]) == g for r in S if r['task'] == t and r['gold'] == g]) for t, g in cells if t in fam]
            return 100 * np.mean(v) if v else float('nan')
        b = sum(am(r[c]) == r['gold'] and am(r['base']) != r['gold'] for r in S); d = sum(am(r[c]) != r['gold'] and am(r['base']) == r['gold'] for r in S)
        print(f'   {c:13s} ' + ' '.join(f'{x:<7s}' for x in row) + f'  {bal(("TA", "TF", "CMTB")):10.1f} {bal(("GSA", "FGSA", "CSMB")):11.1f}  {b}/{d} (p={signp(b, d):.2f})')
    print('   predictions per task (A/B/C):')
    for c in C:
        print(f'     {c:13s} ' + '  '.join(f'{t}:' + '/'.join(str(sum(r["task"] == t and am(r[c]) == L for r in S)) for L in range(len(S[[r["task"] for r in S].index(t)]["base"]))) for t in tasks))


report(R, 'ALL')
if dev_ids:
    report([r for r in R if r['idx'] in dev_ids], 'DEVELOPMENT (J20 items)')
    report([r for r in R if r['idx'] not in dev_ids], 'FRESH (seed 2029)')

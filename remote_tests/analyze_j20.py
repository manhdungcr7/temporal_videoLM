"""Analysis of J20 (knockout map: depth x source x time).   python analyze_j20.py shard1.jsonl shard2.jsonl
Reference = base_mask (same 4-D-mask code path as the knockouts, nothing blocked)."""
import collections, json, sys
import numpy as np

R = []
for f in sys.argv[1:]:
    R += [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
R = [r for r in R if 'error' not in r]
am = lambda v: int(np.argmax(v))
L0S = sorted({int(k.split('@')[1]) for k in R[0] if '@' in k}); KINDS = ('noA', 'noV', 'noSpeech', 'noRest')
FAM = {'temporal': ('TA', 'TF', 'CMTB'), 'semantic': ('GSA', 'FGSA', 'CSMB')}
print('items', len(R), dict(collections.Counter(r['task'] for r in R)))
print('sanity: base vs base_mask same answer on', sum(am(r['base']) == am(r['base_mask']) for r in R), 'of', len(R),
      '| max |score diff|', round(max(float(np.abs(np.array(r['base']) - np.array(r['base_mask'])).max()) for r in R), 3))


def bal(S, key):
    S = [r for r in S if r.get(key) is not None]
    if not S: return float('nan'), 0
    cells = collections.defaultdict(list)
    for r in S: cells[(r['task'], r['gold'])].append(am(r[key]) == r['gold'])
    return 100 * np.mean([np.mean(v) for v in cells.values()]), len(S)


def changed(S, key):
    S = [r for r in S if r.get(key) is not None]
    return (100 * np.mean([am(r[key]) != am(r['base_mask']) for r in S]) if S else float('nan')), len(S)


for fam, tasks in FAM.items():
    S = [r for r in R if r['task'] in tasks]
    print(f'\n===== {fam} ({len(S)} items)   balanced acc: base {bal(S, "base_mask")[0]:.1f}  video-only {bal(S, "vision_only")[0]:.1f}  audio-only {bal(S, "audio_only")[0]:.1f}')
    print('  answers CHANGED vs base (%)  [balanced accuracy]   rows = knockout from layer L0 to the last layer')
    print('  kind       ' + ''.join(f'   L0={l:<10d}' for l in L0S))
    for kind in KINDS:
        cells = []
        for l in L0S:
            c, n = changed(S, f'{kind}@{l}'); b, _ = bal(S, f'{kind}@{l}')
            cells.append(f'{c:5.0f}% [{b:4.0f}] n={n:<3d}')
        print(f'  {kind:9s}' + ''.join(cells))
    # direction: after blocking audio, does the answer move to the video-only answer? after blocking video, to the audio-only answer?
    for kind, ref in (('noA', 'vision_only'), ('noV', 'audio_only')):
        print(f'  {kind}: agreement with the {ref} answer  ' + '  '.join(
            f'L0={l}:{100 * np.mean([am(r[f"{kind}@{l}"]) == am(r[ref]) for r in S]):.0f}%' for l in L0S) +
              f'   (base agrees with {ref}: {100 * np.mean([am(r["base_mask"]) == am(r[ref]) for r in S]):.0f}%)')
    # per task
    print('  per task, answers changed (%) at L0 = ' + ', '.join(map(str, L0S)))
    for t in tasks:
        T = [r for r in S if r['task'] == t]
        row = []
        for kind in KINDS:
            row.append(kind + ' ' + '/'.join(f'{changed(T, f"{kind}@{l}")[0]:.0f}' for l in L0S))
        print(f'    {t:5s} n={len(T):2d}  ' + '   '.join(row))

print('\nPer (task, gold): correct/wrong for base and for each knockout at L0=8 and L0=24')
keys = ['base_mask'] + [f'{k}@{l}' for k in KINDS for l in (8, 24) if l in L0S]
print('   cell      ' + ' '.join(f'{k:>11s}' for k in keys))
for t in ('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CSMB'):
    for g in sorted({r['gold'] for r in R if r['task'] == t}):
        c = [r for r in R if r['task'] == t and r['gold'] == g]
        cells = []
        for k in keys:
            cc = [r for r in c if r.get(k) is not None]
            cells.append(f'{sum(am(r[k]) == g for r in cc)}/{len(cc) - sum(am(r[k]) == g for r in cc)}' if cc else '-')
        print(f'   {t:5s}/{"ABC"[g]}  ' + ' '.join(f'{x:>11s}' for x in cells))

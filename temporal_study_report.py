"""Report for kaggle_temporal_study outputs. Usage: python temporal_study_report.py OUT_DIR"""
import json, sys, random
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')


def load(name):
    p = OUT / name
    if not p.exists(): return [], []
    rs = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    return [r for r in rs if 'error' not in r], [r for r in rs if 'error' in r]


def pred(r, cond):
    if cond == 'lac': return r.get('lac_pred')
    return int(np.argmax(r[cond])) if cond in r else None


def ci(diff, n_boot=4000):
    g = random.Random(0); bs = sorted(np.mean([g.choice(diff) for _ in diff]) for _ in range(n_boot))
    return bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def table(rows, title):
    print(f'\n=== {title} ===')
    for task in ('TA', 'TF', 'CMTB'):
        rs = [r for r in rows if r['task'] == task]
        if not rs: continue
        conds = [c for c in ('base', 'gate_hard', 'gate_soft', 'gate_text', 'lac') if pred(rs[0], c) is not None]
        line = f'{task:5s} n={len(rs):3d} | ' + ' | '.join(f'{c} {100 * np.mean([pred(r, c) == r["gold"] for r in rs]):5.1f}%' for c in conds)
        print(line)
        k = 2 if task == 'TA' else 3
        for c in conds:
            dist = [sum(pred(r, c) == j for r in rs) / len(rs) for j in range(k)]
            per = [np.mean([pred(r, c) == j for r in rs if r['gold'] == j]) * 100 for j in range(k)]
            print(f'      {c:9s} predicted-label share {np.round(dist, 2).tolist()}  recall per gold label {np.round(per, 1).tolist()}')
        for c in conds[1:]:
            d = [int(pred(r, c) == r['gold']) - int(pred(r, 'base') == r['gold']) for r in rs]
            w2c = sum(x == 1 for x in d); c2w = sum(x == -1 for x in d); lo, hi = ci(d)
            print(f'      base -> {c:9s} wrong->correct {w2c:3d} correct->wrong {c2w:3d}  delta {100 * np.mean(d):+5.1f} pts  95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]')
        if task == 'TF':
            # reference: rule using only which half holds the speech (first -> Future, second -> Past, never Present)
            rule = [(2 if (r['win'][0] + r['win'][1]) / 2 < r['dur'] / 2 else 0) == r['gold'] for r in rs]
            print(f'      reference: speech-position-only rule (no video) {100 * np.mean(rule):.1f}%  <- a method must beat this to show it uses the video')


def main():
    orig, e1 = load('orig.jsonl'); noisy, e2 = load('noisy.jsonl'); loc, e3 = load('loc.jsonl')
    print('valid/error: orig', len(orig), len(e1), '| noisy', len(noisy), len(e2), '| loc', len(loc), len(e3))
    for name, errs in (('orig', e1), ('noisy', e2), ('loc', e3)):
        if errs: print(f'  {name} errors (first 3):', [x['error'][:90] for x in errs[:3]])
    if orig: table(orig, 'Original clips (paired, option orders rotated)')
    if noisy:
        table(noisy, 'Silence replaced by low-level noise (artifact control)')
        both = {r['idx']: r for r in orig}
        common = [r for r in noisy if r['idx'] in both]
        if common:
            print('\n  same items, original vs noisy (gate_hard accuracy):',
                  round(100 * np.mean([pred(both[r['idx']], 'gate_hard') == r['gold'] for r in common]), 1), 'vs',
                  round(100 * np.mean([pred(r, 'gate_hard') == r['gold'] for r in common]), 1), f'(n={len(common)})')
    if loc:
        print('\n=== Does the model know WHEN the speech is? ===')
        ious = []
        for r in loc:
            if r.get('ts'):
                a, b = r['ts']; c, d = r['ref']
                inter = max(0, min(b, d) - max(a, c)); uni = max(b, d) - min(a, c)
                ious.append(inter / uni if uni > 0 else 0)
        print(f'  timestamp answers parsed: {len(ious)}/{len(loc)}; IoU with audio-active window: median {np.median(ious):.2f}, mean {np.mean(ious):.2f}, share IoU>0.5 {np.mean(np.array(ious) > 0.5):.2f}')
        half_ok = np.mean([int(np.argmax(r['half_scores'])) == r['ref_half'] for r in loc])
        print(f'  "first or second half" (orders rotated): {100 * half_ok:.1f}% correct (chance 50%)')


if __name__ == '__main__':
    main()

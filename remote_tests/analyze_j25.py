"""Analysis of J25 (checks before training).   python analyze_j25.py reason.jsonl [probe_dir]
Check 1: text-only reasoning ceiling: accuracy per (task, style, gold) with prediction histogram vs the 'none' (no timeline) reference.
Check 2: ridge probe of chunk time from the mean hidden state of the video (and audio) tokens of each 2-s chunk, per layer.
         target = chunk index centred within the clip (removes clip length); metric = Pearson r of held-out predictions (clip-grouped
         5-fold CV) vs the same probe trained on labels shuffled within clips (null)."""
import collections, glob, json, sys
import numpy as np

am = lambda v: int(np.argmax(v))
R = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
print('===== CHECK 1: reasoning ceiling (text only)')
for t in ('TA', 'TF', 'VV'):
    S = [r for r in R if r['task'] == t and r['gold'] is not None]
    if not S: continue
    k = len(S[0]['score']); L = 'ABC'[:k]
    for st in sorted({r['style'] for r in S}):
        C = [r for r in S if r['style'] == st]; row = []
        for g in sorted({r['gold'] for r in C}):
            cc = [r for r in C if r['gold'] == g]; ok = sum(am(r['score']) == g for r in cc)
            hist = '/'.join(str(sum(am(r['score']) == j for r in cc)) for j in range(k)); row.append(f'gold {L[g]}: {ok}/{len(cc)} (pred {hist})')
        bal = np.mean([np.mean([am(r['score']) == g for r in C if r['gold'] == g]) for g in {r['gold'] for r in C}])
        print(f'  {t} {st:20s} bal {100 * bal:5.1f}%   ' + '   '.join(row))
    allb = np.mean([np.mean([am(r['score']) == g for r in S if r['gold'] == g]) for g in {r['gold'] for r in S}])
    print(f'  {t} ALL styles balanced accuracy {100 * allb:.1f}%  (chance {100 / k:.0f}%)')
    if t == 'TF' and any('score_bda' in r for r in S):
        B = [r for r in S if 'score_bda' in r]
        for g in (0, 1, 2):
            cc = [r for r in B if r['gold'] == g]
            print(f'  TF wording Before/During/After  gold {"ABC"[g]}: {sum(am(r["score_bda"]) == g for r in cc)}/{len(cc)} (pred ' + '/'.join(str(sum(am(r['score_bda']) == j for r in cc)) for j in range(3)) + ')')
        print(f'  TF wording Before/During/After  balanced {100 * np.mean([np.mean([am(r["score_bda"]) == g for r in B if r["gold"] == g]) for g in (0, 1, 2)]):.1f}%')
    for r in R:
        if r['task'] == t and r['style'] == 'none': print(f'  {t} no timeline given: prediction {"ABC"[am(r["score"])]}  (log-probs {r["score"]})')

if len(sys.argv) > 2:
    import torch
    D = [torch.load(f) for f in sorted(glob.glob(sys.argv[2].rstrip('/') + '/*.pt'))]
    print(f'\n===== CHECK 2: linear time probe, {len(D)} clips, chunks per clip median {int(np.median([d["nc"] for d in D]))}')
    layers = sorted(D[0]['v'].keys())

    def ridge(Xtr, ytr, Xte, alpha):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean())); return B @ w + ytr.mean()

    def probe(src, li, shuffle, alpha, rng):
        X, y, g = [], [], []
        for k, d in enumerate(D):
            h = d[src][li].float().numpy(); cnt = d['nv' if src == 'v' else 'na'].numpy() > 0
            if cnt.sum() < 3: continue
            t = np.arange(d['nc'])[cnt].astype(float); t = t - t.mean()
            if shuffle: t = rng.permutation(t)
            X.append(h[cnt]); y.append(t); g += [k] * int(cnt.sum())
        X, y, g = np.concatenate(X), np.concatenate(y), np.array(g); ids = np.unique(g); rng.shuffle(ids); folds = np.array_split(ids, 5); pred = np.zeros_like(y)
        for f in folds:
            te = np.isin(g, f); pred[te] = ridge(X[~te], y[~te], X[te], alpha)
        return float(np.corrcoef(pred, y)[0, 1]), len(y)

    for src, name in (('v', 'VIDEO tokens'), ('a', 'AUDIO tokens')):
        print(f'  {name}: held-out Pearson r of predicted vs true (centred) chunk index   [shuffled-label null in brackets]')
        for li in layers:
            best = max((probe(src, li, False, a, np.random.default_rng(0))[0], a) for a in (1e2, 1e3, 1e4))
            nul = np.mean([probe(src, li, True, best[1], np.random.default_rng(s))[0] for s in range(3)])
            print(f'    after layer {li:2d}: r = {best[0]:+.3f} (alpha {best[1]:.0f})   [{nul:+.3f}]   n = {probe(src, li, False, best[1], np.random.default_rng(0))[1]}')

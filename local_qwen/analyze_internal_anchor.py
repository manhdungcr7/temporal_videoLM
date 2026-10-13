import json, numpy as np
R = [json.loads(l) for l in open('internal_anchor_probe.jsonl') if l.strip()]
LF = np.stack([np.array(r['log_focus']) for r in R])          # items x layers x heads
VS = np.stack([np.array(r['video_share']) for r in R]); AS = np.stack([np.array(r['audio_share']) for r in R])
n_in = np.array([r['n_in'] for r in R]); n_out = np.array([r['n_out'] for r in R])
print('items', len(R), 'layers', LF.shape[1], 'heads', LF.shape[2], '| in/out video tokens ratio median', np.median(n_in / n_out).round(2))
print('attention share of text positions onto video / audio tokens, median over items, by layer group:')
for a, b in ((0, 9), (9, 18), (18, 27), (27, 36)):
    print(f'  layers {a}-{b-1}: video {np.median(VS[:, a:b]):.3f}  audio {np.median(AS[:, a:b]):.3f}')
# consistent heads: mean log-focus over items, and fraction of items with positive log-focus
mean = LF.mean(0); frac = (LF > 0).mean(0)
print('overall median log-focus (all items, layers, heads):', round(float(np.median(LF)), 3), '| share >0:', round(float((LF > 0).mean()), 3))
top = np.dstack(np.unravel_index(np.argsort(-mean.ravel())[:12], mean.shape))[0]
print('top 12 (layer, head) by mean log-focus  [mean, share of items >0]:')
for l, h in top: print(f'  L{l:02d} H{h:02d}  mean {mean[l, h]:+.2f}  frac {frac[l, h]:.2f}')
# split-half reliability: pick heads on half of the items, evaluate on the other half
rng = np.random.default_rng(0); perm = rng.permutation(len(R)); a, b = perm[:len(R) // 2], perm[len(R) // 2:]
sel = np.dstack(np.unravel_index(np.argsort(-LF[a].mean(0).ravel())[:20], mean.shape))[0]
held = np.mean([LF[b][:, l, h].mean() for l, h in sel])
print(f'top-20 heads chosen on half A, mean log-focus on held-out half B: {held:+.2f} (0 = no preference for the speech window)')
# does the best-head readout beat chance as an ITEM-level window detector? compare attention mass per token inside vs outside
best = [(l, h) for l, h in sel[:20]]
z = np.array([np.mean([LF[i, l, h] for l, h in best]) for i in range(len(R))])
print('item-level mean log-focus of those heads (held-out items):', np.round(z[b], 2).tolist())
# relation to answer correctness
ok = np.array([r['pred'] == r['gold'] for r in R])
print('plain baseline correct:', int(ok.sum()), '/', len(R))
print('mean log-focus of selected heads, correct items:', round(float(z[ok].mean()), 2) if ok.any() else None, '| wrong items:', round(float(z[~ok].mean()), 2) if (~ok).any() else None)

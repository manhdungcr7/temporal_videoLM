"""Offline analysis of J16 (per-token relevance).   python analyze_j16.py j16_s0.jsonl j16_s1.jsonl ...  (shards are merged by idx)
For every question/answer token and every (layer, head): does the visual / audio probe that receives the most attention belong to
the right chunk?  Targets (benchmark construction, analysis only):
  audio  -> a chunk that overlaps the speech window (VAD)
  visual -> the event half of the video      (TA/TF: where the narrated event happens; CMTB: the speech half, the event is at speech time)
Tokens are aligned across items of the same task by (token string, k-th occurrence in the question stem), which is stable because
the SVHalluc questions are fixed templates. Head selection is cross-validated (choose on one half of the items, test on the other,
20 random splits) and compared with the same procedure on label-shuffled data (null), because tokens x 576 heads is a large search."""
import base64, json, sys, collections
import numpy as np

rows = {}
for f in sys.argv[1:]:
    for l in open(f, encoding='utf-8'):
        if not l.strip(): continue
        r = json.loads(l)
        if 'error' not in r: rows[r['idx']] = r
R = list(rows.values())
print('items', len(R), dict(collections.Counter(r['task'] for r in R)))


def ev_half(r):
    sh = 0 if r['speech_first'] else 1
    if r['task'] == 'CMTB': return sh
    pres = (r['task'] == 'TA' and r['gold'] == 0) or (r['task'] == 'TF' and r['gold'] == 1)
    return sh if pres else 1 - sh


def targets(r):
    n = r['n_chunks']; t = np.arange(n)
    sp = ((2 * t + 2) > r['speech'][0]) & ((2 * t) < r['speech'][1])
    half = (t >= n / 2) if ev_half(r) else (t < n / 2)
    return sp, half


def dec(r, k):
    a = np.frombuffer(base64.b64decode(r[k]), dtype=np.uint8).reshape(r['shape'])        # [L, H, rows]
    return a


# hit arrays per item: [L, H, rows]
for r in R:
    sp, half = targets(r)
    r['_a'] = sp[np.minimum(dec(r, 'a_arg'), r['n_chunks'] - 1)].astype(np.float32)
    r['_v'] = half[np.minimum(dec(r, 'v_arg'), r['n_chunks'] - 1)].astype(np.float32)
    seen = collections.Counter(); keys = []
    for tk in r['tokens']:
        seen[tk] += 1; keys.append((tk, seen[tk]))
    r['_keys'] = keys


def cv(H, rng, reps=20):
    """H: [N, L, Hd] per-item hit of one token; returns held-out hit rate of the best head chosen on the other half."""
    N = H.shape[0]; out = []
    for _ in range(reps):
        p = rng.permutation(N); A, B = p[:N // 2], p[N // 2:]
        for S, T in ((A, B), (B, A)):
            l, h = np.unravel_index(np.argmax(H[S].mean(0)), H.shape[1:])
            out.append(H[T, l, h].mean())
    return float(np.mean(out))


rng = np.random.default_rng(0)
for task in ('TA', 'TF', 'CMTB'):
    S = [r for r in R if r['task'] == task]
    if len(S) < 8: continue
    for mod, name in (('_a', 'audio probes -> speech chunk'), ('_v', 'visual probes -> event half')):
        bykey = collections.defaultdict(list)
        for r in S:
            for j, k in enumerate(r['_keys']): bykey[k].append(r[mod][:, :, j])
        res = []
        for k, lst in bykey.items():
            if len(lst) < 0.8 * len(S): continue
            H = np.stack(lst)
            chance = float(np.mean([targets(r)[0 if mod == '_a' else 1].mean() for r in S]))
            hit_best = cv(H, rng)
            # null: same selection procedure after shuffling hits across items (destroys the item-label link)
            Hn = H.copy()
            for l in range(H.shape[1]):
                for h in range(H.shape[2]): Hn[:, l, h] = rng.permutation(Hn[:, l, h])
            res.append((hit_best, cv(Hn, rng), k, len(lst), float(H.mean())))
        res.sort(reverse=True)
        print(f'\n[{task}] {name}: n={len(S)} items, {len(res)} aligned tokens | token-and-head-averaged hit {np.mean([x[4] for x in res]):.3f}')
        print('  top tokens by held-out hit of the CV-selected head   (null = same procedure with shuffled labels)')
        for hb, hn, k, n, avg in res[:8]:
            print(f'    {hb:.3f}  null {hn:.3f}  avg-over-heads {avg:.3f}  n={n:3d}  token {k[0]!r} #{k[1]}')
        print(f'  best null over all tokens: {max(x[1] for x in res):.3f}   share of tokens above best null: '
              f'{np.mean([x[0] > max(y[1] for y in res) for x in res]):.2f}')

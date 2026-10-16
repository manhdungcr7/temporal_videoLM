"""Cross-seed test of J16: choose (layer, head) for every aligned question token on the TRAIN items (seed 2468), then measure the
hit rate of that choice on fresh TEST items (seed 777). No selection happens on the test items.
    python analyze_j16_xseed.py train1.jsonl train2.jsonl ... -- test1.jsonl test2.jsonl ...
Hit = the most-attended probe of the right modality lies in the target (audio -> speech chunk; visual -> event half; CMTB event = speech half).
Reports per (task, modality): the best tokens on TEST (with a 95% Wilson interval, and chance), the mean over all tokens, and how many of the
aligned tokens have a test interval entirely above chance."""
import base64, collections, json, sys
import numpy as np

a = sys.argv[1:]; sep = a.index('--')
def load(files):
    rows = {}
    for f in files:
        for l in open(f, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                if 'error' not in r: rows[r['idx']] = r
    return list(rows.values())
TR, TE = load(a[:sep]), load(a[sep + 1:])
print('train', len(TR), dict(collections.Counter(r['task'] for r in TR)), '| test', len(TE), dict(collections.Counter(r['task'] for r in TE)))


def ev_half(r):
    sh = 0 if r['speech_first'] else 1
    if r['task'] == 'CMTB': return sh
    pres = (r['task'] == 'TA' and r['gold'] == 0) or (r['task'] == 'TF' and r['gold'] == 1)
    return sh if pres else 1 - sh


def prep(R):
    for r in R:
        n = r['n_chunks']; t = np.arange(n)
        sp = ((2 * t + 2) > r['speech'][0]) & ((2 * t) < r['speech'][1]); half = (t >= n / 2) if ev_half(r) else (t < n / 2)
        dec = lambda k: np.frombuffer(base64.b64decode(r[k]), dtype=np.uint8).reshape(r['shape'])
        r['_a'] = sp[np.minimum(dec('a_arg'), n - 1)].astype(np.float32); r['_v'] = half[np.minimum(dec('v_arg'), n - 1)].astype(np.float32)
        r['_chance'] = {'_a': float(sp.mean()), '_v': float(half.mean())}
        seen = collections.Counter(); keys = []
        for tk in r['tokens']: seen[tk] += 1; keys.append((tk, seen[tk]))
        r['_keys'] = keys
prep(TR); prep(TE)
import os
LMIN = int(os.environ.get('LMIN', 0))                  # LMIN=12: only layers >= 12 (layers 1-11 do not depend on the question, J9)
if LMIN:
    for r in TR + TE: r['_a'] = r['_a'][LMIN:]; r['_v'] = r['_v'][LMIN:]
    print('restricted to layers >=', LMIN, '(printed layer numbers are offset by', LMIN, ')')


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - w) / d, (c + w) / d


for task in ('TA', 'TF', 'CMTB'):
    tr = [r for r in TR if r['task'] == task]; te = [r for r in TE if r['task'] == task]
    if len(tr) < 8 or len(te) < 8: continue
    for mod, name in (('_a', 'audio probes -> speech chunk'), ('_v', 'visual probes -> event half')):
        def gather(S):
            d = collections.defaultdict(list)
            for r in S:
                for j, k in enumerate(r['_keys']): d[k].append(r[mod][:, :, j])
            return d
        dtr, dte = gather(tr), gather(te)
        chance = float(np.mean([r['_chance'][mod] for r in te]))
        out = []
        for k in dtr:
            if k not in dte or len(dtr[k]) < 0.8 * len(tr) or len(dte[k]) < 0.8 * len(te): continue
            Htr, Hte = np.stack(dtr[k]), np.stack(dte[k])
            l, h = np.unravel_index(np.argmax(Htr.mean(0)), Htr.shape[1:])
            hits = Hte[:, l, h]; lo, hi = wilson(hits.sum(), len(hits))
            out.append((hits.mean(), lo, hi, k, (int(l), int(h)), Htr[:, l, h].mean(), Hte.mean()))
        out.sort(reverse=True)
        print(f'\n[{task}] {name}: train {len(tr)} test {len(te)} items, chance on test {chance:.3f}, {len(out)} tokens')
        print(f'  mean test hit over tokens {np.mean([x[0] for x in out]):.3f} | tokens with test CI entirely above chance: '
              f'{sum(x[1] > chance for x in out)} of {len(out)}  (about {int(0.025 * len(out) + 0.5)} expected by luck)')
        for hit, lo, hi, k, (l, h), htr, hall in out[:6]:
            print(f'    test {hit:.3f} [{lo:.2f},{hi:.2f}]  (train {htr:.3f}, head-avg test {hall:.3f})  layer {l} head {h}  token {k[0]!r} #{k[1]}')

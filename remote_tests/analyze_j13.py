"""Offline analysis of J13: does any (layer, head) point the question to the right chunk prompts?
Targets: audio prompts -> speech chunks (VAD window); visual prompts -> event half (benchmark construction).
Head selection is cross-validated (choose on one half of the items, test on the other) to avoid max-of-448 luck."""
import json, sys, numpy as np

R = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
R = [r for r in R if 'error' not in r]
print('items', len(R), {t: sum(r['task'] == t for r in R) for t in ('TA', 'TF')})


def ev_half(r):
    sh = 0 if r['speech_first'] else 1
    pres = (r['task'] == 'TA' and r['gold'] == 0) or (r['task'] == 'TF' and r['gold'] == 1)
    return sh if pres else 1 - sh


def targets(r):
    nc = r['n_chunks']; t = np.arange(nc)
    sp = ((2 * t + 2) > r['speech'][0]) & ((2 * t) < r['speech'][1])                    # chunks overlapping speech
    half = (t >= nc / 2) if ev_half(r) else (t < nc / 2)
    return sp, half


def metrics(key):
    """per item: [L,H] share of audio relevance on speech chunks, share of visual relevance on event-half chunks,
    plus argmax hits; and the chance level of each."""
    out = []
    for r in R:
        x = np.array(r[key]); v = x[..., 0::2]; a = x[..., 1::2]; sp, half = targets(r)
        v = (v + 1e-9) / (v + 1e-9).sum(-1, keepdims=True); a = (a + 1e-9) / (a + 1e-9).sum(-1, keepdims=True)
        out.append(dict(a_share=a[..., sp].sum(-1), a_chance=sp.mean(), a_hit=sp[a.argmax(-1)].astype(float),
                        v_share=v[..., half].sum(-1), v_chance=half.mean(), v_hit=half[v.argmax(-1)].astype(float), task=r['task']))
    return out


def report(key):
    M = metrics(key); n = len(M)
    for tgt in ('a', 'v'):
        share = np.stack([m[f'{tgt}_share'] for m in M]); chance = np.array([m[f'{tgt}_chance'] for m in M])
        hit = np.stack([m[f'{tgt}_hit'] for m in M])
        gain = share - chance[:, None, None]                                          # [n,L,H] above chance
        name = 'audio->speech chunks' if tgt == 'a' else 'visual->event half'
        print(f'\n[{key}] {name}: chance share {chance.mean():.3f}')
        print(f'  head-average share {share.mean((1, 2)).mean():.3f}; layer-18 avg {share[:, 18].mean():.3f}')
        g = gain.mean(0); L, H = np.unravel_index(np.argsort(-g, None)[:5], g.shape)
        print('  top-5 heads (all items, optimistic):', [(int(l), int(h), round(float(share[:, l, h].mean()), 3),
                                                         round(float(hit[:, l, h].mean()), 2)) for l, h in zip(L, H)])
        # cross-validation: choose best head on even items, test on odd items, and vice versa
        cv = []
        rng = np.random.default_rng(0)
        for rep in range(20):
            idx = rng.permutation(n); A, B = idx[:n // 2], idx[n // 2:]
            for S, T in ((A, B), (B, A)):
                l, h = np.unravel_index(np.argmax(gain[S].mean(0)), g.shape)
                l2, h2 = np.unravel_index(np.argmax(hit[S].mean(0)), g.shape)                # selected by argmax hit
                cv.append((share[T, l, h].mean() - chance[T].mean(), hit[T, l, h].mean(), hit[T, l2, h2].mean(), chance[T].mean()))
        cv = np.array(cv)
        print(f'  CROSS-VALIDATED best head: share above chance {cv[:, 0].mean():+.3f}, argmax hit {cv[:, 1].mean():.2f};'
              f' head chosen by hit -> held-out hit {cv[:, 2].mean():.2f} (chance {cv[:, 3].mean():.2f})')
        hm = hit.mean(0); print('  heads with argmax hit >= 0.9:', int((hm >= 0.9).sum()), 'of', hm.size, '| layers', sorted(set(np.where(hm >= 0.9)[0].tolist())))
        for t in ('TA', 'TF'):
            sel = [i for i, m in enumerate(M) if m['task'] == t]
            if sel: print(f'  {t}: head-average share {share[sel].mean():.3f} vs chance {chance[sel].mean():.3f}')


for key in ('last', 'mean'): report(key)

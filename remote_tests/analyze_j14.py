"""Offline analysis of J14 (hop-2 gate): from the anchor (speech) chunks, does probe similarity point to the visual chunk of
the narrated event?   python analyze_j14.py j14_hop2_gate.jsonl [j13d_oneword.jsonl]
alpha1 (hop 1):  'vad'   = chunks overlapping the VAD speech window (a fixed automatic component);
                 'heads' = argmax of the audio-probe relevance of speech heads chosen on the OTHER half of the items (J13d).
S (hop 2): every similarity in the file (cos_L*, ccos_L*, vocab_dot, vocab_bc, vocab_jss), in two forms:
                 raw      alpha2(t') = mean over anchor rows t of S[t, t']
                 contrast alpha2(t') = that minus the mean over ALL audio rows (removes visual "hub" chunks)
Hit = argmax alpha2 lies in the event half (benchmark construction, analysis only). Baselines: always the speech half.
Also: diagonal bias on silent rows (same-chunk similarity without any speech = the number-token / position confound)."""
import json, sys, numpy as np

R = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
R = [r for r in R if 'error' not in r]
H13 = {}
if len(sys.argv) > 2:
    for l in open(sys.argv[2], encoding='utf-8'):
        r = json.loads(l)
        if 'error' not in r: H13[r['idx']] = np.array(r['mean'])[..., 1::2]          # [L, H, n] audio-probe relevance
print('items', len(R), {t: sum(r['task'] == t for r in R) for t in ('TA', 'TF')}, '| with J13d heads', sum(r['idx'] in H13 for r in R))


def ev_half(r):
    sh = 0 if r['speech_first'] else 1
    pres = (r['task'] == 'TA' and r['gold'] == 0) or (r['task'] == 'TF' and r['gold'] == 1)
    return sh if pres else 1 - sh


def chunks(r):
    t = np.arange(r['n_chunks'])
    sp = ((2 * t + 2) > r['speech'][0]) & ((2 * t) < r['speech'][1])
    half = (t >= r['n_chunks'] / 2) if ev_half(r) else (t < r['n_chunks'] / 2)
    sph = (t >= r['n_chunks'] / 2) if not r['speech_first'] else (t < r['n_chunks'] / 2)
    return sp, half, sph


def head_alpha1(rng, reps=20):
    """cross-validated speech heads from J13d: choose the (layer, head) whose argmax most often hits speech on half A,
    use it on half B. Returns {idx: one-hot alpha1} averaged over repetitions (as a distribution)."""
    ids = [r['idx'] for r in R if r['idx'] in H13]; byid = {r['idx']: r for r in R}
    if not ids: return {}
    hit = np.stack([chunks(byid[i])[0][H13[i].argmax(-1)] for i in ids]).astype(float)       # [N, L, H]
    out = {i: np.zeros(byid[i]['n_chunks']) for i in ids}
    for _ in range(reps):
        p = rng.permutation(len(ids)); A, B = p[:len(ids) // 2], p[len(ids) // 2:]
        for S_, T_ in ((A, B), (B, A)):
            l, h = np.unravel_index(np.argmax(hit[S_].mean(0)), hit.shape[1:])
            for j in T_: i = ids[j]; out[i][H13[i][l, h].argmax()] += 1
    return {i: v / v.sum() for i, v in out.items()}


def pmi_alpha1(r, var, layers=range(12, 36), alpha=2.0):
    """hop 1 without choosing heads: question vs NEUTRAL-question attention to the AUDIO probes, averaged over layers 12-35
    and all heads, PMI as in J9: rel = p * (p / p_neutral)^alpha. Returns (one-hot of argmax, raw argmax)."""
    if 'rel_question' not in r[var]: return None, None
    q = np.array(r[var]['rel_question'])[list(layers)].mean((0, 1))[1::2]; n0 = np.array(r[var]['rel_neutral'])[list(layers)].mean((0, 1))[1::2]
    p = q / q.sum(); p0 = n0 / n0.sum(); rel = p * (p / np.maximum(p0, 1e-9)) ** alpha
    w = np.zeros(len(p)); w[int(np.argmax(rel))] = 1.0
    return w, int(np.argmax(p))


A1H = head_alpha1(np.random.default_rng(0))
for var in ('plain', 'quote', 'nonum_quote'):
    if var in R[0] and 'rel_question' in R[0][var]:
        hp = [chunks(r)[0][pmi_alpha1(r, var)[0].argmax()] for r in R]; hr = [chunks(r)[0][pmi_alpha1(r, var)[1]] for r in R]
        print(f'hop-1 ({var}): argmax on a speech chunk  raw {100 * np.mean(hr):.1f}%  PMI {100 * np.mean(hp):.1f}%  (chance {100 * np.mean([chunks(r)[0].mean() for r in R]):.1f}%)')
base_speech = np.mean([np.array_equal(chunks(r)[1], chunks(r)[2]) for r in R])
print(f'baseline "event is in the speech half": {100 * base_speech:.1f}%   (chance 50)')
if A1H:
    print('hop-1 heads: argmax on a speech chunk', round(100 * np.mean([chunks(r)[0][A1H[r['idx']].argmax()] for r in R if r['idx'] in A1H]), 1), '%')

for var in ('plain', 'quote', 'nonum_quote'):
    if var not in R[0]: continue
    keys = list(R[0][var]['S'])
    print(f'\n=== probe variant: {var}   (audio content mass median {np.median([np.median(r[var]["audio_content_mass"]) for r in R]):.3f})')
    rows = []
    for k in keys:
        for form in ('raw', 'contrast'):
            for a1 in ('vad', 'heads', 'pmi'):
                hits = []
                for r in R:
                    S = np.array(r[var]['S'][k]); sp, half, _ = chunks(r)
                    if a1 == 'vad': w = sp.astype(float)
                    elif a1 == 'pmi':
                        w = pmi_alpha1(r, var)[0]
                        if w is None: continue
                    else:
                        if r['idx'] not in A1H: continue
                        w = A1H[r['idx']]
                    if w.sum() == 0: continue
                    w = w / w.sum(); a2 = w @ S
                    if form == 'contrast': a2 = a2 - S.mean(0)
                    hits.append(half[int(np.argmax(a2))])
                if hits: rows.append((k, form, a1, 100 * np.mean(hits), len(hits)))
        # diagonal bias on silent audio rows (no speech there, so a high S[t, t] is not about content)
        diag = []
        for r in R:
            S = np.array(r[var]['S'][k]); sp = chunks(r)[0]; n = len(S)
            for t in np.where(~sp)[0]:
                others = [S[t, j] for j in range(n) if j != t]
                if others: diag.append(S[t, t] - np.mean(others))
        rows.append((k, 'diag_bias_silent', '', float(np.mean(diag)) if diag else float('nan'), len(diag)))
    for k, form, a1, v, n in sorted(rows, key=lambda x: (x[1] == 'diag_bias_silent', -x[3] if x[1] != 'diag_bias_silent' else 0)):
        if form == 'diag_bias_silent': print(f'  {k:12s} diag bias on silent rows: {v:+.4f} (n={n})')
        else: print(f'  {k:12s} {form:8s} alpha1={a1:5s} event-half hit {v:5.1f}%  (n={n})')
    print('  examples of content words (first item, chunk: audio | visual):')
    r = R[0]
    for t in range(min(r['n_chunks'], 8)):
        print(f'    t={t + 1:2d}  {"/".join(r[var]["top_audio"][t][:3]):30s} | {"/".join(r[var]["top_visual"][t][:3])}')

"""Analysis of J19 (layer lens).   python analyze_j19.py j19_layer_lens.jsonl [more shards]
Per layer: accuracy of the answer read from the residual stream (content scores summed over rotations), per (task, gold) at the layers the
friend asked about (16, 22, 30, 35 = last), letter-POSITION bias (does the layer pick the same letter slot whatever the content?), how much the
AV answer follows the video-only vs the audio-only answer, and where the last token looks (video / audio / text attention mass)."""
import collections, json, sys
import numpy as np

R = []
for f in sys.argv[1:]:
    R += [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
R = [r for r in R if 'error' not in r]
NL = len(R[0]['lens_av']); SHOW = (8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 35)
am = lambda v: int(np.argmax(v))
print('items', len(R), dict(collections.Counter(r['task'] for r in R)))
T = [r for r in R if r['data_type'] == 'temporal']; Sm = [r for r in R if r['data_type'] == 'semantic']


def acc(S, key, l=None):
    if not S: return float('nan')
    return 100 * np.mean([am(np.array(r[key])[l] if l is not None else r[key]) == r['gold'] for r in S])


def bal(S, key, l):
    out = []
    for t in sorted({r['task'] for r in S}):
        for g in sorted({r['gold'] for r in S if r['task'] == t}):
            c = [r for r in S if r['task'] == t and r['gold'] == g]
            out.append(np.mean([am(np.array(r[key])[l]) == g for r in c]))
    return 100 * np.mean(out)


print('\n1) Accuracy of the answer read at each layer (video+audio input).  all / temporal / semantic  | balanced over (task,gold) cells')
for l in SHOW:
    print(f'   L{l:2d}  {acc(R, "lens_av", l):5.1f} / {acc(T, "lens_av", l):5.1f} / {acc(Sm, "lens_av", l):5.1f}   | balanced {bal(R, "lens_av", l):5.1f}' + ('   <- final' if l == NL - 1 else ''))

print('\n2) Correct/wrong per (task, gold) at layers 16 / 22 / 30 / 35')
for t in ('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CSMB'):
    for g in sorted({r['gold'] for r in R if r['task'] == t}):
        c = [r for r in R if r['task'] == t and r['gold'] == g]
        cells = []
        for l in (16, 22, 26, 30, NL - 1):
            k = sum(am(np.array(r['lens_av'])[l]) == g for r in c); cells.append(f'{k}/{len(c) - k}')
        print(f'   {t:5s}/{"ABCD"[g]}  n={len(c):2d}   ' + '   '.join(cells))

print('\n3) Letter-POSITION bias per layer: share of rotations where the layer picks letter A / B / C (whatever content is there),')
print('   and share of items where the layer picks the SAME letter in every rotation (a pure position choice, content ignored)')
for l in SHOW:
    picks = collections.Counter(); same = []
    for r in R:
        lets = [int(np.argmax(np.array(x['letter_logprob'])[l])) for x in r['rot_av']]
        picks.update(lets); same.append(len(set(lets)) == 1)
    tot = sum(picks.values())
    print(f'   L{l:2d}  A {100 * picks[0] / tot:5.1f}%  B {100 * picks[1] / tot:5.1f}%  C {100 * picks[2] / tot:5.1f}%   same letter in all rotations: {100 * np.mean(same):5.1f}%')

print('\n4) Content decisiveness: median margin between the best and second option (summed log-prob), per layer')
print('   ' + '  '.join(f'L{l}:{np.median([np.sort(np.array(r["lens_av"])[l])[-1] - np.sort(np.array(r["lens_av"])[l])[-2] for r in R]):.2f}' for l in SHOW))

print('\n5) Which modality does the AV answer follow?  agreement of the AV answer at layer l with the FINAL answer of video-only / audio-only input,')
print('   and accuracy of video-only / audio-only at the same layer')
for l in SHOW:
    av = [am(np.array(r['lens_av'])[l]) for r in R]
    fv = [am(r['final_vision']) for r in R]; fa = [am(r['final_audio']) for r in R]
    print(f'   L{l:2d}  agree w/ video-only {100 * np.mean([x == y for x, y in zip(av, fv)]):5.1f}%  w/ audio-only {100 * np.mean([x == y for x, y in zip(av, fa)]):5.1f}%'
          f'   | acc video-only {acc(R, "lens_vision", l):5.1f}  audio-only {acc(R, "lens_audio", l):5.1f}')
for name, S in (('temporal', T), ('semantic', Sm)):
    print(f'   final answers ({name}): AV {acc(S, "final_av"):.1f}  video-only {acc(S, "final_vision"):.1f}  audio-only {acc(S, "final_audio"):.1f}')

print('\n6) Where the last token looks (attention mass, mean over heads and items): video | audio | text')
RM = [r for r in R if 'mass_av' in r]; M = np.array([r['mass_av'] for r in RM])
nv = np.array([r.get('n_video', np.nan) for r in RM], float); na = np.array([r.get('n_audio', np.nan) for r in RM], float)
print(f'   media tokens per item: video {np.nanmean(nv):.0f}, audio {np.nanmean(na):.0f}  (video/audio token ratio {np.nanmean(nv / na):.2f})')
for l in SHOW:
    ratio = M[:, l, 0].mean() / max(1e-6, M[:, l, 1].mean())
    per_tok = np.nanmean((M[:, l, 0] / nv) / np.maximum(1e-9, M[:, l, 1] / na))
    print(f'   L{l:2d}  {M[:, l, 0].mean():.3f} | {M[:, l, 1].mean():.3f} | {M[:, l, 2].mean():.3f}    mass ratio video/audio {ratio:.2f}   PER-TOKEN ratio {per_tok:.2f}  (<1: one audio token is read more than one video token)')

print('\n7) Found then lost: items where some middle layer (12-30) is right but the final answer is wrong, and the reverse')
lost = [r for r in R if am(r['final_av']) != r['gold'] and any(am(np.array(r['lens_av'])[l]) == r['gold'] for l in range(12, 31))]
gain = [r for r in R if am(r['final_av']) == r['gold'] and all(am(np.array(r['lens_av'])[l]) != r['gold'] for l in range(12, 31))]
print(f'   found then lost: {len(lost)} items   | right only at the end: {len(gain)} items   (of {len(R)})')

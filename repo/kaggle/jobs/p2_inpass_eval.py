"""P2 — module A at scale: ONE-PASS self-routing (relevance measured at layer 12, re-allocation + source gate in layers 13-27)
vs the two-pass version, on a fresh seed. Needs the J8 helpers and inpass.py.
Conditions (rotated options): base | twopass (relevance from layers 12-35, separate pass) | onepass (meas [12] -> apply 13-27,
target modality from the model's own text self-query) | onepass_seen (target forced to 'seen').
The neutral prior is computed once per video (layer band of the measurement). Writes OUT/p2_inpass.jsonl."""
import json, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())
exec((ROOT / 'inpass.py').read_text(encoding='utf-8'), globals())
PER_LABEL = int(globals().get('PER_LABEL', 10)); SEED = int(globals().get('SEED', 777))
MEAS1, APPLY1 = [12], list(range(13, 28))


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    target = self_target(stem); rec = dict(dur=round(dur, 2), target=target)
    rec['base'] = score(media, stem, texts)
    P = profiles(media, prompt_of(stem, texts, tuple(range(k)))); P0 = profiles(media, NEUTRAL); rel2 = pmi(P['av'], P0['av'])
    def two_pass(inp):
        lay = Lay(inp); m = bias_mask(lay, rel2, 4.0, ('v', 'a')); rr = torch.where(lay.rows)[0]
        if target in ('seen', 'heard'):
            cols = torch.where(lay.a if target == 'seen' else lay.v)[0]; m[rr.unsqueeze(1), cols.unsqueeze(0)] = float('-inf')
        return forward(inp, lay, m, layers=ROUTE)[0]
    rec['twopass'] = score(media, stem, texts, fwd=two_pass)
    prior = neutral_prior(media, MEAS1)
    rec['onepass'] = score(media, stem, texts, fwd=lambda inp: inpass_forward(inp, prior, target, MEAS1, APPLY1)[0])
    rec['onepass_seen'] = score(media, stem, texts, fwd=lambda inp: inpass_forward(inp, prior, 'seen', MEAS1, APPLY1)[0])
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
C = ('base', 'twopass', 'onepass', 'onepass_seen')
R = run_loop(items, 'p2_inpass.jsonl', item, show=C)
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in C} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in C}
print(RESULT, flush=True)

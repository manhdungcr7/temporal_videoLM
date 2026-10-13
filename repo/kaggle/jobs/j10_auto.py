"""J10 — FULLY AUTOMATIC in-model method (QTAR + self-decided target modality). Uses the J8 helpers (j8_acar.py).
For any (video, question):
 1 TIME   relevance per 2-s chunk = question-row attention to video+audio tokens, PMI against a neutral question (alpha 2)
 2 SOURCE the model itself, text only: "To answer this question, what must you rely on? seen / heard / both"
 3 READ   layers 13-27: question/answer rows get LAMBDA*log(rel/max rel) on the tokens of each chunk, and
            seen  -> audio tokens blocked for those rows (look with the sound off)
            heard -> video tokens blocked
            both  -> layers 13-20 read both (hear), layers 21-27 audio blocked (then look)
Conditions (rotated options):
  base, oracle_route (known speech window, video only, audio blocked)
  auto       the method above (relevance av)          auto_a   same with audio-only relevance (anchor = sound)
  force_v    target forced to "seen" (shows whether errors come from the source decision or from the mechanism)
  force_both target forced to "both"
  no_gate    time re-weighting only (no source decision)
Writes OUT/j10_auto.jsonl (+ the self-query scores and relevance profiles for failure analysis)."""
import json, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())

PER_LABEL = int(globals().get('PER_LABEL', 8)); SEED = int(globals().get('SEED', 909))
LAM = float(globals().get('LAMBDA', 4.0))
SELFQ = 'To answer the following question about a video, what must you rely on? Question: "{q}"'
SQ_OPTS = ['what is seen in the video frames', 'what is heard in the audio', 'both what is seen and what is heard']
SPLIT = 21


def policy_masks(lay, rel, target):
    both = bias_mask(lay, rel, LAM, ('v', 'a'))
    rr = torch.where(lay.rows)[0]
    def blocked(mod):
        m = both.clone(); cols = torch.where(lay.a if mod == 'a' else lay.v)[0]
        m[rr.unsqueeze(1), cols.unsqueeze(0)] = float('-inf'); return m
    if target == 'seen': b = blocked('a'); return {li: b for li in ROUTE}
    if target == 'heard': b = blocked('v'); return {li: b for li in ROUTE}
    if target == 'none': return {li: both for li in ROUTE}
    b = blocked('a'); return {li: (both if li < SPLIT else b) for li in ROUTE}          # both: hear, then look


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    sq = text_only_scores(SELFQ.format(q=stem), SQ_OPTS); target = ['seen', 'heard', 'both'][int(np.argmax(sq))]
    rec['selfquery'] = [round(x, 3) for x in sq]; rec['target'] = target
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts))))); P0 = profiles(media, NEUTRAL)
    rel_av, rel_a = pmi(P['av'], P0['av']), pmi(P['a'], P0['a'])
    rec['rel_av'] = np.round(rel_av, 4).tolist(); rec['rel_a'] = np.round(rel_a, 4).tolist()
    nc = len(rel_av); act = np.array(chunk_activity(a, nc)) > 0.3; rec['speech_chunks'] = act.astype(int).tolist()

    def mk(fn):
        def f(inp):
            lay = Lay(inp); return forward(inp, lay, fn(lay))[0]
        return f
    rec['base'] = score(media, stem, texts)
    rec['oracle_route'] = score(media, stem, texts, fwd=mk(lambda lay: window_mask(lay, w[0], w[1], True)))
    rec['auto'] = score(media, stem, texts, fwd=mk(lambda lay: policy_masks(lay, rel_av, target)))
    rec['auto_a'] = score(media, stem, texts, fwd=mk(lambda lay: policy_masks(lay, rel_a, target)))
    rec['force_v'] = score(media, stem, texts, fwd=mk(lambda lay: policy_masks(lay, rel_av, 'seen')))
    rec['force_both'] = score(media, stem, texts, fwd=mk(lambda lay: policy_masks(lay, rel_av, 'both')))
    rec['no_gate'] = score(media, stem, texts, fwd=mk(lambda lay: policy_masks(lay, rel_av, 'none')))
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
C = ('base', 'oracle_route', 'auto', 'auto_a', 'force_v', 'force_both', 'no_gate')
R = run_loop(items, 'j10_auto.jsonl', item, show=C)
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in C} | {'n': sum(r['task'] == t for r in OK),
          'targets': {g: sum(r['target'] == g for r in OK if r['task'] == t) for g in ('seen', 'heard', 'both')}} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in C}
print(RESULT, flush=True)

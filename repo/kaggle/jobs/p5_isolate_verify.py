"""P5 — the full "anchor + isolated parallel verification" method at scale (fresh seed). Needs J8 + P1 helpers and the
parallel_verify / locate / TPL definitions of remote_tests/debug_parallel.py.
Per item:  anchor window (question->audio attention, PMI, grown)  |  inner speech  |  ONE forward with isolated, time-aligned
mini-questions for every window x content (heard phrase + content options)  ->  evidence matrix.
Readout from the two pointers (fixed rule by ANSWER FORMAT, no task names; a probe of what the pointers contain):
  options are content phrases  -> the option with the largest (score in the anchor window - best score elsewhere)
  Yes / No options             -> Yes iff the heard phrase is best verified inside the anchor window
  temporal-relation options    -> before / during / after from the window where the heard phrase is best verified
Writes OUT/p5_isolate_verify.jsonl (with the full evidence matrix for learned-readout analysis)."""
import json, re, numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 12)); SEED = int(globals().get('SEED', 4242))
REL_WORDS = {'past': 'before', 'present': 'during', 'future': 'after', 'before': 'before', 'during': 'during', 'after': 'after'}


def answer_format(texts):
    low = [t.lower().strip() for t in texts]
    if set(low) <= {'yes', 'no'}: return 'yesno'
    if all(l in REL_WORDS for l in low): return 'relation'
    return 'content'


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), speech_first=speech_first(w, dur)); rec['base'] = score(media, stem, texts)
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts))))); P0 = profiles(media, NEUTRAL)
    s0, s1 = grow_window(pmi(P['a'], P0['a']), P['a']); T0, T1 = 2.0 * s0, min(dur, 2.0 * s1)
    wins = [(n, t0, t1) for n, t0, t1 in (('before the speech', 0.0, T0), ('while the speech is heard', T0, T1), ('after the speech', T1, dur)) if t1 - t0 >= 1.0]
    names = [x[0] for x in wins]; anc = names.index('while the speech is heard')
    claim = inner_speech(media); fmt = answer_format(texts)
    contents = [claim] + (texts if fmt == 'content' else [])
    ev = parallel_verify(media, wins, contents)
    rec.update(anchor=[T0, T1], windows=names, claim=claim, fmt=fmt, evidence=ev)
    if fmt == 'content':
        con = [ev[t][anc] - max([x for j, x in enumerate(ev[t]) if j != anc] or [-1e9]) for t in texts]
        pred = int(np.argmax(con))
    else:
        best = names[int(np.argmax(ev[claim]))]
        rel = {'before the speech': 'before', 'while the speech is heard': 'during', 'after the speech': 'after'}[best]
        if fmt == 'yesno': pred = [t.lower() for t in texts].index('yes' if rel == 'during' else 'no')
        else: pred = [REL_WORDS[t.lower()] for t in texts].index(rel)
    rec['pred'] = pred; rec['method'] = [1.0 if j == pred else 0.0 for j in range(len(texts))]
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
R = run_loop(items, 'p5_isolate_verify.jsonl', item, show=('base', 'method'))
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in ('base', 'method')} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in ('base', 'method')}
print(RESULT, flush=True)

"""CEILING OF THE REASONING STEP: given a PERFECT evidence timeline (built from the labels; upper bound only, never a method
result), does Qwen2.5-Omni-3B answer the real SVHalluc temporal questions (TA/TF/CMTB) correctly? Text only, no media.
Formats of the timeline:
  rel     segments named relative to the speech ("before the speech" / "while the speech is heard" / "after the speech") + times
  abs     only absolute times (the model has to compare numbers)
  rel_q   rel, plus the question is preceded by "Answer using only the timeline."
The HEARD sentence is the claim in 4 tenses (future / present / past / neutral) to test whether the tense shortcut survives
explicit evidence. Option orders rotated, letter log-probs summed per option content. Writes OUT/timeline_reasoning.json."""
import json, random
import numpy as np

FMTS = ('rel_q', 'list', 'syn', 'list_syn')
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
ACTS = ['add the pork to the pan', 'mash the potatoes', 'slice the tomatoes', 'pour the sauce over the pasta', 'crack the eggs into the bowl',
        'stir the soup', 'spread butter on the bread', 'chop the onions', 'fry the bacon', 'mix the flour and water']
TENSE = {'future': 'Next I will {a}.', 'present': 'Now I am going to {a} right now.', 'past': 'I have already finished: {a}.', 'neutral': '{a}.'}
rng = random.Random(7)
N_PER = int(globals().get('TR_N_PER', 12))
cells = {}
for i, r in enumerate(rows):
    if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))


SYN = {'before the speech': 'before the speech = earlier than the speech = in the past relative to the speech',
       'while the speech is heard': 'while the speech is heard = at the same time as the speech = in the present relative to the speech',
       'after the speech': 'after the speech = later than the speech = in the future relative to the speech'}


def seg_line(fmt, name, t0, t1, parts):
    label = SYN[name] if 'syn' in fmt else name
    when = f'{t0:.1f}-{t1:.1f} s' + (f' ({label})' if fmt != 'abs' else '')
    return f'- {when}: ' + '; '.join(parts)


def timeline(task, gold, texts, fmt, tense):
    dur = round(rng.uniform(20, 40), 1); s0 = round(rng.uniform(0.3, 0.45) * dur, 1) if rng.random() < 0.5 else round(rng.uniform(0.05, 0.15) * dur, 1)
    s1 = round(s0 + rng.uniform(4, 8), 1)
    segs = [('before the speech', 0.0, s0), ('while the speech is heard', s0, s1), ('after the speech', s1, dur)]
    act = rng.choice(ACTS); heard = TENSE[tense].format(a=act)
    seen = {k: [] for k in range(3)}
    if task in ('TA', 'TF'):
        if task == 'TA': where = 1 if gold == 0 else rng.choice([0, 2])
        else: where = {0: 0, 1: 1, 2: 2}[gold]
        for k in range(3): seen[k].append(f'SEEN "{act}": {"yes" if k == where else "no"}')
    else:                                   # CMTB: gold option shown during the speech, distractors in the other segments
        others = [j for j in range(len(texts)) if j != gold]
        place = {gold: 1, others[0]: 0, others[1]: 2}
        for j, t in enumerate(texts):
            for k in range(3): seen[k].append(f'SEEN "{t}": {"yes" if place[j] == k else "no"}')
    if 'list' in fmt:
        seen = {k: ['SEEN: ' + (', '.join('"' + x.split('"')[1] + '"' for x in seen[k] if x.endswith('yes')) or 'none of the listed actions')] for k in seen}
    lines = []
    for k, (name, t0, t1) in enumerate(segs):
        parts = ([f'HEARD: "{heard}"'] if k == 1 else ['HEARD: (no speech)']) + seen[k]
        lines.append(seg_line(fmt, name, t0, t1, parts))
    head = 'Evidence timeline of the video (SEEN = checked on the video with the sound off; HEARD = what is said):\n'
    return head + '\n'.join(lines) + '\n'


out = []
for (task, ans), pool in sorted(cells.items()):
    rng2 = random.Random(3); pool = pool[:]; rng2.shuffle(pool)
    for i, r in pool[:N_PER]:
        stem, texts = parse_question(r['question']); gold = 'ABC'.index(r['answer'])
        for tense in (('neutral', 'future', 'past') if task != 'CMTB' else ['neutral']):
            rec = dict(idx=i, task=task, gold=gold, tense=tense)
            st = rng.getstate()
            for fmt in FMTS:
                rng.setstate(st)                                   # same timeline geometry for every format
                tl = timeline(task, gold, texts, fmt, tense)
                q = tl + 'Answer using only the timeline. ' + stem
                rec[fmt] = text_only_scores(q, texts)
            rng.setstate(st); tl = timeline(task, gold, texts, 'list_syn', tense)
            rec['tf_verify'] = [round(float(np.subtract(*text_only_scores(tl + f'Question: {stem} Proposed answer: {t}. Given only the timeline, is the proposed answer correct?', ['Yes', 'No']))), 3) for t in texts]
            rng.random()
            out.append(rec)
    print(task, ans, 'done', flush=True)
(OUT / 'timeline_reasoning_v2.json').write_text(json.dumps(out), encoding='utf-8')
am = lambda v: int(np.argmax(v))
RESULT = {}
for task in ('TA', 'TF', 'CMTB'):
    S = [r for r in out if r['task'] == task]
    d = {fmt: round(100 * np.mean([am(r[fmt]) == r['gold'] for r in S]), 1) for fmt in FMTS + ('tf_verify',)}
    if task != 'CMTB':
        for f_ in ('list_syn', 'tf_verify'):
            d[f_ + '_by_gold'] = {g: round(100 * np.mean([am(r[f_]) == r['gold'] for r in S if r['gold'] == g]), 1) for g in sorted({r['gold'] for r in S})}
            d[f_ + '_by_tense'] = {t: round(100 * np.mean([am(r[f_]) == r['gold'] for r in S if r['tense'] == t]), 1) for t in ('neutral', 'future', 'past')}
    d['n'] = len(S); RESULT[task] = d
print(json.dumps(RESULT, indent=1), flush=True)

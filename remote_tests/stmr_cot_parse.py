"""Correct CoT scoring: the answer is PARSED from the model's own reasoning (truncated at the first runaway turn), for every
option order, and the orders vote. Same STMR v2 timelines as before (OUT/stmr_v2_offline.json). Automatic, text only.
Prompts: cot (which part of the timeline does the question ask about) and cot2 (evidence first). Writes OUT/stmr_cot_parse.json."""
import json, re
import numpy as np, torch

V2 = json.loads((OUT / 'stmr_v2_offline.json').read_text(encoding='utf-8'))
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
PROMPTS = {'cot': '\nThink step by step about which part of the timeline the question asks about, then give the answer.',
           'cot2': '\nFirst, for every SEEN item, say whether it is seen before, while, or after the speech. Then say what the question asks about. Then give the answer.'}
STOP = re.compile(r'\n\s*(Human|User|Question)\s*:|If you have any other questions', re.I)


def parse_answer(txt, shown):
    """shown: option texts in the order they were displayed (A, B, C, ...). Returns the displayed index or None."""
    txt = STOP.split(txt)[0]
    m = list(re.finditer(r'answer\s+is\W{0,4}\(?([A-D])\b', txt, re.I)) or list(re.finditer(r'\(([A-D])\)', txt))
    if m:
        j = 'ABCD'.index(m[-1].group(1).upper())
        if j < len(shown): return j
    tail = txt[txt.lower().rfind('answer'):] if 'answer' in txt.lower() else txt
    hits = [j for j, t in enumerate(shown) if re.search(r'\b' + re.escape(t.lower()) + r'\b', tail.lower())]
    return hits[-1] if len(hits) == 1 else None


def cot_vote(prefix, texts, suffix):
    k = len(texts); votes = np.zeros(k); outs = []
    for perm in rotations(k):
        shown = [texts[j] for j in perm]
        txt = gen(render(prefix, texts, perm) + suffix, 160)
        p = parse_answer(txt, shown); outs.append(STOP.split(txt)[0][:300])
        if p is not None: votes[perm[p]] += 1
    return votes.tolist(), outs


out = []; am = lambda v: int(np.argmax(v))
for x in V2:
    stem, texts = parse_question(rows[x['idx']]['question'])
    q = x['timeline'] + 'Answer using only the timeline. ' + stem
    rec = dict(idx=x['idx'], task=x['task'], gold=x['gold'], base=x['base'], plain=x['plain'])
    for name, suf in PROMPTS.items():
        v, t = cot_vote(q, texts, suf)
        rec[name + '_votes'] = v; rec[name + '_texts'] = t
        rec[name] = (np.array(v) + 1e-3 * np.exp(np.array(x['plain']))).tolist()      # ties broken by the plain score
    out.append(rec)
    print(x['task'], x['idx'], 'gold', x['gold'], {n: (rec[n + '_votes'], am(rec[n]) == x['gold']) for n in PROMPTS}, flush=True)
(OUT / 'stmr_cot_parse.json').write_text(json.dumps(out), encoding='utf-8')
RESULT = {t: {c: round(100 * np.mean([am(r[c]) == r['gold'] for r in out if r['task'] == t]), 1) for c in ('base', 'plain', 'cot', 'cot2')} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: round(100 * np.mean([am(r[c]) == r['gold'] for r in out]), 1) for c in ('base', 'plain', 'cot', 'cot2')}
RESULT['unparsed_share'] = {n: round(float(np.mean([sum(r[n + '_votes']) < len(r[n + '_votes']) for r in out])), 3) for n in PROMPTS}
print(json.dumps(RESULT, indent=1), flush=True)

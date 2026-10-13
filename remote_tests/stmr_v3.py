"""STMR v3 composition: SOURCE-SEPARATED timeline. Failure analysis of v2 showed (54 pilot items, evidence correct in 51):
 (a) TF label mapping errors in plain reasoning, (b) SOURCE CONFUSION in CoT: the HEARD sentence inside the speech segment is
 read as if it were SEEN there ("the narrated event is X, which is also happening in the video").
v3 keeps WHAT IS SAID out of the timeline: the narration is declared once, separately, and the timeline only says in which
segment speech is present and what is SEEN. Same marks as v2 (symmetric contrast rule), same automatic pipeline otherwise.
Runs on the pilot (OUT/stmr_pilot_rel.jsonl) and the held-out set (OUT/uploads/stmr_heldout.jsonl). Methods: plain and cot
(answer parsed from the reasoning, orders vote). Writes OUT/stmr_v3.json."""
import json, re
import numpy as np, torch

SETS = {'pilot': OUT / 'stmr_pilot_rel.jsonl', 'heldout': OUT / 'uploads' / 'stmr_heldout.jsonl'}
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
am = lambda v: int(np.argmax(v))
COT = '\nThink step by step about which part of the timeline the question asks about, then give the answer.'
STOP = re.compile(r'\n\s*(Human|User|Question)\s*:|If you have any other questions', re.I)


def marks_v2(r, texts):
    K = len(r['segs']); seen = {k: [] for k in range(K)}
    claim = r['contents'][0] if r['contents'] else None
    if claim and claim not in texts: seen[am(r['verify'][claim])].append(claim)
    best = {k: None for k in range(K)}
    for t in texts:
        if t not in r['verify']: continue
        v = r['verify'][t]; k = am(v); con = v[k] - max([x for kk, x in enumerate(v) if kk != k] or [-1e9])
        if best[k] is None or con > best[k][0]: best[k] = (con, t)
    for k in range(K):
        if best[k]: seen[k].append(best[k][1])
    return seen


def timeline_v3(r, seen):
    head = (f'Narration (what the speaker talks about; this is only heard, it is NOT evidence of what is shown): "{r["claim"]}"\n'
            'Timeline of the video (SEEN = checked on the video with the sound off):\n')
    lines = []
    for k, (name, t0, t1) in enumerate(r['segs']):
        sp = 'speech: yes' if name == 'while the speech is heard' else 'speech: no'
        lines.append(f'- {t0:.1f}-{t1:.1f} s ({name}): {sp}; SEEN: ' + (', '.join(f'"{x}"' for x in seen[k]) or 'none of the listed actions'))
    return head + '\n'.join(lines) + '\n'


def chat_text(prompt):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    return PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def gen(prompt, n=160):
    inp = PROC(text=chat_text(prompt), return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=n, do_sample=False)
    return PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


def parse_answer(txt, shown):
    txt = STOP.split(txt)[0]
    m = list(re.finditer(r'answer\s+is\W{0,4}\(?([A-D])\b', txt, re.I)) or list(re.finditer(r'\(([A-D])\)', txt))
    if m:
        j = 'ABCD'.index(m[-1].group(1).upper())
        if j < len(shown): return j
    tail = txt[txt.lower().rfind('answer'):] if 'answer' in txt.lower() else txt
    hits = [j for j, t in enumerate(shown) if re.search(r'\b' + re.escape(t.lower()) + r'\b', tail.lower())]
    return hits[-1] if len(hits) == 1 else None


def cot_vote(prefix, texts, plain):
    k = len(texts); votes = np.zeros(k); outs = []
    for perm in rotations(k):
        txt = gen(render(prefix, texts, perm) + COT); p = parse_answer(txt, [texts[j] for j in perm]); outs.append(STOP.split(txt)[0][:300])
        if p is not None: votes[perm[p]] += 1
    return (votes + 1e-3 * np.exp(np.array(plain))).tolist(), outs


out = []
for set_name, path in SETS.items():
    if not path.exists(): print('missing', path, flush=True); continue
    P = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in [x for x in P if 'error' not in x and x.get('segs')]:
        stem, texts = parse_question(rows[r['idx']]['question'])
        tl = timeline_v3(r, marks_v2(r, texts)); q = tl + 'Answer using only the timeline. ' + stem
        rec = dict(set=set_name, idx=r['idx'], task=r['task'], gold=r['gold'], base=r['base'], timeline=tl)
        rec['plain'] = text_only_scores(q, texts)
        rec['cot'], rec['cot_texts'] = cot_vote(q, texts, rec['plain'])
        out.append(rec)
        print(set_name, r['task'], r['idx'], {c: am(rec[c]) == r['gold'] for c in ('base', 'plain', 'cot')}, flush=True)
(OUT / 'stmr_v3.json').write_text(json.dumps(out), encoding='utf-8')
RESULT = {}
for s in SETS:
    S = [x for x in out if x['set'] == s]
    RESULT[s] = {t: {c: round(100 * np.mean([am(x[c]) == x['gold'] for x in S if x['task'] == t]), 1) for c in ('base', 'plain', 'cot')} for t in ('TA', 'TF', 'CMTB')}
    RESULT[s]['ALL'] = {c: round(100 * np.mean([am(x[c]) == x['gold'] for x in S]), 1) for c in ('base', 'plain', 'cot')} if S else None
print(json.dumps(RESULT, indent=1), flush=True)

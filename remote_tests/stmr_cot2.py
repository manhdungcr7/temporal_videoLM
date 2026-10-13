"""Evidence-first reasoning prompts on the STMR v2 timelines (server 3: OUT/stmr_v2_offline.json). Text only, automatic.
  cot   (v2)  "Think step by step about which part of the timeline the question asks about, then give the answer."
  cot2        "First, for every SEEN item say whether it is seen before, while, or after the speech. Then say what the
               question asks about. Then give the answer."
  cot3        cot2 + the answer must be one of the options copied verbatim
Prints the reasoning of a few TA items to diagnose the Yes-collapse. Writes OUT/stmr_cot2.json."""
import json
import numpy as np, torch

V2 = json.loads((OUT / 'stmr_v2_offline.json').read_text(encoding='utf-8'))
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
PROMPTS = {'cot2': '\nFirst, for every SEEN item, say whether it is seen before, while, or after the speech. Then say what the question asks about. Then give the answer.',
           'cot3': '\nFirst, for every SEEN item, say whether it is seen before, while, or after the speech. Then say what the question asks about. Finally, copy the correct option exactly.'}


def cot_with(prefix, texts, suffix):
    k = len(texts); tot = np.zeros(k); first = None
    for perm in rotations(k):
        prompt = render(prefix, texts, perm) + suffix
        reasoning = gen(prompt, 128); first = first or reasoning
        txt = chat_text(prompt) + reasoning + '\nTherefore, the answer is ('
        inp = PROC(text=txt, return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
        with torch.no_grad(): lg = MODEL_(**inp).logits[0, -1].float()
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
    return tot.round(3).tolist(), first


out = []; shown = 0; am = lambda v: int(np.argmax(v))
for x in V2:
    stem, texts = parse_question(rows[x['idx']]['question'])
    q = x['timeline'] + 'Answer using only the timeline. ' + stem
    rec = dict(idx=x['idx'], task=x['task'], gold=x['gold'], plain=x['plain'], cot=x['cot'])
    for name, suf in PROMPTS.items():
        rec[name], txt = cot_with(q, texts, suf); rec[name + '_text'] = txt[:500]
    if x['task'] == 'TA' and am(x['cot']) != x['gold'] and shown < 3:
        shown += 1; _, t0 = cot_with(q, texts, COT)
        print('==== TA', x['idx'], 'gold', x['gold'], '\n', x['timeline'], '\nCOT:', t0[:500], '\nCOT2:', rec['cot2_text'], '\n', flush=True)
    out.append(rec)
(OUT / 'stmr_cot2.json').write_text(json.dumps(out), encoding='utf-8')
RESULT = {t: {c: round(100 * np.mean([am(r[c]) == r['gold'] for r in out if r['task'] == t]), 1) for c in ('plain', 'cot', 'cot2', 'cot3')} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: round(100 * np.mean([am(r[c]) == r['gold'] for r in out]), 1) for c in ('plain', 'cot', 'cot2', 'cot3')}
print(json.dumps(RESULT, indent=1), flush=True)

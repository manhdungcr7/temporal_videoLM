LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABCD']
import json, random
import numpy as np
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
am = lambda v: int(np.argmax(v))
# ---- router: can the model itself tell which evidence route a question needs? (text only) ----
ROUTES = ['what is visible at the moment the sound or speech is heard',
          'whether the action the speaker talks about happens at the same time as the speech',
          'when the action the speaker talks about happens relative to the speech (before, during or after)',
          'what is seen or heard in the video in general, not tied to the timing of the speech']
GOLD_ROUTE = {'CMTB': 0, 'TA': 1, 'TF': 2, 'GSA': 3, 'FGSA': 3, 'CMSB': 3, 'CSMB': 3}
rng = random.Random(5); sample = rng.sample([(i, r) for i, r in enumerate(rows)], 240)
router = []
for i, r in sample:
    s = text_only_scores(f'Here is a question about a video: "{r["question"]}"\nWhat does this question ask about?', ROUTES)
    router.append(dict(idx=i, task=r['task_type_abbr'], pred=am(s), gold=GOLD_ROUTE[r['task_type_abbr']]))
(OUT / 'router_dev.json').write_text(json.dumps(router))
acc = {t: round(float(np.mean([x['pred'] == x['gold'] for x in router if x['task'] == t])) * 100, 1) for t in sorted({x['task'] for x in router})}
print('router accuracy by task:', acc, flush=True)
RESULT = {'router_acc': acc}

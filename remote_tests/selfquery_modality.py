"""Text-only: what does the model itself say each question needs (seen / heard / both)? SVHalluc temporal (TA/TF/CMTB, 10 each)
and Daily-Omni question types (5 each). No media. Used to decide the target-modality gate of the in-model method."""
import json, random, re
import numpy as np
from pathlib import Path
SELFQ = 'To answer the following question about a video, what must you rely on? Question: "{q}"'
OPTS = ['what is seen in the video frames', 'what is heard in the audio', 'both what is seen and what is heard']
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
out = {}
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in rows if r['data_type'] == 'temporal' and r['task_type_abbr'] == t]; random.Random(0).shuffle(S)
    out[t] = [int(np.argmax(text_only_scores(SELFQ.format(q=parse_question(r['question'])[0]), OPTS))) for r in S[:10]]
qa = json.loads(Path(globals().get('DO_QA', 'qa.json')).read_text(encoding='utf-8'))
by = {}
for q in qa: by.setdefault(q['Type'], []).append(q)
ex = {}
for t, L in by.items():
    random.Random(0).shuffle(L); res = []
    for q in L[:5]:
        res.append(int(np.argmax(text_only_scores(SELFQ.format(q=q['Question']), OPTS)))); ex.setdefault(t, []).append((q['Question'][:90], ['seen', 'heard', 'both'][res[-1]]))
    out['DO:' + t] = res
names = ['seen', 'heard', 'both']
for k, v in out.items(): print(f'{k:28s}', {n: v.count(i) for i, n in enumerate(names)})
for t, L in ex.items():
    for q, a in L[:3]: print('   ', t[:18], '|', a, '|', q)
RESULT = 'done'

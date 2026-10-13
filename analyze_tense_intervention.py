"""Analysis of the causal tense intervention (video fixed, only the tense of the synthetic speech changes)."""
import json, sys, random
import numpy as np

R = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'kaggle_results/tense_intervention.jsonl') if l.strip()]
E = [r for r in R if 'error' in r]; R = [r for r in R if 'error' not in r]
print(f'valid {len(R)} | errors/skipped {len(E)}')
am = lambda v: int(np.argmax(v))
T = ['orig', 'future', 'present', 'past']


def sm(v):
    v = np.array(v); e = np.exp(v - v.max()); return e / e.sum()


for task, key, labels in (('TF', 'q', ['Past', 'Present', 'Future']), ('TF', 'bda', ['Before', 'During', 'After']), ('TA', 'q', ['Yes', 'No'])):
    rs = [r for r in R if r['task'] == task and f'orig_{key}' in r]
    if not rs: continue
    print(f'\n== {task} ({key}) n={len(rs)} — video and timing identical; only the speech tense changes ==')
    for t in T:
        preds = [am(r[f'{t}_{key}']) for r in rs]
        prob = np.mean([sm(r[f'{t}_{key}']) for r in rs], axis=0)
        acc = np.mean([p == r['gold'] for p, r in zip(preds, rs)]) * 100
        print(f'  tts_{t:8s} acc {acc:5.1f} | predicted share {[round(preds.count(j) / len(rs), 2) for j in range(len(labels))]} | mean prob {np.round(prob, 2).tolist()} {labels}')
    # paired shift: how often does the answer change between future and past tense, and in which direction
    fut = [am(r[f'future_{key}']) for r in rs]; pas = [am(r[f'past_{key}']) for r in rs]
    changed = sum(a != b for a, b in zip(fut, pas))
    print(f'  answers differ between future-tense and past-tense speech on {changed}/{len(rs)} items')
    if task == 'TF':
        d = [sm(r[f'past_{key}'])[0] - sm(r[f'future_{key}'])[0] for r in rs]   # P(Past/Before | past tense) - P(... | future tense)
        g = random.Random(0); b = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(4000))
        print(f'  P({labels[0]}) under past-tense minus future-tense speech: {np.mean(d):+.3f}  95% CI [{b[100]:+.3f}, {b[3899]:+.3f}]')
        d2 = [sm(r[f'future_{key}'])[2] - sm(r[f'past_{key}'])[2] for r in rs]
        b2 = sorted(np.mean([g.choice(d2) for _ in d2]) for _ in range(4000))
        print(f'  P({labels[2]}) under future-tense minus past-tense speech: {np.mean(d2):+.3f}  95% CI [{b2[100]:+.3f}, {b2[3899]:+.3f}]')
    else:
        d = [sm(r[f'present_{key}'])[0] - np.mean([sm(r[f'future_{key}'])[0], sm(r[f'past_{key}'])[0]]) for r in rs]
        g = random.Random(0); b = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(4000))
        print(f'  P(Yes) under present-tense minus mean(future, past): {np.mean(d):+.3f}  95% CI [{b[100]:+.3f}, {b[3899]:+.3f}]')
print('\nTTS lengths vs speech window (s):', [(round(r['win'][1] - r['win'][0], 1), r.get('orig_len'), r.get('past_len'), r.get('future_len')) for r in R[:8]])
print('Example rewrites:')
for r in R[:4]: print('  ', json.dumps(r['sentences'])[:300])

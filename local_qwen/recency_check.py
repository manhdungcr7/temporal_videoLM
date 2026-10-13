"""Does the whole-clip answer depend on WHERE the speech window sits? (recency-bias test on existing Kaggle CMTB data)"""
import json, numpy as np
R = [json.loads(l) for l in open('../anchor_kaggle_results/anchor_results.jsonl') if l.strip()]
R = [r for r in R if 'error' not in r]
g = lambda r: 'ABC'[r['gold']]
for name, cond in (('base_av', lambda r: r['speech'][0] < r['dur'] * 0.25), ):
    pass
early = [r for r in R if (r['speech'][0] + r['speech'][1]) / 2 < r['dur'] / 2]
late = [r for r in R if (r['speech'][0] + r['speech'][1]) / 2 >= r['dur'] / 2]
for c in ('base_av', 'vision_full', 'speech_win'):
    a = np.mean([r[c]['pred_rot'] == g(r) for r in early]); b = np.mean([r[c]['pred_rot'] == g(r) for r in late])
    print(f'{c:12s} speech EARLY (distractor late): {a*100:5.1f}% (n={len(early)})  | speech LATE (distractor early): {b*100:5.1f}% (n={len(late)})')
# when wrong, does the whole clip pick the option the model sees in the muted window?
for nm, grp in (('early', early), ('late', late)):
    w = [r for r in grp if r['base_av']['pred_rot'] != g(r)]
    print(nm, 'base wrong', len(w), 'of', len(grp), '; of those = silent-window pick:', sum(r['base_av']['pred_rot'] == r['silent_win']['pred_rot'] for r in w))

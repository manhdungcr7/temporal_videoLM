import json, numpy as np
R = [json.loads(l) for l in open('../anchor_kaggle_results/anchor_results.jsonl') if l.strip()]
rows = json.load(open(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark\samples.json', encoding='utf-8'))
ok = [r for r in R if 'error' not in r]
for r in ok:
    c = 'speech_win'
    if r[c]['pred_rot'] != 'ABC'[r['gold']]:
        q = rows[r['idx']]['question']
        print('idx', r['idx'], 'dur', r['dur'], 'speech', r['speech'], 'silent', r['silent'])
        print('Q:', q)
        print('gold', 'ABC'[r['gold']], 'speech_win pred', r[c]['pred_rot'], 'content scores', [round(x, 1) for x in r[c]['content']])
        for k in ('base_av', 'vision_full', 'silent_win'):
            print('  ', k, 'pred', r[k]['pred_rot'], [round(x, 1) for x in r[k]['content']])
# closest calls among the correct ones (smallest margin)
m = sorted(((r['speech_win']['content'][r['gold']] - max(v for j, v in enumerate(r['speech_win']['content']) if j != r['gold']), r['idx']) for r in ok))
print('smallest speech_win margins (gold minus best other):', [(round(a, 1), i) for a, i in m[:6]])

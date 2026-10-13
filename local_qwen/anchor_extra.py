import json, numpy as np
BS = chr(92)
R = [json.loads(l) for l in open('../anchor_kaggle_results/anchor_results.jsonl')]
A = json.load(open('temporal_structure_audit.json'))
rows = json.load(open(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark\samples.json', encoding='utf-8'))
for r in R:
    if 'error' in r:
        p = rows[r['idx']]['video_path'].replace(BS, '/')
        print('ERR idx', r['idx'], 'dur', round(A['clips'][p]['dur'], 1), r['error'][:60])
ok = [r for r in R if 'error' not in r]
print('valid dur pct', np.percentile([r['dur'] for r in ok], [0, 50, 100]))
fr = [(r['speech'][1] - r['speech'][0]) / r['dur'] for r in ok]
print('speech window / clip dur pct', np.percentile(fr, [10, 50, 90]).round(2))
print('silent_win available', sum('silent_win' in r for r in ok), 'of', len(ok))

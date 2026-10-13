import json
R = [json.loads(l) for l in open('../anchor_kaggle_results/anchor_results.jsonl') if l.strip()]
ok = [r for r in R if 'error' not in r]
g = lambda r: 'ABC'[r['gold']]
wrong = [r for r in ok if r['base_av']['pred_rot'] != g(r)]
print('base_av wrong:', len(wrong), 'of', len(ok))
print('  of those, base_av pick == silent_win pick (the action seen in the muted part):', sum(r['base_av']['pred_rot'] == r['silent_win']['pred_rot'] for r in wrong))
print('  of those, base_av pick == speech_win pick:', sum(r['base_av']['pred_rot'] == r['speech_win']['pred_rot'] for r in wrong))
print('  of those, base_av pick == vision_full pick:', sum(r['base_av']['pred_rot'] == r['vision_full']['pred_rot'] for r in wrong))
right = [r for r in ok if r['base_av']['pred_rot'] == g(r)]
print('base_av right:', len(right), '; of those silent_win pick equals base_av pick:', sum(r['base_av']['pred_rot'] == r['silent_win']['pred_rot'] for r in right))
print('whole-clip pick equals the muted-window pick, all items:', sum(r['base_av']['pred_rot'] == r['silent_win']['pred_rot'] for r in ok), 'of', len(ok))
print('silent_win picks a gold-different option, and is it the same option in silent_win and vision_full?',
      sum(r['silent_win']['pred_rot'] == r['vision_full']['pred_rot'] for r in ok), 'of', len(ok))

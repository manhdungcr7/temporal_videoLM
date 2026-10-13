"""Report for stage-2 shards with the PRE-REGISTERED decision criteria. Usage: python temporal_stage2_report.py DIR..."""
import json, sys, glob, os, random
from pathlib import Path
import numpy as np

paths = []
for a in (sys.argv[1:] or ['.']):
    paths += glob.glob(os.path.join(a, '**', 'stage2_shard*.jsonl'), recursive=True) if os.path.isdir(a) else [a]
recs = {}
for p in paths:
    for l in Path(p).read_text(encoding='utf-8').splitlines():
        if l.strip(): r = json.loads(l); recs[r['idx']] = r
R = [r for r in recs.values() if 'error' not in r]
print(f'files {len(paths)} | items {len(recs)} | valid {len(R)} | skipped/errors {len(recs) - len(R)}')
am = lambda v: int(np.argmax(v))


def p(r, v, c):
    x = r[v]
    if c in ('htl', 'lac'): return x.get(c + '_pred')
    return am(x[c]) if c in x else None


def ci(d, n=4000):
    g = random.Random(0); b = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(n)); return b[int(.025 * n)], b[int(.975 * n)]


def acc(rs, v, c): return np.mean([p(r, v, c) == r['gold'] for r in rs]) * 100


summary = {}
for task, k in (('TA', 2), ('TF', 3), ('CMTB', 3)):
    rs = [r for r in R if r['task'] == task]
    if not rs: continue
    print(f'\n===== {task} n={len(rs)} =====')
    conds = ['base', 'gate', 'audio_block', 'gate_audio'] + (['crop'] if task == 'CMTB' else ['lac', 'htl'])
    for v in ('orig', 'noisy'):
        print(f'  [{v}]')
        for c in conds:
            a = acc(rs, v, c)
            rec_ = [np.mean([p(r, v, c) == j for r in rs if r['gold'] == j]) * 100 for j in range(k)]
            line = f'    {c:11s} {a:5.1f}  recall/label {np.round(rec_, 1).tolist()}'
            if c != 'base':
                d = [int(p(r, v, c) == r['gold']) - int(p(r, v, 'base') == r['gold']) for r in rs]; lo, hi = ci(d)
                line += f'  vs base {np.mean(d) * 100:+5.1f} [{lo * 100:+.1f},{hi * 100:+.1f}] w2c {d.count(1)} c2w {d.count(-1)}'
            summary[(task, v, c)] = a
            print(line)
        if task == 'TF':
            rule = np.mean([(2 if (r[v]['win'][0] + r[v]['win'][1]) / 2 < r[v]['dur'] / 2 else 0) == r['gold'] for r in rs]) * 100
            pres = lambda c: np.mean([(p(r, v, c) == 1) == (r['gold'] == 1) for r in rs]) * 100
            summary[(task, v, 'rule')] = rule
            summary[(task, v, 'htl_present_vs_not')] = pres('htl')
            print(f'    reference speech-position rule (no video) {rule:.1f}% | Present-vs-not accuracy: htl {pres("htl"):.1f}%, lac {pres("lac"):.1f}%, base {pres("base"):.1f}%')
    if task in ('TA', 'TF'):
        ex = [r['orig']['hear'] for r in rs[:6]]
        print('  examples of the self-heard action (orig):', ex)

print('\n===== PRE-REGISTERED DECISION CRITERIA =====')
def g(*key): return summary.get(key, float('nan'))
c1 = (g('CMTB', 'noisy', 'gate_audio') - g('CMTB', 'noisy', 'base') >= 15) and \
     ((g('CMTB', 'noisy', 'gate_audio') - g('CMTB', 'noisy', 'base')) - (g('CMTB', 'noisy', 'gate') - g('CMTB', 'noisy', 'base')) >= 10)
print(f'C1 CMTB without the artifact: gate_audio >= base+15 AND beats gate alone by >=10  -> {"PASS" if c1 else "FAIL"}'
      f'  (noisy base {g("CMTB","noisy","base"):.1f}, gate {g("CMTB","noisy","gate"):.1f}, gate_audio {g("CMTB","noisy","gate_audio"):.1f}, crop {g("CMTB","noisy","crop"):.1f})')
c2 = all(g('TA', v, 'htl') - g('TA', v, 'base') >= 10 for v in ('orig', 'noisy'))
print(f'C2 TA: htl >= base+10 on BOTH versions  -> {"PASS" if c2 else "FAIL"}  (orig {g("TA","orig","base"):.1f}->{g("TA","orig","htl"):.1f}, noisy {g("TA","noisy","base"):.1f}->{g("TA","noisy","htl"):.1f})')
c3 = (g('TF', 'noisy', 'htl_present_vs_not') >= 75) and (g('TF', 'noisy', 'htl') - g('TF', 'noisy', 'rule') >= 10)
print(f'C3 TF without the artifact: Present-vs-not >= 75% AND htl beats the position rule by >= 10  -> {"PASS" if c3 else "FAIL"}'
      f'  (htl {g("TF","noisy","htl"):.1f}, rule {g("TF","noisy","rule"):.1f}, present-vs-not {g("TF","noisy","htl_present_vs_not"):.1f})')
print('Also required for every PASS: recall of each label not below base - 5 (check the tables above).')
print('Decision: scale up (more items, Qwen2.5-Omni-7B, another omni family, general AV benchmarks) only for the criteria that PASS.')

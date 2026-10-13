"""Second audit: is the non-speech half digitally silent, is there a visual cut at the midpoint, and how far does
'which half holds the speech' alone predict labels? Analysis only."""
import json, sys, collections
from pathlib import Path
import numpy as np, av
sys.path.insert(0, r'F:\A_star_paper_NLP_ACL\research_topics_learning\single_sample_prompt_lab')
from lab_media import decode_audio

BASE = Path(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark')
D = json.loads(Path(__file__).with_name('temporal_structure_audit.json').read_text(encoding='utf-8'))
C, I = D['clips'], D['items']

# 1) rule accuracy from speech half only
def half(p): return 'first' if C[p]['centroid'] < 0.5 else 'second'
tf = [it for it in I if it['task'] == 'TF']
rule = {'first': 'C', 'second': 'A'}
acc = np.mean([rule[half(it['path'])] == it['gold'] for it in tf])
print('TF speech-half rule (first->Future, second->Past, never Present):', round(acc * 100, 1), '% on', len(tf))
print('TF gold by speech half:', collections.Counter((it['gold'], half(it['path'])) for it in tf))
print('TA gold by speech half:', collections.Counter((it['gold'], half(it['path'])) for it in I if it['task'] == 'TA'))

# 2) silence level and visual cut on a sample of clips
def frame_diffs(path, n=64):
    with av.open(str(BASE / path)) as c:
        s = c.streams.video[0]
        fr = [f.to_ndarray(format='gray', width=64, height=36).astype(np.float32) for f in c.decode(s)]
    fr = np.stack(fr); d = np.abs(np.diff(fr, axis=0)).mean((1, 2))
    return d, len(fr)
rng = np.random.default_rng(0)
paths = sorted(C)
out = []
for p in rng.choice(paths, 12, replace=False):
    a = decode_audio(BASE / p); n = len(a) // 800
    db = 20 * np.log10(np.sqrt((a[:n * 800].reshape(n, 800) ** 2).mean(1) + 1e-12))
    h = n // 2
    quiet = db[:h] if C[p]['centroid'] >= 0.5 else db[h:]
    loud = db[h:] if C[p]['centroid'] >= 0.5 else db[:h]
    d, nf = frame_diffs(p)
    k = int(np.argmax(d))
    out.append(dict(path=p, dur=round(C[p]['dur'], 1), quiet_half_db_median=round(float(np.median(quiet)), 1),
                    quiet_half_db_max=round(float(quiet.max()), 1), speech_half_db_median=round(float(np.median(loud)), 1),
                    biggest_cut_at=round(k / nf, 3), cut_strength_vs_median=round(float(d[k] / (np.median(d) + 1e-6)), 1)))
for o in out: print(o)
Path(__file__).with_name('temporal_structure_audit2.json').write_text(json.dumps(dict(tf_rule_acc=acc, samples=out), indent=1), encoding='utf-8')

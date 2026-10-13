"""Structural audit of SVHalluc temporal clips: where is the speech inside each clip, and how does that relate to labels?
Speech activity is estimated from short-time RMS of the full PyAV-decoded audio (analysis only, never a model input)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, r'F:\A_star_paper_NLP_ACL\research_topics_learning\single_sample_prompt_lab')
from lab_media import decode_audio

BASE = Path(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark')
OUT = Path(__file__).with_name('temporal_structure_audit.json')
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
stats = json.loads((BASE / 'temporal_video_stats.json').read_text(encoding='utf-8'))

def activity(path):
    a = decode_audio(BASE / path)
    hop = 800  # 50 ms at 16 kHz
    n = len(a) // hop
    rms = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    active = db > lo + 0.5 * (hi - lo)
    t = (np.arange(n) + 0.5) * 0.05
    dur = len(a) / 16000
    if active.sum() == 0:
        return dict(dur=dur, frac=0.0)
    idx = np.where(active)[0]
    return dict(dur=dur, frac=float(active.mean()), first=float(t[idx[0]] / dur), last=float(t[idx[-1]] / dur),
                centroid=float(t[active].mean() / dur),
                profile=[float(active[int(k * n / 10):int((k + 1) * n / 10)].mean()) for k in range(10)])

cache = json.loads(OUT.read_text(encoding='utf-8'))['clips'] if OUT.exists() else {}
paths = sorted({r['video_path'].replace('\\', '/') for r in rows if r['data_type'] == 'temporal'})
for k, p in enumerate(paths):
    if p in cache: continue
    try: cache[p] = activity(p)
    except Exception as e: cache[p] = dict(error=repr(e)[:200])
    if k % 50 == 0:
        print(k, len(paths), flush=True)
        OUT.write_text(json.dumps(dict(clips=cache)), encoding='utf-8')
items = [dict(idx=i, task=r['task_type_abbr'], gold=r['answer'], path=r['video_path'].replace('\\', '/'))
         for i, r in enumerate(rows) if r['data_type'] == 'temporal']
OUT.write_text(json.dumps(dict(clips=cache, items=items)), encoding='utf-8')
print('done', len(cache))

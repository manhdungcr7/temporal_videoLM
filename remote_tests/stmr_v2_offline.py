"""STMR v2 composition, replayed on REAL pilot evidence (stmr_pilot_rel.jsonl: the model's own hearing and routed verification
scores from server 3). Only the composition changes, text only, so it runs on any GPU. Fully automatic:
  marks   options: each option goes to its best segment; each segment keeps the option with the highest contrast
          (score there minus its best score elsewhere); the heard claim goes to its best segment. Symmetric over segments,
          blind to the question.
  timeline 'list' format: per segment (named before / while / after the speech), HEARD and the list of SEEN contents.
  reason  plain | ground (model-side option grounding) | cot | ground_cot  (from tf_bridge.py)
Writes OUT/stmr_v2_offline.json. Needs PILOT=<path to stmr_pilot_rel.jsonl> and tf_bridge.py helpers exec'd first."""
import json, re
import numpy as np

SRC = Path(globals().get('PILOT', 'stmr_pilot_rel.jsonl'))
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
am = lambda v: int(np.argmax(v))


def marks_v2(r, texts):
    segs = r['segs']; K = len(segs); seen = {k: [] for k in range(K)}
    claim = r['contents'][0] if r['contents'] else None
    if claim and claim not in texts: seen[am(r['verify'][claim])].append((1e9, claim))
    best = {k: None for k in range(K)}
    for t in texts:
        if t not in r['verify']: continue
        v = r['verify'][t]; k = am(v); con = v[k] - max([x for kk, x in enumerate(v) if kk != k] or [-1e9])
        if best[k] is None or con > best[k][0]: best[k] = (con, t)
    for k in range(K):
        if best[k]: seen[k].append(best[k])
    return {k: [t for _, t in sorted(v, reverse=True)] for k, v in seen.items()}


def timeline_v2(r, marks):
    lines = []
    for k, (name, t0, t1) in enumerate(r['segs']):
        heard = f'HEARD: "{r["claim"]}"' if name == 'while the speech is heard' else 'HEARD: (no speech)'
        lines.append(f'- {t0:.1f}-{t1:.1f} s ({name}): {heard}; SEEN: ' + (', '.join(f'"{x}"' for x in marks[k]) or 'none of the listed actions'))
    return 'Evidence timeline of the video (SEEN = checked on the video with the sound off; HEARD = what is said):\n' + '\n'.join(lines) + '\n'


P = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
P = [r for r in P if 'error' not in r and r.get('segs')]
out = []
for r in P:
    stem, texts = parse_question(rows[r['idx']]['question'])
    m = marks_v2(r, texts); tl = timeline_v2(r, m); q = tl + 'Answer using only the timeline. ' + stem
    rec = dict(idx=r['idx'], task=r['task'], gold=r['gold'], base=r['base'], v1=r['answer'], timeline=tl)
    rec['plain'] = text_only_scores(q, texts)
    notes = ground_options(stem, texts); at = annotate(texts, notes); rec['ground_notes'] = notes
    rec['ground'] = text_only_scores(q, at)
    if globals().get('WITH_COT', True):
        rec['cot'], _ = cot_scores(q, texts)
        rec['ground_cot'], _ = cot_scores(q, at)
    out.append(rec)
    print(r['task'], r['idx'], {c: am(rec[c]) == r['gold'] for c in ('base', 'v1', 'plain', 'ground', 'cot', 'ground_cot') if c in rec}, flush=True)
(OUT / 'stmr_v2_offline.json').write_text(json.dumps(out), encoding='utf-8')
RESULT = {t: {c: round(100 * np.mean([am(x[c]) == x['gold'] for x in out if x['task'] == t]), 1) for c in ('base', 'v1', 'plain', 'ground', 'cot', 'ground_cot') if c in out[0]}
          | {'n': sum(x['task'] == t for x in out)} for t in ('TA', 'TF', 'CMTB')}
print(json.dumps(RESULT, indent=1), flush=True)

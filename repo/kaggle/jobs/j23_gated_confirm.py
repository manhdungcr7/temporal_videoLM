"""J23 — PRE-REGISTERED confirmation of gated_anchor (J21), on items never used before.  Needs the J8 helpers and the J21 routing code.
Rule (fixed before the run, no tuning):  anchor from the model's own attention (J21 `anchor`); gate = normalised entropy of the anchor < 0.8;
gate open -> anchor_video routing (layers 13..24, block audio for the question rows, video bias 4*log(rel/max)); gate closed -> the model as is.
Conditions stored per item: base, null (parity guard), anchor_video (ungated, so the would-be damage when the gate is closed is visible).
gated_anchor is derived offline from these (exactly the rule above).  Items: seed 2030, J23_PER per (task, gold), excluding every item
used in J20/J21 (J23_EXCLUDE).  Primary outcome and tests: see results/j23/protocol.md.  Writes OUT/j23_gated_confirm.jsonl."""
import json, random, numpy as np, torch
if 'Lay' not in globals():
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())
if 'route_attention' not in globals():
    exec((ROOT / 'j21_temporal_split.py').read_text(encoding='utf-8').split(chr(10) + 'items = []')[0], globals())

J23_OUT = globals().get('J23_OUT', 'j23_gated_confirm.jsonl')
GATE23 = 0.8
PER23 = dict(globals().get('J23_PER', {'TA': 10, 'TF': 10, 'CMTB': 15, 'GSA': 8, 'FGSA': 8, 'CSMB': 8}))
SEED23 = int(globals().get('J23_SEED', 2030))
EXCL23 = set(globals().get('J23_EXCLUDE', ()))


def item23(i, r):
    src, a, dur, w = prep_clip(r, MAXD) if r['data_type'] == 'temporal' else prep_any(r)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rel, win, ent = anchor(media, prompt_of(stem, texts, tuple(range(len(texts)))))
    rec = dict(dur=round(dur, 2), data_type=r['data_type'], n_chunks=len(rel), anchor_rel=np.round(rel, 4).tolist(), anchor_win=list(win),
               anchor_entropy=round(ent, 4), gate_open=bool(ent < GATE23),
               vad_speech=None if w is None else [round(float(w[0]), 2), round(float(w[1]), 2)])
    rec['base'] = score(media, stem, texts)
    rec['null'] = score(media, stem, texts, fwd=fwd_kind('null', rel, win))
    drift = float(np.abs(np.array(rec['null']) - np.array(rec['base'])).max())
    if drift > 0.15: raise RuntimeError(f'null path does not reproduce base (max diff {drift:.3f})')
    rec['anchor_video'] = score(media, stem, texts, fwd=fwd_kind('anchor_video', rel, win))
    rec['gated_anchor'] = rec['anchor_video'] if rec['gate_open'] else rec['base']
    return rec


def balanced23(per, seed):
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8')); cells = {}
    for i, r in enumerate(rows):
        if r['task_type_abbr'] in per and i not in EXCL23: cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    pools = []
    for key in sorted(cells):
        pool = cells[key][:]; random.Random(f'{seed}-{key}').shuffle(pool); got = []
        for i, r in pool:
            if usable(r): got.append((i, r))
            if len(got) == per[key[0]]: break
        pools.append(got)
    out = []
    for kk in range(max(len(p) for p in pools)):
        for p in pools:
            if kk < len(p): out.append(p[kk])
    return out


items = balanced23(PER23, SEED23)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
import collections as _c
print('skipped before selection:', dict(_c.Counter((t, a_) for t, a_, _ in SKIPPED21)), '| items', len(items),
      dict(_c.Counter(r['task_type_abbr'] + '/' + r['answer'] for _, r in items)), flush=True)
R = run_loop(items, J23_OUT, item23, show=('base', 'anchor_video', 'gated_anchor'))
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)

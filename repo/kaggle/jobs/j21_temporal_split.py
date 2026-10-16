"""J21 — INTERNAL temporal routing test (training-free, no VAD / ASR / cropping in the method).  Needs the J8 helpers (Lay).
Anchor (from the model itself): question-row attention mass on each 2-s AUDIO chunk, mean over layers 12..35, for the real question and
for a neutral question; PMI  rel(c) = p(c) * (p(c)/p0(c))^2, normalised (J9: follows the question, works for non-speech sounds).
Anchor window W = contiguous chunks around argmax rel with rel >= 0.5 * max.  Gate signal: normalised entropy of rel (logged; gating is
evaluated offline with thresholds fixed in advance).  In decoder layers J21_BAND (default 13..24, where J20 shows evidence is still read),
only the question/answer rows are recomputed with an additive per-head bias (all other rows untouched):
    C1 block_audio   question rows cannot read audio tokens                                  (source control only)
    C2 anchor_video  C1 + video bias LAMBDA*log(rel/max rel) per chunk (all heads)             (J8-style routing, internal anchor)
    C3 split         C1 + heads 0-4 may read video only BEFORE W, heads 5-10 only INSIDE W, heads 11-15 only AFTER W
                     (an empty region falls back to W)                                        (proposed past/sync/future split)
    C4 split_random  C3 with W replaced by a random window of the same length                (control: does the anchor matter?)
plus C0 base and 'null' (same recomputation path, zero bias: must equal base).  Rotated options.  The VAD window is stored for ANALYSIS only (is the anchor on the speech?).
Writes OUT/j21_temporal_split.jsonl."""
import json, random, numpy as np, torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
if 'Lay' not in globals():
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())

J21_OUT = globals().get('J21_OUT', 'j21_temporal_split.jsonl')
BAND = tuple(globals().get('J21_BAND', tuple(range(13, 25))))
REC_LAYERS = tuple(range(12, 36))
LAM21 = float(globals().get('J21_LAMBDA', 4.0))
SEEDS21 = tuple(globals().get('J21_SEEDS', (2028, 2029))); PER21 = int(globals().get('J21_PER_CELL', 2))
NL21 = len(MODEL_.model.layers)
for _li, _layer in enumerate(MODEL_.model.layers): _layer.self_attn._rt_layer = _li
RT = {}
NEG = -1e4


def route_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = RT.get('cfg'); li = getattr(module, '_rt_layer', None)
    if not c or li is None: return out, None
    if c['mode'] == 'record' and li not in REC_LAYERS: return out, None
    if c['mode'] != 'record' and li not in BAND: return out, None
    rows = c['rows'].to(query.device); rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(rep, 1).float()
    sc = scaling if scaling is not None else query.shape[-1] ** -0.5
    s = (query[:, :, rows, :].float() @ k.transpose(-1, -2)) * sc
    if attention_mask is not None: s = s + attention_mask[:, :, rows, :].float().to(s.device)
    else:                                   # native SDPA path: the mask is None and causality is applied inside the kernel -> apply it here too
        cols = torch.arange(key.shape[2], device=s.device)
        s = s.masked_fill(cols[None, None, None, :] > rows[None, None, :, None], float('-inf'))
    if c['mode'] == 'record':
        p = torch.softmax(s, -1)[0].mean((0, 1))                                    # [L] mean over heads and rows
        c['rec'][li] = (p @ c['G'].to(p.device)).cpu()                               # [nc] mass per audio chunk
        return out, None
    s = s + c['bias'].to(s.device)[None, :, None, :]                                 # [1,H,1,L]
    o = (torch.softmax(s, -1) @ value.repeat_interleave(rep, 1).float()).to(out.dtype)
    out = out.clone(); out[:, rows] = o.transpose(1, 2)
    return out, None


def run_cfg(inp, cfg):
    RT['cfg'] = cfg; old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = route_attention
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],))
    try:
        with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = old; hk.remove(); RT.pop('cfg', None)
    return lg


def audio_profile(media, prompt):
    inp, _ = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    lay = Lay(inp); G = torch.zeros(lay.L, lay.nc); G[torch.where(lay.a)[0], lay.chunk[lay.a]] = 1
    cfg = dict(mode='record', rows=torch.where(lay.rows)[0], G=G, rec={})
    run_cfg(inp, cfg); del inp
    p = torch.stack([cfg['rec'][l] for l in REC_LAYERS]).mean(0).numpy()
    return p / max(p.sum(), 1e-12), lay.nc


def anchor(media, prompt):
    p, nc = audio_profile(media, prompt); p0, _ = audio_profile(media, NEUTRAL)
    rel = p * (p / np.maximum(p0, 1e-9)) ** 2; rel = rel / max(rel.sum(), 1e-12)
    c = int(np.argmax(rel)); lo = hi = c
    while lo - 1 >= 0 and rel[lo - 1] >= 0.5 * rel[c]: lo -= 1
    while hi + 1 < nc and rel[hi + 1] >= 0.5 * rel[c]: hi += 1
    ent = float(-(rel * np.log(np.maximum(rel, 1e-12))).sum() / np.log(max(nc, 2)))
    return rel, (lo, hi), ent


def make_bias(lay, kind, rel, win):
    H = MODEL_.model.layers[0].self_attn.config.num_attention_heads
    bias = torch.zeros(H, lay.L)
    if kind == 'null': return bias                                                   # sanity: same code path, nothing changed
    bias[:, lay.a] = NEG                                                             # every other condition blocks audio for the question rows
    if kind == 'anchor_video':
        r = torch.tensor(rel, dtype=torch.float32); lb = LAM21 * torch.log(r / r.max() + 1e-6)
        bias[:, lay.v] = lb[lay.chunk[lay.v]][None, :]
    elif kind in ('split', 'split_random'):
        lo, hi = win; ch = lay.chunk
        regions = [lay.v & (ch < lo), lay.v & (ch >= lo) & (ch <= hi), lay.v & (ch > hi)]
        groups = [range(0, 5), range(5, 11), range(11, H)]
        for g, reg in zip(groups, regions):
            if not bool(reg.any()): reg = regions[1]
            for h in g: bias[h, lay.v & ~reg] = NEG
    return bias


def fwd_kind(kind, rel, win):
    def f(inp):
        lay = Lay(inp)
        cfg = dict(mode='route', rows=torch.where(lay.rows)[0], bias=make_bias(lay, kind, rel, win))
        return run_cfg(inp, cfg)
    return f


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD) if r['data_type'] == 'temporal' else prep_any(r)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rel, win, ent = anchor(media, prompt_of(stem, texts, tuple(range(len(texts)))))
    nc = len(rel); span = win[1] - win[0]
    rng = random.Random(i); s0 = rng.randrange(0, max(1, nc - span)); rwin = (s0, s0 + span)
    rec = dict(dur=round(dur, 2), data_type=r['data_type'], n_chunks=nc, anchor_rel=np.round(rel, 4).tolist(), anchor_win=list(win),
               anchor_entropy=round(ent, 4), random_win=list(rwin), vad_speech=None if w is None else [round(float(w[0]), 2), round(float(w[1]), 2)])
    rec['base'] = score(media, stem, texts)
    rec['null'] = score(media, stem, texts, fwd=fwd_kind('null', rel, win))
    drift = float(np.abs(np.array(rec['null']) - np.array(rec['base'])).max())
    if drift > 0.15:                        # hard guard: the recomputation path must reproduce the model (fp16 noise is ~0.01-0.05)
        raise RuntimeError(f'null path does not reproduce base (max diff {drift:.3f})')
    for kind in ('block_audio', 'anchor_video', 'split'):
        rec[kind] = score(media, stem, texts, fwd=fwd_kind(kind, rel, win))
    rec['split_random'] = score(media, stem, texts, fwd=fwd_kind('split', rel, rwin))
    return rec


def prep_any(r):
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
    a = decode_audio(src); dur = len(a) / SR
    if dur > MAXD: raise RuntimeError('long')
    return src, a, dur, speech_window(a)


SKIPPED21 = []


def usable(r):
    try: (prep_clip(r, MAXD) if r['data_type'] == 'temporal' else prep_any(r)); return True
    except Exception as e: SKIPPED21.append((r['task_type_abbr'], r['answer'], repr(e)[:40])); return False


def balanced(tasks, per, seed):
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8')); cells = {}
    for i, r in enumerate(rows):
        if r['task_type_abbr'] in tasks: cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    pools = []
    for key in sorted(cells):
        pool = cells[key][:]; random.Random(f'{seed}-{key}').shuffle(pool); got = []
        for i, r in pool:
            if usable(r): got.append((i, r))
            if len(got) == per: break
        pools.append(got)
    out = []
    for kk in range(max(len(p) for p in pools)):
        for p in pools:
            if kk < len(p): out.append(p[kk])
    return out


items = []
for sd in SEEDS21:                     # 2028 = the J20 items (development); 2029 = fresh items
    for it in balanced(('TA', 'TF', 'CMTB', 'GSA', 'FGSA', 'CSMB'), PER21, sd):
        if it[0] not in {x[0] for x in items}: items.append(it)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
import collections as _c
print('skipped before selection:', dict(_c.Counter((t, a_) for t, a_, _ in SKIPPED21)), '| items', len(items), flush=True)
R = run_loop(items, J21_OUT, item, show=('base', 'block_audio', 'anchor_video', 'split', 'split_random'))
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)

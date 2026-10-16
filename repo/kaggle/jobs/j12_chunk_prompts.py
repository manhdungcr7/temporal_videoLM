"""J12 — CHUNK PROMPTS v2 (team specification). Needs the J8 helpers (Lay, ...) and locate().
For every 2-s chunk t (= one TMRoPE chunk) two prompts are appended after the media:
    C_{t,v}: "The essential visual content of chunk t in one short phrase is"   -> may read ONLY the video tokens of chunk t
    C_{t,a}: "The essential audio content of chunk t in one short phrase is"    -> may read ONLY the audio tokens of chunk t
Prompts never see each other or the question. The question and the answer see all prompts and all audio/video.
Positions: all prompts share the same position ids, as if each prompt stood directly before [question][answer].
Conditions (rotated options):
  base              no prompts
  cq                prompts, normal attention
  cq_boost_p        in every layer, for every question/answer token, relevance of prompt (t,s) = that token's attention to
                    the prompt's last token ("is"); the token's attention logits to the prompt's tokens get
                    LAMBDA*log(r / max r)  (the most relevant prompt keeps full weight, the others are pushed down)
  cq_boost_pr       the same bias also applied to the raw tokens of the prompt's chunk and modality
Writes OUT/j12_chunk_prompts.jsonl (with the per-prompt relevance of the last token in a middle layer)."""
import json, numpy as np, torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

if 'Lay' not in globals():                       # notebook run: load the J8 helpers (server jobs prepend them instead)
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('\nitems = sample_items')[0], globals())
PER_LABEL = int(globals().get('PER_LABEL', 2)); SEED = int(globals().get('SEED', 2468)); LAMBDA = float(globals().get('LAMBDA', 2.0))
TOK = PROC.tokenizer
P_V = globals().get('J12_PV', ' Visual information from {t0}s to {t1}s, summarized in one short phrase: ')
P_A = globals().get('J12_PA', ' Audio information from {t0}s to {t1}s, summarized in one short phrase: ')
BOOST_LAYERS = tuple(globals().get('J12_BOOST_LAYERS', tuple(range(13, 28))))


if 'locate' not in globals():
    def locate(input_ids, text, pieces, after):
        """token spans [start, end) of each piece of `text`, searched after position `after` (the last media token)."""
        seq = input_ids[0].tolist()[after + 1:]
        strs = [TOK.decode([t]) for t in seq]; starts = np.cumsum([0] + [len(s) for s in strs[:-1]]); full = ''.join(strs)
        p0 = full.find(text)
        if p0 < 0: raise RuntimeError('probe text not found')
        bounds = np.cumsum([p0] + [len(p) for p in pieces]); spans = []
        for k in range(len(pieces)):
            toks = [t for t, c in enumerate(starts) if bounds[k] <= c < bounds[k + 1]]
            spans.append((after + 1 + toks[0], after + 1 + toks[-1] + 1))
        return spans
CTX = {}                              # per-forward context for the custom attention


def cq_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """standard SDPA for all rows; question/answer rows recomputed with relevance-driven boosts."""
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = CTX.get('cfg')
    if not c or c['mode'] == 'none' or getattr(module, '_cq_layer', None) is None: return out, None   # encoders untouched
    if module._cq_layer not in c['boost_layers']: return out, None
    rows = c['rows'].to(query.device)                                       # question/answer positions
    H, KV = query.shape[1], key.shape[1]; rep = H // KV
    k = key.repeat_interleave(rep, 1); v = value.repeat_interleave(rep, 1)
    q = query[:, :, rows, :]                                                # [1,H,R,d]
    s = (q.float() @ k.float().transpose(-1, -2)) * (scaling if scaling is not None else q.shape[-1] ** -0.5)
    s = s + attention_mask[:, :, rows, :].float().to(s.device)
    p = torch.softmax(s, -1)
    is_idx = c['is_idx'].to(s.device)                                       # [G] last token of each prompt
    if c['mode'] == 'vcenter':                                              # no boost; probe values centred per modality
        vv = v.float().clone()
        for par in (0, 1):                                                  # 0 = visual probes, 1 = audio probes
            sel = is_idx[par::2]; vv[..., sel, :] -= vv[..., sel, :].mean(-2, keepdim=True)
        out = out.clone(); out[:, rows] = (p @ vv).to(out.dtype).transpose(1, 2)
        return out, None
    r = p[..., is_idx]                                                      # [1,H,R,G] relevance per prompt
    bias_g = LAMBDA * torch.log(r / r.amax(-1, keepdim=True).clamp(min=1e-9) + 1e-9)   # <= 0
    G = c['groups'].to(s.device)                                            # [G, L] bool: columns to boost per prompt
    B = bias_g @ G.float()                                                  # [1,H,R,L]
    if c['mode'] == 'boost_pr': B = B + bias_g @ c['raw'].to(s.device).float()
    p2 = torch.softmax(s + B, -1)
    o = (p2 @ v.float()).to(out.dtype)                                      # [1,H,R,d]
    if 'record' in c and module._cq_layer == c['record_layer']: c['record'] = r[0, :, -1, :].mean(0).cpu()
    out = out.clone(); out[:, rows] = o.transpose(1, 2)                     # out is [1, L, H, d]
    return out, None


for li, layer in enumerate(MODEL_.model.layers): layer.self_attn._cq_layer = li   # only the LLM layers are modified


def cq_forward(inp, pieces, mode, record=False, layout='shared', variant=None):
    """layout 'shared': every prompt at p0 (right after all media).
    layout 'aligned': prompt (t, v|a) starts right after the last position of chunk t, so every prompt is equally far
    from its own media; the question/answer still start at p0 + maxlen.
    variant (bias-source controls, J15): 'blind' = probes see no media at all; 'lastonly' = question/answer see only the
    last token of each probe; 'hidden' = question/answer see no probe token (probes still occupy the sequence)."""
    lay = Lay(inp); text = ''.join(pieces)
    spans = locate(inp['input_ids'].cpu(), text, pieces, lay.last_media)
    L = lay.L; m = lay.m.clone(); pos = lay.pos.clone()
    pcols = torch.zeros(L, dtype=torch.bool)
    for a_, b_ in spans: pcols[a_:b_] = True
    q_start = spans[-1][1]; maxlen = max(b_ - a_ for a_, b_ in spans); p0 = int(pos[0, 0, spans[0][0]])
    groups = torch.zeros(len(spans), L, dtype=torch.bool); raw = torch.zeros(len(spans), L, dtype=torch.bool); is_idx = []
    prompt_cursor = p0
    for g, ((a_, b_), piece) in enumerate(zip(spans, pieces)):
        t = g // 2; src = 'v' if g % 2 == 0 else 'a'
        own = (lay.v if src == 'v' else lay.a) & (lay.chunk == t)
        m[a_:b_][:, (lay.v | lay.a) & (~own if variant != 'blind' else torch.ones_like(own))] = float('-inf')   # only its chunk and modality
        m[a_:b_][:, pcols] = float('-inf'); m[a_:b_, a_:b_] = torch.triu(torch.full((b_ - a_, b_ - a_), float('-inf')), 1)
        if layout == 'shared':
            start = p0
        else:
            # Prompts are ordered in time. The audio probe follows the visual
            # probe in the same chunk; later chunks start after their own media
            # and after any preceding prompt tokens.
            media_end = int(lay.pos.cpu()[:, 0, (lay.v | lay.a) & (lay.chunk == t)].max()) + 1
            start = max(media_end, prompt_cursor)
        pos[:, :, a_:b_] = (start + torch.arange(b_ - a_)).view(1, 1, -1).to(pos.dtype)
        prompt_cursor = start + (b_ - a_)
        groups[g, a_:b_] = True; raw[g] = own; is_idx.append(b_ - 1)
    if layout == 'shared':
        q_pos = p0 + maxlen
    else:
        # Question and answer must follow every aligned prompt, including the
        # prompt belonging to the last temporal chunk.
        q_pos = max(prompt_cursor, int(pos.cpu()[:, 0, (lay.v | lay.a)].max()) + 1)
    pos[:, :, q_start:] = (q_pos + torch.arange(L - q_start)).view(1, 1, -1).to(pos.dtype)
    if variant == 'lastonly':                                   # (J15 controls) question/answer see only each probe's last token
        keep = torch.zeros(L, dtype=torch.bool); keep[torch.tensor(is_idx)] = True
        m[q_start:][:, pcols & ~keep] = float('-inf')
    elif variant == 'hidden':                                   # question/answer see no probe token
        m[q_start:][:, pcols] = float('-inf')
    rows = torch.arange(q_start, L)
    CTX['cfg'] = dict(mode=mode, rows=rows, is_idx=torch.tensor(is_idx), groups=groups, raw=raw,
                      record_layer=18, boost_layers=BOOST_LAYERS)
    if record: CTX['cfg']['record'] = None
    old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = cq_attention     # mask logic stays 'sdpa'
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev), use_audio_in_video=True).logits[0, -1].float()
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = old
    rec = CTX['cfg'].get('record'); CTX.clear(); return lg, rec


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2)); rec['base'] = score(media, stem, texts)
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av'); inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc; del inp0
    pieces = [x for t in range(nc)
              for x in (P_V.format(t0=2 * t, t1=min(2 * (t + 1), dur)),
                        P_A.format(t0=2 * t, t1=min(2 * (t + 1), dur)))]
    prompt = ''.join(pieces) + globals().get('J12_SEP', '\n') + stem
    for mode, name, lo in CONDS:
        rec[name] = score(media, prompt, texts, fwd=lambda inp, mode=mode, lo=lo: cq_forward(inp, pieces, mode, layout=lo)[0])
    inp, _ = qwen_inputs(PROC, prompt_of(prompt, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
    for lo in sorted({c[2] for c in CONDS}):
        _, rel = cq_forward(inp, pieces, 'boost_p', record=True, layout=lo)
        if rel is not None: rec['relevance_L18_' + lo] = [round(float(x), 4) for x in rel]
    del inp
    rec['n_chunks'] = nc
    inp, _ = qwen_inputs(PROC, prompt_of(prompt, texts, tuple(range(len(texts)))), media, 'av')
    sp_ = locate(inp['input_ids'].cpu(), ''.join(pieces), pieces, Lay(move_inputs(inp, MODEL_)).last_media)
    rec['last_tokens'] = [TOK.decode(inp['input_ids'][0, b_ - 1:b_]) for a_, b_ in sp_[:4]]; rec['prompt0'] = TOK.decode(inp['input_ids'][0, sp_[0][0]:sp_[0][1]])
    return rec


TASKS = tuple(globals().get('J12_TASKS', ('TA', 'TF', 'CMTB'))); OUT_NAME = globals().get('J12_OUT', 'j12_chunk_prompts.jsonl')
CONDS = globals().get('J12_CONDS', (('none', 'cq', 'shared'), ('boost_p', 'cq_boost_p', 'shared'), ('boost_pr', 'cq_boost_pr', 'shared')))
items = sample_items(TASKS, PER_LABEL, seed=SEED)
SHARD = globals().get('J12_SHARD')                       # (k, n): this server takes items k, k+n, k+2n, ...
if SHARD: items = items[SHARD[0]::SHARD[1]]
C =('base',) + tuple(c[1] for c in CONDS)
R = run_loop(items, OUT_NAME, item, show=C)
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in C} | {'n': sum(r['task'] == t for r in OK)} for t in TASKS}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in C}
print(RESULT, flush=True)

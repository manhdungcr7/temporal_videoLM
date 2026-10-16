"""J16 — PER-TOKEN relevance. Which question/answer token (if any) sends attention to the right chunk probe, in which layer/head?
J13 averaged over the question tokens ("mean") or took only the last one ("last"); a token that points correctly can be diluted by
the ~40 others. Needs the J8 helpers, locate() and the J12 definitions. Same layout as J12d: one visual + one audio probe per 2-s
chunk ("...in one word:" / "...in one word is:"), isolated, shared positions, no boost; the question is asked once with the
options in their original order (so that token positions are comparable across items of the same task).
Stored per item (all uint8, base64 of a [layers, heads, rows] array):  v_arg / a_arg = index of the chunk whose visual / audio
probe receives the most attention from that row in that head (argmax over chunks), plus the decoded row tokens and the targets:
  speech chunks (VAD window) and the event half (benchmark construction; analysis only; for CMTB the event is at the speech time).
Analysis: remote_tests/analyze_j16.py.  Writes OUT/j16_token_relevance.jsonl (J16_OUT)."""
import base64, json, numpy as np, torch
if 'cq_forward' not in globals():                 # notebook run: load the J12 definitions (and through them the J8 helpers)
    exec((ROOT / 'j12_chunk_prompts.py').read_text(encoding='utf-8').split(chr(10) + 'TASKS = ')[0], globals())

J16_OUT = globals().get('J16_OUT', 'j16_token_relevance.jsonl')
PV = ' The essential visual content of chunk {t} in one word:'
PA = ' The essential audio content of chunk {t} in one word is:'
REC16 = {}


def tok_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """plain SDPA output; in LLM layers store, per head and per question/answer row, the argmax chunk of the visual / audio probes."""
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = REC16.get('cfg'); li = getattr(module, '_cq_layer', None)
    if not c or li is None: return out, None
    rows = c['rows'].to(query.device); rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(rep, 1)
    s = (query[:, :, rows, :].float() @ k.float().transpose(-1, -2)) * (scaling if scaling is not None else query.shape[-1] ** -0.5)
    s = s + attention_mask[:, :, rows, :].float().to(s.device)
    r = torch.softmax(s, -1)[..., c['is_idx'].to(s.device)][0]              # [H, R, G]
    c['rec'][li] = (r[..., 0::2].argmax(-1).to(torch.uint8).cpu(), r[..., 1::2].argmax(-1).to(torch.uint8).cpu())   # [H, R] each (visual, audio)
    return out, None


def token_forward(inp, pieces):
    lay = Lay(inp); text = ''.join(pieces)
    spans = locate(inp['input_ids'].cpu(), text, pieces, lay.last_media)
    L = lay.L; m = lay.m.clone(); pos = lay.pos.clone()
    pcols = torch.zeros(L, dtype=torch.bool)
    for a_, b_ in spans: pcols[a_:b_] = True
    q_start = spans[-1][1]; maxlen = max(b_ - a_ for a_, b_ in spans); p0 = int(pos[0, 0, spans[0][0]])
    for g, (a_, b_) in enumerate(spans):
        t = g // 2; own = (lay.v if g % 2 == 0 else lay.a) & (lay.chunk == t)
        m[a_:b_][:, (lay.v | lay.a) & ~own] = float('-inf')
        m[a_:b_][:, pcols] = float('-inf'); m[a_:b_, a_:b_] = torch.triu(torch.full((b_ - a_, b_ - a_), float('-inf')), 1)
        pos[:, :, a_:b_] = (p0 + torch.arange(b_ - a_)).view(1, 1, -1).to(pos.dtype)
    pos[:, :, q_start:] = (p0 + maxlen + torch.arange(L - q_start)).view(1, 1, -1).to(pos.dtype)
    REC16['cfg'] = dict(rows=torch.arange(q_start, L), is_idx=torch.tensor([b_ - 1 for a_, b_ in spans]), rec={})
    old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = tok_attention
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    sel = torch.tensor([L - 1])
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, sel.to(args[0].device), :],))
    try:
        with torch.no_grad():
            MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev), use_audio_in_video=True)
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = old; hk.remove()
    rec = REC16.pop('cfg')['rec']; nl = len(MODEL_.model.layers)
    va = torch.stack([rec[l][0] for l in range(nl)]).numpy(); aa = torch.stack([rec[l][1] for l in range(nl)]).numpy()    # [L, H, R]
    toks = [TOK.decode(inp['input_ids'][0, j:j + 1]) for j in range(q_start, L)]
    return va, aa, toks


b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av'); inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc; del inp0
    pieces = [x for t in range(nc) for x in (PV.format(t=t + 1), PA.format(t=t + 1))]
    prompt = ''.join(pieces) + ' ' + stem
    inp, _ = qwen_inputs(PROC, prompt_of(prompt, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
    va, aa, toks = token_forward(inp, pieces); del inp
    return dict(dur=round(dur, 2), speech=[round(float(w[0]), 2), round(float(w[1]), 2)], speech_first=bool(speech_first(w, dur)),
                n_chunks=nc, shape=list(va.shape), v_arg=b64(va), a_arg=b64(aa), tokens=toks, options=texts)


items = sample_items(tuple(globals().get('J16_TASKS', ('TA', 'TF', 'CMTB'))), PER_LABEL, seed=SEED)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
R = run_loop(items, J16_OUT, item, show=())
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)

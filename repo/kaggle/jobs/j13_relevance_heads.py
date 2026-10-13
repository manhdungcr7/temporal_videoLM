"""J13 — WHERE does chunk-prompt relevance point? (diagnostic, no scoring). Needs J8 helpers, locate() and j12 cq_forward.
Same layout as J12 (two isolated prompts per 2-s chunk, shared positions, no boost). For every layer and head, record the
relevance r = attention of the question/answer tokens to each prompt's last token ("is"):
    last[l][h][g]  from the last token (the one that produces the answer letter)
    mean[l][h][g]  averaged over all question/answer tokens
Analysis (offline) compares the argmax prompt per (layer, head) with the speech window (anchor, from VAD) and with the
event half (from the benchmark construction; analysis only). Writes OUT/j13_relevance_heads.jsonl."""
import json, numpy as np, torch

J13_OUT = globals().get('J13_OUT', 'j13_relevance_heads.jsonl')
MEASURE = globals().get('J13_MEASURE', 'is')            # 'is' | 'full' (sum over all tokens of each prompt)
DECODE = bool(globals().get('J13_DECODE', False))       # also decode the top-5 next tokens at every prompt's "is"


def rec_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """plain SDPA output; in LLM layers additionally store the relevance of every head."""
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = CTX.get('cfg'); li = getattr(module, '_cq_layer', None)
    if not c or li is None: return out, None
    rows = c['rows'].to(query.device); H, KV = query.shape[1], key.shape[1]; rep = H // KV
    k = key.repeat_interleave(rep, 1)
    s = (query[:, :, rows, :].float() @ k.float().transpose(-1, -2)) * (scaling if scaling is not None else query.shape[-1] ** -0.5)
    s = s + attention_mask[:, :, rows, :].float().to(s.device)
    p = torch.softmax(s, -1)
    if MEASURE == 'full': r = (p @ c['groups'].to(s.device).float().T)[0]     # [H, R, G] attention summed over the whole prompt
    else: r = p[..., c['is_idx'].to(s.device)][0]                            # [H, R, G] attention to the last token ("is")
    c['rec'][li] = (r[:, -1, :].cpu(), r.mean(1).cpu())
    return out, None


LAST_REC = {}


def cq_forward_rec(inp, pieces):
    """cq_forward with mode 'record' (outputs unchanged) that keeps the per-layer relevance dict."""
    global LAST_REC
    lay = Lay(inp); text = ''.join(pieces)
    spans = locate(inp['input_ids'].cpu(), text, pieces, lay.last_media)
    L = lay.L; m = lay.m.clone(); pos = lay.pos.clone()
    pcols = torch.zeros(L, dtype=torch.bool)
    for a_, b_ in spans: pcols[a_:b_] = True
    q_start = spans[-1][1]; maxlen = max(b_ - a_ for a_, b_ in spans); p0 = int(pos[0, 0, spans[0][0]])
    is_idx = []; groups = torch.zeros(len(spans), L, dtype=torch.bool)
    for g, (a_, b_) in enumerate(spans):
        t = g // 2; src = 'v' if g % 2 == 0 else 'a'
        own = (lay.v if src == 'v' else lay.a) & (lay.chunk == t)
        m[a_:b_][:, (lay.v | lay.a) & ~own] = float('-inf')
        m[a_:b_][:, pcols] = float('-inf'); m[a_:b_, a_:b_] = torch.triu(torch.full((b_ - a_, b_ - a_), float('-inf')), 1)
        pos[:, :, a_:b_] = (p0 + torch.arange(b_ - a_)).view(1, 1, -1).to(pos.dtype)
        is_idx.append(b_ - 1); groups[g, a_:b_] = True
    pos[:, :, q_start:] = (p0 + maxlen + torch.arange(L - q_start)).view(1, 1, -1).to(pos.dtype)
    CTX['cfg'] = dict(mode='record', rows=torch.arange(q_start, L), is_idx=torch.tensor(is_idx), groups=groups, rec={})
    old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = rec_attention
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    sel = torch.tensor(is_idx + [L - 1])
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, sel.to(args[0].device), :],))
    global LAST_TOP
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev), use_audio_in_video=True).logits[0, :-1].float()
        LAST_TOP = [[TOK.decode([t]) for t in row] for row in lg.topk(5, -1).indices.tolist()]
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = old; hk.remove()
    LAST_REC = CTX['cfg']['rec']; CTX.clear()
    return None, None


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av'); inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc; del inp0
    pieces = [x for t in range(nc) for x in (P_V.format(t=t + 1), P_A.format(t=t + 1))]
    prompt = ''.join(pieces) + globals().get('J12_SEP', '\n') + stem
    inp, _ = qwen_inputs(PROC, prompt_of(prompt, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
    cq_forward_rec(inp, pieces); del inp
    nl = len(MODEL_.model.layers)
    last = torch.stack([LAST_REC[l][0] for l in range(nl)]); mean = torch.stack([LAST_REC[l][1] for l in range(nl)])
    extra = dict(top5=LAST_TOP) if DECODE else {}
    return extra | dict(question=stem, options=texts, dur=round(dur, 2), speech=[round(float(w[0]), 2), round(float(w[1]), 2)], speech_first=bool(speech_first(w, dur)),
                n_chunks=nc, last=np.round(last.numpy(), 5).tolist(), mean=np.round(mean.numpy(), 5).tolist())


items = sample_items(('TA', 'TF'), PER_LABEL, seed=SEED)
R = run_loop(items, J13_OUT, item, show=())
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)

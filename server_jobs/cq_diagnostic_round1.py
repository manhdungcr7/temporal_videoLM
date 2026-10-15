"""CQ diagnosis. All inference paths take media/question only; labels score outputs."""
from transformers.integrations.sdpa_attention import sdpa_attention_forward

MANIFEST = json.loads((OUT / f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
DATA = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
ITEMS = [(d['idx'], DATA[d['idx']]) for d in MANIFEST['selected']]
P_V = ' The essential video content from {t0}s to {t1}s of the video, in one short phrase'
P_A = ' The essential audio content from {t0}s to {t1}s of the video, in one short phrase'
LAB_CTX = {}


def lab_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask,
                                    scaling=scaling, dropout=dropout, **kw)
    c = LAB_CTX.get('cfg')
    if not c or c['mode'] != 'balanced' or getattr(module, '_cq_layer', -1) not in range(13, 28):
        return out, None
    rep = query.shape[1] // key.shape[1]
    k, v = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
    rr = c['rows'].to(query.device)
    q = query[:, :, rr, :]
    s = (q.float() @ k.float().transpose(-1, -2)) * (scaling or q.shape[-1] ** -0.5)
    s += attention_mask[:, :, rr, :].float().to(s.device)
    p = s.softmax(-1)
    # Per-head and per-query relevance. Divide endpoint attention by each
    # modality's total so video/audio token-count disparities do not win.
    r = p[..., c['ends'].to(p.device)].clamp_min(1e-10)
    r = r.reshape(*r.shape[:-1], -1, 2)
    r = r / r.sum(-2, keepdim=True).clamp_min(1e-9)
    r = 0.9 * r + 0.1 / r.shape[-2]
    r = r.reshape(*r.shape[:-2], -1)
    routed = torch.zeros_like(p)
    for group, mass in ((c['groups'], 0.2), (c['raw'], 0.4)):
        G = group.to(p.device).float()
        old = p @ G.T
        # Keep within-group attention structure; allocate group mass rather
        # than giving every raw token an identical multiplicative bonus.
        weights = mass * 0.5 * r / old.clamp_min(1e-12)
        routed += p * (weights @ G)
    rest = ~(c['groups'].any(0) | c['raw'].any(0))
    residual = p * rest.to(p.device)
    routed += 0.4 * residual / residual.sum(-1, keepdim=True).clamp_min(1e-12)
    routed /= routed.sum(-1, keepdim=True).clamp_min(1e-12)
    z = (routed @ v.float()).to(out.dtype)
    out = out.clone()
    out[:, rr] = z.transpose(1, 2)
    return out, None


def lab_forward(inp, pieces, variant):
    lay = Lay(inp)
    spans = locate(inp['input_ids'].cpu(), ''.join(pieces), pieces, lay.last_media)
    m, pos = lay.m.clone(), lay.pos.clone()
    L = lay.L
    pcols = torch.zeros(L, dtype=torch.bool)
    groups = torch.zeros(len(spans), L, dtype=torch.bool)
    raw = torch.zeros(len(spans), L, dtype=torch.bool)
    for s, e in spans:
        pcols[s:e] = True
    first_media = int(torch.where(lay.v | lay.a)[0][0])
    safe_prefix = torch.arange(L) < first_media
    q_start = spans[-1][1]
    for g, (s, e) in enumerate(spans):
        own = (lay.v if g % 2 == 0 else lay.a) & (lay.chunk == g // 2)
        allowed = own | safe_prefix
        m[s:e, ~allowed] = float('-inf')
        m[s:e, s:e] = torch.triu(torch.full((e - s, e - s), float('-inf')), 1)
        if variant == 'null':
            m[s:e, own] = float('-inf')
        if variant in ('local', 'isolated', 'balanced', 'null'):
            # A parallel branch has its own clock; branches never consume
            # positions of other probes. Preserve raw TMRoPE coordinates.
            start = int(lay.pos[:, 0, own].max()) + 1 if own.any() else int(lay.pos[:, 0, lay.v | lay.a].max()) + 1
            pos[:, :, s:e] = (start + torch.arange(e - s)).view(1, 1, -1).to(pos.dtype)
        groups[g, s:e] = True
        raw[g] = own
    if variant in ('local', 'isolated', 'balanced', 'null'):
        q_pos = int(pos[:, 0, (lay.v | lay.a) | pcols].max()) + 1
        pos[:, :, q_start:] = (q_pos + torch.arange(L - q_start)).view(1, 1, -1).to(pos.dtype)
    if variant in ('isolated', 'balanced'):
        # Blocking only the final probe does not prevent earlier raw video
        # hidden states from importing audio. Enforce source provenance in
        # every decoder layer; both streams can fuse in question/answer.
        for own in (lay.v, lay.a):
            rr = torch.where(own)[0]
            cols = torch.where(~(own | safe_prefix))[0]
            m[rr[:, None], cols[None, :]] = float('-inf')
    LAB_CTX['cfg'] = dict(mode=variant, rows=torch.arange(q_start, L),
                          groups=groups, raw=raw, ends=torch.tensor([e - 1 for s, e in spans]))
    old = ALL_ATTENTION_FUNCTIONS['sdpa']
    ALL_ATTENTION_FUNCTIONS['sdpa'] = lab_attention
    # lm_head on the final row avoids materializing L x vocabulary logits.
    hook = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev),
                        use_audio_in_video=True, use_cache=False).logits[0, -1].float()
    finally:
        hook.remove()
        ALL_ATTENTION_FUNCTIONS['sdpa'] = old
        LAB_CTX.clear()
    return lg


def diagnostic_item(i, row):
    src, audio, dur, win = prep_clip(row, 50)
    media = prepare_video(src)
    stem, options = parse_question(row['question'])
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av')
    inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc
    del inp0
    pieces = [p.format(t0=2*t, t1=round(min(2*(t+1), dur), 2))
              for t in range(nc) for p in (P_V, P_A)]
    prefix = ''.join(pieces) + '\n' + stem
    rec = dict(dur=round(dur, 3), question=stem, options=options, n_chunks=nc)
    rec['base'] = score(media, stem, options)
    rec['cq_old'] = score(media, prefix, options,
                          fwd=lambda inp: cq_forward(inp, pieces, 'none', layout='aligned')[0])
    for variant in ('local', 'null', 'isolated', 'balanced'):
        rec['cq_' + variant] = score(media, prefix, options,
                                     fwd=lambda inp, v=variant: lab_forward(inp, pieces, v))
    # Paired calibration against a prompt-identical counterfactual path.
    # No label, class-specific constant, or task mapping is used.
    for alpha in (1, 2):
        rec[f'cq_cal{alpha}'] = (np.array(rec['cq_local']) + alpha *
            (np.array(rec['cq_local']) - np.array(rec['cq_null']))).round(3).tolist()
    print('DIAGNOSTIC', i, json.dumps({k:v for k,v in rec.items() if k.startswith('cq_') or k == 'base'}), flush=True)
    return rec


CONDITIONS = ('base', 'cq_old', 'cq_local', 'cq_isolated', 'cq_balanced', 'cq_cal1', 'cq_cal2')
R = run_loop(ITEMS, f'cq_diagnostic_round1_s{LAB_SERVER}.jsonl', diagnostic_item, show=CONDITIONS, min_left=0)
RESULT = {c: {'accuracy': acc(R, c)[0], 'n': acc(R, c)[1]} for c in CONDITIONS}
print('ROUND1_RESULT', json.dumps(RESULT), flush=True)

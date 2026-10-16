"""J14 — GATE for the second hop (audio probe -> matching visual probe). Needs J8 helpers, locate() and j12 (definitions only).
Same probe layout as J12/J13 (one visual + one audio probe per 2-s chunk, isolated, shared positions). Probe states do not
depend on the question (probes never see it), so everything here is a property of the video alone (cacheable per video).
For every item, recorded per probe (its last token):
  - hidden states at LAYERS -> similarity matrices S[t_audio, t_visual]: raw cosine and modality-mean-centred cosine;
  - next-token distribution restricted to CONTENT words (alphabetic, >= 3 letters, not a stop word), renormalised ->
    overlap S = sum_v p_a(v) p_v(v) and Jensen-Shannon similarity; plus the top content words (qualitative);
  - three probe variants: 'plain' ("...in one word is:"), 'quote' (audio probe ends with an opening quote, so the next token
    must be a content word) and 'nonum_quote' (no chunk number in either probe: the number token shared by the visual and
    audio probe of the same chunk could make them similar for a reason that has nothing to do with content);
  - per-chunk audio RMS (dB) and the VAD speech window (hop-1 check: are the speech heads just a VAD?).
  - hop 1: the question/answer rows' attention to every probe's last token, per layer and head, for the real question and
    for a NEUTRAL question ("What is happening in the video?") -> PMI anchoring on probes, as J9 did on raw chunks;
Analysis offline (remote_tests/analyze_j14.py): alpha2 = alpha1 . S, does argmax alpha2 fall in the event half?
Writes OUT/j14_hop2_gate.jsonl."""
import json, re, numpy as np, torch
if 'cq_forward' not in globals():                 # notebook run: load the J12 definitions (and through them the J8 helpers)
    exec((ROOT / 'j12_chunk_prompts.py').read_text(encoding='utf-8').split(chr(10) + 'TASKS = ')[0], globals())

J14_OUT = globals().get('J14_OUT', 'j14_hop2_gate.jsonl')
LAYERS = list(globals().get('J14_LAYERS', (8, 12, 16, 20, 24, 28, 32, 35)))
PV1 = ' The essential visual content of chunk {t} in one word:'
PA1 = ' The essential audio content of chunk {t} in one word is:'
PA2 = ' The essential audio content of chunk {t} in one word is: "'
PV3 = ' The essential visual content in one word:'          # no chunk number: controls a same-number (diagonal) similarity
PA3 = ' The essential audio content in one word is: "'
VARIANTS = (('plain', PV1, PA1), ('quote', PV1, PA2), ('nonum_quote', PV3, PA3))
STOP = set('the and for are but not you all any can her was one our out has him his how its may new now old see two way who '
           'did get let put say she too use that with have this will your from they know want been good much some time very '
           'when come here just like long make many over such take than them well were what into more also there their would '
           'about which these other could after first never where those while being every under still might shall should '
           'yes video audio visual content chunk word essential phrase sound speech person people thing'.split())


def content_mask():
    V = MODEL_.lm_head.weight.shape[0]; m = torch.zeros(V, dtype=torch.bool)
    for i in range(min(V, len(TOK))):
        s = TOK.decode([i]).strip()
        if len(s) >= 3 and s.isascii() and s.isalpha() and s.lower() not in STOP: m[i] = True
    return m


CMASK = content_mask()
print('content vocabulary', int(CMASK.sum()), flush=True)


REC = {}


def rel_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """plain SDPA; in LLM layers also store, per head, the question/answer rows' mean attention to each probe's last token."""
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = REC.get('cfg'); li = getattr(module, '_cq_layer', None)
    if not c or li is None: return out, None
    rows = c['rows'].to(query.device); rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(rep, 1)
    s = (query[:, :, rows, :].float() @ k.float().transpose(-1, -2)) * (scaling if scaling is not None else query.shape[-1] ** -0.5)
    s = s + attention_mask[:, :, rows, :].float().to(s.device)
    c['rec'][li] = torch.softmax(s, -1)[..., c['is_idx'].to(s.device)][0].mean(1).cpu()     # [H, G]
    return out, None


def probe_states(inp, pieces, record_rel=False):
    """record_rel: also return the question->probe relevance [layers, heads, G] (for hop-1 PMI)."""
    """forward with the J12 probe layout; returns hidden [len(LAYERS), G, d] (float32, cpu) and logits [G, V] at each probe's last token."""
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
    sel = torch.tensor([b_ - 1 for a_, b_ in spans]); got = {}
    def mk(li):
        def hk(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            got[li] = h[0, sel.to(h.device)].float().cpu()
        return hk
    hooks = [MODEL_.model.layers[li].register_forward_hook(mk(li)) for li in LAYERS]
    hooks.append(MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, sel.to(args[0].device), :],)))
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    if record_rel:
        REC['cfg'] = dict(rows=torch.arange(q_start, L), is_idx=sel, rec={})
        old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = rel_attention
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev), use_audio_in_video=True).logits[0].float().cpu()
    finally:
        for h in hooks: h.remove()
        if record_rel: ALL_ATTENTION_FUNCTIONS['sdpa'] = old
    last_tok = [TOK.decode(inp['input_ids'][0, b_ - 1:b_]) for a_, b_ in spans[:2]]
    rel = None
    if record_rel:
        rec = REC.pop('cfg')['rec']; rel = torch.stack([rec[l] for l in range(len(MODEL_.model.layers))])
    return torch.stack([got[li] for li in LAYERS]), lg, last_tok, rel


def cos(a, b): a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6); b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6); return a @ b.T


def sims(Ha, Hv, La, Lv):
    """Ha/Hv [nl, n, d]; La/Lv [n, V] logits -> dict of n x n similarity matrices (rows = audio chunk, cols = visual chunk)."""
    out = {}
    for k, li in enumerate(LAYERS):
        out[f'cos_L{li}'] = cos(Ha[k], Hv[k])
        out[f'ccos_L{li}'] = cos(Ha[k] - Ha[k].mean(0, keepdim=True), Hv[k] - Hv[k].mean(0, keepdim=True))
    pa = torch.softmax(La, -1) * CMASK; pv = torch.softmax(Lv, -1) * CMASK
    pa = pa / pa.sum(-1, keepdim=True).clamp(min=1e-12); pv = pv / pv.sum(-1, keepdim=True).clamp(min=1e-12)
    out['vocab_dot'] = pa @ pv.T
    out['vocab_bc'] = pa.sqrt() @ pv.sqrt().T                                # Bhattacharyya coefficient
    M = 0.5 * (pa[:, None, :] + pv[None, :, :])
    kl = lambda p, q: (p * (torch.log(p.clamp(min=1e-12)) - torch.log(q.clamp(min=1e-12)))).sum(-1)
    out['vocab_jss'] = 1 - 0.5 * (kl(pa[:, None, :].expand_as(M), M) + kl(pv[None, :, :].expand_as(M), M)) / np.log(2)
    tops = lambda p: [[TOK.decode([int(i)]).strip() for i in row] for row in p.topk(5, -1).indices]
    return {k: np.round(v.numpy(), 4).tolist() for k, v in out.items()}, tops(pa), tops(pv), \
        [round(float(x), 3) for x in (torch.softmax(La, -1) * CMASK).sum(-1)]                # content mass of audio probes


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av'); inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc; del inp0
    rec = dict(dur=round(dur, 2), speech=[round(float(w[0]), 2), round(float(w[1]), 2)], speech_first=bool(speech_first(w, dur)), n_chunks=nc)
    seg = [a[int(2 * t * SR):int(min(2 * t + 2, dur) * SR)] for t in range(nc)]
    rec['rms_db'] = [round(float(20 * np.log10(np.sqrt(np.mean(s.astype(np.float64) ** 2)) + 1e-9)), 1) if len(s) else -200.0 for s in seg]
    for name, pv_, pa_ in VARIANTS:
        pieces = [x for t in range(nc) for x in (pv_.format(t=t + 1), pa_.format(t=t + 1))]
        prompt = ''.join(pieces) + ' ' + stem
        inp, _ = qwen_inputs(PROC, prompt_of(prompt, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
        H, lg, last_tok, rel_q = probe_states(inp, pieces, record_rel=True); del inp
        S, top_a, top_v, cmass = sims(H[:, 1::2], H[:, 0::2], lg[1::2], lg[0::2])
        # hop 1: the same probes read by a NEUTRAL question (probe states are question-independent, only the relevance changes)
        inp, _ = qwen_inputs(PROC, ''.join(pieces) + ' ' + NEUTRAL, media, 'av'); inp = move_inputs(inp, MODEL_)
        _, _, _, rel_n = probe_states(inp, pieces, record_rel=True); del inp
        rec[name] = dict(S=S, top_audio=top_a, top_visual=top_v, audio_content_mass=cmass, last_tokens=last_tok,
                         rel_question=np.round(rel_q.numpy(), 5).tolist(), rel_neutral=np.round(rel_n.numpy(), 5).tolist())
    return rec


items = sample_items(('TA', 'TF'), PER_LABEL, seed=SEED)
R = run_loop(items, J14_OUT, item, show=())
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)

"""J7 — 2-SECOND CHUNK QUERIES + SELF-SELECTED ROUTING (training-free). Team proposal, implemented as specified:
  * every 2-s chunk (= one TMRoPE chunk of the omni model) gets a text query "Summarize the information from the video and
    audio."; query c may read ONLY the video+audio tokens of chunk c (plus the system/user prefix); queries never see each
    other; the question/answer text sees the queries AND all audio/video (so it cannot misread the media).
Conditions (one item, rotated options, all automatic):
  base            native model
  cq              chunk queries, normal attention
  cq_noiso        control: same queries but each may read ALL media (tests whether ISOLATION is what matters)
  cq_boost_mid    + BETA on question/answer -> query attention logits in layers 12-23
  cq_boost_late   + BETA in layers 24-35
  cq_select_soft  the model's own attention from the question/answer rows to each chunk query (eager pass, layers 12-35,
                  original option order) gives a relevance p_c; then in layers 13-27 the question/answer rows get a bias
                  LAMBDA*log(p_c / max p) on the MEDIA tokens of chunk c: the most relevant chunks keep full weight, the
                  others are pushed towards zero contribution (lambda 1)
  cq_select_sharp same with lambda 4
Also saved per item: p_c and the speech activity per chunk (analysis only) -> does the model's choice match the speech time?
Writes OUT/j7_chunk_queries.jsonl."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 8))
SEED = int(globals().get('SEED', 909))
BETA = float(globals().get('BETA', 2.0))
NL = len(MODEL_.model.layers)
MID_B, LATE_B = list(range(12, 24)), list(range(24, NL))
SEL_LAYERS, ROUTE_LAYERS = list(range(12, NL)), list(range(13, 28))
CONDS = globals().get('CONDS', ['cq', 'cq_noiso', 'cq_boost_mid', 'cq_boost_late', 'cq_select_soft', 'cq_select_sharp'])
TOK = PROC.tokenizer
QTEXT = 'Summarize the information from the video and audio.'


def locate(input_ids, text, pieces, after):
    seq = input_ids[0].tolist()[after + 1:]
    strs = [TOK.decode([t]) for t in seq]
    starts = np.cumsum([0] + [len(s) for s in strs[:-1]]); full = ''.join(strs)
    p0 = full.find(text)
    if p0 < 0: raise RuntimeError('query text not found in the token sequence')
    bounds = np.cumsum([p0] + [len(p) for p in pieces]); spans = []
    for k in range(len(pieces)):
        toks = [t for t, c in enumerate(starts) if bounds[k] <= c < bounds[k + 1]]
        if not toks: raise RuntimeError('empty query span')
        spans.append((after + 1 + toks[0], after + 1 + toks[-1] + 1))
    return spans, after + 1 + max(t for t, c in enumerate(starts) if c < bounds[-1]) + 1


def chunk_queries(n_chunks):
    return [f'\n[part {c + 1:02d}] {QTEXT}' for c in range(n_chunks)]          # 2-digit numbers: equal token length


class Layout:
    """token groups and the base mask for one tokenised prompt."""
    def __init__(self, inp, pieces, iso=True, par=False):
        emb = MODEL_.get_input_embeddings().weight; self.dev, self.dtype = emb.device, emb.dtype
        self.pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        L = inp['input_ids'].shape[1]; self.L = L
        self.spans, self.q_end = locate(inp['input_ids'].cpu(), ''.join(pieces), pieces, last_media)
        if par:   # PARALLEL queries: all queries share the same RoPE positions (they cannot see each other anyway), so the
                  # question sees every query at the same distance -> removes the recency bias towards later queries
            self.pos = self.pos.clone(); s0 = self.spans[0][0]
            for s, e in self.spans: self.pos[:, :, s:e] = self.pos[:, :, s0:s0 + (e - s)]
        sec, v, a = sec.cpu(), is_v.cpu(), is_a.cpu(); self.media = v | a
        nc = len(pieces); self.chunk = torch.clamp((sec // 2).long(), 0, nc - 1)
        m = torch.zeros(L, L); m = m.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float('-inf'))
        self.qcols = torch.zeros(L, dtype=torch.bool)
        for c, (s, e) in enumerate(self.spans):
            if iso: m[s:e][:, self.media & (self.chunk != c)] = float('-inf')
            self.qcols[s:e] = True
        for (s, e) in self.spans:
            m[s:e][:, self.qcols] = float('-inf')
            m[s:e, s:e] = torch.triu(torch.full((e - s, e - s), float('-inf')), 1)
        self.m = m

    def t(self, m):
        return m.clamp(min=torch.finfo(self.dtype).min).to(self.dtype).to(self.dev).view(1, 1, self.L, self.L)


def run(inp, lay, layer_masks=None, record=None):
    """forward with the base mask; layer_masks: {layer: mask} replaces it in those layers. record: layers whose attention
    (question/answer rows -> query tokens) is returned (eager)."""
    hooks, rec = [], {}
    if layer_masks:
        for li, mk in layer_masks.items():
            def pre(mod, args, kwargs, mk=mk):
                am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = mk.to(am_.device if am_ is not None else mk.device); return args, kwargs
            hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    if record:
        MODEL_.set_attn_implementation('eager')
        for li in record:
            def hk(mod, args, out, li=li):
                w = out[1]
                if w is None: return
                rows = w[0, :, lay.q_end:, :].float().sum(0).mean(0)                    # heads summed, rows averaged
                rec[li] = torch.stack([rows[s:e].sum() for s, e in lay.spans]).cpu()
                return (out[0], None) + tuple(out[2:])
            hooks.append(MODEL_.model.layers[li].self_attn.register_forward_hook(hk))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(lay.m), position_ids=lay.pos.to(lay.dev), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()
        if record: MODEL_.set_attn_implementation('sdpa')
    return lg, rec


def boost_masks(lay, layers):
    mb = lay.m.clone(); mb[lay.q_end:][:, lay.qcols] += BETA; t = lay.t(mb)
    return {li: t for li in layers}


def select_masks(lay, p, lam):
    bias = lam * torch.log(torch.as_tensor(p, dtype=torch.float32) / max(p))                 # 0 for the best chunk, < 0 others
    mb = lay.m.clone(); cols = torch.where(lay.media)[0]
    mb[lay.q_end:, cols] += bias[lay.chunk[cols]].unsqueeze(0)
    t = lay.t(mb); return {li: t for li in ROUTE_LAYERS}


def chunk_activity(a, n):
    hop = 800; k = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95); act = db > lo + 0.5 * (hi - lo)
    c = np.minimum(((np.arange(k) + 0.5) * 0.05 // 2).astype(int), n - 1)
    return [round(float(act[c == j].mean()), 3) if (c == j).any() else 0.0 for j in range(n)]


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    rec['base'] = score(media, stem, texts)
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av')
    _, sec0, v0, a0, _ = media_layout(MODEL_, PROC, move_inputs(inp0, MODEL_)); del inp0
    nc = int(sec0[(v0 | a0)].max().item() // 2) + 1
    pieces = chunk_queries(nc); prompt = ''.join(pieces) + '\n' + stem
    rec['n_chunks'] = nc; rec['speech_activity'] = chunk_activity(a, nc)
    def mk(cond):
        par = cond.startswith('cqp'); c_ = cond.replace('cqp', 'cq', 1)          # cqp* = the same with parallel positions
        def f(inp):
            lay = Layout(inp, pieces, iso=(c_ != 'cq_noiso'), par=par)
            if c_ in ('cq', 'cq_noiso'): return run(inp, lay)[0]
            if c_ == 'cq_boost_mid': return run(inp, lay, boost_masks(lay, MID_B))[0]
            if c_ == 'cq_boost_late': return run(inp, lay, boost_masks(lay, LATE_B))[0]
            lam = 1.0 if c_ == 'cq_select_soft' else 4.0
            return run(inp, lay, select_masks(lay, f.p, lam))[0]
        return f
    for par, key in ((False, 'relevance'), (True, 'relevance_par')):   # the model's own attention, original option order
        if not any(c.startswith('cqp_select' if par else 'cq_select') for c in CONDS): continue
        inp, _ = qwen_inputs(PROC, prompt_of(prompt, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
        lay = Layout(inp, pieces, par=par); _, att = run(inp, lay, record=SEL_LAYERS)
        p = torch.stack([att[li] for li in SEL_LAYERS]).mean(0).numpy(); p = np.maximum(p / p.sum(), 1e-6)
        rec[key] = p.round(4).tolist(); del inp; torch.cuda.empty_cache()
    for c in CONDS:
        f = mk(c)
        if c.startswith('cq_select'): f.p = rec['relevance']
        if c.startswith('cqp_select'): f.p = rec['relevance_par']
        rec[c] = score(media, prompt, texts, fwd=f)
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
R = run_loop(items, globals().get('OUTNAME', 'j7_chunk_queries.jsonl'), item, show=['base'] + list(CONDS))
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in ['base'] + list(CONDS)} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in ['base'] + list(CONDS)}
for key in ('relevance', 'relevance_par'):   # does the model's own chunk choice fall on the speech? (analysis only)
    S = [r for r in OK if key in r]
    if S: RESULT[f'top_chunk_has_speech%[{key}]'] = round(100 * float(np.mean([r['speech_activity'][int(np.argmax(r[key]))] > 0.3 for r in S])), 1)
S = [r for r in OK if 'speech_activity' in r]
if S:
    RESULT['chance_speech_chunk%'] = round(100 * float(np.mean([np.mean(np.array(r['speech_activity']) > 0.3) for r in S])), 1)
print(RESULT, flush=True)

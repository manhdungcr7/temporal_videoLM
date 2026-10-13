"""J6 — SEGMENT SUMMARY QUERIES (training-free, ONE forward pass).
Proposal: text "summary queries" are appended after the media; query i may read ONLY the media tokens of segment i (queries do
not see each other); the question then reads the queries. Optionally the question's attention to the queries is boosted in the
late layers. Everything is done with a 4-D attention mask + explicit TMRoPE position ids; no generation, no training.
Variants (all automatic):
  segmentation   grid  : fixed SEG_S-second parts ("part i of K")
                 anchor: before / while / after the speech (fixed audio-activity detector)
  query type     av    : one query per segment that sees video AND audio of the segment ("Summarize ...")
                 sep   : two queries per segment, sight-only (video tokens) and sound-only (audio tokens)  [source monitoring]
  question reads bn    : only the queries (bottleneck: no direct access to media)
                 media : queries + all media
  boost          +BETA added to the question rows' attention logits on query tokens in layers >= BOOST_FROM
Labels are used only for scoring. Writes OUT/j6_summary_queries.jsonl."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 8))
SEED = int(globals().get('SEED', 909))
SEG_S = float(globals().get('SEG_S', 10.0))
BETA = float(globals().get('BETA', 2.0))
NL = len(MODEL_.model.layers)
BOOST_FROM = int(globals().get('BOOST_FROM', NL // 2))
CONDS = globals().get('CONDS', ['av_grid_bn', 'av_grid_media', 'sep_grid_bn', 'sep_grid_media_boost',
                                'av_anchor_bn', 'sep_anchor_bn', 'sep_anchor_media', 'sep_anchor_media_boost'])
TOK = PROC.tokenizer


def segments(kind, dur, w):
    if kind == 'grid':
        edges = list(np.arange(0.0, dur, SEG_S)) + [dur]
        if len(edges) > 2 and edges[-1] - edges[-2] < 3.0: edges.pop(-2)       # merge a short tail
        K = len(edges) - 1
        return [(f'part {k + 1} of {K}', edges[k], edges[k + 1]) for k in range(K)]
    s0, s1 = w
    segs = [('the part before the speech', 0.0, s0), ('the part while the speech is heard', s0, s1), ('the part after the speech', s1, dur)]
    return [s for s in segs if s[2] - s[1] >= 1.0]


def queries(segs, qtype):
    """list of (text, seg index, 'av'|'v'|'a'); each text starts with a newline so the spans are easy to separate."""
    out = []
    for k, (name, _, _) in enumerate(segs):
        if qtype == 'av': out.append((f'\n[{name}] Summarize the content of this video segment.', k, 'av'))
        else:
            out.append((f'\n[{name}, sight only] Describe what is seen in this segment.', k, 'v'))
            out.append((f'\n[{name}, sound only] Describe what is heard in this segment.', k, 'a'))
    return out


def locate(input_ids, text, pieces, after):
    """token span of every piece of `text` inside input_ids, by decoding the tokens after the media one by one and matching
    character offsets (robust to BPE merges across piece boundaries: a token belongs to the piece holding its first char)."""
    seq = input_ids[0].tolist()[after + 1:]
    strs = [TOK.decode([t]) for t in seq]
    starts = np.cumsum([0] + [len(s) for s in strs[:-1]]); full = ''.join(strs)
    p0 = full.find(text)
    if p0 < 0: raise RuntimeError('query text not found in the token sequence')
    bounds = np.cumsum([p0] + [len(p) for p in pieces])
    spans = []
    for k in range(len(pieces)):
        toks = [t for t, c in enumerate(starts) if bounds[k] <= c < bounds[k + 1]]
        if not toks: raise RuntimeError('empty query span')
        spans.append((after + 1 + toks[0], after + 1 + toks[-1] + 1))
    q_end = after + 1 + max(t for t, c in enumerate(starts) if c < bounds[-1]) + 1
    return spans, q_end


def sq_forward(qs, segs, read, boost):
    text_q = ''.join(q for q, _, _ in qs)
    def f(inp):
        emb = MODEL_.get_input_embeddings().weight; dev = emb.device
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        L = inp['input_ids'].shape[1]
        spans, q_end = locate(inp['input_ids'].cpu(), text_q, [q for q, _, _ in qs], last_media)
        sec_c, v, a = sec.cpu(), is_v.cpu(), is_a.cpu(); media = v | a
        m = torch.zeros(L, L); m = m.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float('-inf'))
        qcols = torch.zeros(L, dtype=torch.bool)
        for (s, e), (_, k, kind) in zip(spans, qs):
            _, t0, t1 = segs[k]
            inseg = (sec_c >= t0) & (sec_c < t1 + 1e-6)
            allow = torch.zeros(L, dtype=torch.bool)
            if kind in ('av', 'v'): allow |= v & inseg
            if kind in ('av', 'a'): allow |= a & inseg
            m[s:e][:, media & ~allow] = float('-inf')                       # only its own segment and modality
            qcols[s:e] = True
        for (s, e) in spans:                                                 # queries do not see each other
            m[s:e][:, qcols] = float('-inf')
            m[s:e, s:e] = torch.triu(torch.full((e - s, e - s), float('-inf')), 1)
        if read == 'bn': m[q_end:][:, media] = float('-inf')                 # question/answer rows: no direct media
        neg = torch.finfo(emb.dtype).min
        base = m.clamp(min=neg).to(emb.dtype).to(dev).view(1, 1, L, L)
        hooks = []
        if boost:
            mb = m.clone(); mb[q_end:][:, qcols] += BETA
            boosted = mb.clamp(min=neg).to(emb.dtype).to(dev).view(1, 1, L, L)
            def pre(mod, args, kwargs):
                am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = boosted.to(am_.device if am_ is not None else boosted.device); return args, kwargs
            hooks = [MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for li in range(BOOST_FROM, NL)]
        kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
        try:
            with torch.no_grad():
                return MODEL_(**kw, attention_mask=base, position_ids=pos.to(dev), use_audio_in_video=True).logits[0, -1].float()
        finally:
            for h in hooks: h.remove()
    return f, text_q


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    rec['base'] = score(media, stem, texts)
    for c in CONDS:
        qtype, segk, read = c.split('_')[:3]; boost = c.endswith('_boost')
        segs = segments(segk, dur, w); qs = queries(segs, qtype)
        fwd, text_q = sq_forward(qs, segs, read, boost)
        rec[c] = score(media, text_q + '\n' + stem, texts, fwd=fwd)
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
R = run_loop(items, 'j6_summary_queries.jsonl', item, show=['base'] + list(CONDS))
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in ['base'] + list(CONDS)} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in ['base'] + list(CONDS)}
print(RESULT, flush=True)

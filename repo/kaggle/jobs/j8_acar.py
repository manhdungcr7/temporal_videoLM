"""J8 — QUESTION-GUIDED TEMPORAL ATTENTION RE-ALLOCATION (training-free, inside the model; no ASR, no text pipeline).
Goal (team): for any text question, the video LM itself should focus on the part of the video - frames AND audio - that is
needed to answer, and the rest should contribute almost nothing.
Pass 1 (prefill, eager): relevance of every 2-s chunk c = attention of the question/answer rows to the tokens of c, averaged
over all heads of layers 12-35 (label-free), measured separately on video tokens (v), audio tokens (a) and both (av).
Salience correction (PMI): the same profile for a neutral question ("What is happening in the video?") is divided out,
  rel(c) = p(c|q) * (p(c|q) / p(c|neutral))^ALPHA
so chunks that are always salient (e.g. speech) only win when the question is about them (J9: raw attention always lands on
the speech even when the question asks about a bell).
Pass 2: in layers 13-27 the question/answer rows get LAMBDA*log(rel(c)/max rel) on the tokens of chunk c.
Conditions (rotated options):
  base
  oracle_route        hard window from the clip's audio activity, video only + audio blocked (upper bound for SVHalluc)
  acar_a_l4_ab        audio-attention anchor (raw), re-weight video, audio blocked                  (speech-anchor design)
  acar_a_pmi_l4_ab    same with PMI anchor
  qtar_av_pmi_l1      GENERAL: av relevance with PMI, re-weight video AND audio tokens, lambda 1      (team proposal)
  qtar_av_pmi_l4      same, lambda 4
  qtar_av_pmi_l4_sg   same + source gating: audio tokens get the bias too but the video re-weighting is applied alone in the
                      upper half of the routed layers (late layers read the selected scene with the sound off)
  (TA/TF) look        "Is the action that the speaker talks about shown in the video?" with the question rows reading all audio
                      but video only before / during / after the anchor window (analysis + rule readout, reported separately)
Analysis only: IoU of every relevance profile's window with the true speech chunks. Writes OUT/j8_acar.jsonl."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 8))
SEED = int(globals().get('SEED', 909))
MAXD = float(globals().get('J8_MAX_DUR', 34.0))
ALPHA = float(globals().get('ALPHA', 2.0))
NL = len(MODEL_.model.layers)
ROUTE = list(range(13, 28))
REL_LAYERS = list(range(12, NL))
NEUTRAL = 'What is happening in the video?'
LOOK_Q = 'Is the action that the speaker talks about shown in the video?'


class Lay:
    def __init__(self, inp):
        emb = MODEL_.get_input_embeddings().weight; self.dev, self.dtype = emb.device, emb.dtype
        self.pos, sec, v, a, self.last_media = media_layout(MODEL_, PROC, inp)
        self.L = inp['input_ids'].shape[1]; sec, v, a = sec.cpu(), v.cpu(), a.cpu()
        self.v, self.a = v, a; self.nc = int(sec[v | a].max() // 2) + 1
        self.chunk = torch.clamp((sec // 2).long(), 0, self.nc - 1)
        self.rows = torch.arange(self.L) > self.last_media
        m = torch.zeros(self.L, self.L); self.m = m.masked_fill(torch.triu(torch.ones(self.L, self.L, dtype=torch.bool), 1), float('-inf'))

    def t(self, m): return m.clamp(min=torch.finfo(self.dtype).min).to(self.dtype).to(self.dev).view(1, 1, self.L, self.L)


def forward(inp, lay, special=None, layers=ROUTE, record=None):
    """special: mask used in `layers` (a dict {layer: mask} is also accepted); record: layers whose question-row attention
    mass per chunk on video and audio tokens is returned (eager)."""
    hooks, rec = [], {}
    if special is not None:
        per = special if isinstance(special, dict) else {li: special for li in layers}
        per_t = {}
        for li, mk in per.items():
            key = id(mk)
            if key not in per_t: per_t[key] = lay.t(mk)
            sp = per_t[key]
            def pre(mod, args, kwargs, sp=sp):
                am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = sp.to(am_.device if am_ is not None else sp.device); return args, kwargs
            hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    if record:
        MODEL_.set_attn_implementation('eager')
        Ga = torch.zeros(lay.L, lay.nc); Ga[torch.where(lay.a)[0], lay.chunk[lay.a]] = 1
        Gv = torch.zeros(lay.L, lay.nc); Gv[torch.where(lay.v)[0], lay.chunk[lay.v]] = 1
        for li in record:
            def hk(mod, args, out, li=li):
                w = out[1]
                if w is None: return
                q = w[0][:, lay.rows.to(w.device), :].float().mean(1)                  # [H, L]
                rec[li] = ((q @ Ga.to(w.device)).mean(0).cpu(), (q @ Gv.to(w.device)).mean(0).cpu())
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


def profiles(media, prompt):
    """relevance per chunk from the question rows: {'a','v','av'} normalised to sum 1."""
    inp, _ = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    lay = Lay(inp); _, rec = forward(inp, lay, record=REL_LAYERS)
    a = torch.stack([rec[li][0] for li in REL_LAYERS]).mean(0).numpy(); v = torch.stack([rec[li][1] for li in REL_LAYERS]).mean(0).numpy()
    del inp; torch.cuda.empty_cache()
    n = lambda x: np.maximum(x / x.sum(), 1e-6)
    return {'a': n(a), 'v': n(v), 'av': n(n(a) + n(v))}      # each modality normalised first: ~144 video vs 25 audio tokens/s


def pmi(p, p0): r = p * (p / np.maximum(p0, 1e-6)) ** ALPHA; return np.maximum(r / r.sum(), 1e-6)


def bias_mask(lay, rel, lam, on=('v',), block_audio=False):
    m = lay.m.clone(); r = torch.where(lay.rows)[0]
    bias = lam * torch.log(torch.as_tensor(rel, dtype=torch.float32) / float(max(rel)))
    for mod in on:
        cols = torch.where(lay.v if mod == 'v' else lay.a)[0]
        m[r.unsqueeze(1), cols.unsqueeze(0)] += bias[lay.chunk[cols]].unsqueeze(0)
    if block_audio: m[r.unsqueeze(1), torch.where(lay.a)[0].unsqueeze(0)] = float('-inf')
    return m


def window_mask(lay, t0, t1, block_audio):
    m = lay.m.clone(); c0, c1 = int(t0 // 2), max(int(t0 // 2) + 1, int(np.ceil(t1 / 2)))
    out_v = lay.v & ((lay.chunk < c0) | (lay.chunk >= c1)); r = torch.where(lay.rows)[0]
    m[r.unsqueeze(1), torch.where(out_v)[0].unsqueeze(0)] = float('-inf')
    if block_audio: m[r.unsqueeze(1), torch.where(lay.a)[0].unsqueeze(0)] = float('-inf')
    return m


def top_window(p):
    on = list(p > p.mean()) + [False]; best = (0, 0, 0); st = None
    for k, x in enumerate(on):
        if x and st is None: st = k
        if not x and st is not None:
            if k - st > best[0]: best = (k - st, st, k)
            st = None
    return best[1], best[2]


def chunk_activity(a, n):
    hop = 800; k = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95); act = db > lo + 0.5 * (hi - lo)
    c = np.minimum(((np.arange(k) + 0.5) * 0.05 // 2).astype(int), n - 1)
    return [round(float(act[c == j].mean()), 3) if (c == j).any() else 0.0 for j in range(n)]


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question']); task = r['task_type_abbr']
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts)))))
    P0 = profiles(media, NEUTRAL)
    rel = {'a': P['a'], 'a_pmi': pmi(P['a'], P0['a']), 'av_pmi': pmi(P['av'], P0['av']), 'v_pmi': pmi(P['v'], P0['v'])}
    nc = len(P['a']); act = np.array(chunk_activity(a, nc)) > 0.3; rec['speech_chunks'] = act.astype(int).tolist()
    for k_, p in list(rel.items()) + [('v', P['v']), ('av', P['av'])]:
        s0, s1 = top_window(p); wv = np.zeros(nc, bool); wv[s0:s1] = True
        rec[f'rel_{k_}'] = np.round(p, 4).tolist(); rec[f'rel_{k_}_iou'] = round(float((wv & act).sum() / max(1, (wv | act).sum())), 3)
    rec['neutral_a'] = np.round(P0['a'], 4).tolist(); rec['neutral_v'] = np.round(P0['v'], 4).tolist()

    def mk(fn):
        def f(inp):
            lay = Lay(inp); return forward(inp, lay, fn(lay))[0]
        return f
    def sg(lay):   # both modalities re-weighted in layers 13-20; the upper routed layers additionally read with the sound off
        both = bias_mask(lay, rel['av_pmi'], 4.0, ('v', 'a')); off = bias_mask(lay, rel['av_pmi'], 4.0, ('v',), block_audio=True)
        return {li: (both if li < 21 else off) for li in ROUTE}
    rec['base'] = score(media, stem, texts)
    rec['oracle_route'] = score(media, stem, texts, fwd=mk(lambda lay: window_mask(lay, w[0], w[1], True)))
    rec['acar_a_l4_ab'] = score(media, stem, texts, fwd=mk(lambda lay: bias_mask(lay, rel['a'], 4.0, ('v',), True)))
    rec['acar_a_pmi_l4_ab'] = score(media, stem, texts, fwd=mk(lambda lay: bias_mask(lay, rel['a_pmi'], 4.0, ('v',), True)))
    rec['qtar_av_pmi_l1'] = score(media, stem, texts, fwd=mk(lambda lay: bias_mask(lay, rel['av_pmi'], 1.0, ('v', 'a'))))
    rec['qtar_av_pmi_l4'] = score(media, stem, texts, fwd=mk(lambda lay: bias_mask(lay, rel['av_pmi'], 4.0, ('v', 'a'))))
    rec['qtar_av_pmi_l4_sg'] = score(media, stem, texts, fwd=mk(sg))
    if task in ('TA', 'TF'):
        s0, s1 = top_window(rel['a_pmi']); T0, T1 = 2.0 * s0, min(dur, 2.0 * s1); look = {}
        for name, t0, t1 in (('before', 0.0, T0), ('during', T0, T1), ('after', T1, dur)):
            if t1 - t0 < 1.0: continue
            s = score(media, LOOK_Q, ['Yes', 'No'], fwd=mk(lambda lay, t0=t0, t1=t1: window_mask(lay, t0, t1, False)))
            look[name] = round(s[0] - s[1], 3)
        rec['look'] = look
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
C = ('base', 'oracle_route', 'acar_a_l4_ab', 'acar_a_pmi_l4_ab', 'qtar_av_pmi_l1', 'qtar_av_pmi_l4', 'qtar_av_pmi_l4_sg')
R = run_loop(items, 'j8_acar.jsonl', item, show=C)
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in C} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in C}
RESULT['relevance_window_iou_with_speech_median'] = {k: round(float(np.median([r[f'rel_{k}_iou'] for r in OK])), 3)
                                                    for k in ('a', 'a_pmi', 'v', 'v_pmi', 'av', 'av_pmi')} if OK else None


def look_rule(r):
    lk = r.get('look') or {}
    if not lk: return None
    best = max(lk, key=lk.get)
    if r['task'] == 'TA': return 0 if best == 'during' else 1
    return {'before': 0, 'during': 1, 'after': 2}[best]
L = [r for r in OK if r.get('look')]
RESULT['look_rule_acc'] = {t: round(100 * float(np.mean([look_rule(r) == r['gold'] for r in L if r['task'] == t])), 1) for t in ('TA', 'TF') if any(r['task'] == t for r in L)}
print(RESULT, flush=True)

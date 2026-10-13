"""J4 — CAUSAL ORDER: CAN THE MODEL RE-INTERPRET WHAT IT SAW AFTER HEARING THE SPEECH?  (TA + TF + CMTB)
Hypothesis (untested so far): Qwen2.5-Omni reads media under a causal mask, so the video tokens of an event that happens
BEFORE the speech can never be contextualised by that speech (it comes later in the sequence). Humans do this routinely:
perception is revised by later input (postdiction). Evidence that fits: accuracy is much higher when the speech is in the
first half (CMTB 70 vs 50, TA 68 vs 56, TF 48 vs 11). But attention front-loading predicts the same split, so this job
separates the two:
  base          native causal mask
  v2a_all       video tokens may also read ALL audio tokens, including later ones (retrospective binding), all layers
  v2a_early     the same only in layers 1-14 (later layers native), limiting the distribution shift
  media_bidir   every media token reads every media token (fully bidirectional media block), all layers
Text rows are never changed and TMRoPE position ids are kept explicitly, so only "who may read whom" inside the media changes.
The binding lab already showed that LOCAL edges (same chunk, +-1 chunk) do nothing; here the edges span the whole clip.
Representation: per 2-s chunk, mean video and audio hidden states at layers 9/18/27/36 for base and v2a_all
  -> offline probes: is "the narrated event is here" decodable from the event chunk's video tokens, and is it decodable for
     events BEFORE the speech only under v2a_all?   (event half from the benchmark construction; analysis only)
Decision: v2a_* helps mainly on items whose event precedes the speech (TF Past, speech-second TA/CMTB) and the event-chunk
probe rises -> causal order is a root cause (missing postdiction). No change -> front-loading / readout is the cause."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 20))
NL = len(MODEL_.model.layers)
REC_LAYERS = [9, 18, 27, NL]
HID4 = OUT / 'hidden_j4'; HID4.mkdir(exist_ok=True)


def media_mask(L, is_v, is_a, kind, dtype, device):
    m = torch.zeros(L, L, dtype=torch.float32, device=device)
    m = m.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1), float('-inf'))
    v, a = is_v.to(device), is_a.to(device)
    if kind == 'v2a':
        m[v.unsqueeze(1) & a.unsqueeze(0)] = 0.0
    elif kind == 'bidir':
        med = v | a; m[med.unsqueeze(1) & med.unsqueeze(0)] = 0.0
    return m.clamp(min=torch.finfo(dtype).min).to(dtype).view(1, 1, L, L)


def masked_fwd(inp, kind, layers=None, record=None):
    """kind applied in `layers` (None = all); the other layers keep the causal mask. record: (layers, chunk one-hots)."""
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]
    causal = media_mask(L, is_v, is_a, 'none', emb.dtype, emb.device)
    special = media_mask(L, is_v, is_a, kind, emb.dtype, emb.device)
    hooks = []
    if layers is None: base_mask = special
    else:
        base_mask = causal
        def pre(mod, args, kwargs):
            am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = special.to(am_.device if am_ is not None else special.device); return args, kwargs
        hooks += [MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for li in layers]
    store = {}
    if record is not None:
        rl, Gv, Ga = record
        for li in rl:
            def hook(mod, args, out, li=li):
                h = layer_out(out).float(); d = h.device
                store[li] = ((Gv.to(d).T @ h) / Gv.to(d).sum(0).clamp(min=1).unsqueeze(1)).cpu().half(), \
                            ((Ga.to(d).T @ h) / Ga.to(d).sum(0).clamp(min=1).unsqueeze(1)).cpu().half()
            hooks.append(MODEL_.model.layers[li - 1].register_forward_hook(hook))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=base_mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()
    return lg, store


def chunk_onehots(inp):
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    sec, is_v, is_a = sec.cpu(), is_v.cpu(), is_a.cpu(); L = inp['input_ids'].shape[1]
    nc = int(sec[is_v | is_a].max() // 2) + 1; ch = torch.clamp((sec // 2).long(), 0, nc - 1)
    Gv = torch.zeros(L, nc); Ga = torch.zeros(L, nc)
    Gv[torch.where(is_v)[0], ch[is_v]] = 1; Ga[torch.where(is_a)[0], ch[is_a]] = 1
    return Gv, Ga, nc


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    saved = {}
    def fwd_for(kind, layers=None, save_as=None):
        def f(inp):
            if save_as and save_as not in saved:
                Gv, Ga, nc = chunk_onehots(inp)
                lg, st = masked_fwd(inp, kind, layers, (REC_LAYERS, Gv, Ga)); saved[save_as] = st; saved['nc'] = nc; return lg
            return masked_fwd(inp, kind, layers)[0]
        return f
    rec['base'] = score(media, stem, texts, fwd=fwd_for('none', None, 'base'))
    rec['v2a_all'] = score(media, stem, texts, fwd=fwd_for('v2a', None, 'v2a'))
    rec['v2a_early'] = score(media, stem, texts, fwd=fwd_for('v2a', list(range(0, 14))))
    rec['media_bidir'] = score(media, stem, texts, fwd=fwd_for('bidir'))
    rec['n_chunks'] = saved['nc']
    np.savez_compressed(HID4 / f'c_{i}.npz', layers=np.array(REC_LAYERS),
                        **{f'{c}_{m}': np.stack([saved[c][l][k].numpy() for l in REC_LAYERS]) for c in ('base', 'v2a') for k, m in ((0, 'video'), (1, 'audio'))})
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=404)
R = run_loop(items, 'j4_postdiction.jsonl', item, show=('base', 'v2a_all', 'v2a_early', 'media_bidir'))


def event_before_speech(r):
    """True if the narrated/target event precedes the speech (TF Past, TA No with speech second); None if co-timed."""
    if r['task'] == 'TF': return {0: True, 1: None, 2: False}[r['gold']]
    if r['task'] == 'TA': return None if r['gold'] == 0 else (not r['speech_first'])
    return None


OK = [r for r in R if 'error' not in r]
summary = {}
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in OK if r['task'] == t]
    if not S: continue
    d = {'n': len(S)}
    for c in ('base', 'v2a_all', 'v2a_early', 'media_bidir'):
        d[c] = acc(S, c)[0]
        for sf_ in (True, False): d[f'{c}|speech_{"first" if sf_ else "second"}'] = acc([r for r in S if r['speech_first'] == sf_], c)
    E = [r for r in S if event_before_speech(r)]
    if E: d['event_before_speech'] = {c: acc(E, c) for c in ('base', 'v2a_all', 'v2a_early', 'media_bidir')}
    summary[t] = d
RESULT = {'summary': summary, 'errors': sum('error' in r for r in R)}
print(RESULT, flush=True)

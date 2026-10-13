"""Module A — IN-PASS SELF-ROUTING (training-free, one forward pass per question). Needs the J8 helpers.
Inside ONE prefill:
  layers MEAS (default 12-20) run with eager attention; their question/answer-row attention to the video and audio tokens of
  every 2-s chunk is accumulated  ->  after the last MEAS layer: relevance rel(c) = PMI against a cached, question-independent
  neutral prior (one extra prefill per VIDEO, reused for every question)
  layers APPLY (default 21-27): the question/answer rows get LAMBDA*log(rel/max rel) on the chunk tokens, and the modality the
  model said it does not need (text-only self-query: seen / heard / both) is blocked for those rows.
Exposes:  neutral_prior(media, band)  ->  dict a/v/av   (cache per video)
          inpass_forward(inp, prior, target, meas, apply_layers, lam)  ->  logits, relevance used
          self_target(stem)  ->  'seen' | 'heard' | 'both'"""
import numpy as np, torch

MEAS = list(range(12, 21)); APPLY = list(range(21, 28))
SELFQ = 'To answer the following question about a video, what must you rely on? Question: "{q}"'
SQ_OPTS = ['what is seen in the video frames', 'what is heard in the audio', 'both what is seen and what is heard']
ATTN_CFG = MODEL_.model.layers[0].self_attn.config


def self_target(stem):
    return ['seen', 'heard', 'both'][int(np.argmax(text_only_scores(SELFQ.format(q=stem), SQ_OPTS)))]


def band_profiles(media, prompt, band):
    inp, _ = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    lay = Lay(inp); _, rec = forward(inp, lay, record=band)
    a = torch.stack([rec[li][0] for li in band]).mean(0).numpy(); v = torch.stack([rec[li][1] for li in band]).mean(0).numpy()
    del inp; torch.cuda.empty_cache()
    n = lambda x: np.maximum(x / x.sum(), 1e-6)
    return {'a': n(a), 'v': n(v), 'av': n(n(a) + n(v))}


def neutral_prior(media, band=MEAS):
    return band_profiles(media, NEUTRAL, band)


def inpass_forward(inp, prior, target='both', meas=MEAS, apply_layers=APPLY, lam=4.0, key='av'):
    lay = Lay(inp); dev = lay.dev
    Ga = torch.zeros(lay.L, lay.nc); Ga[torch.where(lay.a)[0], lay.chunk[lay.a]] = 1
    Gv = torch.zeros(lay.L, lay.nc); Gv[torch.where(lay.v)[0], lay.chunk[lay.v]] = 1
    st = {'a': [], 'v': [], 'mask': None, 'rel': None}
    hooks = []
    def to_eager(mod, args, kwargs): ATTN_CFG._attn_implementation = 'eager'; return args, kwargs
    def to_sdpa(mod, args, out): ATTN_CFG._attn_implementation = 'sdpa'
    hooks.append(MODEL_.model.layers[meas[0]].register_forward_pre_hook(to_eager, with_kwargs=True))
    hooks.append(MODEL_.model.layers[meas[-1]].register_forward_hook(to_sdpa))
    for li in meas:
        def hk(mod, args, out, li=li):
            w = out[1]
            if w is None: return
            q = w[0][:, lay.rows.to(w.device), :].float().mean(1)
            st['a'].append((q @ Ga.to(w.device)).mean(0).cpu()); st['v'].append((q @ Gv.to(w.device)).mean(0).cpu())
            if li == meas[-1]:                                   # relevance is ready: build the mask for the APPLY layers
                n = lambda x: np.maximum(x / x.sum(), 1e-6)
                a = n(torch.stack(st['a']).mean(0).numpy()); v = n(torch.stack(st['v']).mean(0).numpy())
                p = {'a': a, 'v': v, 'av': n(n(a) + n(v))}[key]
                rel = pmi(p, prior[key]); st['rel'] = rel
                m = bias_mask(lay, rel, lam, ('v', 'a'))
                rr = torch.where(lay.rows)[0]
                if target in ('seen', 'heard'):
                    cols = torch.where(lay.a if target == 'seen' else lay.v)[0]
                    m[rr.unsqueeze(1), cols.unsqueeze(0)] = float('-inf')
                st['mask'] = lay.t(m)
            return (out[0], None) + tuple(out[2:])
        hooks.append(MODEL_.model.layers[li].self_attn.register_forward_hook(hk))
    for li in apply_layers:
        def pre(mod, args, kwargs):
            if st['mask'] is not None:
                am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = st['mask'].to(am_.device if am_ is not None else st['mask'].device)
            return args, kwargs
        hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(lay.m), position_ids=lay.pos.to(dev), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()
        ATTN_CFG._attn_implementation = 'sdpa'
    return lg, st['rel']

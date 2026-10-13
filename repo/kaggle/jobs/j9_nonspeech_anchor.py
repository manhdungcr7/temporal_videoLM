"""J9 — DOES THE ATTENTION ANCHOR WORK FOR NON-SPEECH SOUNDS, AND DOES IT FOLLOW THE QUESTION?
A synthetic sound (bell / beep / clap / whistle, 2-3 s, generated procedurally so its timing is exact) is mixed into the
NON-speech half of SVHalluc clips at the speech's loudness. Every clip then has two audio events: the speech and the sound.
For each clip, three questions (prefill only, no answer needed):
  q_sound   "What is happening in the video when the <sound> is heard?"
  q_speech  "What is happening in the video when the person is speaking?"
  q_none    "What is happening in the video?"                         (control)
Anchor profile = question rows' attention to the audio tokens of every 2-s chunk (all heads; bands: layers 12-35 / 1-12),
exactly as in ACAR (J8). Measured: does the anchor land on the sound chunks for q_sound and on the speech chunks for
q_speech (question-conditioned anchoring)? Writes OUT/j9_nonspeech_anchor.jsonl."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 8))
SEED = int(globals().get('SEED', 77))
MAXD = float(globals().get('J9_MAX_DUR', 34.0))
NL = len(MODEL_.model.layers)
BANDS = {'mid': list(range(12, NL)), 'early': list(range(0, 12))}
SOUNDS = ['bell', 'beeping sound', 'clapping', 'whistle']


def synth(kind, dur, seed):
    t = np.arange(int(dur * SR)) / SR; g = np.random.default_rng(seed)
    if kind == 'bell':
        x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((523, 1.5), (1046, 2.5), (1568, 3.5)))
    elif kind == 'beeping sound':
        x = np.sin(2 * np.pi * 1000 * t) * ((t * 4) % 1 < 0.5)
    elif kind == 'clapping':
        x = np.zeros_like(t)
        for c in np.arange(0, dur, 0.4):
            i = int(c * SR); n = int(0.05 * SR); x[i:i + n] += g.standard_normal(min(n, len(x) - i)) * np.exp(-np.arange(min(n, len(x) - i)) / (0.01 * SR))
    else:
        x = np.sin(2 * np.pi * np.cumsum(1500 + 400 * np.sin(2 * np.pi * 2 * t)) / SR)
    return (x / (np.abs(x).max() + 1e-9)).astype('float32')


def chunk_mask(t0, t1, nc):
    m = np.zeros(nc, bool)
    for c in range(nc):
        if (2 * c < t1) and (2 * c + 2 > t0): m[c] = True
    return m


def profile(media, question):
    inp, _ = qwen_inputs(PROC, question, media, 'av'); inp = move_inputs(inp, MODEL_)
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, v, a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]; sec, v, a = sec.cpu(), v.cpu(), a.cpu()
    nc = int(sec[v | a].max() // 2) + 1; ch = torch.clamp((sec // 2).long(), 0, nc - 1)
    Ga = torch.zeros(L, nc); Ga[torch.where(a)[0], ch[a]] = 1; rows = torch.arange(L) > last_media
    rec = {}
    def mk(li):
        def hk(mod, args, out):
            w = out[1]
            if w is None: return
            q = w[0][:, rows.to(w.device), :].float().mean(1); rec[li] = (q @ Ga.to(w.device)).mean(0).cpu()
            return (out[0], None) + tuple(out[2:])
        return hk
    MODEL_.set_attn_implementation('eager')
    hooks = [MODEL_.model.layers[li].self_attn.register_forward_hook(mk(li)) for li in range(NL)]
    try:
        with torch.no_grad(): MODEL_(**inp, use_audio_in_video=True)
    finally:
        for h in hooks: h.remove()
        MODEL_.set_attn_implementation('sdpa')
    del inp; torch.cuda.empty_cache()
    out = {}
    for b, ls in BANDS.items():
        p = torch.stack([rec[li] for li in ls]).mean(0).numpy(); out[b] = p / p.sum()
    return out, nc


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    s0, s1 = w; kind = SOUNDS[i % len(SOUNDS)]; L_s = 2.5
    free = (0.5, s0 - 0.5) if speech_first(w, dur) is False else (s1 + 0.5, dur - 0.5)    # the non-speech half
    if free[1] - free[0] < L_s + 0.5: raise RuntimeError('no room for the sound')
    t0 = round(float(np.random.default_rng(i).uniform(free[0], free[1] - L_s)), 2); t1 = t0 + L_s
    seg = a[int(s0 * SR): int(s1 * SR)]; rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    x = synth(kind, L_s, i); x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * rms
    a2 = a.copy(); a2[int(t0 * SR): int(t0 * SR) + len(x)] += x[: len(a2) - int(t0 * SR)]
    media = prepare_video(with_audio(src, np.clip(a2, -1, 1), 'snd'))
    rec = dict(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)], sound=kind, sound_t=[t0, round(t1, 2)])
    qs = {'q_sound': f'What is happening in the video when the {kind} is heard?',
          'q_speech': 'What is happening in the video when the person is speaking?',
          'q_none': 'What is happening in the video?'}
    for qn, q in qs.items():
        prof, nc = profile(media, q)
        snd, sp = chunk_mask(t0, t1, nc), chunk_mask(s0, s1, nc)
        for b, p in prof.items():
            rec[f'{qn}_{b}'] = p.round(4).tolist()
            rec[f'{qn}_{b}_mass'] = {'sound': round(float(p[snd].sum()), 3), 'speech': round(float(p[sp & ~snd].sum()), 3),
                                    'sound_share_of_time': round(float(snd.mean()), 3), 'speech_share_of_time': round(float((sp & ~snd).mean()), 3)}
            rec[f'{qn}_{b}_top'] = 'sound' if snd[int(np.argmax(p))] else ('speech' if sp[int(np.argmax(p))] else 'other')
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
R = run_loop(items, 'j9_nonspeech_anchor.jsonl', item, show=())
OK = [r for r in R if 'error' not in r]
RESULT = {'n': len(OK)}


def pmi_top(r, qn, b, alpha):
    """anchor = p(c|q) * (p(c|q) / p(c|neutral))^alpha : removes the salience prior shared with the neutral question."""
    p, p0 = np.array(r[f'{qn}_{b}']), np.array(r[f'q_none_{b}'])
    s = p * (p / np.maximum(p0, 1e-6)) ** alpha; nc = len(p); c = int(np.argmax(s))
    return 'sound' if chunk_mask(*r['sound_t'], nc)[c] else ('speech' if chunk_mask(*r['win'], nc)[c] else 'other')


for b in BANDS:
    for alpha in (1, 2, 4):
        for qn in ('q_sound', 'q_speech'):
            tops = [pmi_top(r, qn, b, alpha) for r in OK]
            RESULT[f'{b}:{qn}:pmi{alpha}'] = {k: round(100 * tops.count(k) / max(1, len(tops)), 1) for k in ('sound', 'speech', 'other')}
for b in BANDS:
    for qn in ('q_sound', 'q_speech', 'q_none'):
        tops = [r[f'{qn}_{b}_top'] for r in OK]
        RESULT[f'{b}:{qn}'] = {k: round(100 * tops.count(k) / max(1, len(tops)), 1) for k in ('sound', 'speech', 'other')} | \
            {'mass_sound': round(float(np.mean([r[f'{qn}_{b}_mass']['sound'] for r in OK])), 3) if OK else None,
             'mass_speech': round(float(np.mean([r[f'{qn}_{b}_mass']['speech'] for r in OK])), 3) if OK else None}
print(RESULT, flush=True)

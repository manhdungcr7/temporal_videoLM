"""J5 — DOES THE MODEL KNOW, INTERNALLY, WHEN THE SPEECH IS AND WHICH VIDEO MOMENT GOES WITH IT?  (TA + TF + CMTB)
Observations: routing the question to the speech-time video (+ blocking audio) fixes CMTB (base 51 -> 96) but the window
came from an external RMS heuristic that only works because the non-speech half is digital silence. A fully automatic,
general method needs the window from the model itself, and it must survive real (noisy) audio.
Per item, clean and noisy (pink noise 10 dB below the speech):
  attention (eager) of every layer x head, from the last token and from the question rows, summed over the audio tokens
  and over the video tokens of every 2-s chunk -> arrays [layers, heads, chunks]   (saved to attn_j5/a_<idx>.npz)
  speech activity per chunk (fraction of active 50-ms frames, from the clean audio; analysis only)
  base and gate_audio scores (window: RMS on clean audio; and RMS re-detected on the noisy audio)
Offline: find 'speech-locating' heads (audio attention tracks speech activity), check whether they also do it under noise,
derive a window from them (no external VAD), and test whether any heads move video attention to the speech-time chunks."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 15))
MAXD = float(globals().get('J5_MAX_DUR', 34.0))
ATT = OUT / 'attn_j5'; ATT.mkdir(exist_ok=True)


def chunk_activity(a, dur, n_chunks):
    hop = 800; n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95); act = db > lo + 0.5 * (hi - lo)
    t = (np.arange(n) + 0.5) * 0.05; c = np.minimum((t // 2).astype(int), n_chunks - 1)
    return [round(float(act[c == k].mean()), 3) if (c == k).any() else 0.0 for k in range(n_chunks)]


def attention_maps(media, stem, texts):
    inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, tuple(range(len(texts)))), media, 'av'); inp = move_inputs(inp, MODEL_)
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    sec, is_v, is_a = sec.cpu(), is_v.cpu(), is_a.cpu()
    L = inp['input_ids'].shape[1]; rows_q = torch.arange(L) > last_media
    n_chunks = int(sec[is_v | is_a].max() // 2) + 1
    ch = torch.clamp((sec // 2).long(), 0, n_chunks - 1)
    Ga = torch.zeros(L, n_chunks); Gv = torch.zeros(L, n_chunks)
    Ga[torch.where(is_a)[0], ch[is_a]] = 1; Gv[torch.where(is_v)[0], ch[is_v]] = 1
    out = {'a_last': [], 'v_last': [], 'a_q': [], 'v_q': []}
    def mk(li):
        def hook(mod, args, o):
            w = o[1]
            if w is None: return
            dv = w.device; w = w[0]                                          # [H, L, L], no full fp32 copy
            last = w[:, -1, :].float(); q = w[:, rows_q.to(dv), :].float().mean(1)
            ga, gv = Ga.to(dv), Gv.to(dv)
            out['a_last'].append((last @ ga).cpu()); out['v_last'].append((last @ gv).cpu())
            out['a_q'].append((q @ ga).cpu()); out['v_q'].append((q @ gv).cpu())
            return (o[0], None) + tuple(o[2:])
        return hook
    MODEL_.set_attn_implementation('eager')
    hooks = [MODEL_.model.layers[li].self_attn.register_forward_hook(mk(li)) for li in range(len(MODEL_.model.layers))]
    try:
        with torch.no_grad(): MODEL_(**inp, use_audio_in_video=True)
    finally:
        for h in hooks: h.remove()
        MODEL_.set_attn_implementation('sdpa')
    del inp; torch.cuda.empty_cache()
    return {k: torch.stack(v).numpy().astype('float16') for k, v in out.items()}, n_chunks


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    stem, texts = parse_question(r['question']); task = r['task_type_abbr']
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    media = prepare_video(src)
    an = noisy_audio(a, w, 10.0, seed=i); nz = with_audio(src, an, 'noisy10'); mn = prepare_video(nz)
    maps_c, nc = attention_maps(media, stem, texts)
    maps_n, _ = attention_maps(mn, stem, texts)
    np.savez_compressed(ATT / f'a_{i}.npz', **{f'clean_{k}': v for k, v in maps_c.items()}, **{f'noisy_{k}': v for k, v in maps_n.items()})
    rec['n_chunks'] = nc; rec['speech_activity'] = chunk_activity(a, dur, nc)
    nw = speech_window(decode_audio(nz)); rec['noisy_detected_win'] = None if nw is None else [round(nw[0], 2), round(nw[1], 2)]
    rec['base'] = score(media, stem, texts); rec['noisy'] = score(mn, stem, texts)
    rec['gate_audio'] = scores(media, stem, texts, mask_kind='gate_audio', window=w)
    if nw is not None: rec['noisy_gate_audio_detected'] = scores(mn, stem, texts, mask_kind='gate_audio', window=nw)
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=505)
R = run_loop(items, 'j5_internal_localization.jsonl', item, show=('base', 'noisy', 'gate_audio'))

# quick on-device summary: which heads' audio attention (last token) correlates with speech activity, clean vs noisy
OK = [r for r in R if 'error' not in r]
corr = {'clean': [], 'noisy': []}
for r in OK:
    z = np.load(ATT / f'a_{r["idx"]}.npz'); s = np.array(r['speech_activity'], dtype='float32')
    if s.std() < 1e-6: continue
    for cond in corr:
        A = z[f'{cond}_a_last'].astype('float32')                            # [layers, heads, chunks]
        A = A / (A.sum(-1, keepdims=True) + 1e-9)
        Ac = A - A.mean(-1, keepdims=True); sc = s - s.mean()
        corr[cond].append((Ac * sc).sum(-1) / (np.sqrt((Ac ** 2).sum(-1) * (sc ** 2).sum()) + 1e-9))
summary = {'n': len(OK)}
if corr['clean']:
    C, Nz = np.mean(corr['clean'], 0), np.mean(corr['noisy'], 0)
    top = np.dstack(np.unravel_index(np.argsort(-C.ravel())[:10], C.shape))[0]
    summary['top_speech_heads(layer,head,clean_r,noisy_r)'] = [(int(l), int(h), round(float(C[l, h]), 3), round(float(Nz[l, h]), 3)) for l, h in top]
    summary['heads_with_clean_r>0.5'] = int((C > 0.5).sum()); summary['heads_with_noisy_r>0.5'] = int((Nz > 0.5).sum())
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in OK if r['task'] == t]
    if S: summary[t] = {c: acc(S, c) for c in ('base', 'noisy', 'gate_audio', 'noisy_gate_audio_detected')}
RESULT = {'summary': summary, 'errors': sum('error' in r for r in R)}
print(RESULT, flush=True)

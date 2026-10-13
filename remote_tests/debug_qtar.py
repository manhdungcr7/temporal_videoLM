"""Single-item mechanism debug for QTAR (local GPU). Needs the J8 helpers (exec'd from j8_acar.py, up to its item loop).
A) Does the bias really move the question rows' attention to the selected chunks?  Per routed layer band, the mass of
   question-row attention on each 2-s chunk (video and audio tokens) without and with the QTAR bias, plus option probs.
B) Scene anchoring (visual analogue of J9): a red square is drawn in a known 2.5-s window of the non-speech half; the
   question "What is happening in the video when a red square appears on the screen?" vs the neutral question -> does the
   VIDEO relevance (raw / PMI) land on the square's chunks?"""
import json, numpy as np, torch
src_j8 = (ROOT / 'j8_acar.py').read_text(encoding='utf-8')
exec(src_j8[:src_j8.index('items = sample_items')], globals())
IDX = int(globals().get('DEBUG_IDX', 1713))
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8')); r = rows[IDX]
src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / SR; w = speech_window(a)
media = prepare_video(src); stem, texts = parse_question(r['question']); gold = 'ABC'.index(r['answer'])
print('item', IDX, r['task_type_abbr'], 'dur', round(dur, 1), 'speech', [round(x, 1) for x in w], 'gold', gold, texts, flush=True)
P = profiles(media, prompt_of(stem, texts, (0, 1, 2))); P0 = profiles(media, NEUTRAL)
rel = pmi(P['av'], P0['av']); nc = len(rel)
act = np.array(chunk_activity(a, nc)) > 0.3
fmt = lambda x: ' '.join(f'{v:.2f}' for v in x)
print('speech chunks    ', ''.join('S' if s else '.' for s in act))
for k in ('a', 'v', 'av'): print(f'raw  {k:3s}         ', fmt(P[k]))
print('neutral av        ', fmt(P0['av'])); print('PMI  av (rel)     ', fmt(rel), flush=True)


def mass(inp, lay, special, layers):
    """question-row attention mass per chunk on video/audio tokens, averaged over `layers` (eager)."""
    MODEL_.set_attn_implementation('eager')
    Ga = torch.zeros(lay.L, lay.nc); Ga[torch.where(lay.a)[0], lay.chunk[lay.a]] = 1
    Gv = torch.zeros(lay.L, lay.nc); Gv[torch.where(lay.v)[0], lay.chunk[lay.v]] = 1
    acc_ = {}; hooks = []
    sp = None if special is None else lay.t(special)
    for li in layers:
        if sp is not None:
            def pre(mod, args, kwargs):
                am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = sp.to(am_.device if am_ is not None else sp.device); return args, kwargs
            hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        def hk(mod, args, out, li=li):
            ww = out[1]; q = ww[0][:, lay.rows.to(ww.device), :].float().mean(1).mean(0)
            acc_[li] = ((q @ Gv.to(ww.device)).cpu(), (q @ Ga.to(ww.device)).cpu(), float(q.sum()))
            return (out[0], None) + tuple(out[2:])
        hooks.append(MODEL_.model.layers[li].self_attn.register_forward_hook(hk))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad(): lg = MODEL_(**kw, attention_mask=lay.t(lay.m), position_ids=lay.pos.to(lay.dev), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()
        MODEL_.set_attn_implementation('sdpa')
    v = torch.stack([acc_[li][0] for li in layers]).mean(0).numpy(); au = torch.stack([acc_[li][1] for li in layers]).mean(0).numpy()
    return v, au, torch.softmax(lg[LETTERS[:3]].cpu(), -1).numpy()


inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, (0, 1, 2)), media, 'av'); inp = move_inputs(inp, MODEL_); lay = Lay(inp)
for name, special in (('base', None), ('qtar_l1', bias_mask(lay, rel, 1.0, ('v', 'a'))), ('qtar_l4', bias_mask(lay, rel, 4.0, ('v', 'a'))),
                      ('oracle', window_mask(lay, w[0], w[1], True))):
    v, au, pr = mass(inp, lay, special, ROUTE)
    sel = rel >= np.sort(rel)[-max(1, int(act.sum()))]
    print(f'\n[{name}] option probs {np.round(pr, 3)} (gold {gold})')
    print('   video mass/chunk ', fmt(v / max(v.sum(), 1e-9)), f'| total media mass {v.sum() + au.sum():.3f}')
    print('   audio mass/chunk ', fmt(au / max(au.sum(), 1e-9)))
    print(f'   share of video mass on the selected chunks {v[sel].sum() / max(v.sum(), 1e-9):.2f}; on the speech chunks {v[act].sum() / max(v.sum(), 1e-9):.2f}', flush=True)
del inp; torch.cuda.empty_cache()

# ---- B) scene anchoring with a synthetic visual event
free = (0.3, w[0] - 0.3) if not speech_first(w, dur) else (w[1] + 0.3, dur - 0.3)
if free[1] - free[0] >= 2.5:
    t0 = round(free[0] + 0.2, 2); t1 = t0 + 2.5
    dst = CLIPS / f'{src.stem}_redsq.mp4'
    ffmpeg('-i', str(src), '-vf', f"drawbox=x=iw*0.35:y=ih*0.3:w=iw*0.3:h=ih*0.4:color=red@1:t=fill:enable='between(t,{t0},{t1})'",
           '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'copy', str(dst))
    m2 = prepare_video(dst)
    Q = profiles(m2, 'What is happening in the video when a red square appears on the screen?'); Q0 = profiles(m2, NEUTRAL)
    sq = np.array([(2 * c < t1) and (2 * c + 2 > t0) for c in range(len(Q['v']))])
    print(f'\nB) red square at {t0}-{t1} s  chunks', ''.join('R' if s else '.' for s in sq), '| speech', ''.join('S' if s else '.' for s in act))
    for k in ('v', 'a', 'av'):
        pm = pmi(Q[k], Q0[k])
        print(f'   {k:3s} raw  {fmt(Q[k])}  top on square: {bool(sq[int(np.argmax(Q[k]))])}')
        print(f'   {k:3s} PMI  {fmt(pm)}  top on square: {bool(sq[int(np.argmax(pm))])}')
else:
    print('B) no room for the visual event')
RESULT = 'done'

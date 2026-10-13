"""Dry-run debug of the 'look' step for TA/TF (single items, local GPU). Uses J8 helpers.
Question: "Is the action that the speaker talks about shown in the video?" asked once per window (before / during / after the
attention anchor). Regimes for the question/answer rows:
  allaudio : audio visible in every layer, video only in the window (J8 'look')
  split    : layers 0-12 see everything (HEAR), layers 13-27 see only the window's video and NO audio (LOOK with sound off)
  split_lo : same but the look starts at layer 9
  noaudio  : audio blocked in every layer (cannot know what was said: control)
  ref_text : J1-style reference: the model's own hearing (generated phrase) + vision-only crop of the window
For each regime: Yes-minus-No per window (rotated Yes/No), the chosen window, and the logit-lens Yes-minus-No per layer for the
gold window and the best wrong window (where the evidence enters / gets overwritten)."""
import json, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())
IDS = [int(x) for x in str(globals().get('DEBUG_IDS', '1472,1485,1513')).split(',')]
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
YN = [PROC.tokenizer.convert_tokens_to_ids(c) for c in 'AB']


def masks_for(lay, t0, t1, regime):
    win = window_mask(lay, t0, t1, False); win_na = window_mask(lay, t0, t1, True)
    na = lay.m.clone(); rr = torch.where(lay.rows)[0]; na[rr.unsqueeze(1), torch.where(lay.a)[0].unsqueeze(0)] = float('-inf')
    if regime == 'allaudio': return {li: win for li in ROUTE}
    if regime == 'split': return {li: win_na for li in ROUTE}
    if regime == 'split_lo': return {li: win_na for li in range(9, 28)}
    if regime == 'noaudio': return {li: (win_na if li in ROUTE else na) for li in range(NL)}


def yn_with_lens(media, q, fn):
    """rotated Yes-No (letters) and per-layer lens Yes-No for the original order."""
    tot = 0.0; lens = None
    for perm in rotations(2):
        inp, _ = qwen_inputs(PROC, prompt_of(q, ['Yes', 'No'], perm), media, 'av'); inp = move_inputs(inp, MODEL_); lay = Lay(inp)
        store = {}
        hooks = []
        if perm == (0, 1):
            for li in range(NL):
                def hk(mod, args, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out; store[li] = h[0, -1].detach().float()
                hooks.append(MODEL_.model.layers[li].register_forward_hook(hk))
        try:
            lg, _ = forward(inp, lay, fn(lay) if fn else None)
        finally:
            for h in hooks: h.remove()
        lp = torch.log_softmax(lg[YN].cpu(), -1).numpy(); tot += (lp[0] - lp[1]) if perm == (0, 1) else (lp[1] - lp[0])
        if store:
            nm = MODEL_.model.norm; W = MODEL_.lm_head.weight; lens = []
            for li in range(NL):
                x = nm(store[li].to(nm.weight.device, nm.weight.dtype)[None])[0].to(W.device)
                l2 = (W[YN].float() @ x.float()).cpu(); lens.append(round(float(l2[0] - l2[1]), 2))
        del inp
    torch.cuda.empty_cache(); return round(tot, 3), lens


for IDX in IDS:
    r = rows[IDX]; task = r['task_type_abbr']; gold = 'ABC'.index(r['answer'])
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / SR; w = speech_window(a)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts))))); P0 = profiles(media, NEUTRAL); rel = pmi(P['a'], P0['a'])
    s0, s1 = top_window(rel); T0, T1 = 2.0 * s0, min(dur, 2.0 * s1)
    wins = [(n, t0, t1) for n, t0, t1 in (('before', 0.0, T0), ('during', T0, T1), ('after', T1, dur)) if t1 - t0 >= 1.0]
    sf = speech_first(w, dur); present = (task == 'TA' and gold == 0) or (task == 'TF' and gold == 1)
    gold_win = 'during' if present else (('after' if sf else 'before') if task == 'TA' else {0: 'before', 2: 'after'}.get(gold))
    print(f'\n######## {task} {IDX} gold {texts[gold]!r} | speech {w[0]:.1f}-{w[1]:.1f}s | anchor window {T0:.0f}-{T1:.0f}s | event should be in: {gold_win}', flush=True)
    claim = hear(media); print('   model hears:', repr(claim), flush=True)
    for regime in ('allaudio', 'split', 'split_lo', 'noaudio', 'ref_text'):
        sc, lensd = {}, {}
        for n, t0, t1 in wins:
            if regime == 'ref_text':
                crop = prepare_video(cut(src, t0, t1, f'w{int(t0)}'))
                sc[n], lensd[n] = yn_with_lens(crop, f'Does this video show someone doing the following: "{claim}"?', None) if False else (yes_minus_no(crop, f'Does this video show someone doing the following: "{claim}"?'), None)
            else:
                sc[n], lensd[n] = yn_with_lens(media, LOOK_Q, lambda lay, t0=t0, t1=t1: masks_for(lay, t0, t1, regime))
        best = max(sc, key=sc.get)
        print(f'   {regime:9s} Yes-No per window {sc} -> picks {best} {"OK" if best == gold_win else "WRONG"}', flush=True)
        if regime in ('allaudio', 'split') and lensd.get(gold_win):
            wrong = max((n for n in sc if n != gold_win), key=lambda n: sc[n], default=None)
            g = lensd[gold_win]; o = lensd.get(wrong)
            print(f'      lens (Yes-No) by layer, gold window [{gold_win}] :', ' '.join(f'{x:+.1f}' for x in g[::3]), f'| L{NL - 1} {g[-1]:+.1f}')
            if o: print(f'      lens (Yes-No) by layer, other window [{wrong}]:', ' '.join(f'{x:+.1f}' for x in o[::3]), f'| L{NL - 1} {o[-1]:+.1f}', flush=True)
RESULT = 'done'

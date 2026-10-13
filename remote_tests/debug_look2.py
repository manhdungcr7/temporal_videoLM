"""Debug 2 of the look step: explicit INNER-SPEECH template (the model's own description of what it heard) + in-model routing.
Question: 'Does the video show someone doing the following: "<heard phrase>"?' per window. Regimes for the question rows:
  tmpl_mid  : audio blocked in every layer, video only inside the window in layers 13-27 (attention routing, full clip kept)
  tmpl_all  : audio blocked + video only inside the window in every layer
  tmpl_crop : the window cut out as its own clip, vision only (reference = J1 / ref_text)
Prints Yes-No per window and the margin, to compare routing with cropping."""
import json, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())
IDS = [int(x) for x in str(globals().get('DEBUG_IDS', '1472,1485,1513')).split(',')]
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))


def regime_mask(lay, t0, t1, regime):
    rr = torch.where(lay.rows)[0]; A = torch.where(lay.a)[0]
    na = lay.m.clone(); na[rr.unsqueeze(1), A.unsqueeze(0)] = float('-inf')
    win = window_mask(lay, t0, t1, True)
    if regime == 'tmpl_mid': return {li: (win if li in ROUTE else na) for li in range(NL)}
    return {li: win for li in range(NL)}


for IDX in IDS:
    r = rows[IDX]; task = r['task_type_abbr']; gold = 'ABC'.index(r['answer'])
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / SR; w = speech_window(a)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts))))); P0 = profiles(media, NEUTRAL); rel = pmi(P['a'], P0['a'])
    s0, s1 = top_window(rel); T0, T1 = 2.0 * s0, min(dur, 2.0 * s1)
    wins = [(n, t0, t1) for n, t0, t1 in (('before', 0.0, T0), ('during', T0, T1), ('after', T1, dur)) if t1 - t0 >= 1.0]
    sf = speech_first(w, dur); present = (task == 'TA' and gold == 0) or (task == 'TF' and gold == 1)
    gold_win = 'during' if present else (('after' if sf else 'before') if task == 'TA' else {0: 'before', 2: 'after'}.get(gold))
    claim = hear(media).strip().strip('."')
    q = f'Does the video show someone doing the following: "{claim}"?'
    print(f'\n######## {task} {IDX} event should be in: {gold_win} | heard: {claim!r}', flush=True)
    for regime in ('tmpl_mid', 'tmpl_all', 'tmpl_crop'):
        sc = {}
        for n, t0, t1 in wins:
            if regime == 'tmpl_crop': sc[n] = yes_minus_no(prepare_video(cut(src, t0, t1, f'w{int(t0)}')), q)
            else:
                s = score(media, q, ['Yes', 'No'], fwd=lambda inp, t0=t0, t1=t1: (lambda lay: forward(inp, lay, regime_mask(lay, t0, t1, regime))[0])(Lay(inp)))
                sc[n] = round(s[0] - s[1], 3)
        best = max(sc, key=sc.get); vals = sorted(sc.values())
        print(f'   {regime:9s} {sc} -> {best} {"OK" if best == gold_win else "WRONG"} | margin {vals[-1] - vals[-2]:.2f}', flush=True)
RESULT = 'done'

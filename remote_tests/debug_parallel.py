"""P3 debug — PARALLEL ISOLATED VERIFICATION in ONE forward pass (single items, local GPU).
All window x content mini-questions are appended after the media in one sequence:
    \n[<window>] Does the video show someone doing the following: "<content>"? Answer:
Mask: each mini-question reads the system/user prefix, ONLY the video tokens of its window (no audio), and its own tokens;
never the other mini-questions. Positions: each mini-question is placed right after its window on the TMRoPE clock.
Readout: logit(" Yes") - logit(" No") at the last token of each mini-question (lm_head applied only to those rows).
Reference: the same open Yes/No asked separately for each window with attention routing (one forward per window)."""
import json, numpy as np, torch
exec((ROOT / 'p1_two_pointer.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())
IDS = [int(x) for x in str(globals().get('DEBUG_IDS', '1513,1485,1713')).split(',')]
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
TOK = PROC.tokenizer
YES, NO = TOK.encode(' Yes', add_special_tokens=False)[0], TOK.encode(' No', add_special_tokens=False)[0]
TPL = '\n[{w}] Does the video show someone doing the following: "{c}"? Answer:'


def locate(input_ids, text, pieces, after):
    seq = input_ids[0].tolist()[after + 1:]
    strs = [TOK.decode([t]) for t in seq]; starts = np.cumsum([0] + [len(s) for s in strs[:-1]]); full = ''.join(strs)
    p0 = full.find(text)
    if p0 < 0: raise RuntimeError('mini-questions not found')
    bounds = np.cumsum([p0] + [len(p) for p in pieces]); spans = []
    for k in range(len(pieces)):
        toks = [t for t, c in enumerate(starts) if bounds[k] <= c < bounds[k + 1]]
        spans.append((after + 1 + toks[0], after + 1 + toks[-1] + 1))
    return spans


def parallel_verify(media, wins, contents):
    queries = [(wi, c) for wi in range(len(wins)) for c in contents]
    pieces = [TPL.format(w=wins[wi][0], c=c) for wi, c in queries]; text = ''.join(pieces)
    inp, _ = qwen_inputs(PROC, text + '\nThank you.', media, 'av'); inp = move_inputs(inp, MODEL_)
    lay = Lay(inp); spans = locate(inp['input_ids'].cpu(), text, pieces, lay.last_media)
    m = lay.m.clone(); pos = lay.pos.clone(); qcols = torch.zeros(lay.L, dtype=torch.bool)
    for s, e in spans: qcols[s:e] = True
    sec_chunk = lay.chunk
    for (s, e), (wi, c) in zip(spans, queries):
        _, t0, t1 = wins[wi]; c0, c1 = int(t0 // 2), max(int(t0 // 2) + 1, int(np.ceil(t1 / 2)))
        inwin = lay.v & (sec_chunk >= c0) & (sec_chunk < c1)
        m[s:e][:, (lay.v | lay.a) & ~inwin] = float('-inf')                       # only this window's video, no audio
        m[s:e][:, qcols] = float('-inf'); m[s:e, s:e] = torch.triu(torch.full((e - s, e - s), float('-inf')), 1)
        last_media_pos = int(pos[0, 0, torch.where(inwin)[0]].max())               # time-aligned: right after the window
        pos[:, :, s:e] = (last_media_pos + 1 + torch.arange(e - s)).view(1, 1, -1).to(pos.dtype)
    sel = torch.tensor([e - 1 for s, e in spans])
    cap = {}
    def pre(mod, args):
        h = args[0]; return (h[:, sel.to(h.device), :],)
    hk = MODEL_.lm_head.register_forward_pre_hook(pre)
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev), use_audio_in_video=True).logits[0].float()
    finally:
        hk.remove()
    out = {}
    for j, (wi, c) in enumerate(queries): out.setdefault(c, [0.0] * len(wins))[wi] = round(float(lg[j, YES] - lg[j, NO]), 2)
    del inp; torch.cuda.empty_cache(); return out


def separate_verify(media, wins, contents):
    out = {}
    for c in contents:
        out[c] = []
        for _, t0, t1 in wins:
            q = TPL.format(w='video', c=c).strip()
            inp, _ = qwen_inputs(PROC, q, media, 'av'); inp = move_inputs(inp, MODEL_); lay = Lay(inp)
            rr = torch.where(lay.rows)[0]; na = lay.m.clone(); na[rr.unsqueeze(1), torch.where(lay.a)[0].unsqueeze(0)] = float('-inf')
            win = window_mask(lay, t0, t1, True); masks = {li: (win if li in ROUTE else na) for li in range(NL)}
            lg = forward(inp, lay, masks)[0].cpu()
            out[c].append(round(float(lg[YES] - lg[NO]), 2)); del inp
    torch.cuda.empty_cache(); return out


for IDX in IDS:
    r = rows[IDX]; task = r['task_type_abbr']; gold = 'ABC'.index(r['answer'])
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / SR
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts))))); P0 = profiles(media, NEUTRAL)
    s0, s1 = grow_window(pmi(P['a'], P0['a']), P['a']); T0, T1 = 2.0 * s0, min(dur, 2.0 * s1)
    wins = [(n, t0, t1) for n, t0, t1 in (('before the speech', 0.0, T0), ('while the speech is heard', T0, T1), ('after the speech', T1, dur)) if t1 - t0 >= 1.0]
    claim = inner_speech(media)
    contents = [claim] + ([t for t in texts] if task == 'CMTB' else [])
    print(f'\n######## {task} {IDX} gold {texts[gold]!r} | windows {[(n[:12], round(t0), round(t1)) for n, t0, t1 in wins]} | heard {claim!r}', flush=True)
    par = parallel_verify(media, wins, contents); sepv = separate_verify(media, wins, contents)
    agree = []
    for c in contents:
        a_, b_ = np.array(par[c]), np.array(sepv[c]); agree.append(int(np.argmax(a_)) == int(np.argmax(b_)))
        print(f'   {c[:45]:45s} parallel {par[c]}  separate {sepv[c]}  same best window: {agree[-1]}', flush=True)
    allp = np.concatenate([par[c] for c in contents]); alls = np.concatenate([sepv[c] for c in contents])
    print(f'   correlation parallel vs separate: {np.corrcoef(allp, alls)[0, 1]:.2f} | best-window agreement {sum(agree)}/{len(agree)}', flush=True)
RESULT = 'done'

"""Does telling the model WHEN the speech is (as text) fix the temporal tasks? Isolates cause #5 (speech localization).
Conditions (same media, rotated options):
  base            original question
  anchor          "Note: in this video the speech is heard from S0 to S1 seconds; the video lasts D seconds." + question
  anchor_bda      (TF) anchor + before/during/after wording
  anchor_route    anchor + evidence routing in layers 13-27 (CMTB: time+source; TA: time only)
The S0/S1 come from the clip's own audio activity (no labels). Writes OUT/anchor_text.jsonl."""
import json, random, time
import numpy as np, torch

MID = list(range(13, 28)); NL = len(MODEL_.model.layers)


def route_logits(inp, window, block_audio, layers):
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]; spg = float(inp['video_second_per_grid'].reshape(-1)[0])
    base = gated_mask(L, is_v, sec, last_media, window, 'none', 0.0, emb.dtype, emb.device, spg, 'all')
    routed = gated_mask(L, is_v, sec, last_media, window, 'hard', 0.0, emb.dtype, emb.device, spg, 'all', is_a, block_audio)
    hooks = []
    def pre(mod, args, kwargs):
        am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = routed.to(am_.device if am_ is not None else routed.device); return args, kwargs
    for li in layers: hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad(): return MODEL_(**kw, attention_mask=base, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()


def sc(media, stem, texts, window=None, block_audio=False, route=False):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_)
        if route: lg = route_logits(inp, window, block_audio, MID)
        else:
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache(); return tot.round(3).tolist()


BDA_Q = 'Is the action that the speaker talks about shown in the video before, during, or after the speech?'
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = {}
for i, r in enumerate(rows):
    if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
items = []
for key in sorted(cells):
    pool = cells[key]; random.Random(31).shuffle(pool); items += pool[:int(globals().get('ANCHOR_N_PER', 10))]
res = OUT / 'anchor_text.jsonl'
done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
am = lambda v: int(np.argmax(v))
for n, (i, r) in enumerate(items):
    if i in done: continue
    t0 = time.time(); task = r['task_type_abbr']; rec = dict(idx=i, task=task, gold='ABC'.index(r['answer']))
    try:
        src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
        a = decode_audio(src); dur = len(a) / 16000
        if dur > MAX_DUR: raise RuntimeError('long')
        s0, s1 = speech_window(a); win = (s0, s1)
        media = prepare_video(src); stem, texts = parse_question(r['question'])
        note = f'Note: in this video the speech is heard from {s0:.1f} to {s1:.1f} seconds; the video lasts {dur:.1f} seconds. '
        rec.update(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)])
        rec['base'] = sc(media, stem, texts)
        rec['anchor'] = sc(media, note + stem, texts)
        if task == 'TF':
            rec['bda'] = sc(media, BDA_Q, ['Before', 'During', 'After'])
            rec['anchor_bda'] = sc(media, note + BDA_Q, ['Before', 'During', 'After'])
        else:
            rec['anchor_route'] = sc(media, note + stem, texts, win, task == 'CMTB', True)
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:200]
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    if 'error' in rec: print(n + 1, len(items), task, i, rec['error'][:50], flush=True); continue
    g = rec['gold']
    print(n + 1, len(items), task, i, {k: am(v) == g for k, v in rec.items() if isinstance(v, list) and k not in ('win',)}, rec['sec'], 's', flush=True)
RESULT = {'written': str(res)}

"""Method dev test on the remote GPU (namespace of kaggle_rootcause.py). Not the final evaluation: chooses variants.
Variants:
  route(window, block_audio, layers): in the given decoder layers only, question/answer rows (and, for completeness, every
     non-audio row) cannot read video outside the cue window; optionally cannot read audio. Other layers are native.
  CMTB : base | route_all (time+source, all layers) | route_mid (time+source, layers 13-27) | route_mid_time (time only)
  TA   : base | gate_time_all | gate_time_mid | htl_mid (hear -> look per window with audio blocked in 13-27 -> compare)
  TF   : base | bda (before/during/after wording) | htl_mid (relation from comparing look margins; no grammar)
  router: the model itself classifies each question (text only, rotated) into 4 evidence routes; accuracy on a sample of
          temporal AND semantic questions."""
import json, random, time
import numpy as np, torch

MID = list(range(13, 28))
NL = len(MODEL_.model.layers)


def route_logits(inp, window, block_audio, layers):
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]
    spg = float(inp['video_second_per_grid'].reshape(-1)[0])
    base = gated_mask(L, is_v, sec, last_media, window, 'none', 0.0, emb.dtype, emb.device, spg, 'all')
    routed = gated_mask(L, is_v, sec, last_media, window, 'hard', 0.0, emb.dtype, emb.device, spg, 'all', is_a, block_audio)
    hooks = []
    if len(layers) == NL: mask = routed
    else:
        mask = base
        def pre(mod, args, kwargs):
            am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = routed.to(am_.device if am_ is not None else routed.device); return args, kwargs
        for li in layers: hooks.append(MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad(): return MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()


def route_scores(media, stem, texts, window, block_audio, layers):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_)
        lp = torch.log_softmax(route_logits(inp, window, block_audio, layers)[LETTERS[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache(); return tot.round(3).tolist()


def windows_of(s0, s1, dur):
    w = {'speech': (s0, s1)}
    if s0 >= 1.0: w['before'] = (0.0, s0)
    if dur - s1 >= 1.0: w['after'] = (s1, dur)
    return w


def htl(media, s0, s1, dur, task):
    e = hear(media); q = f'Is the following action happening in this video: {e}?'
    m = {}
    for name, w in windows_of(s0, s1, dur).items():
        s = route_scores(media, q, ['Yes', 'No'], w, True, MID); m[name] = float(s[0] - s[1])
    others = {k: v for k, v in m.items() if k != 'speech'}
    present = m['speech'] >= (max(others.values()) if others else 0.0)
    if task == 'TA': pred = 0 if present else 1
    elif present: pred = 1
    else: pred = 0 if max(others, key=others.get) == 'before' else 2
    return dict(heard=e, margins=m, pred=pred)


BDA_Q = 'Is the action that the speaker talks about shown in the video before, during, or after the speech?'
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = {}
for i, r in enumerate(rows):
    if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
items = []
for key in sorted(cells):
    pool = cells[key]; random.Random(23).shuffle(pool); items += pool[:int(globals().get('DEV_N_PER', 8))]
res = OUT / 'method_dev.jsonl'
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
        rec.update(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)])
        rec['base'] = scores(media, stem, texts)
        if task == 'CMTB':
            rec['route_all'] = route_scores(media, stem, texts, win, True, list(range(NL)))
            rec['route_mid'] = route_scores(media, stem, texts, win, True, MID)
            rec['route_mid_time'] = route_scores(media, stem, texts, win, False, MID)
        elif task == 'TA':
            rec['gate_time_all'] = route_scores(media, stem, texts, win, False, list(range(NL)))
            rec['gate_time_mid'] = route_scores(media, stem, texts, win, False, MID)
            rec['htl_mid'] = htl(media, s0, s1, dur, 'TA')
        else:
            rec['bda'] = scores(media, BDA_Q, ['Before', 'During', 'After'])
            rec['htl_mid'] = htl(media, s0, s1, dur, 'TF')
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:300]
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    if 'error' in rec: print(n + 1, len(items), task, i, rec['error'][:50], flush=True); continue
    g = rec['gold']
    msg = {k: (am(v) == g if isinstance(v, list) else v['pred'] == g) for k, v in rec.items() if k in ('base', 'route_all', 'route_mid', 'route_mid_time', 'gate_time_all', 'gate_time_mid', 'bda', 'htl_mid')}
    print(n + 1, len(items), task, i, msg, rec['sec'], 's', flush=True)

# ---- router: can the model itself tell which evidence route a question needs? (text only) ----
ROUTES = ['what is visible at the moment the sound or speech is heard',
          'whether the action the speaker talks about happens at the same time as the speech',
          'when the action the speaker talks about happens relative to the speech (before, during or after)',
          'what is seen or heard in the video in general, not tied to the timing of the speech']
GOLD_ROUTE = {'CMTB': 0, 'TA': 1, 'TF': 2, 'GSA': 3, 'FGSA': 3, 'CMSB': 3, 'CSMB': 3}
rng = random.Random(5); sample = rng.sample([(i, r) for i, r in enumerate(rows)], 240)
router = []
for i, r in sample:
    s = text_only_scores(f'Here is a question about a video: "{r["question"]}"\nWhat does this question ask about?', ROUTES)
    router.append(dict(idx=i, task=r['task_type_abbr'], pred=am(s), gold=GOLD_ROUTE[r['task_type_abbr']]))
(OUT / 'router_dev.json').write_text(json.dumps(router))
acc = {t: round(float(np.mean([x['pred'] == x['gold'] for x in router if x['task'] == t])) * 100, 1) for t in sorted({x['task'] for x in router})}
print('router accuracy by task:', acc, flush=True)
RESULT = {'router_acc': acc}

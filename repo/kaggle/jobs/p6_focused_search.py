"""P6 v2 — FOCUSED TEMPORAL SEARCH by isolated parallel inspection (general: no speech-specific logic, no answer rules).
For any (video, question [, options]):
 1 REFERENCE MOMENT  the model says (text only) whether the moment the question is about is identified by something HEARD or
                     SEEN; its own attention (layer 12, PMI vs a neutral question) on that source locates the moment.
 2 WHAT HAPPENS THERE the model describes, in one short phrase, what happens at that moment (inner speech for sound,
                     a caption of the moment for sight). Search targets = that phrase + content answer options.
 3 FOCUSED SEARCH    uniform windows over the whole video; ONE forward with isolated mini-questions for every
                     (window x target x source): "Is the following shown here?" (video of the window only) and "Is the
                     following said or heard here?" (audio of the window only); placed right after their window on the clock.
 4 ANSWER            per target and source, the best-supported windows (relative: within 1 logit of the best) form a
                     timeline; the model answers the original question from it (one-step and step-by-step).
Writes OUT/p6_focused_search.jsonl with every intermediate result."""
import json, re, numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 1)); SEED = int(globals().get('SEED', 31337)); WIN_S = float(globals().get('WIN_S', 4.0))
REF_BAND = [12]
TOK = PROC.tokenizer
YES, NO = TOK.encode(' Yes', add_special_tokens=False)[0], TOK.encode(' No', add_special_tokens=False)[0]
Q_SEEN = '\n[part {k}, sound off] Is the following shown here: "{t}"? Answer:'
Q_HEARD = '\n[part {k}, eyes closed] Is the following said or heard here: "{t}"? Answer:'
REF_Q = 'In the following question about a video, the moment the question is about is identified by something that is: Question: "{q}"'
DESCRIBE_AUDIO = 'Listen to the audio. If a person speaks, give only the action they talk about (verb and object). Otherwise name the sound. Answer with one short phrase.'
DESCRIBE_CLIP = 'Describe the main action shown in this clip with one short phrase (verb and object).'
NON_CONTENT = {'yes', 'no', 'past', 'present', 'future', 'before', 'during', 'after', 'true', 'false'}
COT = '\nThink step by step about which parts of the timeline the question is about, then give the answer.'
STOP = re.compile(r'\n\s*(Human|User|Question)\s*:|If you have any other questions', re.I)
FIRST = re.compile(r'(?<=[.!?])\s')


def short_gen(inp, n=20):
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=n, do_sample=False)
    t = PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
    return FIRST.split(STOP.split(t)[0])[0].strip().strip('."')[:80]


def describe_audio(media):
    inp, _ = qwen_inputs(PROC, DESCRIBE_AUDIO, media, 'audio'); inp = move_inputs(inp, MODEL_)
    t = short_gen(inp); del inp; torch.cuda.empty_cache(); return t


def describe_clip(m2):
    inp, _ = qwen_inputs(PROC, DESCRIBE_CLIP, m2, 'vision'); inp = move_inputs(inp, MODEL_)
    t = short_gen(inp); del inp; torch.cuda.empty_cache(); return t


def reference_moment(media, stem, texts):
    sq = text_only_scores(REF_Q.format(q=stem), ['heard', 'seen']); src = 'audio' if sq[0] >= sq[1] else 'video'
    P = band_profiles(media, prompt_of(stem, texts, tuple(range(len(texts)))), REF_BAND); P0 = band_profiles(media, NEUTRAL, REF_BAND)
    rel, raw = (pmi(P['a'], P0['a']), P['a']) if src == 'audio' else (pmi(P['v'], P0['v']), P['v'])
    s0, s1 = grow_window(rel, raw)
    return src, s0, s1, [round(x, 2) for x in sq]


def isolated_search(media, wins, targets):
    queries = [(src, k, t) for src in ('v', 'a') for k in range(len(wins)) for t in targets]
    pieces = [(Q_SEEN if s_ == 'v' else Q_HEARD).format(k=k + 1, t=t) for s_, k, t in queries]; text = ''.join(pieces)
    inp, _ = qwen_inputs(PROC, text + '\nThank you.', media, 'av'); inp = move_inputs(inp, MODEL_)
    lay = Lay(inp); spans = locate(inp['input_ids'].cpu(), text, pieces, lay.last_media)
    m = lay.m.clone(); pos = lay.pos.clone(); qcols = torch.zeros(lay.L, dtype=torch.bool)
    for a_, b_ in spans: qcols[a_:b_] = True
    for (a_, b_), (src, k, t) in zip(spans, queries):
        t0, t1 = wins[k]; c0, c1 = int(t0 // 2), max(int(t0 // 2) + 1, int(np.ceil(t1 / 2)))
        inwin = (lay.v if src == 'v' else lay.a) & (lay.chunk >= c0) & (lay.chunk < c1)
        m[a_:b_][:, (lay.v | lay.a) & ~inwin] = float('-inf')
        m[a_:b_][:, qcols] = float('-inf'); m[a_:b_, a_:b_] = torch.triu(torch.full((b_ - a_, b_ - a_), float('-inf')), 1)
        anchor_pos = int(pos[0, 0, torch.where(inwin)[0]].max()) if inwin.any() else int(pos[0, 0, lay.last_media])
        pos[:, :, a_:b_] = (anchor_pos + 1 + torch.arange(b_ - a_)).view(1, 1, -1).to(pos.dtype)
    sel = torch.tensor([b_ - 1 for a_, b_ in spans])
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, sel.to(args[0].device), :],))
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev), use_audio_in_video=True).logits[0].float()
    finally:
        hk.remove()
    out = {}
    for j, (src, k, t) in enumerate(queries): out.setdefault(f'{src}|{t}', [0.0] * len(wins))[k] = round(float(lg[j, YES] - lg[j, NO]), 2)
    del inp; torch.cuda.empty_cache(); return out


def best_windows(scores):
    sc = np.array(scores); return set(np.where(sc >= sc.max() - 1.0)[0].tolist()) if sc.max() > -3.0 else set()


def timeline(wins, targets, ev, ref, ref_content, rsrc):
    lines = [f'What happens at the moment the question is about ({"heard" if rsrc == "audio" else "seen"}): "{ref_content}"',
             'Timeline of the video (SEEN = checked on the frames with the sound off; HEARD = checked on the audio with the eyes closed):']
    bw = {key: best_windows(v) for key, v in ev.items()}
    for k, (t0, t1) in enumerate(wins):
        seen = [t for t in targets if k in bw[f'v|{t}']]; heard = [t for t in targets if k in bw[f'a|{t}']]
        tag = ' (the moment the question is about)' if ref[k] else ''
        lines.append(f'- part {k + 1}, {t0:.0f}-{t1:.0f} s{tag}: SEEN: ' + (', '.join(f'"{x}"' for x in seen) or 'none of them') +
                     '; HEARD: ' + (', '.join(f'"{x}"' for x in heard) or 'none of them'))
    return '\n'.join(lines) + '\n'


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2)); rec['base'] = score(media, stem, texts)
    rsrc, s0, s1, sq = reference_moment(media, stem, texts)
    ref_content = describe_audio(media) if rsrc == 'audio' else describe_clip(prepare_video(cut(src, 2.0 * s0, min(dur, 2.0 * s1), f'ref{i}')))
    targets = [ref_content] + [t for t in texts if len(re.findall(r"[A-Za-z']+", t)) >= 2 and t.lower() not in NON_CONTENT and t != ref_content]
    edges = list(np.arange(0.0, dur, WIN_S)) + [dur]
    if len(edges) > 2 and edges[-1] - edges[-2] < 1.5: edges.pop(-2)
    wins = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
    ref = [(t1 > 2.0 * s0) and (t0 < 2.0 * s1) for t0, t1 in wins]
    ev = isolated_search(media, wins, targets)
    tl = timeline(wins, targets, ev, ref, ref_content, rsrc); q = tl + 'Answer using only the timeline. ' + stem
    rec.update(ref_source=rsrc, ref_selfquery=sq, ref_window=[2.0 * s0, 2.0 * s1], ref_content=ref_content, targets=targets,
               windows=wins, evidence=ev, timeline=tl)
    rec['plain'] = text_only_scores(q, texts)
    votes = np.zeros(k); outs = []
    for perm in rotations(k):
        conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': render(q, texts, perm) + COT}]}]
        inp = PROC(text=PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False), return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
        with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=160, do_sample=False)
        txt = STOP.split(PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0])[0]; outs.append(txt[:300])
        mm = list(re.finditer(r'answer\s+is\W{0,4}\(?([A-D])\b', txt, re.I)) or list(re.finditer(r'\(([A-D])\)', txt))
        if mm and 'ABCD'.index(mm[-1].group(1).upper()) < k: votes[perm['ABCD'.index(mm[-1].group(1).upper())]] += 1
    rec['cot'] = (votes + 1e-3 * np.exp(np.array(rec['plain']))).tolist(); rec['cot_texts'] = outs
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
R = run_loop(items, 'p6_focused_search.jsonl', item, show=('base', 'plain', 'cot'))
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in ('base', 'plain', 'cot')} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
print(RESULT, flush=True)

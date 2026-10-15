"""Materialized CQ: independent branch completions, then normal question reading.

Every chunk summary is produced without question/options. No task-specific
relation or polarity rules are used. Memory can be cached across questions.
"""
MANIFEST = json.loads((OUT / f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
DATA = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
ITEMS = [(d['idx'], DATA[d['idx']]) for d in MANIFEST['selected']]
MAX_PROBE_TOKENS = 12
PROBE_V = ' Video {t0}-{t1}s, visible action in a short phrase:'
PROBE_A = ' Audio {t0}-{t1}s, spoken words or sound in a short phrase:'


def branch_forward(inp, pieces, select_ends=False, endpoint_only=False):
    lay = Lay(inp)
    spans = locate(inp['input_ids'].cpu(), ''.join(pieces), pieces, lay.last_media)
    m, pos = lay.m.clone(), lay.pos.clone()
    first_media = int(torch.where(lay.v | lay.a)[0][0])
    safe = torch.arange(lay.L) < first_media
    pcols = torch.zeros(lay.L, dtype=torch.bool)
    ends = torch.tensor([e - 1 for s, e in spans])
    for s, e in spans:
        pcols[s:e] = True
    # All-layer separation prevents a visual memory node importing speech.
    for own in (lay.v, lay.a):
        rr, cc = torch.where(own)[0], torch.where(~(own | safe))[0]
        m[rr[:, None], cc[None, :]] = float('-inf')
    for g, (s, e) in enumerate(spans):
        own = (lay.v if g % 2 == 0 else lay.a) & (lay.chunk == g // 2)
        m[s:e, ~(own | safe)] = float('-inf')
        m[s:e, s:e] = torch.triu(torch.full((e-s, e-s), float('-inf')), 1)
        start = int(lay.pos[:, 0, own].max()) + 1 if own.any() else int(lay.pos[:, 0, lay.v | lay.a].max()) + 1
        pos[:, :, s:e] = (start + torch.arange(e-s)).view(1, 1, -1).to(pos.dtype)
    q_start = spans[-1][1]
    start = int(pos[:, 0, pcols | lay.v | lay.a].max()) + 1
    pos[:, :, q_start:] = (start + torch.arange(lay.L-q_start)).view(1, 1, -1).to(pos.dtype)
    if endpoint_only:
        qrows = torch.arange(q_start, lay.L)
        blocked = pcols.clone()
        blocked[ends] = False
        m[qrows[:, None], torch.where(blocked)[0][None, :]] = float('-inf')
    selection = ends if select_ends else torch.tensor([lay.L - 1])
    hook = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, selection.to(args[0].device), :],))
    kw = {k:v for k,v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad():
            lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=pos.to(lay.dev),
                        use_audio_in_video=True, use_cache=False).logits[0].float()
    finally:
        hook.remove()
    return lg if select_ends else lg[-1]


def complete_branches(media, nc, dur):
    base = [p.format(t0=2*t, t1=round(min(2*(t+1), dur), 2))
            for t in range(nc) for p in (PROBE_V, PROBE_A)]
    tokens = [[] for _ in base]
    stopped = [False for _ in base]
    specials = set(TOK.all_special_ids)
    for step in range(MAX_PROBE_TOKENS):
        pieces = [b + TOK.decode(ts, skip_special_tokens=True) for b,ts in zip(base,tokens)]
        inp, _ = qwen_inputs(PROC, ''.join(pieces) + '\nEnd of observations.', media, 'av')
        inp = move_inputs(inp, MODEL_)
        lg = branch_forward(inp, pieces, select_ends=True)
        next_ids = lg.argmax(-1).tolist()
        for g, tok in enumerate(next_ids):
            if stopped[g]:
                continue
            word = TOK.decode([tok])
            if tok in specials or '\n' in word or (step >= 3 and word.strip() in ('.',';')):
                stopped[g] = True
            else:
                tokens[g].append(tok)
        del inp, lg
        if all(stopped):
            break
    caps = [TOK.decode(ts, skip_special_tokens=True).strip() for ts in tokens]
    return base, caps


def memory_text(caps, dur):
    lines = ['Observations from independent video/audio chunks. VIDEO is directly visible; AUDIO is only heard.']
    for t in range(len(caps)//2):
        end = round(min(2*(t+1), dur), 2)
        lines.append(f'{2*t}-{end}s | VIDEO: {caps[2*t]} | AUDIO: {caps[2*t+1]}')
    return '\n'.join(lines) + '\n'


GROUNDING = ('Read the observations as measurements from different sources. '
             'A spoken description does not establish a visible event. '
             'Locate evidence for each referenced event using its source and timestamp, '
             'then answer the question from those observations. ')


def materialized_item(i, row):
    src, audio, dur, win = prep_clip(row, 50)
    media = prepare_video(src)
    stem, options = parse_question(row['question'])
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av')
    inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc
    del inp0
    base, caps = complete_branches(media, nc, dur)
    pieces = [b+' '+cap for b,cap in zip(base,caps)]
    memory = memory_text(caps, dur)
    rec = dict(dur=round(dur,3), captions=caps, memory=memory, options=options,
               independent_of_question=True)
    for endpoint, name in ((False, 'cq_completed'), (True, 'cq_completed_ends')):
        rec[name] = score(media, ''.join(pieces)+'\n'+stem, options,
                         fwd=lambda inp, ep=endpoint: branch_forward(inp, pieces, endpoint_only=ep))
    rec['cq_memory'] = text_only_scores(memory+stem, options)
    rec['cq_memory_grounded'] = text_only_scores(memory+GROUNDING+stem, options)
    print('CAPTIONS', i, json.dumps(caps), flush=True)
    return rec


CONDITIONS = ('cq_completed', 'cq_completed_ends', 'cq_memory', 'cq_memory_grounded')
R = run_loop(ITEMS, f'cq_diagnostic_round2_s{LAB_SERVER}.jsonl', materialized_item, show=CONDITIONS, min_left=0)
RESULT = {c: {'accuracy':acc(R,c)[0], 'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND2_RESULT', json.dumps(RESULT), flush=True)

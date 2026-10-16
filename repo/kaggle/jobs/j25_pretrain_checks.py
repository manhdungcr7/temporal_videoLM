"""J25 — two checks BEFORE any training (diagnostic only; decides where the light training must act).  Needs the J8 helpers (Lay).
Check 1  REASONING CEILING (text only, no media).  The exact SVHalluc TA/TF question templates, preceded by a textual timeline
         (speech interval, narrated-event interval) with random times; gold follows from the timeline.  Three phrasings (seconds with
         speech mentioned first, seconds with the event mentioned first, mm:ss).  Plus a visual-visual order question (two named actions).
         Plus 'none' (no timeline) as the collapse reference.  If the frozen model answers these well, the deficit is perception of time
         (training only needs the evidence-reading layers); if not, the late layers must be trained too.
Check 2  LINEAR TIME PROBE.  Natural SVHalluc clips with the neutral question; mean hidden state of the video tokens and of the audio
         tokens of every 2-s chunk at the decoder input (layer 0) and after layers 1, 6, 13, 20, 28, 36.  Saved for an offline ridge probe
         (grouped by clip): can the time of a chunk be read linearly from its video tokens, beyond the content-only baseline of layer 0?
J25_PART = 'reason' | 'probe' | 'both'.  Writes OUT/j25_reason.jsonl and OUT/j25_probe/<idx>.pt."""
import json, random, numpy as np, torch
if 'Lay' not in globals():
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())

J25_PART = globals().get('J25_PART', 'both')
J25_N = int(globals().get('J25_PER_CELL', 30)); J25_CLIPS = int(globals().get('J25_CLIPS', 48)); SEED25 = int(globals().get('J25_SEED', 2040))
SYS25 = globals().get('SYSTEM', 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.')
TA_Q = 'When the speech is heard, is the narrated event happening at the same time in the video?'
TF_Q = 'Relative to the moment when the speech is heard, when does the narrated event occur in the video?'
VV_Q = 'Which of these two events happens first in the video?'
TF_Q2 = 'When does the narrated event occur in the video, relative to the speech?'
ACTS = ['chopping onions', 'pouring water into a glass', 'opening a door', 'kicking a ball', 'writing on a whiteboard', 'stirring a pot',
        'tying a shoelace', 'washing a car', 'playing a guitar', 'folding a shirt', 'cutting paper with scissors', 'brushing a dog']


# ---------------------------------------------------------------- check 1
def text_score(prefix, stem, texts):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYS25}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': prefix + prompt_of(stem, texts, perm)}]}]
        inp = PROC(text=PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False), return_tensors='pt', padding=True)
        inp = move_inputs(inp, MODEL_)
        with torch.no_grad(): lg = MODEL_(**inp).logits[0, -1].float()
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
    return tot.round(3).tolist()


def mmss(x): return f'{int(x) // 60}:{int(x) % 60:02d}'


def timeline(rng, rel):
    """rel in {'past', 'present', 'future'}: position of the narrated event relative to the speech."""
    D = rng.uniform(20, 40); sl = rng.uniform(2, 4); el = rng.uniform(2, 5)
    while True:
        s0 = rng.uniform(0, D - sl); s1 = s0 + sl
        if rel == 'present':
            e0 = rng.uniform(max(0, s0 - el + 0.8), min(s1 - 0.8, D - el)) if max(0, s0 - el + 0.8) < min(s1 - 0.8, D - el) else None
        elif rel == 'past':
            e0 = rng.uniform(0, s0 - el - 1.0) if s0 - el - 1.0 > 0 else None
        else:
            e0 = rng.uniform(s1 + 1.0, D - el) if D - el > s1 + 1.0 else None
        if e0 is not None: return D, s0, s1, e0, e0 + el


def phrase(style, D, s0, s1, e0, e1):
    if style == 'sec_speech_first':
        return f'Timeline of the video: the speech is heard from {s0:.1f} s to {s1:.1f} s; the narrated event is shown from {e0:.1f} s to {e1:.1f} s.\n'
    if style == 'sec_event_first':
        return f'In this {D:.0f}-second video, the narrated event appears on screen between {e0:.1f} s and {e1:.1f} s, and the speech is heard between {s0:.1f} s and {s1:.1f} s.\n'
    return f'Speech: {mmss(s0)}-{mmss(s1)}. Narrated event on screen: {mmss(e0)}-{mmss(e1)}.\n'


STYLES = ('sec_speech_first', 'sec_event_first', 'mmss')


def run_reason():
    out = OUT / 'j25_reason.jsonl'; rng = random.Random(SEED25); recs = []
    for task, stem, texts in (('TA', TA_Q, ['Yes', 'No']), ('TF', TF_Q, ['Past', 'Present', 'Future'])):
        recs.append(dict(task=task, style='none', gold=None, score=text_score('', stem, texts)))
        rels = ('present', 'past', 'future')
        for n in range(J25_N * (2 if task == 'TA' else 3)):
            rel = rels[n % 3] if task == 'TF' else ('present' if n % 2 == 0 else ('past' if n % 4 == 1 else 'future'))   # TA: half Yes, half No (past/future)
            D, s0, s1, e0, e1 = timeline(rng, rel); style = STYLES[(n // 3) % 3] if task == 'TF' else STYLES[n % 3]   # TF: style not tied to gold
            gold = (0 if rel == 'present' else 1) if task == 'TA' else {'past': 0, 'present': 1, 'future': 2}[rel]
            pre = phrase(style, D, s0, s1, e0, e1)
            rec = dict(task=task, style=style, rel=rel, gold=gold, times=[round(x, 1) for x in (D, s0, s1, e0, e1)], score=text_score(pre, stem, texts))
            if task == 'TF':                                   # same timeline, wording without tense words: is the failure the words Past/Present/Future?
                rec['score_bda'] = text_score(pre, TF_Q2, ['Before the speech', 'During the speech', 'After the speech'])
            recs.append(rec)
    for n in range(J25_N * 2):
        a1, a2 = rng.sample(ACTS, 2); D = rng.uniform(20, 40); l1, l2 = rng.uniform(2, 5), rng.uniform(2, 5)
        t1 = rng.uniform(0, D / 2 - l1); t2 = rng.uniform(t1 + l1 + 1.0, D - l2)       # a1 strictly first
        first_named = n % 2 == 0                                                        # mention order independent of time order
        tl = (f'Timeline of the video: {a1} from {t1:.1f} s to {t1 + l1:.1f} s; {a2} from {t2:.1f} s to {t2 + l2:.1f} s.\n' if first_named else
              f'Timeline of the video: {a2} from {t2:.1f} s to {t2 + l2:.1f} s; {a1} from {t1:.1f} s to {t1 + l1:.1f} s.\n')
        texts = [a1, a2] if n % 4 < 2 else [a2, a1]; gold = texts.index(a1)
        recs.append(dict(task='VV', style='sec_first_mentioned' if first_named else 'sec_second_mentioned', gold=gold, score=text_score(tl, VV_Q, texts)))
    with out.open('w') as f:
        for r in recs: f.write(json.dumps(r) + '\n')
    print('check 1 written', len(recs), flush=True)


# ---------------------------------------------------------------- check 2
PROBE_LAYERS = (0, 1, 6, 13, 20, 28, 36)


def probe_clip(i, r):
    src, a, dur, w = prep_clip(r, MAXD); media = prepare_video(src)
    inp, _ = qwen_inputs(PROC, NEUTRAL, media, 'av'); inp = move_inputs(inp, MODEL_); lay = Lay(inp)
    Gv = torch.zeros(lay.L, lay.nc); Gv[torch.where(lay.v)[0], lay.chunk[lay.v]] = 1
    Ga = torch.zeros(lay.L, lay.nc); Ga[torch.where(lay.a)[0], lay.chunk[lay.a]] = 1
    nv, na = Gv.sum(0), Ga.sum(0); store, hooks = {}, []
    def grab(li, h):
        h = (h[0] if h.dim() == 3 else h).float()
        store[li] = ((Gv.to(h.device).T @ h) / nv.clamp(min=1).to(h.device)[:, None]).half().cpu(), \
                    ((Ga.to(h.device).T @ h) / na.clamp(min=1).to(h.device)[:, None]).half().cpu()
    layers = MODEL_.model.layers
    hooks.append(layers[0].register_forward_pre_hook(lambda m, args, kw: grab(0, args[0] if args else kw['hidden_states']), with_kwargs=True))
    for li in PROBE_LAYERS[1:]:
        hooks.append(layers[li - 1].register_forward_hook(lambda m, args, out, li=li: grab(li, out[0] if isinstance(out, tuple) else out)))
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],))
    try:
        with torch.no_grad(): MODEL_(**inp, use_audio_in_video=True)
    finally:
        for h in hooks: h.remove()
        hk.remove()
    d = OUT / 'j25_probe'; d.mkdir(exist_ok=True)
    torch.save(dict(idx=i, task=r['task_type_abbr'], dur=dur, nc=lay.nc, nv=nv, na=na, speech=list(map(float, w)),
                    v={li: store[li][0] for li in PROBE_LAYERS}, a={li: store[li][1] for li in PROBE_LAYERS}), d / f'{i}.pt')
    return dict(dur=round(dur, 2), nc=lay.nc)


def run_probe():
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    pool = [(i, r) for i, r in enumerate(rows) if r['data_type'] == 'temporal']; random.Random(SEED25).shuffle(pool)
    seen, items = set(), []
    for i, r in pool:                                   # one question per recording (the probe is about clips, not questions)
        if r['video_path'] in seen: continue
        seen.add(r['video_path']); items.append((i, r))
        if len(items) >= J25_CLIPS * 2: break
    SH = globals().get('J12_SHARD')
    if SH: items = items[SH[0]::SH[1]]
    run_loop(items[:J25_CLIPS], 'j25_probe_log.jsonl', probe_clip)


if J25_PART in ('reason', 'both') and not globals().get('J12_SHARD', (0, 1))[0]: run_reason()
if J25_PART in ('probe', 'both'): run_probe()
RESULT = {'done': J25_PART}
print(RESULT, flush=True)

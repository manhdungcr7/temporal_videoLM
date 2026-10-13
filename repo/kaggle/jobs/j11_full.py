"""J11 — the full automatic method (human-inspired), every step recorded for failure analysis. Uses J8 helpers.
 1 ORIENT   anchor = the question's attention to the AUDIO tokens per 2-s chunk, PMI vs a neutral question (alpha 2)
 2 REHEARSE inner speech: the model itself says, in one short phrase, what it hears (any sound; the action if a person speaks)
 3 LOOK     for each window (before / during / after the anchor) and each content (heard phrase + content options):
            Yes/No "is this shown?" with the question rows' audio blocked in every layer and video restricted to the window in
            layers 13-27 (attention routing; the full clip stays in context)
 4 REMEMBER source-separated episodic timeline: the narration declared once and marked as heard-only; per window only
            "speech: yes/no" and what is SEEN (each content in its best window; each window keeps its top-contrast option)
 5 REASON   (a) plain one-step answer from the timeline   (b) step-by-step reasoning, answer parsed from the reasoning, the
            three option orders vote
Reference conditions: base; acar = single-pass in-model anchor + audio blocked (J8 best).
Writes OUT/j11_full.jsonl."""
import json, re, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())

PER_LABEL = int(globals().get('PER_LABEL', 8)); SEED = int(globals().get('SEED', 909))
INNER_Q = 'Listen to the audio. If a person speaks, give only the action they talk about (verb and object). Otherwise name the sound you hear. Answer with one short phrase.'
LOOK_T = 'Does the video show someone doing the following: "{c}"?'
NON_CONTENT = {'yes', 'no', 'past', 'present', 'future', 'before', 'during', 'after', 'true', 'false'}
COT = '\nThink step by step about which part of the timeline the question asks about, then give the answer.'
STOP = re.compile(r'\n\s*(Human|User|Question)\s*:|If you have any other questions', re.I)


def chat_text(prompt):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    return PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def gen(prompt, n=160):
    inp = PROC(text=chat_text(prompt), return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=n, do_sample=False)
    return PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


def inner_speech(media):
    inp, _ = qwen_inputs(PROC, INNER_Q, media, 'audio'); inp = move_inputs(inp, MODEL_)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=20, do_sample=False)
    t = PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
    t = re.split(r'(?<=[.!?])\s', STOP.split(t)[0])[0]                              # first sentence only, no run-on
    del inp; torch.cuda.empty_cache(); return t.strip().strip('."')[:100]


def grow_window(rel, raw, frac=0.3):
    """seed = the PMI peak (WHICH event), extent = contiguous chunks whose raw attention >= frac * max (HOW LONG it lasts)."""
    seed = int(np.argmax(rel)); on = raw >= frac * raw.max(); on[seed] = True; lo = hi = seed
    while lo - 1 >= 0 and on[lo - 1]: lo -= 1
    while hi + 1 < len(on) and on[hi + 1]: hi += 1
    return lo, hi + 1


def look_mask(lay, t0, t1):
    rr = torch.where(lay.rows)[0]; A = torch.where(lay.a)[0]
    na = lay.m.clone(); na[rr.unsqueeze(1), A.unsqueeze(0)] = float('-inf')
    win = window_mask(lay, t0, t1, True)
    return {li: (win if li in ROUTE else na) for li in range(NL)}


def parse_answer(txt, shown):
    txt = STOP.split(txt)[0]
    m = list(re.finditer(r'answer\s+is\W{0,4}\(?([A-D])\b', txt, re.I)) or list(re.finditer(r'\(([A-D])\)', txt))
    if m:
        j = 'ABCD'.index(m[-1].group(1).upper())
        if j < len(shown): return j
    tail = txt[txt.lower().rfind('answer'):] if 'answer' in txt.lower() else txt
    hits = [j for j, t in enumerate(shown) if re.search(r'\b' + re.escape(t.lower()) + r'\b', tail.lower())]
    return hits[-1] if len(hits) == 1 else None


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    rec['base'] = score(media, stem, texts)
    # 1 orient
    P = profiles(media, prompt_of(stem, texts, tuple(range(k)))); P0 = profiles(media, NEUTRAL); rel = pmi(P['a'], P0['a'])
    s0, s1 = grow_window(rel, P['a']); T0, T1 = 2.0 * s0, min(dur, 2.0 * s1); rec['anchor'] = [T0, T1]; rec['rel_a'] = np.round(rel, 4).tolist()
    rec['acar'] = score(media, stem, texts, fwd=lambda inp: (lambda lay: forward(inp, lay, bias_mask(lay, rel, 4.0, ('v',), True))[0])(Lay(inp)))
    # 2 rehearse
    claim = inner_speech(media); rec['claim'] = claim
    contents = [claim] + [t for t in texts if len(re.findall(r"[A-Za-z']+", t)) >= 2 and t.lower().strip() not in NON_CONTENT and t != claim]
    # 3 look
    segs = [(n, t0, t1) for n, t0, t1 in (('before the speech', 0.0, T0), ('while the speech is heard', T0, T1), ('after the speech', T1, dur)) if t1 - t0 >= 1.0]
    ver = {}
    for c in contents:
        ver[c] = []
        for _, t0, t1 in segs:
            s = score(media, LOOK_T.format(c=c), ['Yes', 'No'], fwd=lambda inp, t0=t0, t1=t1: (lambda lay: forward(inp, lay, look_mask(lay, t0, t1))[0])(Lay(inp)))
            ver[c].append(round(s[0] - s[1], 3))
    rec['segs'] = [[n, round(t0, 2), round(t1, 2)] for n, t0, t1 in segs]; rec['verify'] = ver
    # 4 remember
    seen = {j: [] for j in range(len(segs))}
    if max(ver[claim]) > 0: seen[int(np.argmax(ver[claim]))].append(claim)          # only if actually seen somewhere
    best = {j: None for j in range(len(segs))}
    for t in contents[1:]:
        v = ver[t]; j = int(np.argmax(v)); con = v[j] - max([x for jj, x in enumerate(v) if jj != j] or [-1e9])
        if v[j] <= 0: continue                                                       # never seen -> not in the timeline
        if best[j] is None or con > best[j][0]: best[j] = (con, t)
    for j in best:
        if best[j]: seen[j].append(best[j][1])
    tl = (f'Narration (what the speaker talks about; this is only heard, it is NOT evidence of what is shown): "{claim}"\n'
          'Timeline of the video (SEEN = checked on the video with the sound off):\n' +
          '\n'.join(f'- {t0:.1f}-{t1:.1f} s ({n}): speech: {"yes" if n == "while the speech is heard" else "no"}; SEEN: ' +
                    (', '.join(f'"{x}"' for x in seen[j]) or 'none of the listed actions') for j, (n, t0, t1) in enumerate(segs)) + '\n')
    rec['timeline'] = tl; q = tl + 'Answer using only the timeline. ' + stem
    # 5 reason
    rec['plain'] = text_only_scores(q, texts)
    votes = np.zeros(k); outs = []
    for perm in rotations(k):
        txt = gen(render(q, texts, perm) + COT); p = parse_answer(txt, [texts[j] for j in perm]); outs.append(STOP.split(txt)[0][:300])
        if p is not None: votes[perm[p]] += 1
    rec['cot'] = (votes + 1e-3 * np.exp(np.array(rec['plain']))).tolist(); rec['cot_texts'] = outs
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
C = ('base', 'acar', 'plain', 'cot')
R = run_loop(items, 'j11_full.jsonl', item, show=C)
OK = [r for r in R if 'error' not in r]
RESULT = {t: {c: acc(OK, c, t)[0] for c in C} | {'n': sum(r['task'] == t for r in OK)} for t in ('TA', 'TF', 'CMTB')}
RESULT['ALL'] = {c: acc(OK, c)[0] for c in C}
print(RESULT, flush=True)

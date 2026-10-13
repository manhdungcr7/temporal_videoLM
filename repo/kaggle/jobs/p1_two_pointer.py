"""P1 — PROOF OF PRINCIPLE for "Anchor -> Target" two-pointer reading (before any training). Needs the J8 helpers.
For each TA/TF item (fresh seed, never used for design):
  1 ANCHOR pointer: question-row attention to AUDIO tokens per 2-s chunk, PMI vs a neutral question (alpha 2);
                    window = PMI peak grown along the raw attention (>= 0.3 max)                  [J8/J9/J11 design]
  2 REHEARSE: the model's own short phrase of what it hears (inner speech)
  3 TARGET pointer: attention of the query 'When is someone "<phrase>" shown in the video?' to VIDEO tokens per chunk,
                    PMI vs the neutral question  ->  t* = argmax  (same mechanism that found the red square)
  4 RELATION read off the two pointers (fixed rule, a probe of whether the information is there):
        TA: t* inside the anchor window -> Yes, else No        TF: t* before / inside / after the anchor window -> Past/Present/Future
Also: target pointer from the raw (non-PMI) profile, and whether the pointers fall in the true speech / true event half.
Writes OUT/p1_two_pointer.jsonl."""
import json, re, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())

PER_LABEL = int(globals().get('PER_LABEL', 25)); SEED = int(globals().get('SEED', 777))
INNER_Q = 'Listen to the audio. If a person speaks, give only the action they talk about (verb and object). Otherwise name the sound you hear. Answer with one short phrase.'
STOP = re.compile(r'\n\s*(Human|User|Question)\s*:|If you have any other questions', re.I)


def inner_speech(media):
    inp, _ = qwen_inputs(PROC, INNER_Q, media, 'audio'); inp = move_inputs(inp, MODEL_)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=20, do_sample=False)
    t = PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
    t = re.split(r'(?<=[.!?])\s', STOP.split(t)[0])[0]
    del inp; torch.cuda.empty_cache(); return t.strip().strip('."')[:100]


def grow_window(rel, raw, frac=0.3):
    seed = int(np.argmax(rel)); on = raw >= frac * raw.max(); on[seed] = True; lo = hi = seed
    while lo - 1 >= 0 and on[lo - 1]: lo -= 1
    while hi + 1 < len(on) and on[hi + 1]: hi += 1
    return lo, hi + 1


def relation(t, s0, s1):
    return 'before' if t < s0 else ('during' if t < s1 else 'after')


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question']); task = r['task_type_abbr']
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    P = profiles(media, prompt_of(stem, texts, tuple(range(len(texts))))); P0 = profiles(media, NEUTRAL)
    ra = pmi(P['a'], P0['a']); s0, s1 = grow_window(ra, P['a'])
    claim = inner_speech(media)
    Q = profiles(media, f'When is someone "{claim}" shown in the video?')
    rv = pmi(Q['v'], P0['v']); t_pmi = int(np.argmax(rv)); t_raw = int(np.argmax(Q['v']))
    nc = len(ra); act = np.array(chunk_activity(a, nc)) > 0.3
    rec.update(claim=claim, anchor=[s0, s1], t_pmi=t_pmi, t_raw=t_raw, n_chunks=nc, speech_chunks=act.astype(int).tolist(),
               rel_a=np.round(ra, 4).tolist(), rel_v_target=np.round(rv, 4).tolist(), raw_v_target=np.round(Q['v'], 4).tolist())
    rec['base'] = score(media, stem, texts)
    for name, t in (('pmi', t_pmi), ('raw', t_raw)):
        rel_ = relation(t, s0, s1)
        rec[f'rel_{name}'] = rel_
        rec[f'pred_{name}'] = (0 if rel_ == 'during' else 1) if task == 'TA' else {'before': 0, 'during': 1, 'after': 2}[rel_]
    return rec


items = sample_items(('TA', 'TF'), PER_LABEL, seed=SEED)
R = run_loop(items, 'p1_two_pointer.jsonl', item, show=('base',))
OK = [r for r in R if 'error' not in r]


def ev_half(r):
    sh = 0 if r['speech_first'] else 1
    pres = (r['task'] == 'TA' and r['gold'] == 0) or (r['task'] == 'TF' and r['gold'] == 1)
    return sh if pres else 1 - sh


RESULT = {}
for t in ('TA', 'TF'):
    S = [r for r in OK if r['task'] == t]
    if not S: continue
    d = {'n': len(S), 'base': acc(S, 'base')[0]}
    for name in ('pmi', 'raw'):
        d[f'two_pointer_{name}'] = round(100 * float(np.mean([r[f'pred_{name}'] == r['gold'] for r in S])), 1)
        d[f'target_in_event_half_{name}'] = round(100 * float(np.mean([(r[f't_{name}'] >= r['n_chunks'] / 2) == bool(ev_half(r)) for r in S])), 1)
    d['anchor_top_in_speech'] = round(100 * float(np.mean([r['speech_chunks'][int(np.argmax(r['rel_a']))] for r in S])), 1)
    RESULT[t] = d
print(RESULT, flush=True)

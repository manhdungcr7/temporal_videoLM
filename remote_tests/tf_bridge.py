"""Can the model bridge relation labels (Past/Present/Future) to the evidence timeline by itself?  Text only, oracle timeline
('list' format: per segment, what is SEEN), real SVHalluc questions (TA/TF/CMTB, so that a fix for TF is also checked for
harm on the others). Variants, all fully automatic and task-agnostic:
  list      timeline + question (reference)
  cot       the model first writes its reasoning (greedy, <= 96 tokens), then the option is scored after the reasoning
  ground    OPTION GROUNDING: for every option the model itself decides which part of the timeline it refers to
            (before / while / after the speech / not a time); options are then annotated with that reading and scored
  ground_cot both
Writes OUT/tf_bridge.json."""
import json, random
import numpy as np, torch

rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
ACTS = ['add the pork to the pan', 'mash the potatoes', 'slice the tomatoes', 'pour the sauce over the pasta', 'crack the eggs into the bowl',
        'stir the soup', 'spread butter on the bread', 'chop the onions', 'fry the bacon', 'mix the flour and water']
SEG = ['before the speech', 'while the speech is heard', 'after the speech']
GROUND = ['before the speech', 'while the speech is heard', 'after the speech', 'not a time']
rng = random.Random(5)
N_PER = int(globals().get('TB_N_PER', 12))


def oracle_timeline(task, gold, texts):
    dur = round(rng.uniform(20, 40), 1); s0 = round(rng.uniform(0.05, 0.45) * dur, 1); s1 = round(s0 + rng.uniform(4, 8), 1)
    times = [(0.0, s0), (s0, s1), (s1, dur)]
    act = rng.choice(ACTS); seen = {k: [] for k in range(3)}
    if task == 'TA': seen[1 if gold == 0 else rng.choice([0, 2])].append(act)
    elif task == 'TF': seen[gold].append(act)
    else:
        others = [j for j in range(len(texts)) if j != gold]
        seen[1].append(texts[gold]); seen[0].append(texts[others[0]]); seen[2].append(texts[others[1]])
    lines = []
    for k in range(3):
        heard = f'HEARD: "{act}"' if k == 1 else 'HEARD: (no speech)'
        lines.append(f'- {times[k][0]:.1f}-{times[k][1]:.1f} s ({SEG[k]}): {heard}; SEEN: ' + (', '.join(f'"{x}"' for x in seen[k]) or 'none of the listed actions'))
    return 'Evidence timeline of the video (SEEN = checked on the video with the sound off; HEARD = what is said):\n' + '\n'.join(lines) + '\n'


def chat_text(prompt):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    return PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def gen(prompt, n=96):
    inp = PROC(text=chat_text(prompt), return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=n, do_sample=False)
    return PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


COT = '\nThink step by step about which part of the timeline the question asks about, then give the answer.'


def cot_scores(prefix_prompt, texts):
    """for every option order: the model writes its own reasoning, then the answer letter is scored after it."""
    k = len(texts); tot = np.zeros(k); first = None
    for perm in rotations(k):
        prompt = render(prefix_prompt, texts, perm) + COT
        reasoning = gen(prompt); first = first or reasoning
        txt = chat_text(prompt) + reasoning + '\nTherefore, the answer is ('
        inp = PROC(text=txt, return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
        with torch.no_grad(): lg = MODEL_(**inp).logits[0, -1].float()
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
    return tot.round(3).tolist(), first[:400]


def ground_options(stem, texts):
    """the model's own reading of each option in the timeline vocabulary (no labels, no task names)."""
    notes = []
    for t in texts:
        sc = np.zeros(4)
        for perm in rotations(4):
            q = f'Question: "{stem}"\nOne answer option is "{t}". Which part of a video timeline does this option refer to? ' + ' '.join(f'({L}) {GROUND[j]}' for L, j in zip('ABCD', perm))
            inp = PROC(text=chat_text(q), return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
            with torch.no_grad(): lg = MODEL_(**inp).logits[0, -1].float()
            lp = torch.log_softmax(lg[L4].cpu(), -1).numpy()
            for p_, j in enumerate(perm): sc[j] += lp[p_]
        notes.append(int(np.argmax(sc)))
    return notes


def annotate(texts, notes):
    return [t if n == 3 else f'{t} (= {GROUND[n]})' for t, n in zip(texts, notes)]


cells = {}
for i, r in enumerate(rows):
    if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
out = []
for (task, ans), pool in sorted(cells.items()):
    pool = pool[:]; random.Random(3).shuffle(pool)
    for i, r in pool[:N_PER]:
        stem, texts = parse_question(r['question']); gold = 'ABC'.index(r['answer'])
        tl = oracle_timeline(task, gold, texts); q = tl + 'Answer using only the timeline. ' + stem
        rec = dict(idx=i, task=task, gold=gold)
        rec['list'] = text_only_scores(q, texts)
        rec['cot'], rec['cot_text'] = cot_scores(q, texts)
        notes = ground_options(stem, texts); rec['ground_notes'] = notes
        at = annotate(texts, notes)
        rec['ground'] = text_only_scores(q, at)
        rec['ground_cot'], rec['ground_cot_text'] = cot_scores(q, at)
        out.append(rec)
    print(task, ans, 'done', flush=True)
(OUT / 'tf_bridge.json').write_text(json.dumps(out), encoding='utf-8')
am = lambda v: int(np.argmax(v))
RESULT = {}
for task in ('TA', 'TF', 'CMTB'):
    S = [r for r in out if r['task'] == task]
    d = {f: round(100 * np.mean([am(r[f]) == r['gold'] for r in S]), 1) for f in ('list', 'cot', 'ground', 'ground_cot')}
    d['by_gold'] = {f: {g: round(100 * np.mean([am(r[f]) == r['gold'] for r in S if r['gold'] == g]), 1) for g in sorted({r['gold'] for r in S})} for f in ('cot', 'ground', 'ground_cot')}
    d['ground_notes_by_option_index'] = [np.bincount([r['ground_notes'][j] for r in S], minlength=4).tolist() for j in range(len(parse_question(rows[S[0]['idx']]['question'])[1]))]
    d['n'] = len(S); RESULT[task] = d
print(json.dumps(RESULT, indent=1, default=float), flush=True)

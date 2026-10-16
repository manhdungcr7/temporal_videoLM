"""J15 — WHERE does the "collapse to one answer" with chunk probes come from? Needs J8 helpers, locate() and j12 (definitions).
J12/J12b/J12d: with probes, TA answers "Yes" for every item and TF answers "Present" for almost every item. Conditions (all
without boost, shared positions, "one word" probes; rotated option orders):
  base            no probes
  cq              probes, question/answer see every probe token                       (reproduces the collapse)
  cq_lastonly     question/answer see ONLY the last token of each probe               (is it the repeated probe text?)
  cq_past         probe text in the past tense ("... was, in one word:")              (is it the present tense?)
  cq_blind        probes see NO media (text only)                                     (is it probe content or probe presence?)
  cq_hidden       question/answer see NO probe token; probes only sit in the sequence (is it reading the probes at all?)
  cq_lastonly_vc  as cq_lastonly, and the probes' value vectors are centred per modality (removes a shared "affirmative" push)
Report per (task, gold). Writes OUT/j15_bias_source.jsonl (J15_OUT)."""
import json, numpy as np, torch
if 'cq_forward' not in globals():                 # notebook run: load the J12 definitions (and through them the J8 helpers)
    exec((ROOT / 'j12_chunk_prompts.py').read_text(encoding='utf-8').split(chr(10) + 'TASKS = ')[0], globals())

J15_OUT = globals().get('J15_OUT', 'j15_bias_source.jsonl')
PV = ' The essential visual content of chunk {t} in one word:'
PA = ' The essential audio content of chunk {t} in one word is:'
PV_PAST = ' The essential visual content of chunk {t} was, in one word:'
PA_PAST = ' The essential audio content of chunk {t} was, in one word:'
CONDS15 = (('cq', PV, PA, None, 'none'), ('cq_lastonly', PV, PA, 'lastonly', 'none'), ('cq_past', PV_PAST, PA_PAST, None, 'none'),
           ('cq_blind', PV, PA, 'blind', 'none'), ('cq_hidden', PV, PA, 'hidden', 'none'),
           ('cq_lastonly_vc', PV, PA, 'lastonly', 'vcenter'))    # + probe values centred per modality (removes a shared push)


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2)); rec['base'] = score(media, stem, texts)
    inp0, _ = qwen_inputs(PROC, 'x', media, 'av'); inp0 = move_inputs(inp0, MODEL_)
    nc = Lay(inp0).nc; del inp0
    for name, pv, pa, variant, mode in CONDS15:
        pieces = [x for t in range(nc) for x in (pv.format(t=t + 1), pa.format(t=t + 1))]
        prompt = ''.join(pieces) + ' ' + stem
        rec[name] = score(media, prompt, texts, fwd=lambda inp, pieces=pieces, variant=variant, mode=mode: cq_forward(inp, pieces, mode, variant=variant)[0])
    rec['n_chunks'] = nc
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=SEED)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
C = ('base',) + tuple(c[0] for c in CONDS15)
R = run_loop(items, J15_OUT, item, show=C)
OK = [r for r in R if 'error' not in r]
am = lambda v: int(np.argmax(v))
RESULT = {}
for t in ('TA', 'TF', 'CMTB'):
    for g in sorted({r['gold'] for r in OK if r['task'] == t}):
        S = [r for r in OK if r['task'] == t and r['gold'] == g]
        RESULT[f'{t}_{"ABCD"[g]}'] = {c: f'{sum(am(r[c]) == g for r in S)}/{len(S)}' for c in C}
print(RESULT, flush=True)

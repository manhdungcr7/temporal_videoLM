"""Debug 2 (single item): automatic TARGET-MODALITY gating on top of QTAR.
Modality PMI: total question-row attention mass on video vs audio tokens, relative to the neutral question:
  m_v = mass_v(q) / mass_v(neutral), m_a = mass_a(q) / mass_a(neutral); target = argmax.
Variants (time relevance = PMI av, lambda 4, layers 13-27):
  qtar          both modalities re-weighted (as J8)
  qtar_gate     re-weighted, and the NON-target modality blocked for the question rows in layers 13-27
  qtar_gate_up  ... blocked only in layers 21-27
Prints option probabilities (original order) and the masses, so we can see whether the cause is removed."""
import json, numpy as np, torch
src_j8 = (ROOT / 'j8_acar.py').read_text(encoding='utf-8')
exec(src_j8[:src_j8.index('items = sample_items')], globals())
IDX = int(globals().get('DEBUG_IDX', 1713))
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8')); r = rows[IDX]
src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / SR; w = speech_window(a)
media = prepare_video(src); stem, texts = parse_question(r['question']); gold = 'ABC'.index(r['answer']); k = len(texts)
print('item', IDX, r['task_type_abbr'], 'gold', gold, texts, flush=True)


def masses(prompt):
    inp, _ = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    lay = Lay(inp); _, rec = forward(inp, lay, record=REL_LAYERS)
    a_ = torch.stack([rec[li][0] for li in REL_LAYERS]).mean(0).numpy(); v_ = torch.stack([rec[li][1] for li in REL_LAYERS]).mean(0).numpy()
    del inp; torch.cuda.empty_cache(); return a_, v_


aq, vq = masses(prompt_of(stem, texts, tuple(range(k)))); a0, v0 = masses(NEUTRAL)
m_v, m_a = vq.sum() / v0.sum(), aq.sum() / a0.sum(); target = globals().get('FORCE_TARGET') or ('v' if m_v >= m_a else 'a')
SELFQ = 'To answer the following question about a video, what must you rely on? Question: "' + stem + '"'
sq = text_only_scores(SELFQ, ['what is seen in the video frames', 'what is heard in the audio', 'both what is seen and what is heard'])
print('model self-query (seen / heard / both):', [round(x, 2) for x in sq], '->', ['v', 'a', 'both'][int(np.argmax(sq))])
print(f'total mass question: video {vq.sum():.4f} audio {aq.sum():.4f} | neutral: video {v0.sum():.4f} audio {a0.sum():.4f}')
print(f'modality PMI: video x{m_v:.2f}  audio x{m_a:.2f}  -> target modality = {target}', flush=True)
n = lambda x: np.maximum(x / x.sum(), 1e-6)
rel = pmi(n(n(aq) + n(vq)), n(n(a0) + n(v0)))
print('time relevance (PMI av)', ' '.join(f'{x:.2f}' for x in rel), flush=True)


def gate(lay, layers_block, lam=4.0):
    base = bias_mask(lay, rel, lam, ('v', 'a'))
    blk = base.clone(); rr = torch.where(lay.rows)[0]; other = lay.a if target == 'v' else lay.v
    blk[rr.unsqueeze(1), torch.where(other)[0].unsqueeze(0)] = float('-inf')
    return {li: (blk if li in layers_block else base) for li in ROUTE}


def probs(fn):
    tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_); lay = Lay(inp)
        lg, _ = forward(inp, lay, fn(lay) if fn else None)
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache(); e = np.exp(tot / k); return np.round(e / e.sum(), 3)


for name, fn in (('base', None), ('qtar', lambda lay: bias_mask(lay, rel, 4.0, ('v', 'a'))),
                 ('qtar_gate', lambda lay: gate(lay, set(ROUTE))), ('qtar_gate_up', lambda lay: gate(lay, set(range(21, 28)))),
                 ('qtar_gate_l8', lambda lay: gate(lay, set(ROUTE), 8.0)), ('qtar_gate_l16', lambda lay: gate(lay, set(ROUTE), 16.0)),
                 ('oracle', lambda lay: window_mask(lay, w[0], w[1], True))):
    pr = probs(fn); print(f'{name:13s} rotated probs {pr}  -> {"OK" if int(np.argmax(pr)) == gold else "wrong"}', flush=True)
RESULT = 'done'

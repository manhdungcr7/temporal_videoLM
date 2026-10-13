"""Debug module A on single items: in-pass self-routing (one forward) vs the two-pass version (J8/J10 style).
For each item: self-query target; relevance from the two-pass profile (layers 12-35) vs the in-pass profile (layers 12-20);
rotated option probabilities for base / two-pass / in-pass (several APPLY bands)."""
import json, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())
exec((ROOT / 'inpass.py').read_text(encoding='utf-8'), globals())
IDS = [int(x) for x in str(globals().get('DEBUG_IDS', '1713,1772,1472')).split(',')]
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
fmt = lambda x: ' '.join(f'{v:.2f}' for v in x)


def probs(media, stem, texts, fn):
    k = len(texts); tot = np.zeros(k); rels = []
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_)
        out = fn(inp); lg = out[0] if isinstance(out, tuple) else out
        if isinstance(out, tuple) and out[1] is not None: rels.append(out[1])
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache(); e = np.exp(tot / k); return np.round(e / e.sum(), 3), rels


for IDX in IDS:
    r = rows[IDX]; task = r['task_type_abbr']; gold = 'ABC'.index(r['answer'])
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / SR
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    target = self_target(stem)
    act = np.array(chunk_activity(a, int(dur // 2) + 1)) > 0.3
    print(f'\n######## {task} {IDX} gold={gold} target={target} speech chunks', ''.join('S' if s else '.' for s in act), flush=True)
    # two-pass reference
    P = profiles(media, prompt_of(stem, texts, tuple(range(k)))); P0 = profiles(media, NEUTRAL); rel2 = pmi(P['av'], P0['av'])
    print('   two-pass rel (12-35):', fmt(rel2))
    def two_pass(inp):
        lay = Lay(inp); m = bias_mask(lay, rel2, 4.0, ('v', 'a')); rr = torch.where(lay.rows)[0]
        if target in ('seen', 'heard'):
            cols = torch.where(lay.a if target == 'seen' else lay.v)[0]; m[rr.unsqueeze(1), cols.unsqueeze(0)] = float('-inf')
        return forward(inp, lay, m, layers=ROUTE)[0]
    prior = neutral_prior(media)                                    # cached per video in deployment
    res = {'base': probs(media, stem, texts, lambda inp: forward(inp, Lay(inp))[0])[0],
           'two_pass(13-27)': probs(media, stem, texts, two_pass)[0]}
    for name, ap in (('inpass apply 21-27', list(range(21, 28))), ('inpass apply 21-35', list(range(21, NL)))):
        pr, rels = probs(media, stem, texts, lambda inp, ap=ap: inpass_forward(inp, prior, target, MEAS, ap))
        res[name] = pr
        if rels: print(f'   in-pass rel (12-20), orig order:', fmt(rels[0]), '| corr with two-pass', round(float(np.corrcoef(rels[0], rel2)[0, 1]), 2))
    for n, pr in res.items(): print(f'   {n:20s} {pr} -> {"OK" if int(np.argmax(pr)) == gold else "wrong"}', flush=True)
RESULT = 'done'

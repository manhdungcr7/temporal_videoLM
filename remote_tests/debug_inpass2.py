"""Debug 2 of module A (in-pass self-routing): which (measure band, apply band) split works in ONE forward pass?
J9: layers 1-11 are question-independent (cannot measure there). debug_inpass: applying from layer 21 is too late.
Splits tried: meas [12] -> apply 13-27; meas 12-14 -> apply 15-27; meas 12-16 -> apply 17-27; reference: two-pass 13-27."""
import json, numpy as np, torch
exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split('items = sample_items')[0], globals())
exec((ROOT / 'inpass.py').read_text(encoding='utf-8'), globals())
IDS = [int(x) for x in str(globals().get('DEBUG_IDS', '1713,1772,1472')).split(',')]
SPLITS = [([12], list(range(13, 28))), (list(range(12, 15)), list(range(15, 28))), (list(range(12, 17)), list(range(17, 28)))]
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
    print(f'\n######## {task} {IDX} gold={gold} target={target}', flush=True)
    P = profiles(media, prompt_of(stem, texts, tuple(range(k)))); P0 = profiles(media, NEUTRAL); rel2 = pmi(P['av'], P0['av'])
    def two_pass(inp):
        lay = Lay(inp); m = bias_mask(lay, rel2, 4.0, ('v', 'a')); rr = torch.where(lay.rows)[0]
        if target in ('seen', 'heard'):
            cols = torch.where(lay.a if target == 'seen' else lay.v)[0]; m[rr.unsqueeze(1), cols.unsqueeze(0)] = float('-inf')
        return forward(inp, lay, m, layers=ROUTE)[0]
    res = {'base': probs(media, stem, texts, lambda inp: forward(inp, Lay(inp))[0])[0], 'two_pass 13-27': probs(media, stem, texts, two_pass)[0]}
    print('   two-pass rel:', fmt(rel2), flush=True)
    for meas, ap in SPLITS:
        prior = neutral_prior(media, meas)
        pr, rels = probs(media, stem, texts, lambda inp, meas=meas, ap=ap, prior=prior: inpass_forward(inp, prior, target, meas, ap))
        name = f'in-pass meas {meas[0]}-{meas[-1]} / apply {ap[0]}-{ap[-1]}'; res[name] = pr
        if rels: print(f'   {name}: rel', fmt(rels[0]), '| corr two-pass', round(float(np.corrcoef(rels[0], rel2)[0, 1]), 2), flush=True)
    for n, pr in res.items(): print(f'   {n:36s} {pr} -> {"OK" if int(np.argmax(pr)) == gold else "wrong"}', flush=True)
RESULT = 'done'

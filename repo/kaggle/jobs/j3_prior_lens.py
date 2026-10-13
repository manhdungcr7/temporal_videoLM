"""J3 — WHERE DO THE ANSWER PRIORS COME FROM, AND AT WHICH LAYER?  (TA + TF + CMTB)
Observations: on TA the content-free answer is always "No" and 85% of TA errors equal it; on TF Past recall is ~0 and the
answer collapses to Present/Future; vision-only ~= base. Unknown: is the prior a TEXT prior (question wording alone),
a MEDIA-PRESENCE prior, or does it override correct evidence late in the network?
Per item:
  base, prior (black video + silence), text_only (no media at all), vision_only      -> source of the prior
  logit lens of base and prior: per layer, the option scores read through final norm + lm_head (rotated, summed)
                                                                                         -> where the decision forms / flips
  (TA) polarity: the same question asked negatively ("... at a DIFFERENT time ...?").  If the model still answers "No",
                 the error is a Yes/No response bias, not a temporal judgement.
  (TF) neutral labels: options "earlier than the speech / while the speech is heard / later than the speech"
Offline: contextual calibration (base - prior, base - text_only; both fully automatic, no labels) and the layer at which
the lens answer of base and prior diverge."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 30))
NL = len(MODEL_.model.layers)
NEG_TA = 'When the speech is heard, is the narrated event happening at a DIFFERENT time in the video (earlier or later), rather than at the same time?'
TF_NEUTRAL = ['Earlier than the speech', 'While the speech is heard', 'Later than the speech']


@torch.no_grad()
def lens_letters(h, k):
    nm = MODEL_.model.norm; W = MODEL_.lm_head.weight
    x = nm(h.to(nm.weight.device, nm.weight.dtype)[None])[0]
    return (W[L4[:k]].float() @ x.to(W.device).float()).cpu()


def score_lens(media, stem, texts, mode='av'):
    """rotated scores + per-layer lens scores [NL, k]."""
    k = len(texts); lens = np.zeros((NL, k))
    def fwd(inp):
        L = inp['input_ids'].shape[1]; g = {'last': torch.arange(L) == L - 1}
        lg, st = forward_record(inp, list(range(1, NL + 1)), g, use_audio=(mode == 'av'))
        fwd.st = st; return lg
    def on(perm, lg):
        for li in range(1, NL + 1):
            lp = torch.log_softmax(lens_letters(fwd.st[li]['last'], k), -1).numpy()
            for p_, j in enumerate(perm): lens[li - 1, j] += lp[p_]
    s = score(media, stem, texts, mode=mode, fwd=fwd, on_logits=on)
    return s, lens.round(2).tolist()


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    stem, texts = parse_question(r['question']); task = r['task_type_abbr']
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    media = prepare_video(src)
    rec['base'], rec['lens_base'] = score_lens(media, stem, texts)
    mp = prepare_video(blank_video(src, dur, False))
    rec['prior'], rec['lens_prior'] = score_lens(mp, stem, texts)
    rec['text_only'] = text_only_scores(stem, texts)
    rec['vision_only'] = score(media, stem, texts, mode='vision')
    if task == 'TA':
        rec['polarity'] = score(media, NEG_TA, ['Yes', 'No'])            # gold flips: Yes(0) <-> No(1)
        rec['polarity_prior'] = score(mp, NEG_TA, ['Yes', 'No'])
    if task == 'TF':
        rec['neutral'] = score(media, stem, TF_NEUTRAL)
        rec['neutral_prior'] = score(mp, stem, TF_NEUTRAL)
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=303)
R = run_loop(items, 'j3_prior_lens.jsonl', item, show=('base', 'prior', 'text_only'))

OK = [r for r in R if 'error' not in r]
am = lambda v: int(np.argmax(v))
summary = {}
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in OK if r['task'] == t]
    if not S: continue
    d = {'n': len(S)}
    for c in ('base', 'prior', 'text_only', 'vision_only'): d[c] = acc(S, c)[0]
    d['calib_prior'] = round(100 * np.mean([am(np.array(r['base']) - np.array(r['prior'])) == r['gold'] for r in S]), 1)
    d['calib_text'] = round(100 * np.mean([am(np.array(r['base']) - np.array(r['text_only'])) == r['gold'] for r in S]), 1)
    d['pred_dist_base'] = np.bincount([am(r['base']) for r in S], minlength=3).tolist()
    d['pred_dist_prior'] = np.bincount([am(r['prior']) for r in S], minlength=3).tolist()
    d['pred_dist_text'] = np.bincount([am(r['text_only']) for r in S], minlength=3).tolist()
    lb = np.array([[am(x) == r['gold'] for x in r['lens_base']] for r in S]).mean(0)
    d['lens_acc_by_layer'] = {li + 1: round(100 * float(lb[li]), 1) for li in range(0, NL, 3)} | {NL: round(100 * float(lb[-1]), 1)}
    if t == 'TA':
        d['polarity(expected=flipped)'] = acc(S, 'polarity', expect=lambda r: 1 - r['gold'])[0]
        d['polarity_prior_pred'] = np.bincount([am(r['polarity_prior']) for r in S], minlength=2).tolist()
    if t == 'TF':
        d['neutral'] = acc(S, 'neutral')[0]
        d['neutral_calib'] = round(100 * np.mean([am(np.array(r['neutral']) - np.array(r['neutral_prior'])) == r['gold'] for r in S]), 1)
    summary[t] = d
RESULT = {'summary': summary, 'errors': sum('error' in r for r in R)}
print(RESULT, flush=True)

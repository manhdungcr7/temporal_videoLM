"""J2 — DOES THE MODEL USE TIME, OR ONLY POSITION / SILENCE?  (TA + TF + CMTB)
Observations behind it: CMTB base is 70% when the speech is in the first half but 50% when it is in the second half
(rootcause, n=44); attention is front-loaded; the non-speech half of every clip is digital silence (a possible shortcut).
Causal manipulations of the same clip (rotated options, nothing fed from labels):
  base        original
  swap_av     the two halves exchanged, video and audio together. Content and relations are preserved, only the absolute
              position changes. Expected: TA same, CMTB same, TF Past<->Future (Present same).
              A drop vs base = position bias (e.g. front-loading), not a temporal-reasoning failure.
  swap_v      only the video halves exchanged, audio kept: the event moves relative to the speech.
              Expected: TA flips; TF Present -> Future (speech first) / Past (speech second), Past/Future -> Present;
              CMTB: the answer should move to the action of the other half (we record whether the prediction changes).
  rope_rev    same tokens, TMRoPE time ids of all media mirrored: does the model's clock matter at all?
  noisy       pink noise over the whole track at 10 dB below the speech (no more digital silence). Expected: same as base.
  (CMTB) gate_audio on base and noisy, window from the clean audio: does the routing gain survive realistic audio?"""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 20))


def rope_rev_fwd(inp):
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    pos = pos.clone(); media_m = (is_v | is_a).to(pos.device); t = pos[0, 0]
    lo, hi = t[media_m].min(), t[media_m].max(); t2 = t.clone(); t2[media_m] = lo + (hi - t[media_m]); pos[0, 0] = t2
    L = inp['input_ids'].shape[1]
    mask = gated_mask(L, is_v, sec, last_media, None, 'none', 0.0, emb.dtype, emb.device, None, 'all')
    kw = {kk: v for kk, v in inp.items() if kk != 'attention_mask'}
    with torch.no_grad():
        return MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    stem, texts = parse_question(r['question']); task = r['task_type_abbr']
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    media = prepare_video(src)
    rec['base'] = score(media, stem, texts)
    rec['rope_rev'] = score(media, stem, texts, fwd=rope_rev_fwd)
    h1, h2 = halves(src, dur)
    rec['swap_av'] = score(prepare_video(concat_clips([h2, h1], 'swapav')), stem, texts)
    rec['swap_v'] = score(prepare_video(concat_clips([h2, h1], 'swapv', audio_from=src)), stem, texts)
    nz = with_audio(src, noisy_audio(a, w, 10.0, seed=i), 'noisy10'); mn = prepare_video(nz)
    rec['noisy'] = score(mn, stem, texts)
    if task == 'CMTB':
        rec['gate_audio'] = scores(media, stem, texts, mask_kind='gate_audio', window=w)
        rec['noisy_gate_audio'] = scores(mn, stem, texts, mask_kind='gate_audio', window=w)
        nw = speech_window(decode_audio(nz)); rec['noisy_detected_win'] = None if nw is None else [round(nw[0], 2), round(nw[1], 2)]
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), PER_LABEL, seed=202)
R = run_loop(items, 'j2_position_causal.jsonl', item, show=('base', 'swap_av', 'noisy', 'gate_audio'))


def exp_swap_av(r):
    if r['task'] == 'TF': return {0: 2, 1: 1, 2: 0}[r['gold']]
    return r['gold']


def exp_swap_v(r):
    if r['task'] == 'TA': return 1 - r['gold']
    if r['task'] == 'TF': return (2 if r['speech_first'] else 0) if r['gold'] == 1 else 1
    return None


OK = [r for r in R if 'error' not in r]
summary = {}
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in OK if r['task'] == t]
    if not S: continue
    d = {'n': len(S)}
    for c in ('base', 'rope_rev', 'noisy', 'gate_audio', 'noisy_gate_audio'): d[c] = acc(S, c)[0]
    d['swap_av(expected)'] = acc(S, 'swap_av', expect=exp_swap_av)[0]
    if t != 'CMTB': d['swap_v(expected)'] = acc(S, 'swap_v', expect=exp_swap_v)[0]
    for c in ('rope_rev', 'swap_v', 'noisy'):
        d[f'{c}_changed%'] = round(100 * np.mean([int(np.argmax(r[c])) != int(np.argmax(r['base'])) for r in S]), 1)
    for sf_ in (True, False):
        d[f'base_speech_{"first" if sf_ else "second"}'] = acc([r for r in S if r['speech_first'] == sf_], 'base')
    summary[t] = d
RESULT = {'summary': summary, 'errors': sum('error' in r for r in R)}
print(RESULT, flush=True)

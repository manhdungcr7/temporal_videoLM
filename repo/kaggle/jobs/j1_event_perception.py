"""J1 — PERCEPTION or TEMPORAL BINDING? (TA + TF)
Local finding that motivates it (rootcause hidden states, n=35-51): the speech half is linearly decodable at 94-100%, but the
TF label only at 71% (= the speech-position rule) and the TA label at chance. So the question-token states seem NOT to encode
where the narrated event is. Two explanations remain:
  (P) perception: the model cannot even see the narrated action in the right part of the video;
  (B) binding:    it sees the action, but never relates it to the time of the speech.
Per item (no labels fed to the model):
  base           native answer (rotated options) + hidden states at layers 9/15/21/27/33: last token, question rows,
                 video tokens of each half, audio tokens inside/outside the speech window  -> position-balanced probes offline
  hear           the model's own audio-only answer "what action does the speaker say"
  whisper        analysis-only transcript of the speech window (upper bound for hearing; never shown to the model)
  see_h1/see_h2  vision-only crop of each half: "Does this video show someone <action>?"  Yes-minus-No, for hear and whisper
Analysis: event half = Present ? speech half : other half (benchmark construction). Perception accuracy = the half with the
higher Yes score is the event half. If perception >> base, cause (B) is confirmed and the fix must be in temporal binding.
The automatic hear-then-look rule (event half from see_*, relation from halves) is also computed offline."""
import numpy as np, torch

PER_LABEL = int(globals().get('PER_LABEL', 50))
LAYERS = [9, 15, 21, 27, 33]
HID1 = OUT / 'hidden_j1'; HID1.mkdir(exist_ok=True)
SEE_Q = 'Does this video show someone doing the following: "{a}"?'


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)], speech_first=speech_first(w, dur))
    store0 = {}
    def rec_fwd(inp):
        if store0:
            with torch.no_grad(): return MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        groups, _ = token_groups(inp, dur, w)
        lg, st = forward_record(inp, LAYERS, groups); store0.update(st); return lg
    rec['base'] = score(media, stem, texts, fwd=rec_fwd)
    np.savez_compressed(HID1 / f'h_{i}.npz', layers=np.array(LAYERS),
                        **{g: np.stack([store0[l][g].numpy() for l in LAYERS]).astype('float16') for g in store0[LAYERS[0]]})
    rec['hear'] = hear(media)
    try: rec['whisper'] = whisper_text(a, *w)
    except Exception as e: rec['whisper'] = None; rec['whisper_error'] = repr(e)[:120]
    h1, h2 = halves(src, dur); m1, m2 = prepare_video(h1), prepare_video(h2)
    for src_name in ('hear', 'whisper'):
        act = rec.get(src_name)
        if not act: continue
        act = act.strip().strip('."')[:100]
        rec[f'see_{src_name}'] = [yes_minus_no(m1, SEE_Q.format(a=act)), yes_minus_no(m2, SEE_Q.format(a=act))]
    return rec


items = sample_items(('TA', 'TF'), PER_LABEL, seed=101)
R = run_loop(items, 'j1_event_perception.jsonl', item, show=('base',))


def event_half(r):            # 0 = first half, 1 = second half (from the benchmark construction; analysis only)
    sh = 0 if r['speech_first'] else 1
    present = (r['task'] == 'TA' and r['gold'] == 0) or (r['task'] == 'TF' and r['gold'] == 1)
    return sh if present else 1 - sh


def rule(r, src_name):        # automatic hear-then-look relation from the two halves
    s = r.get(f'see_{src_name}')
    if not s: return None
    eh = int(np.argmax(s)); sh = 0 if r['speech_first'] else 1
    if r['task'] == 'TA': return 0 if eh == sh else 1
    return 1 if eh == sh else (0 if eh < sh else 2)


OK = [r for r in R if 'error' not in r]
summary = {}
for t in ('TA', 'TF'):
    S = [r for r in OK if r['task'] == t]
    if not S: continue
    d = {'n': len(S), 'base': acc(S, 'base')[0]}
    for sn in ('hear', 'whisper'):
        P = [r for r in S if r.get(f'see_{sn}')]
        if P:
            d[f'perception_{sn}'] = round(100 * np.mean([int(np.argmax(r[f'see_{sn}'])) == event_half(r) for r in P]), 1)
            d[f'rule_{sn}'] = round(100 * np.mean([rule(r, sn) == r['gold'] for r in P]), 1)
    summary[t] = d
RESULT = {'summary': summary, 'errors': sum('error' in r for r in R)}
print(RESULT, flush=True)

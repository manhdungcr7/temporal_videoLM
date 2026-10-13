"""STMR pilot — Source- and Time-Monitored Reading, FULLY AUTOMATIC (no task names, no labels, no human choices).
For any (video, question):
 1 ANCHOR   fixed audio-activity detector -> speech span; segments = before / during / after the speech (segments < 1 s dropped).
            No speech found -> the method returns the native answer (no-harm by construction).
 2 HEAR     the model itself, audio only: "what action does the speaker say is being done?" -> claim phrase.
 3 CONTENTS what to verify = the heard claim + every answer option that describes content (fixed rule: >= 2 words and not
            a relation/polarity word). Nothing depends on the benchmark task.
 4 VERIFY   for each segment x content: Yes/No "is this shown?" read by the SAME model with attention routing in layers
            13-27: text rows see video only inside the segment and no audio (seeing with the sound off). Rotated Yes/No.
            Each content is marked "yes" in its best-scoring segment, "no" elsewhere.
 5 REASON   the model answers the ORIGINAL question from the evidence timeline (text only, rotated options).
Pilot: 8 items per (task, label) of SVHalluc temporal (TA 16, TF 24, CMTB 24). Records base, every intermediate step and the
pipeline answer. Pre-registered success: pipeline >= 90% on every task. Writes OUT/stmr_pilot.jsonl."""
import json, random, time, re
import numpy as np, torch

MID = list(range(13, 28))
TL_FMT = globals().get('TL_FMT', 'rel')
NON_CONTENT = {'yes', 'no', 'past', 'present', 'future', 'before', 'during', 'after', 'true', 'false'}
VERIFY_Q = 'Is someone doing the following in this video: "{c}"?'


def route_fwd(window):
    def f(inp):
        emb = MODEL_.get_input_embeddings().weight
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        L = inp['input_ids'].shape[1]; spg = float(inp['video_second_per_grid'].reshape(-1)[0])
        base = gated_mask(L, is_v, sec, last_media, window, 'none', 0.0, emb.dtype, emb.device, spg, 'all')
        routed = gated_mask(L, is_v, sec, last_media, window, 'hard', 0.0, emb.dtype, emb.device, spg, 'all', is_a, True)
        def pre(mod, args, kwargs):
            am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = routed.to(am_.device if am_ is not None else routed.device); return args, kwargs
        hooks = [MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for li in MID]
        kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
        try:
            with torch.no_grad(): return MODEL_(**kw, attention_mask=base, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
        finally:
            for h in hooks: h.remove()
    return f


def contents_of(claim, texts):
    cs = [claim.strip().strip('."').rstrip('.')] if claim else []
    for t in texts:
        w = re.findall(r"[a-zA-Z']+", t)
        if len(w) >= 2 and t.lower().strip() not in NON_CONTENT and t not in cs: cs.append(t)
    return cs


def build_timeline(segs, claim, contents, marks, fmt):
    lines = []
    for k, (name, t0, t1) in enumerate(segs):
        when = f'{t0:.1f}-{t1:.1f} s' + (f' ({name})' if fmt != 'abs' else '')
        parts = [f'HEARD: "{claim}"' if name == 'while the speech is heard' else 'HEARD: (no speech)']
        parts += [f'SEEN "{c}": {"yes" if marks[c] == k else "no"}' for c in contents]
        lines.append(f'- {when}: ' + '; '.join(parts))
    return 'Evidence timeline of the video (SEEN = checked on the video with the sound off; HEARD = what is said):\n' + '\n'.join(lines) + '\n'


def stmr(src, a, dur, stem, texts):
    w = speech_window(a)
    media = prepare_video(src)
    if w is None: return dict(anchor=None, answer=score(media, stem, texts))
    s0, s1 = w
    segs = [s for s in (('before the speech', 0.0, s0), ('while the speech is heard', s0, s1), ('after the speech', s1, dur)) if s[2] - s[1] >= 1.0]
    claim = hear(media)
    contents = contents_of(claim, texts)
    ver = {c: [yes_minus_no_routed(media, c, (t0, t1)) for _, t0, t1 in segs] for c in contents}
    marks = {c: int(np.argmax(v)) for c, v in ver.items()}
    tl = build_timeline(segs, claim, contents, marks, TL_FMT)
    ans = text_only_scores(tl + ('Answer using only the timeline. ' if TL_FMT == 'rel_q' else '') + stem, texts)
    return dict(anchor=[round(s0, 2), round(s1, 2)], segs=[[n, round(t0, 2), round(t1, 2)] for n, t0, t1 in segs], claim=claim,
                contents=contents, verify=ver, marks=marks, timeline=tl, answer=ans)


def yes_minus_no_routed(media, c, window):
    s = score(media, VERIFY_Q.format(c=c), ['Yes', 'No'], fwd=route_fwd(window)); return round(s[0] - s[1], 3)


def item(i, r):
    src, a, dur, w = prep_clip(r, MAX_DUR)
    stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2), speech_first=speech_first(w, dur))
    rec['base'] = score(prepare_video(src), stem, texts)
    rec.update(stmr(src, a, dur, stem, texts))
    return rec


items = sample_items(('TA', 'TF', 'CMTB'), int(globals().get('PILOT_N_PER', 8)), seed=int(globals().get('PILOT_SEED', 909)))
R = run_loop(items, f'stmr_pilot_{TL_FMT}' + globals().get('PILOT_TAG', '') + '.jsonl', item, show=('base', 'answer'))
OK = [r for r in R if 'error' not in r]
RESULT = {t: {'n': len([r for r in OK if r['task'] == t]), 'base': acc(OK, 'base', t)[0], 'stmr': acc(OK, 'answer', t)[0]} for t in ('TA', 'TF', 'CMTB')}
print(RESULT, flush=True)

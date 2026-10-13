"""Does the model use TIME at all? Three causal manipulations on TA/TF clips (server namespace of kaggle_rootcause.py).
The benchmark clips put the speech in one half and (for Past/Future/No) the narrated event in the other half.
  base            original clip
  speech_moved    the speech audio moved to the mirrored position (other half); video untouched
  video_reversed  frames played backwards (event moves to the other half); audio untouched
  rope_reversed   same clip, but the TMRoPE temporal ids of all audio/video tokens are mirrored (content and token order
                  unchanged; only the model's clock is reversed)
Expected answer after speech_moved / video_reversed (event and speech exchange their relative position):
  TA: Yes <-> No.   TF: Present -> Past if the speech now sits in the second half, Future if in the first half;
                        Past/Future -> Present.
rope_reversed has no fixed expected answer; we measure whether the answer moves at all.
Option orders rotated. Writes OUT/timing_diag.jsonl."""
import json, random, time, subprocess
import numpy as np, soundfile as sf, torch

rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = {}
for i, r in enumerate(rows):
    if r['task_type_abbr'] in ('TA', 'TF'): cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
items = []
for key in sorted(cells):
    pool = cells[key]; random.Random(41).shuffle(pool); items += pool[:int(globals().get('TD_N_PER', 10))]


def run_ff(args):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-y', *args], capture_output=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.decode(errors='replace')[-300:])


def moved_speech(src, a, s0, s1, dur):
    out = np.zeros_like(a); seg = a[int(s0 * 16000): int(s1 * 16000)]
    n0 = int((dur - s1) * 16000); out[n0:n0 + len(seg)] = seg[: len(out) - n0]
    w = CLIPS / f'{src.stem}_mv.wav'; sf.write(w, out, 16000); dst = CLIPS / f'{src.stem}_speech_moved.mp4'
    run_ff(['-i', str(src), '-i', str(w), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(dst)])
    return dst, (dur - s1, dur - s0)


def reversed_video(src):
    dst = CLIPS / f'{src.stem}_vrev.mp4'
    run_ff(['-i', str(src), '-map', '0:v:0', '-map', '0:a:0', '-vf', 'reverse', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'copy', str(dst)])
    return dst


def rope_rev_scores(media, stem, texts):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_)
        emb = MODEL_.get_input_embeddings().weight
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        pos = pos.clone(); media_m = (is_v | is_a).to(pos.device); t = pos[0, 0]
        lo, hi = t[media_m].min(), t[media_m].max()
        t2 = t.clone(); t2[media_m] = lo + (hi - t[media_m]); pos[0, 0] = t2
        L = inp['input_ids'].shape[1]
        mask = gated_mask(L, is_v, sec, last_media, None, 'none', 0.0, emb.dtype, emb.device, None, 'all')
        kw = {kk: v for kk, v in inp.items() if kk != 'attention_mask'}
        with torch.no_grad(): lg = MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache(); return tot.round(3).tolist()


def expected_after_swap(task, gold, speech_first_now):
    if task == 'TA': return 1 - gold
    if gold == 1: return 0 if not speech_first_now else 2     # Present -> Past if speech now second half, Future if first
    return 1                                                  # Past/Future -> Present


res = OUT / 'timing_diag.jsonl'
done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
am = lambda v: int(np.argmax(v))
for n, (i, r) in enumerate(items):
    if i in done: continue
    t0 = time.time(); task = r['task_type_abbr']; rec = dict(idx=i, task=task, gold='ABC'.index(r['answer']))
    try:
        src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
        a = decode_audio(src); dur = len(a) / 16000
        if dur > MAX_DUR: raise RuntimeError('long')
        s0, s1 = speech_window(a); speech_first = (s0 + s1) / 2 < dur / 2
        stem, texts = parse_question(r['question'])
        rec.update(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)], speech_first=speech_first)
        media = prepare_video(src)
        rec['base'] = scores(media, stem, texts)
        rec['rope_reversed'] = rope_rev_scores(media, stem, texts)
        mv, w2 = moved_speech(src, a, s0, s1, dur)
        rec['speech_moved'] = scores(prepare_video(mv), stem, texts)
        rec['video_reversed'] = scores(prepare_video(reversed_video(src)), stem, texts)
        rec['expected_after_swap'] = expected_after_swap(task, rec['gold'], not speech_first)   # speech_moved: speech now in the other half
        rec['expected_after_vrev'] = expected_after_swap(task, rec['gold'], speech_first)       # video_reversed: speech stays where it was
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:200]
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    if 'error' in rec: print(n + 1, len(items), task, i, rec['error'][:50], flush=True); continue
    print(n + 1, len(items), task, i, 'gold', rec['gold'], 'base', am(rec['base']), '| moved', am(rec['speech_moved']), 'exp', rec['expected_after_swap'],
          '| vrev', am(rec['video_reversed']), 'exp', rec['expected_after_vrev'], '| rope_rev', am(rec['rope_reversed']), rec['sec'], 's', flush=True)
RESULT = {'written': str(res)}

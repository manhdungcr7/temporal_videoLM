"""Do the root causes persist at 7B?  Swaps the server's model to Qwen2.5-Omni-7B (FP16, balanced over both T4s, clips <= 28 s) and
measures the same core quantities as the 3B root-cause run on a balanced TA/TF/CMTB sample:
  base, prior (black video + silence), vision_only, audio_only; (CMTB) gate_audio and crop (speech-window video only).
Option orders rotated and summed per content. Labels are only used for scoring. Writes OUT/scale7b.jsonl.
After this job the server holds the 7B model (restart the server to get 3B back)."""
import json, random, time, gc
import numpy as np, torch

if 'SCALE_LOADED' not in globals():
    MODEL_ = None; gc.collect(); torch.cuda.empty_cache()
    MODEL, MREV = 'Qwen/Qwen2.5-Omni-7B', None
    MODEL_, PROC = load_model(); LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    SCALE_LOADED = True
    print('loaded', MODEL, 'layers', len(MODEL_.model.layers), [round(torch.cuda.memory_allocated(d) / 2 ** 30, 1) for d in range(2)], flush=True)
MAXD = 28.0
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = {}
for i, r in enumerate(rows):
    if r['data_type'] == 'temporal': cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
pools = []
for key in sorted(cells):
    p = cells[key]; random.Random(1).shuffle(p); pools.append(p[:int(globals().get('SCALE_N_PER', 20))])
items = [x for k in range(max(map(len, pools))) for p in pools for x in p[k:k + 1]]
res = OUT / 'scale7b.jsonl'
done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
am = lambda v: int(np.argmax(v))
for n, (i, r) in enumerate(items):
    if i in done: continue
    t0 = time.time(); task = r['task_type_abbr']; rec = dict(idx=i, task=task, gold='ABC'.index(r['answer']))
    try:
        src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full'); a = decode_audio(src); dur = len(a) / 16000
        if dur > MAXD: raise RuntimeError('long')
        w = speech_window(a)
        if w is None: raise RuntimeError('no speech')
        media = prepare_video(src); stem, texts = parse_question(r['question'])
        rec.update(dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)])
        rec['base'] = scores(media, stem, texts)
        rec['prior'] = scores(prepare_video(blank_video(src, dur, False)), stem, texts)
        rec['vision_only'] = scores(media, stem, texts, mode='vision')
        rec['audio_only'] = scores(prepare_video(blank_video(src, dur, True)), stem, texts)
        if task == 'CMTB':
            rec['gate_audio'] = scores(media, stem, texts, mask_kind='gate_audio', window=w)
            rec['crop'] = scores(prepare_video(cut(src, w[0], w[1], 'speech')), stem, texts, mode='vision')
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:300]; torch.cuda.empty_cache()
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(n + 1, len(items), task, i, 'ERR ' + rec['error'][:60] if 'error' in rec else {c: am(rec[c]) == rec['gold'] for c in ('base', 'prior', 'vision_only', 'gate_audio', 'crop') if c in rec}, rec.get('sec'), flush=True)
R = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]; R = [r for r in R if 'error' not in r]
RESULT = {}
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in R if r['task'] == t]
    if S: RESULT[t] = {'n': len(S), **{c: round(100 * float(np.mean([am(r[c]) == r['gold'] for r in S if c in r])), 1) for c in ('base', 'prior', 'vision_only', 'audio_only', 'gate_audio', 'crop') if any(c in r for r in S)}}
print(RESULT, flush=True)

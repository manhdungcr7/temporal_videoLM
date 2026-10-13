import json, random, time
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = {}
for i, r in enumerate(rows):
    if r['task_type_abbr'] == 'TF': cells.setdefault(r['answer'], []).append((i, r))
items = []
for g in 'ABC':
    pool = cells[g]; random.Random(7).shuffle(pool); items += pool[:24]
res_path = OUT / 'tf_framing.jsonl'
done = {json.loads(l)['idx'] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
BDA_Q = 'Is the action that the speaker talks about shown in the video before, during, or after the speech?'
for n, (i, r) in enumerate(items):
    if i in done: continue
    t0 = time.time()
    try:
        src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10**4, 'full')
        a = decode_audio(src); dur = len(a) / 16000
        if dur > MAX_DUR: raise RuntimeError('long')
        w = speech_window(a)
        media = prepare_video(src)
        stem, texts = parse_question(r['question'])
        rec = dict(idx=i, gold='ABC'.index(r['answer']), dur=round(dur, 2), win=[round(w[0], 2), round(w[1], 2)])
        rec['ppf'] = scores(media, stem, texts)                                  # original Past/Present/Future
        rec['bda'] = scores(media, BDA_Q, ['Before', 'During', 'After'])           # same evidence, plain ordering words
        rec['ppf_vision'] = scores(media, stem, texts, mode='vision')
        rec['bda_vision'] = scores(media, BDA_Q, ['Before', 'During', 'After'], mode='vision')
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec = dict(idx=i, error=repr(e)[:200])
    with res_path.open('a') as f: f.write(json.dumps(rec) + '\n')
    am = lambda v: int(max(range(len(v)), key=lambda j: v[j]))
    if 'error' not in rec:
        print(n + 1, len(items), i, 'gold', rec['gold'], 'ppf', am(rec['ppf']), 'bda', am(rec['bda']), rec['sec'], 's', flush=True)
    else:
        print(n + 1, len(items), i, rec['error'][:40], flush=True)
RESULT = {'written': str(res_path)}

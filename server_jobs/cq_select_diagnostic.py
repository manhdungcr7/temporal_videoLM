"""Freeze a diagnostic set by duration, never by model success or difficulty."""
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
task, answer = ('TA', 'B') if LAB_SERVER == 1 else ('TF', 'C')
pool = [(i, r) for i, r in enumerate(rows)
        if r['data_type'] == 'temporal' and r['task_type_abbr'] == task and r['answer'] == answer]
random.Random(20261014).shuffle(pool)
chosen, rejected = [], []
for i, r in pool:
    if len(chosen) == 5:
        break
    try:
        path = fetch(r['video_path'].replace(BS, '/'))
        with av.open(str(path)) as container:
            dur = float(container.duration / av.time_base) if container.duration else None
        if dur is None:
            dur = len(decode_audio(path)) / 16000
        if dur > 50:
            rejected.append(dict(idx=i, reason='duration', dur=dur))
            continue
        chosen.append(dict(idx=i, task=task, gold=answer, duration=round(dur, 3),
                           video_path=r['video_path'], question=r['question']))
        print('SELECTED', json.dumps(chosen[-1]), flush=True)
    except Exception as e:
        rejected.append(dict(idx=i, reason=repr(e)))
assert len(chosen) == 5, 'Unable to select five readable clips'
manifest = dict(seed=20261014, max_duration=50, selected=chosen, rejected=rejected,
                selection='random permutation then duration/readability filter; no predictions inspected')
(OUT / f'cq_diagnostic_manifest_s{LAB_SERVER}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
RESULT = manifest

"""Small untouched controls detect replacing Yes/Present bias with No/Future."""
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = [('TA', 'A')] if LAB_SERVER == 1 else [('TF', 'A'), ('TF', 'B')]
chosen = []
for task, answer in cells:
    pool = [(i,r) for i,r in enumerate(rows) if r['data_type']=='temporal'
            and r['task_type_abbr']==task and r['answer']==answer]
    random.Random(20261015).shuffle(pool)
    count = 0
    for i,r in pool:
        if count == 2:
            break
        try:
            path = fetch(r['video_path'].replace(BS,'/'))
            with av.open(str(path)) as container:
                dur = float(container.duration/av.time_base) if container.duration else len(decode_audio(path))/16000
            if dur > 50:
                continue
            chosen.append(dict(idx=i,task=task,gold=answer,duration=round(dur,3),
                               video_path=r['video_path'],question=r['question']))
            count += 1
        except Exception:
            continue
    assert count == 2
manifest = dict(seed=20261015,selected=chosen,selection='random+duration; untouched during method selection')
(OUT/f'cq_control_manifest_s{LAB_SERVER}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
RESULT = manifest

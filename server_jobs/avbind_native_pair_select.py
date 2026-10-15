"""Independent AV pretraining split, no label/question-based selection."""
import av,hashlib,random
rows=json.loads(fetch('samples.json').read_text())
excluded=set()
for server in (1,2):
    manifest=json.loads((OUT/f'uploads/cq_large_manifest_s{server}.json').read_text())
    excluded.update(r['video_path'].replace(BS,'/') for r in manifest['selected'])
excluded.update(json.loads((OUT/'uploads/cq_large_exclusions.json').read_text())['video_paths'])
excluded.update(rows[i]['video_path'].replace(BS,'/') for i in (1434,2380,2133))
source=(OUT/'uploads/avbind_native_pairs.py').read_text()
pair_ns={};exec(compile(source,'native_pairs','exec'),pair_ns)
recording_key=pair_ns['recording_key']
excluded_origins={recording_key(path) for path in excluded}
path=OUT/f'avbind_native_pair_selection_s{LAB_SERVER}.json'
if path.exists():
    manifest=json.loads(path.read_text())
    assert set(manifest['excluded_recording_ids'])==excluded_origins
    selected=manifest['selected']
else:
    pool=sorted({r['video_path'].replace(BS,'/') for r in rows
        if recording_key(r['video_path']) not in excluded_origins and
        int(hashlib.sha256(recording_key(r['video_path']).encode()).hexdigest(),16)%2==LAB_SERVER-1})
    random.Random(2026101410+LAB_SERVER).shuffle(pool);selected=[];seen=set()
    for candidate in pool:
        origin=recording_key(candidate)
        if origin in seen:continue
        try:
            src=fetch(candidate)
            with av.open(str(src)) as container:
                duration=float(container.duration/av.time_base)
                if not container.streams.audio or not container.streams.video:continue
            if not 4<=duration<=20:continue
        except Exception:continue
        seen.add(origin)
        selected.append(dict(video_path=candidate,recording_id=origin,duration=duration,
            split='train' if len(selected)<8 else 'heldout'))
        manifest=dict(server=LAB_SERVER,selected=selected,excluded_recording_ids=sorted(excluded_origins),
            seed=2026101410+LAB_SERVER,selection='recording-separated; 4-20s duration only; no question/answer/category',
            desired_train=8,desired_heldout=2,status='selecting')
        path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('NATIVE_PAIR_SELECT',json.dumps(dict(server=LAB_SERVER,n=len(selected),target=10,recording_id=origin)),flush=True)
        if len(selected)==10:break
    if len(selected)<10:raise RuntimeError(f'Only {len(selected)} eligible independent recordings')
    manifest['status']='done';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
RESULT=dict(file=path.name,n=len(selected),status=manifest['status'],exclusion_level='original recording ID')

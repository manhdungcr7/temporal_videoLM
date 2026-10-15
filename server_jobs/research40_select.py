"""Freeze a random, duration-verified 40-example development set before inference."""
import json, random, hashlib, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import av
import numpy as np

R40_SEED=2026101419
R40_CELLS=[('TA','A'),('TA','B'),('TF','A'),('TF','B'),('TF','C'),('CMTB','A'),('CMTB','B'),('CMTB','C')]
R40_PATH=OUT/'research40_manifest.json'
if R40_PATH.exists():
    RESULT=json.loads(R40_PATH.read_text())
    assert RESULT['total']==40 and RESULT['max_duration']==45
    print('REUSING_FROZEN_MANIFEST',RESULT['manifest_sha256'],flush=True)
else:
    source=fetch('samples.json')
    rows=json.loads(source.read_text(encoding='utf-8'))
    candidates=[(i,r) for i,r in enumerate(rows) if r.get('data_type')=='temporal'
                and (r.get('task_type_abbr'),r.get('answer')) in R40_CELLS]
    by_path={r['video_path'].replace(chr(92),'/'):None for _,r in candidates}
    def inspect_duration(name):
        try:
            path=fetch(name)
            with av.open(str(path)) as container:
                if not container.streams.video or not container.streams.audio:
                    return name,dict(ok=False,error='missing AV stream')
                durations=[float(container.duration/av.time_base)] if container.duration else []
                for stream in container.streams:
                    if stream.duration is not None and stream.time_base is not None:
                        durations.append(float(stream.duration*stream.time_base))
                if not durations: return name,dict(ok=False,error='no duration metadata')
                duration=max(durations)
            return name,dict(ok=0.2<duration<=45.0,duration=duration)
        except Exception as exc:
            return name,dict(ok=False,error=repr(exc)[-200:])
    started=time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for n,(name,info) in enumerate(ex.map(inspect_duration,by_path)):
            by_path[name]=info
            if n%40==0: print('DURATION_PREFILTER',n+1,'/',len(by_path),flush=True)
    (OUT/'research40_duration_catalog.json').write_text(json.dumps(by_path),encoding='utf-8')
    eligible=[(i,r) for i,r in candidates if by_path[r['video_path'].replace(chr(92),'/')]['ok']]
    used=set();selected=[];shards={1:[],2:[]}
    for cell_n,(task,gold) in enumerate(R40_CELLS):
        pool=[(i,r) for i,r in eligible if (r['task_type_abbr'],r['answer'])==(task,gold)]
        random.Random(f'{R40_SEED}:{task}:{gold}').shuffle(pool)
        chosen=[]
        for i,r in pool:
            name=r['video_path'].replace(chr(92),'/')
            if name in used: continue
            used.add(name)
            item=dict(idx=i,task=task,gold=gold,video_path=name,question=r['question'],
                      duration=by_path[name]['duration'])
            # Verify actual decoded audio before freezing; never silently skip at evaluation.
            audio=decode_audio(fetch(name))
            if len(audio)/16000>45.0:
                used.remove(name); continue
            item['decoded_audio_duration']=len(audio)/16000
            chosen.append(item)
            if len(chosen)==5:break
        if len(chosen)!=5:raise RuntimeError(f'Insufficient eligible unique videos for {task}/{gold}')
        selected.extend(chosen)
        for j,item in enumerate(chosen):shards[1+(j+cell_n)%2].append(item)
    for s in (1,2):random.Random(R40_SEED+s).shuffle(shards[s])
    assert len(selected)==40 and len(used)==40 and all(x['duration']<=45 for x in selected)
    digest=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
    RESULT=dict(seed=R40_SEED,total=40,max_duration=45,
                counts={f'{t}/{g}':5 for t,g in R40_CELLS},selected=selected,
                server1=shards[1],server2=shards[2],manifest_sha256=digest,
                samples_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                prefiltered_candidates=len(candidates),
                eligible_candidates=len(eligible),seconds=round(time.time()-started,2),
                selection='Random within label cell after duration prefilter; distinct videos; no predictions used')
    R40_PATH.write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
    print('FROZEN40',json.dumps({k:RESULT[k] for k in ('total','manifest_sha256','eligible_candidates','counts','seconds')}),flush=True)

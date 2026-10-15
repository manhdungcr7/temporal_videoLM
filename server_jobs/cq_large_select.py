"""Select held-out-by-video duration-filtered cells; never inspect predictions."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib

RUN_ID='cq_large_20261014'
SEED=2026101401
ROWS=json.loads(fetch('samples.json').read_text(encoding='utf-8'))
EXCLUSIONS=json.loads((OUT/'uploads/cq_large_exclusions.json').read_text())
EXCLUDED_PATHS={x.replace(BS,'/') for x in EXCLUSIONS['video_paths']}
CELLS=([('TA','A'),('TA','B'),('TF','A'),('TF','B')] if LAB_SERVER==1 else
       [('CMTB','A'),('CMTB','B'),('CMTB','C'),('TF','C')])
selection_path=OUT/f'{RUN_ID}_selection_s{LAB_SERVER}.json'
duration_cache={}
selected,rejected=[],[]

def inspect_candidate(pair):
    i,row=pair
    name=row['video_path'].replace(BS,'/')
    try:
        path=fetch(name)
        if name not in duration_cache:
            with av.open(str(path)) as container:
                durations=[]
                if container.duration:durations.append(float(container.duration/av.time_base))
                for stream in container.streams:
                    if stream.duration is not None and stream.time_base is not None:
                        durations.append(float(stream.duration*stream.time_base))
                if not durations:raise ValueError('missing duration metadata')
                if not container.streams.video or not container.streams.audio:
                    raise ValueError('missing video or audio stream')
                duration_cache[name]=max(durations)
        dur=duration_cache[name]
        if not 0.2<dur<=50:
            return None,dict(idx=i,reason='duration',duration=round(dur,5))
        return dict(idx=i,task=row['task_type_abbr'],gold=row['answer'],
                    duration=round(dur,5),video_path=name,question=row['question']),None
    except Exception as exc:
        return None,dict(idx=i,reason='readability',error=str(exc)[-500:])

def selection_snapshot(status):
    payload=dict(run_id=RUN_ID,server=LAB_SERVER,seed=SEED,max_duration=50,
                 selected=selected,rejected=rejected,status=status,
                 excluded_video_paths=sorted(EXCLUDED_PATHS),
                 samples_sha256=hashlib.sha256(fetch('samples.json').read_bytes()).hexdigest(),
                 selection='fixed random order; metadata duration/readability only; no predictions')
    temp=selection_path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    temp.replace(selection_path)
    return payload

selection_snapshot('selecting')
with ThreadPoolExecutor(max_workers=4) as executor:
    for task,gold in CELLS:
        pool=[(i,r) for i,r in enumerate(ROWS) if r['data_type']=='temporal'
              and r['task_type_abbr']==task and r['answer']==gold
              and r['video_path'].replace(BS,'/') not in EXCLUDED_PATHS]
        random.Random(f'{SEED}:{task}:{gold}').shuffle(pool)
        count=0
        for offset in range(0,len(pool),8):
            for chosen,reason in executor.map(inspect_candidate,pool[offset:offset+8]):
                if reason is not None:rejected.append(reason)
                elif count<50:
                    selected.append(chosen);count+=1
            selection_snapshot('selecting')
            print('SELECT_PROGRESS',task,gold,count,'/50',flush=True)
            if count==50:break
        if count!=50:
            selection_snapshot('insufficient_pool')
            raise RuntimeError(f'Only {count}/50 readable <=50s held-out-by-video samples for {task}/{gold}')
RESULT=selection_snapshot('done')
print('SELECTION_DONE',json.dumps(dict(server=LAB_SERVER,n=len(selected),
                                    groups={f'{t}/{g}':n for (t,g),n in Counter((x['task'],x['gold']) for x in selected).items()})),flush=True)

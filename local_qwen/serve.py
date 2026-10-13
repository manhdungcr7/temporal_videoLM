"""Persistent worker: loads model once; polls job.json {prompt, ids, max_new} -> writes job_out.json."""
import json,time,sys,os
from pathlib import Path
import runner
from runner import *
D=Path(__file__).parent
rows=json.loads((ROOT/'data/svhalluc_benchmark/samples.json').read_text(encoding='utf-8'))
model,proc=load();CACHE_MEDIA={}
print('ready',flush=True)
while True:
    j=D/'job.json'
    if not j.exists():time.sleep(1);continue
    time.sleep(0.5);job=json.loads(j.read_text(encoding='utf-8'));j.unlink()
    out=[]
    for i in job['ids']:
        r=rows[i];t=time.time()
        try:
            if i not in CACHE_MEDIA:
                if len(CACHE_MEDIA)>6:CACHE_MEDIA.clear()
                CACHE_MEDIA[i]=prepare_video(ROOT/'data/svhalluc_benchmark'/r['video_path'])
            media=CACHE_MEDIA[i]
            p=job['prompt'].replace('{question}',r['question']) if job['prompt'] else r['question']
            inp,_=qwen_inputs(proc,p,media);inp=move_inputs(inp,model)
            with torch.no_grad():o=model.generate(**inp,max_new_tokens=job.get('max_new',8),do_sample=False,use_audio_in_video=True)
            txt=proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0]
            rec=dict(idx=i,gold=r['answer'],text=txt,sec=round(time.time()-t))
        except Exception as e:
            rec=dict(idx=i,gold=r['answer'],text='ERR '+repr(e)[:150],sec=0)
        torch.cuda.empty_cache();out.append(rec)
        (D/'job_progress.json').write_text(json.dumps(out),encoding='utf-8')
    (D/'job_out.json').write_text(json.dumps(out),encoding='utf-8')

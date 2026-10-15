"""CMTB controls; no task code is passed to the inference API."""
import traceback,gc
rows=json.loads(fetch('samples.json').read_text());records=[]
path=OUT/f'avbind_native_cmtb_r3_s{LAB_SERVER}.jsonl'
# Previous prediction-independent selection excluded all 400 reserve videos.
for i in (1434,2380,2133):
    row=rows[i];rec=dict(idx=i,task=row['task_type_abbr'],gold=row['answer'])
    try:
        src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full')
        wave=decode_audio(src)
        if len(wave)/16000>50:raise ValueError('Development clip exceeds 50 seconds')
        question,options=parse_question(row['question'])
        media=prepare_video(src);rec['base']=score(media,question,options,mode='av');del media
        for enabled,name in ((False,'memory'),(True,'binding')):
            result=NEURAL_NS['avbind_native_answer'](src,question,options,
                OUT/f'avbind_cmtb_observations_{i}.json',OUT/f'avbind_cmtb_utterances_{i}.json',enabled=enabled)
            rec[name]=result['scores'];rec[name+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-2400:]
    with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(rec)+'\n');handle.flush()
    records.append(rec)
    print('AVBIND_CMTB',json.dumps(dict(idx=i,gold=rec['gold'],
        predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','memory','binding') if v in rec},error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),file=path.name)

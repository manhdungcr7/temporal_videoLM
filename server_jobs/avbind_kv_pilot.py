"""R7 all development cells, fixed gain 1 on both matched checkpoints."""
import traceback,gc
indices=[1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER==1 else [2139,2327,1603,2097,1532,1580,2040,1666,2124]
rows=json.loads(fetch('samples.json').read_text());records=[]
path=OUT/f'avbind_independent_kv_r7_s{LAB_SERVER}.jsonl'
for i in indices:
    row=rows[i];rec=dict(idx=i,task=row['task_type_abbr'],gold=row['answer'])
    try:
        src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
        for gain,name in ((0.0,'base'),(1.0,'binding')):
            result=NEURAL_NS['avbind_kv_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',OUT/f'avbind_raw_values_{i}.pt',gain=gain)
            rec[name]=result['scores'];rec[name+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-1800:]
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
    records.append(rec)
    print('BIND_REAL_PILOT',json.dumps(dict(idx=i,task=rec['task'],gold=rec['gold'],
        predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','binding') if v in rec},error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),file=path.name)

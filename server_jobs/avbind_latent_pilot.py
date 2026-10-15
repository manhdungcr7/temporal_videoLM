"""Full native input + raw isolated slots; task/gold belong only to evaluator."""
import traceback,gc
exec(compile((OUT/'uploads/avbind_sdpa_compat.py').read_text(),'avbind_sdpa_compat.py','exec'),globals())
indices=[1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER==1 else [2139,2327,1603,2097,1532,1580,2040,1666,2124]
rows=json.loads(fetch('samples.json').read_text());records=[]
path=OUT/f'avbind_latent_r5_s{LAB_SERVER}.jsonl'
for i in indices:
    row=rows[i];rec=dict(idx=i,task=row['task_type_abbr'],gold=row['answer'])
    try:
        src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
        for enabled,name in ((False,'base'),(True,'binding')):
            result=NEURAL_NS['avbind_latent_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',enabled=enabled)
            rec[name]=result['scores'];rec[name+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-1600:]
    with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(rec)+'\n');handle.flush()
    records.append(rec)
    print('BIND_REAL_PILOT',json.dumps(dict(idx=i,task=rec['task'],gold=rec['gold'],
        predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','binding') if v in rec},error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),file=path.name)

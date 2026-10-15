"""Real-media transfer; native base, same-memory no adapter, learned binding."""
import traceback,gc
compat=OUT/'uploads/avbind_sdpa_compat.py'
if compat.exists():exec(compile(compat.read_text(),'avbind_sdpa_compat.py','exec'),globals())
if not NEURAL_NS['BIND_METADATA'].get('trained'):
    raise RuntimeError('Refusing to evaluate an unfinished adapter as a trained candidate')
indices=[1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER==1 else [2139,2327,1603,2097,1532,1580,2040,1666,2124]
rows=json.loads(fetch('samples.json').read_text());records=[]
path=OUT/f'avbind_neural_pilot_s{LAB_SERVER}.jsonl'
if '_retry_' in path.name:
    original=path.with_name(path.name.replace('_retry_','_pilot_'))
    prior=[json.loads(line) for line in original.read_text().splitlines() if line.strip()] if original.exists() else []
    # Reuse complete same-method evaluations; retry infrastructure errors only.
    valid=[row for row in prior if 'memory' in row and 'binding' in row and not row.get('error')]
    completed={row['idx'] for row in valid}
    indices=[i for i in indices if i not in completed]
    if not path.exists():path.write_text(''.join(json.dumps(row)+'\n' for row in valid),encoding='utf-8')
for i in indices:
    row=rows[i];rec=dict(idx=i,task=row['task_type_abbr'],gold=row['answer'])
    try:
        src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full')
        question,options=parse_question(row['question'])
        media=prepare_video(src);rec['base']=score(media,question,options,mode='av');del media
        for enabled,name in ((False,'memory'),(True,'binding')):
            result=NEURAL_NS['avbind_neural_answer'](src,question,options,OUT/f'cq_physical_memory_{i}.json',
                OUT/f'avbind_utterances_{i}.json',enabled=enabled)
            rec[name]=result['scores'];rec[name+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-2400:]
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
    records.append(rec)
    print('BIND_REAL_PILOT',json.dumps(dict(idx=i,task=rec['task'],gold=rec['gold'],
        predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','memory','binding') if v in rec},error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),file=path.name)

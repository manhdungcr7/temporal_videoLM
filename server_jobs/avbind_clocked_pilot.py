import traceback,gc
indices=([1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER==1
         else [2139,2327,1603,2097,1532,1580,2040,1666,2124])
rows=json.loads(fetch('samples.json').read_text())
baseline_path=OUT/f'avbind_independent_kv_r7_s{LAB_SERVER}.jsonl'
if not baseline_path.exists():baseline_path=OUT/f'uploads/r7-independent-kv_s{LAB_SERVER}.jsonl'
previous={r['idx']:r for r in (json.loads(line) for line in baseline_path.read_text().splitlines() if line.strip())}
path=OUT/f'avbind_clocked_r12_s{LAB_SERVER}.jsonl';records={}
path.touch(exist_ok=True)
for line in path.read_text().splitlines():
    if line.strip():r=json.loads(line);records[r['idx']]=r
for index in indices:
    if index in records and 'binding' in records[index]:continue
    row=rows[index];r=dict(idx=index,task=row['task_type_abbr'],gold=row['answer'],base=previous[index]['base'])
    try:
        src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
        result=TIMELINE_NS['avbind_clocked_answer'](src,question,options)
        r.update(binding=result['scores'],binding_trace=result)
    except Exception:r['error']=traceback.format_exc()[-1800:]
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n');f.flush()
    records[index]=r
    print('BIND_REAL_PILOT',json.dumps(dict(idx=index,task=r['task'],gold=r['gold'],
        predictions={m:'ABCD'[int(np.argmax(r[m]))] for m in ('base','binding') if m in r},error=r.get('error'),
        completed=sum('binding' in x for x in records.values()),target=len(indices))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(file=path.name,n=len(records),completed=sum('binding' in x for x in records.values()))

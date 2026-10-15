import traceback,gc
indices=([1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER==1
         else [2139,2327,1603,2097,1532,1580,2040,1666,2124])
rows=json.loads(fetch('samples.json').read_text())
baseline_path=OUT/f'avbind_independent_kv_r7_s{LAB_SERVER}.jsonl'
if not baseline_path.exists():baseline_path=OUT/f'uploads/r7-independent-kv_s{LAB_SERVER}.jsonl'
previous={r['idx']:r for r in (json.loads(line) for line in baseline_path.read_text().splitlines() if line.strip())}
path=OUT/f'avbind_context_clock_r13_s{LAB_SERVER}.jsonl';path.touch(exist_ok=True);records={}
for line in path.read_text().splitlines():
    if line.strip():r=json.loads(line);records[r['idx']]=r
for index in indices:
    r=records.get(index,dict(idx=index,task=rows[index]['task_type_abbr'],gold=rows[index]['answer'],base=previous[index]['base'],errors_by_condition={}))
    if 'memory' in r and 'binding' in r:continue
    src=cut(fetch(rows[index]['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(rows[index]['question'])
    for physical_clock,name in ((False,'memory'),(True,'binding')):
        if name in r:continue
        try:
            result=TIMELINE_NS['avbind_context_clock_answer'](src,question,options,physical_clock)
            r.update({name:result['scores'],name+'_trace':result});r['errors_by_condition'].pop(name,None)
        except Exception:r['errors_by_condition'][name]=traceback.format_exc()[-1800:]
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n');f.flush()
        records[index]=r
        print('BIND_REAL_PILOT',json.dumps(dict(idx=index,task=r['task'],gold=r['gold'],condition=name,
            predictions={m:'ABCD'[int(np.argmax(r[m]))] for m in ('base','memory','binding') if m in r},errors=r['errors_by_condition'],
            completed={m:sum(m in x for x in records.values()) for m in ('memory','binding')},target=len(indices))),flush=True)
        gc.collect();torch.cuda.empty_cache()
RESULT=dict(file=path.name,n=len(records),completed={m:sum(m in r for r in records.values()) for m in ('memory','binding')})

import traceback,gc
indices=([1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER==1 else [2139,2327,1603,2097,1532,1580,2040,1666,2124])
rows=json.loads(fetch('samples.json').read_text())
baseline_path=OUT/f'avbind_independent_kv_r7_s{LAB_SERVER}.jsonl'
if not baseline_path.exists():baseline_path=OUT/f'uploads/r7-independent-kv_s{LAB_SERVER}.jsonl'
previous={r['idx']:r for r in (json.loads(line) for line in baseline_path.read_text().splitlines() if line.strip())}
path=OUT/f'avbind_witness_r15_s{LAB_SERVER}.jsonl';path.touch(exist_ok=True);records={}
for line in path.read_text().splitlines():
    if line.strip():
        r=json.loads(line);merged=records.setdefault(r['idx'],{});merged.update(r)
for index in indices:
    row=rows[index]
    r=records.get(index,dict(idx=index,task=row['task_type_abbr'],gold=row['answer'],base=previous[index]['base']))
    for condition,guided in (('memory',False),('binding',True)):
        if isinstance(r.get(condition),list):continue
        try:
            src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
            result=TIMELINE_NS['avbind_witness_answer'](src,question,options,guided=guided,
                slot_cache=OUT/f'avbind_guided_raw_{index}.pt',witness_cache=OUT/f'avbind_witness_r15_{index}_{condition}.json')
            r.update({condition:result['scores'],condition+'_trace':result})
            r.setdefault('errors_by_condition',{}).pop(condition,None)
        except Exception:r.setdefault('errors_by_condition',{})[condition]=traceback.format_exc()[-1800:]
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n');f.flush()
        records[index]=r
        print('BIND_REAL_PILOT',json.dumps(dict(idx=index,task=r['task'],gold=r['gold'],condition=condition,
            predictions={m:'ABCD'[int(np.argmax(r[m]))] for m in ('base','memory','binding') if isinstance(r.get(m),list)},
            errors=r.get('errors_by_condition'),completed={m:sum(isinstance(x.get(m),list) for x in records.values()) for m in ('memory','binding')},target=len(indices))),flush=True)
        gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),completed={m:sum(isinstance(x.get(m),list) for x in records.values()) for m in ('memory','binding')},file=path.name)

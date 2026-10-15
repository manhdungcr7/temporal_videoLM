"""Initial symmetric-method ablation; labels only in the evaluation wrapper."""
import gc,traceback
indices=[1454,2396,2348,2141] if LAB_SERVER==1 else [2139,2327,1580,2040,1666,2124]
ROWS=json.loads(fetch('samples.json').read_text(encoding='utf-8'))
records=[]
path=OUT/f'avbind_pilot_r1_s{LAB_SERVER}.jsonl'
done={json.loads(line)['idx'] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
for i in indices:
    if i in done:continue
    row=ROWS[i];rec=dict(idx=i,task=row['task_type_abbr'],gold=row['answer'])
    try:
        src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full')
        question,options=parse_question(row['question'])
        physical=OUT/f'cq_physical_memory_{i}.json'
        utterance=OUT/f'avbind_utterances_{i}.json'
        # Independent caches may be reused, but no graph/answers from old CQ.
        if not utterance.exists():
            for name in (f'cq_diagnostic_round7_s{LAB_SERVER}.jsonl',f'cq_provenance_controls_s{LAB_SERVER}.jsonl'):
                previous=OUT/name
                if previous.exists():
                    saved=next((json.loads(line) for line in previous.read_text().splitlines()
                                if line.strip() and json.loads(line).get('idx')==i),None)
                    if saved and saved.get('utterances'):
                        utterance.write_text(json.dumps(saved['utterances']),encoding='utf-8');break
        media=prepare_video(src);rec['base']=score(media,question,options,mode='av');del media
        for variant in ('ledger','linked'):
            result=AVB_NS['avbind_answer'](src,question,options,physical,utterance,max_steps=3,variant=variant)
            rec[variant]=result['scores'];rec[variant+'_trace']=result
        rec['duration']=result['duration']
    except Exception:rec['error']=traceback.format_exc()[-2400:]
    with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(rec)+'\n');handle.flush()
    records.append(rec)
    print('AVBIND_PILOT',json.dumps(dict(server=LAB_SERVER,idx=i,task=rec['task'],gold=rec['gold'],
                                      prediction={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','ledger','linked') if v in rec},
                                      error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(server=LAB_SERVER,attempted=len(records),file=path.name)

"""Same general API on unseen query family, no task passed to inference."""
import traceback
manifest=json.loads((OUT/'avbind_cmtb_manifest.json').read_text())
rows=json.loads(fetch('samples.json').read_text());records=[]
path=OUT/f'avbind_cmtb_pilot_s{LAB_SERVER}.jsonl'
for item in manifest['selected']:
    i=item['idx'];row=rows[i];rec=dict(idx=i,task='CMTB',gold=row['answer'])
    try:
        src=cut(fetch(item['video_path']),0,10**4,'full')
        question,options=parse_question(row['question'])
        media=prepare_video(src);rec['base']=score(media,question,options,mode='av');del media
        for variant in ('ledger','linked'):
            result=AVB_NS['avbind_answer'](src,question,options,OUT/f'avbind_cmtb_observations_{i}.json',
                                          OUT/f'avbind_cmtb_utterances_{i}.json',max_steps=3,variant=variant)
            rec[variant]=result['scores'];rec[variant+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-2400:]
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
    records.append(rec)
    print('AVBIND_CMTB',json.dumps(dict(idx=i,gold=rec['gold'],
        predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','ledger','linked') if v in rec},error=rec.get('error'))),flush=True)
RESULT=dict(attempted=len(records),file=path.name)

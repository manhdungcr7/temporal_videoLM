"""Identical generic native binding API on six cross-domain probe questions."""
import traceback,gc
daily=OUT/'avbind_daily_examples'
manifest=json.loads((daily/'manifest.json').read_text());qa=json.loads((daily/'qa.json').read_text())
path=OUT/f'avbind_native_daily_r3_s{LAB_SERVER}.jsonl';records=[]
for item in manifest['selected']:
    row=qa[item['idx']];rec=dict(idx=item['idx'],dataset='Daily-Omni public examples',type=row['Type'],gold=row['Answer'])
    try:
        src=cut(item['path'],0,10**4,'full')
        question=row['Question'];options=[re.sub(r'^[A-D]\.\s*','',x).strip() for x in row['Choice']]
        media=prepare_video(src);rec['base']=score(media,question,options,mode='av');del media
        for enabled,name in ((False,'memory'),(True,'binding')):
            result=NEURAL_NS['avbind_native_answer'](src,question,options,
                daily/f'{item["video_id"]}_observations.json',daily/f'{item["video_id"]}_utterances.json',enabled=enabled)
            rec[name]=result['scores'];rec[name+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-2400:]
    with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(rec)+'\n');handle.flush()
    records.append(rec)
    print('AVBIND_DAILY',json.dumps(dict(idx=rec['idx'],type=rec['type'],gold=rec['gold'],
        predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','memory','binding') if v in rec},error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),file=path.name)

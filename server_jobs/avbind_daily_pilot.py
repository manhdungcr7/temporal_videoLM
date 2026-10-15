"""Cross-domain pilot, unchanged symmetric API, no task provided to inference."""
import traceback,gc
daily=OUT/'avbind_daily_examples'
manifest=json.loads((daily/'manifest.json').read_text())
qa=json.loads((daily/'qa.json').read_text())
path=OUT/f'avbind_daily_pilot_s{LAB_SERVER}.jsonl'
done={json.loads(line)['idx'] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
records=[]
for item in manifest['selected']:
    if item['idx'] in done:continue
    row=qa[item['idx']]
    rec=dict(idx=item['idx'],dataset='Daily-Omni official examples',type=row['Type'],gold=row['Answer'])
    try:
        src=cut(item['path'],0,10**4,'full')
        question=row['Question'];options=[re.sub(r'^[A-D]\.\s*','',x).strip() for x in row['Choice']]
        media=prepare_video(src);rec['base']=score(media,question,options,mode='av');del media
        for variant in ('ledger','linked'):
            result=AVB_NS['avbind_answer'](src,question,options,daily/f'{item["video_id"]}_observations.json',
                                          daily/f'{item["video_id"]}_utterances.json',max_steps=3,variant=variant)
            rec[variant]=result['scores'];rec[variant+'_trace']=result
    except Exception:rec['error']=traceback.format_exc()[-2400:]
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
    records.append(rec)
    print('AVBIND_DAILY',json.dumps(dict(idx=item['idx'],type=rec['type'],gold=rec['gold'],
                                      predictions={v:'ABCD'[int(np.argmax(rec[v]))] for v in ('base','ledger','linked') if v in rec},
                                      error=rec.get('error'))),flush=True)
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(records),file=path.name)

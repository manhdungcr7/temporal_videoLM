"""Frozen CQ and native baseline; durable per-sample records and live counts."""
import gc,hashlib,os,traceback
from collections import Counter

MANIFEST=json.loads((OUT/f'uploads/cq_large_manifest_s{LAB_SERVER}.json').read_text())
assert MANIFEST['source_sha256']==CQ_INFERENCE_METADATA['source_sha256']
assert CQ_INFERENCE_METADATA['model_revision']=='f75b40e3da2003cdd6e1829b1f420ca70797c34e'
assert CQ_INFERENCE_METADATA['model']=='Qwen/Qwen2.5-Omni-3B'
assert CQ_INFERENCE_METADATA['vad_version']=='6.2.3'
assert CQ_INFERENCE_METADATA['layers']==36 and CQ_INFERENCE_METADATA['dtype']=='torch.float16'
assert CQ_INFERENCE_METADATA['benchmark_variables_absent']
ROWS=json.loads(fetch('samples.json').read_text(encoding='utf-8'))
assert hashlib.sha256(fetch('samples.json').read_bytes()).hexdigest()==MANIFEST['samples_sha256']
RUN_ID=MANIFEST['run_id']
RUN_DIR=OUT/RUN_ID
RUN_DIR.mkdir(exist_ok=True)
CACHE_DIR=RUN_DIR/'observations'
CACHE_DIR.mkdir(exist_ok=True)
RESULT_PATH=RUN_DIR/f'results_s{LAB_SERVER}.jsonl'
PROGRESS_PATH=RUN_DIR/f'progress_s{LAB_SERVER}.json'
RECORDS=[json.loads(line) for line in RESULT_PATH.read_text().splitlines() if line.strip()] if RESULT_PATH.exists() else []
assert len({r['idx'] for r in RECORDS})==len(RECORDS),'Duplicate resume records'
assert {r['idx'] for r in RECORDS}<={x['idx'] for x in MANIFEST['selected']}
DONE={r['idx'] for r in RECORDS}
TARGETS=Counter((x['task'],x['gold']) for x in MANIFEST['selected'])
CONDITIONS=('base','cq')
RUN_STARTED=time.time()
current=None

def write_progress(status='running',halt_reason=None):
    cells=[]
    for (task,gold),target in sorted(TARGETS.items()):
        subset=[r for r in RECORDS if r['task']==task and r['gold']==gold]
        methods={}
        for condition in CONDITIONS:
            valid=[r for r in subset if isinstance(r.get(condition),list)]
            correct=sum(int(np.argmax(r[condition]))=='ABCD'.index(gold) for r in valid)
            histogram=Counter('ABCD'[int(np.argmax(r[condition]))] for r in valid)
            methods[condition]=dict(correct=correct,wrong=len(valid)-correct,valid=len(valid),
                                    errors=len(subset)-len(valid),predictions=dict(histogram))
        cells.append(dict(task=task,gold=gold,target=target,attempted=len(subset),
                          remaining=target-len(subset),methods=methods))
    payload=dict(run_id=RUN_ID,server=LAB_SERVER,status=status,total=len(MANIFEST['selected']),
                 attempted=len(RECORDS),remaining=len(MANIFEST['selected'])-len(RECORDS),
                 current=current,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 session_elapsed_sec=round(time.time()-RUN_STARTED,1),groups=cells,
                 halt_reason=halt_reason,metadata=CQ_INFERENCE_METADATA,
                 result_file=str(RESULT_PATH.relative_to(OUT)))
    temp=PROGRESS_PATH.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    temp.replace(PROGRESS_PATH)
    return payload

write_progress()
consecutive_errors=0
fatal=None
for item in MANIFEST['selected']:
    i=item['idx']
    if i in DONE:continue
    row=ROWS[i]
    assert row['task_type_abbr']==item['task'] and row['answer']==item['gold']
    assert row['video_path'].replace(BS,'/')==item['video_path']
    assert item['video_path'] not in MANIFEST['excluded_video_paths']
    started=time.time()
    rec=dict(idx=i,task=item['task'],gold=item['gold'],video_path=item['video_path'],
             selected_duration=item['duration'],errors={})
    current=dict(idx=i,task=item['task'],gold=item['gold'],stage='media',
                 stage_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_progress()
    try:
        src=cut(fetch(item['video_path']),0,10**4,'full')
        wave=decode_audio(src)
        duration=len(wave)/16000
        if duration>50:raise RuntimeError(f'decoded duration {duration:.5f}s exceeds 50s; no truncation allowed')
        rec['duration']=round(duration,5)
        question,options=parse_question(row['question'])
        if len(options) not in (2,3,4):raise RuntimeError('Unexpected answer option count')
        # The scoring wrapper reads task/gold; neither is passed into inference.
        rec.update(question=question,options=options)
        current.update(stage='base',stage_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write_progress()
        base_start=time.time()
        try:
            media=prepare_video(src)
            rec['base']=score(media,question,options,mode='av')
            del media
        except Exception:
            rec['errors']['base']=traceback.format_exc()[-2400:]
        rec['base_latency_sec']=round(time.time()-base_start,2)
        gc.collect();torch.cuda.empty_cache()
        current.update(stage='cq',stage_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write_progress()
        cache_name=hashlib.sha256(item['video_path'].encode()).hexdigest()[:20]+'.json'
        result=CQ_NS['cq_provenance_answer'](src,question,options,CACHE_DIR/cache_name,
                                            mode='episode_constraints')
        assert not result['label_or_task_used_for_inference']
        rec.update(cq=result['scores'],cq_latency_sec=result['latency_sec'],evidence=result)
    except Exception:
        rec['errors']['cq']=traceback.format_exc()[-2400:]
    rec['elapsed_sec']=round(time.time()-started,2)
    with RESULT_PATH.open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(rec)+'\n');handle.flush();os.fsync(handle.fileno())
    RECORDS.append(rec);DONE.add(i)
    current=None
    consecutive_errors=0 if isinstance(rec.get('cq'),list) else consecutive_errors+1
    errors=' '.join(rec['errors'].values()).lower()
    if 'device-side assert' in errors or 'illegal memory access' in errors:
        fatal='CUDA context failure; restart required; remaining samples NOT attempted'
    elif consecutive_errors>=3:
        fatal='Three consecutive CQ execution errors; halted to preserve unattempted samples'
    progress=write_progress('halted' if fatal else 'running',fatal)
    print('CQ_LARGE_PROGRESS',json.dumps({k:progress[k] for k in ('server','attempted','total','status','groups')}),flush=True)
    gc.collect();torch.cuda.empty_cache()
    if fatal:break
RESULT=write_progress('halted' if fatal else 'done',fatal)
print('CQ_LARGE_FINISHED',json.dumps(dict(server=LAB_SERVER,status=RESULT['status'],
                                       attempted=len(RECORDS),remaining=RESULT['remaining'])),flush=True)

"""Resumable physical AV/text pseudo pairs, no benchmark questions or labels."""
import hashlib,traceback,gc,math
source=(OUT/'uploads/avbind_native_pairs.py').read_text()
PAIR_NS={name:globals()[name] for name in ('MODEL_','PROC','SYSTEM','cut','decode_audio','prepare_video','qwen_inputs','move_inputs')}
PAIR_NS['TOK']=PROC.tokenizer
exec(compile(source,'avbind_native_pairs.py','exec'),PAIR_NS)
assert not {'task','gold','rows','DATA','question','answer_options'} & PAIR_NS.keys()
manifest=json.loads((OUT/f'avbind_native_pair_selection_s{LAB_SERVER}.json').read_text())
assert manifest['status']=='done'
output=OUT/f'avbind_native_pairs_s{LAB_SERVER}.pt'
log=OUT/f'avbind_native_pairs_s{LAB_SERVER}.jsonl';log.touch(exist_ok=True)
metadata=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),model_revision=globals().get('MREV'),layer=28,
    question_used=False,benchmark_labels_used=False,pseudo_text_not_ground_truth=True,selection=manifest)
if output.exists():
    saved=torch.load(output,map_location='cpu',weights_only=True);assert saved['metadata']==metadata
    examples=saved['examples']
else:examples=[]
done={(x['video_path'],x['start'],x['source']) for x in examples}
for video in manifest['selected']:
    src=cut(fetch(video['video_path']),0,10**4,'full');duration=len(decode_audio(src))/16000
    for index in range(int(math.ceil(duration/2))):
        start,end=2*index,min(2*index+2,duration)
        if end-start<.2:continue
        media=prepare_video(cut(src,start,end,f'avbind_latent_{index}'))
        for source_name,mode in (('video','vision'),('audio','audio')):
            if (video['video_path'],start,source_name) in done:continue
            try:
                raw=PAIR_NS['native_pair_feature'](media,mode)
                content=PAIR_NS['native_pair_description'](media,mode)
                if not content:raise ValueError('Empty pseudo annotation')
                text=PAIR_NS['native_pair_text_feature'](content)
                example=dict(video_path=video['video_path'],recording_id=video['recording_id'],split=video['split'],
                    start=start,end=end,source=source_name,content=content,raw=raw,text=text)
                examples.append(example);done.add((video['video_path'],start,source_name))
                temporary=output.with_suffix('.tmp');torch.save(dict(metadata=metadata,examples=examples),temporary);temporary.replace(output)
                record={key:value for key,value in example.items() if key not in ('raw','text')};record['completed_pairs']=len(examples)
            except Exception:record=dict(video_path=video['video_path'],start=start,source=source_name,error=traceback.format_exc()[-1400:])
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n');f.flush()
            print('NATIVE_PAIR_EXTRACT',json.dumps(record),flush=True)
            gc.collect();torch.cuda.empty_cache()
RESULT=dict(n=len(examples),file=output.name,log=log.name,pseudo_annotations_not_ground_truth=True)

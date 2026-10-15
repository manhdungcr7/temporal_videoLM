"""New media-state representation; reuse existing splits/crops/text/annotations."""
import hashlib,traceback,gc
source=(OUT/'uploads/avbind_pooled_pairs.py').read_text()
POOL_NS={name:globals()[name] for name in ('MODEL_','PROC','qwen_inputs','move_inputs')};POOL_NS['TOK']=PROC.tokenizer
exec(compile(source,'avbind_pooled_pairs.py','exec'),POOL_NS)
original=torch.load(OUT/f'avbind_native_pairs_s{LAB_SERVER}.pt',map_location='cpu',weights_only=True)
metadata=dict(original['metadata']);metadata.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    original_pair_source_sha256=original['metadata']['source_sha256'],representation='mean native media-token hidden states',
    descriptions_reused=True,text_features_reused=True)
path=OUT/f'avbind_pooled_pairs_s{LAB_SERVER}.pt';log=OUT/f'avbind_pooled_pairs_s{LAB_SERVER}.jsonl';log.touch(exist_ok=True)
if path.exists():
    saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['metadata']==metadata;examples=saved['examples']
else:examples=[]
done={(e['video_path'],e['start'],e['source']) for e in examples};last=None
for old in original['examples']:
    identity=(old['video_path'],old['start'],old['source'])
    if identity in done:continue
    try:
        key=(old['video_path'],old['start'])
        if key!=last:
            src=cut(fetch(old['video_path']),0,10**4,'full');media=prepare_video(cut(src,old['start'],old['end'],f"avbind_latent_{int(old['start']/2)}"));last=key
        mode='vision' if old['source']=='video' else 'audio'
        example=dict(old);example['raw']=POOL_NS['pooled_native_feature'](media,mode)
        examples.append(example);done.add(identity)
        temporary=path.with_suffix('.tmp');torch.save(dict(metadata=metadata,examples=examples),temporary);temporary.replace(path)
        r=dict(video_path=old['video_path'],start=old['start'],source=old['source'],completed_pairs=len(examples),target=len(original['examples']))
    except Exception:r=dict(video_path=old['video_path'],start=old['start'],source=old['source'],error=traceback.format_exc()[-1400:])
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n');f.flush()
    print('NATIVE_POOL_EXTRACT',json.dumps(r),flush=True);gc.collect();torch.cuda.empty_cache()
RESULT=dict(file=path.name,completed=len(examples),target=len(original['examples']),annotations_reused=True)

"""Prepare all forty frozen IDs at rotation0, retaining identical media budget."""
import gc,json,time,torch
CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'cache40_definitions','exec'),globals())
D=OUT/'pcebr_stage0_v8';records=[]
for row in CD_MANIFEST[f'server{PC_SERVER}']:
    assert row['duration']<=45 and row['decoded_audio_duration']<=45
    stem,texts=parse_question(row['question']);perm=next(iter(rotations(len(texts))))
    path=CD_CACHE/f"{row['idx']}_rot0.pt"
    if not path.exists():
        media=prepare_video(fetch(row['video_path']),max_pixels=int(160*min(1.,32/row['duration']))*28*28)
        inp,_=qwen_inputs(PROC,cd_prompt(stem,texts,perm),media,'av');inp=move_inputs(inp,MODEL_)
        p=cd_encode(inp,len(texts));p['input_ids']=inp['input_ids'].detach().cpu();torch.save(p,path)
        del inp,media,p;gc.collect();torch.cuda.empty_cache()
    records.append(dict(idx=row['idx'],bytes=path.stat().st_size))
    (D/'prepare40_progress.json').write_text(json.dumps(records))
    print('V8_PREP40',row['idx'],len(records),flush=True)
RESULT=dict(n=len(records),manifest_sha256=CD_SHA)

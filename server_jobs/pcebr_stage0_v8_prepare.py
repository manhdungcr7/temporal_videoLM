"""Reproduce M0, prepare the fixed micro8 shards, and retain exact native inputs."""
import ast, gc, hashlib, json, time, traceback
from pathlib import Path
import torch

V8_DIR = OUT/'pcebr_stage0_v8'
V8_DIR.mkdir(exist_ok=True)
source = (OUT/'uploads/stage0_pcebr_v8_original.py').read_text(encoding='utf-8')
assert hashlib.sha256(source.encode()).hexdigest() == 'fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895'
tree = ast.parse(source)
tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='RESULT' for t in n.targets))]
ns = dict(OUT=V8_DIR)
exec(compile(tree,'v8_original_definitions','exec'), ns)
checks = {}
for dev in ('cpu','cuda:0'):
    try:
        t0=time.time()
        checks[dev]=dict(result=ns['run_algebraic_unit_checks'](dev), seconds=time.time()-t0)
    except Exception:
        checks[dev]=dict(error=traceback.format_exc())
    (V8_DIR/'m0_reproduction.json').write_text(json.dumps(checks,indent=2))
    print('V8_M0',dev,json.dumps(checks[dev]),flush=True)

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
records=[]
for row in CD_MANIFEST[f'server{PC_SERVER}'][:4]:
    assert row['duration']<=45 and row['decoded_audio_duration']<=45
    stem,texts=parse_question(row['question'])
    perm=next(iter(rotations(len(texts))))
    path=CD_CACHE/f"{row['idx']}_rot0.pt"
    if not path.exists():
        media=prepare_video(fetch(row['video_path']),max_pixels=int(160*min(1.,32/row['duration']))*28*28)
        inputs,_=qwen_inputs(PROC,cd_prompt(stem,texts,perm),media,'av')
        inputs=move_inputs(inputs,MODEL_)
        payload=cd_encode(inputs,len(texts))
        payload['input_ids']=inputs['input_ids'].detach().cpu()
        torch.save(payload,path)
        del inputs,media,payload
        gc.collect();torch.cuda.empty_cache()
    p=torch.load(path,map_location='cpu',weights_only=True)
    record=dict(idx=row['idx'],task=row['task'],gold=row.get('gold'),duration=row['duration'],tokens=p['tokens'],native_logits=p['native_logits'].tolist(),bytes=path.stat().st_size)
    records.append(record)
    print('V8_MICRO_CACHE',json.dumps(record),flush=True)
    (V8_DIR/'prepare_progress.json').write_text(json.dumps(records,indent=2))
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,rows=records,manifest_sha256=CD_SHA,sdpa_compat=AVB_SDPA_COMPAT,m0_status={k:'error' if 'error' in v else 'pass' for k,v in checks.items()})
(V8_DIR/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2))

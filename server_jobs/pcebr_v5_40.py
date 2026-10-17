"""Frozen research40 evaluation of the unchanged OB-TVOR v5 inference path.

All cyclic option rotations, native paired against v5; no fitting or tuning.
Use the supplied nested run_routed function verbatim. Generate no free-form text.
"""
import ast
import copy
import gc
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

O40_DIR=OUT/'pcebr_v5_40';O40_DIR.mkdir(exist_ok=True)
(O40_DIR/'rotations').mkdir(exist_ok=True)
(O40_DIR/'states').mkdir(exist_ok=True)
O40_SOURCE=OUT/'uploads/stage0_pcebr_v5_original.py'
O40_SHA=hashlib.sha256(O40_SOURCE.read_bytes()).hexdigest()
O40_TREE=ast.parse(O40_SOURCE.read_text(encoding='utf-8'))
O40_TREE.body=[n for n in O40_TREE.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RESULT' for t in n.targets))]
O40_NS=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=O40_SERVER,__name__='obtvor_v5_40_original')
exec(compile(O40_TREE,str(O40_SOURCE),'exec'),O40_NS)
O40_ROUTE=next(copy.deepcopy(n) for n in ast.walk(O40_TREE) if isinstance(n,ast.FunctionDef) and n.name=='run_routed')
O40_MANIFEST=json.loads((OUT/'uploads/research40_manifest.json').read_text(encoding='utf-8'))
O40_SAMPLES=O40_MANIFEST[f'server{O40_SERVER}']
O40_IDS=[r['idx'] for r in O40_SAMPLES]
assert len(O40_IDS)==20 and len(set(O40_IDS))==20
assert O40_MANIFEST['total']==40
assert all(r['duration']<=45 and r['decoded_audio_duration']<=45 for r in O40_SAMPLES)
O40_MSHA=O40_MANIFEST['manifest_sha256']
O40_LABELS=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD']
O40_DEVICE=MODEL_.get_input_embeddings().weight.device
CD_SERVER=O40_SERVER;CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())

def o40_write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def o40_capture(cap,index):
    def hook(mod,args,out):
        h=out[0] if isinstance(out,tuple) else out
        cap[index]=h[0].detach().float().cpu()
        return out
    return hook

def o40_rotation(sample,rotation,perm,media):
    idx=sample['idx'];path=O40_DIR/'rotations'/f'{idx}_rot{rotation}.json'
    if path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        assert result['manifest_sha256']==O40_MSHA and result['source_sha256']==O40_SHA and result['perm']==list(perm)
        return result,media
    started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
    cache=CD_CACHE/f'{idx}_rot{rotation}.pt'
    cap={};handles=[MODEL_.model.layers[i].register_forward_hook(o40_capture(cap,i)) for i in (18,27)]
    created=False
    try:
        if cache.exists():
            payload=torch.load(cache,map_location='cpu',weights_only=True)
            assert 'input_ids' in payload
            kw={key:value.to(O40_DEVICE) for key,value in payload['decoder'].items()}
            with torch.inference_mode():
                out=MODEL_.model(**kw,use_cache=False,return_dict=True)
                logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
            native=logits[O40_LABELS[:k]]
        else:
            if media is None:
                media=prepare_video(fetch(sample['video_path']),max_pixels=int(160*min(1.,32/sample['duration']))*28*28)
            inp,_=qwen_inputs(PROC,cd_prompt(stem,texts,perm),media,'av')
            inp=move_inputs(inp,MODEL_)
            payload=cd_encode(inp,k)
            payload['input_ids']=inp['input_ids'].detach().cpu()
            torch.save(payload,cache)
            native=payload['native_logits'].float();created=True
            del inp
    finally:
        for handle in handles:handle.remove()
    assert payload['manifest_sha256']==O40_MSHA and payload['prompt']=='letter_only_v1'
    assert torch.isfinite(native).all()
    replay=float((native-payload['native_logits'].float()).abs().max())
    kw={key:value.to(O40_DEVICE) for key,value in payload['decoder'].items()}
    clock=O40_NS['audit_and_interpolate_grid_clocks'](payload,sample)
    stem_text=sample['question'].split('(A)')[0].split('A.')[0].strip()
    stem_rows,option_rows,assistant_rows,hook_rows=O40_NS['find_option_free_and_option_spans'](payload,stem_text,PROC.tokenizer)
    assert not set(hook_rows)&set(option_rows)
    assert all(0<=i<payload['tokens'] for i in hook_rows)
    V,A,lv,la,iv=O40_NS['extract_query_conditioned_salient_slots'](cap[18],cap[27],stem_rows,clock,top_patch_frac=.15)
    Vvo=O40_NS['project_to_layer_vo_subspace'](MODEL_.model.layers[18],V)
    Avo=O40_NS['project_to_layer_vo_subspace'](MODEL_.model.layers[18],A)
    vec,gate,phi,aud=O40_NS['build_orthogonal_bivector_routed_state'](Vvo,Avo,lv,la,iv,reverse_timeline=False)
    rev,r_gate,r_phi,r_aud=O40_NS['build_orthogonal_bivector_routed_state'](Vvo,Avo,lv,la,iv,reverse_timeline=True)
    even_error=float((aud['c_ev']-r_aud['c_ev']).abs().max())
    odd_error=float((aud['tau_unit']+r_aud['tau_unit']).abs().max())
    cos=float(vec@rev)
    parity_checks=dict(content_even_error=even_error,order_odd_error=odd_error,cosine=cos,
                       original_micro_asserts_pass=even_error<1e-5 and odd_error<1e-5 and cos<.5,
                       zero_gate=gate<1e-6)
    assert torch.isfinite(vec).all() and np.isfinite(gate)
    route_ns=dict(O40_NS,model_llm=MODEL_,kw=kw,eff_scale=.045*gate,hook_rows_t=torch.tensor(hook_rows,dtype=torch.long))
    exec(compile(ast.Module(body=[copy.deepcopy(O40_ROUTE)],type_ignores=[]),'<original_v5_run_routed>','exec'),route_ns)
    routed=route_ns['run_routed']((14,16,18,20,22,24),vec)[O40_LABELS[:k]]
    assert torch.isfinite(routed).all()
    raw=dict(V18_raw=V,A18_raw=A,V18_vo=Vvo,A18_vo=Avo,logits_v=lv,logits_a=la,intervals=iv,
             vec=vec,vec_reversed=rev,phi=phi,native_logits=native,routed_logits=routed,
             hook_rows=hook_rows,stem_rows=stem_rows,assistant_rows=assistant_rows,option_rows=option_rows)
    torch.save(raw,O40_DIR/'states'/f'{idx}_rot{rotation}.pt')
    if rotation==0:
        (O40_DIR/'states'/f'{idx}_decoded_stem.txt').write_text(PROC.tokenizer.decode(payload['input_ids'][0,stem_rows],skip_special_tokens=False),encoding='utf-8')
    result=dict(idx=idx,rotation=rotation,perm=list(perm),tokens=payload['tokens'],created_native_cache=created,
                manifest_sha256=O40_MSHA,source_sha256=O40_SHA,native_replay_max_error=replay,
                logits={'native':native.tolist(),'v5':routed.tolist()},
                logprobs={'native':F.log_softmax(native,dim=-1).tolist(),'v5':F.log_softmax(routed,dim=-1).tolist()},
                gate=gate,effective_scale=.045*gate,clock=clock['clock_audit'],parity_checks=parity_checks,
                seconds=round(time.time()-started,3))
    o40_write(path,result)
    print('OBTVOR40_ROT',idx,rotation,'gate',round(gate,4),'replay',replay,'seconds',result['seconds'],flush=True)
    del cap,kw,payload,raw,route_ns
    gc.collect();torch.cuda.empty_cache()
    return result,media

O40_ROWS=[]
audit=dict(server=O40_SERVER,manifest_sha256=O40_MSHA,source_sha256=O40_SHA,
           runner_sha256=hashlib.sha256((OUT/'uploads/pcebr_v5_40.py').read_bytes()).hexdigest(),
           model=MODEL_.config._name_or_path,torch=torch.__version__,out=str(OUT),ids=O40_IDS,
           methods=['native','v5'],layers=[14,16,18,20,22,24],scale=.045,option_rotations='all cyclic',
           pooling='sum mapped log_softmax answer-letter logits',training=False,freeform=False)
o40_write(O40_DIR/'protocol.json',audit)
O40_START=time.time()
try:
    for sample in O40_SAMPLES:
        started=time.time();media=None;stem,texts=parse_question(sample['question']);k=len(texts)
        total={name:np.zeros(k,dtype=np.float64) for name in ('native','v5')};rlist=[]
        try:
            for rotation,perm in enumerate(rotations(k)):
                row,media=o40_rotation(sample,rotation,perm,media)
                for name in total:
                    for slot,original in enumerate(perm):total[name][original]+=row['logprobs'][name][slot]
                rlist.append(row)
        finally:
            del media;gc.collect();torch.cuda.empty_cache()
        # Gold is read only after all inference on the sample, solely for scoring.
        row=dict(idx=sample['idx'],task=sample['task'],gold=sample['gold'],duration=sample['duration'],
                 decoded_audio_duration=sample['decoded_audio_duration'],rotation_count=len(rlist),
                 scores={name:values.tolist() for name,values in total.items()},
                 predictions={name:'ABCD'[int(values.argmax())] for name,values in total.items()},
                 rotation0_predictions={name:'ABCD'[rlist[0]['perm'][int(np.argmax(rlist[0]['logprobs'][name]))]] for name in total},
                 max_replay_error=max(r['native_replay_max_error'] for r in rlist),
                 gates=[r['gate'] for r in rlist],seconds=round(time.time()-started,2))
        O40_ROWS.append(row)
        o40_write(O40_DIR/'results.json',O40_ROWS)
        print('OBTVOR40_SAMPLE',len(O40_ROWS),'/',len(O40_SAMPLES),json.dumps(row),flush=True)
    RESULT=dict(server=O40_SERVER,completed=len(O40_ROWS),rotations=sum(r['rotation_count'] for r in O40_ROWS),seconds=round(time.time()-O40_START,2),manifest_sha256=O40_MSHA,source_sha256=O40_SHA)
    o40_write(O40_DIR/'completion.json',RESULT)
    print('OBTVOR40_DONE',json.dumps(RESULT),flush=True)
except Exception:
    error=dict(server=O40_SERVER,completed=len(O40_ROWS),traceback=traceback.format_exc())
    o40_write(O40_DIR/'error.json',error)
    raise
finally:
    o40_write(O40_DIR/'hook_audit.json',dict(layer_hooks=sum(len(l._forward_hooks) for l in MODEL_.model.layers),cuda_ok=bool(torch.isfinite(torch.ones(1,device='cuda:0')).all())))

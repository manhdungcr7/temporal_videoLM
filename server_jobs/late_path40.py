"""Task-independent late attention/MLP interventions; fixed development40."""
import gc,hashlib,json,time,traceback
import numpy as np
import torch
from transformers import DynamicCache

CD_SERVER=globals().get('LP_SERVER',1)
LP_LIMIT=globals().get('LP_LIMIT')
_source=(OUT/'uploads/chronodesmear40.py').read_text().split('try: RESULT=cd_run()',1)[0]
exec(compile(_source,'chronodesmear_definitions_only','exec'),globals())
LP_DIR=OUT/'late_path40';LP_DIR.mkdir(exist_ok=True)
LP_SHA=hashlib.sha256((OUT/'uploads/late_path40.py').read_bytes()
    +(OUT/'uploads/chronodesmear40.py').read_bytes()+(OUT/'uploads/chronodesmear_math.py').read_bytes()).hexdigest()
LP_METHODS=['P0_native_reference','P1_last_mlp_half','P2_last_attn_half',
    'P3_last_mlp_amplify','P4_stem_mlp_half','P5_last_both_half','P6_last_mlp_random']
LP_PARITY_CHECKED=False
LP_FULL_ONLY=set()

def lp_decode(payload,bank=None,build_bank=False):
    """Reuse only the causal media prefix, which query-only interventions cannot change."""
    device=MODEL_.get_input_embeddings().weight.device
    kwargs={key:value.to(device) for key,value in payload['decoder'].items()}
    prefix=payload['last_media']+1
    if bank is not None:
        kwargs['inputs_embeds']=kwargs['inputs_embeds'][:,prefix:]
        kwargs['position_ids']=kwargs['position_ids'][:,:,prefix:]
        kwargs['cache_position']=torch.arange(prefix,payload['tokens'],device=device)
        kwargs['past_key_values']=DynamicCache(ddp_cache_data=bank,config=MODEL_.model.config)
    with torch.inference_mode():
        output=MODEL_.model(**kwargs,use_cache=build_bank or bank is not None,return_dict=True)
        logits=MODEL_.lm_head(output.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
        stored=tuple((key[:,:,:prefix].detach(),value[:,:,:prefix].detach())
            for key,value in output.past_key_values.to_legacy_cache()) if build_bank else None
    assert torch.isfinite(logits).all()
    return logits,stored

def lp_forward(payload,stem,mode,bank=None):
    span=cd_find_span(payload,stem)
    rows=list(range(span['lexical_stem_start'],span['lexical_stem_end'])) if mode=='stem_mlp_half' else [payload['tokens']-1]
    if bank is not None: rows=[row-payload['last_media']-1 for row in rows]
    handles=[];changes={};before={}
    def hook_for(layer,kind):
        def hook(module,args,out):
            value=out[0] if isinstance(out,tuple) else out
            new=value.clone();original=value[0,rows].float()
            if mode=='last_mlp_random':
                random=torch.randn(original.shape,generator=torch.Generator().manual_seed(2027+layer))
                direction=(random/random.norm(dim=-1,keepdim=True).clamp_min(1e-8)).to(original.device)
                correction=.5*original.norm(dim=-1,keepdim=True)*direction
            else:
                factor=.5 if mode=='last_mlp_amplify' else -.5
                correction=factor*original
            new[0,rows]=(original+correction).to(value.dtype)
            changes[f'{layer}_{kind}']=dict(native_update_norm=float(original.norm(dim=-1).mean()),
                correction_norm=float(correction.norm(dim=-1).mean()))
            return (new,)+out[1:] if isinstance(out,tuple) else new
        return hook
    modules=[]
    try:
        for li in range(28,36):
            layer=MODEL_.model.layers[li]
            kinds=('attention','mlp') if mode=='last_both_half' else ('attention',) if mode=='last_attn_half' else ('mlp',)
            for kind in kinds:
                module=layer.self_attn if kind=='attention' else layer.mlp
                before[id(module)]=len(module._forward_hooks);modules.append(module)
                handles.append(module.register_forward_hook(hook_for(li,kind)))
        logits,_=lp_decode(payload,bank=bank)
    finally:
        for handle in handles: handle.remove()
        assert all(len(module._forward_hooks)==before[id(module)] for module in modules)
    return logits,changes

def lp_rotation(payload,k,stem,reference):
    global LP_PARITY_CHECKED
    scores={'P0_native_reference':reference['scores']['R5_C0_baseline']}
    native,bank=lp_decode(payload,build_bank=True)
    values=torch.log_softmax(native[CD_LABELS[:k]],dim=-1).numpy()
    parity=float(np.max(np.abs(values-np.array(scores['P0_native_reference']))))
    build_replay_error=parity
    if build_replay_error>.03:
        native,_=lp_decode(payload)
        parity=float(np.max(np.abs(torch.log_softmax(native[CD_LABELS[:k]],dim=-1).numpy()
            -np.array(scores['P0_native_reference']))))
    assert parity<=.03,('Full native replay',parity)
    cached_native,_=lp_decode(payload,bank=bank)
    prefix_error=float((native[CD_LABELS[:k]]-cached_native[CD_LABELS[:k]]).abs().max())
    use_bank=build_replay_error<=.03 and prefix_error<=.03 and native[CD_LABELS[:k]].argmax()==cached_native[CD_LABELS[:k]].argmax()
    audit={}
    for method,mode in zip(LP_METHODS[1:],('last_mlp_half','last_attn_half','last_mlp_amplify',
        'stem_mlp_half','last_both_half','last_mlp_random')):
        started=time.perf_counter();selected_bank=bank if use_bank and method not in LP_FULL_ONLY else None
        lg,changes=lp_forward(payload,stem,mode,bank=selected_bank)
        condition_parity=None
        if not LP_PARITY_CHECKED and selected_bank is not None:
            full,full_changes=lp_forward(payload,stem,mode)
            condition_parity=float((full[CD_LABELS[:k]]-lg[CD_LABELS[:k]]).abs().max())
            if condition_parity>.03 or full[CD_LABELS[:k]].argmax()!=lg[CD_LABELS[:k]].argmax():
                LP_FULL_ONLY.add(method);lg=full;changes=full_changes;selected_bank=None
        scores[method]=torch.log_softmax(lg[CD_LABELS[:k]],dim=-1).numpy().tolist()
        audit[method]=dict(seconds=time.perf_counter()-started,changes=changes,
            prefix_used=selected_bank is not None,full_parity=condition_parity)
    LP_PARITY_CHECKED=True
    return dict(scores=scores,condition_audit=audit,native_spotcheck=parity,
        build_native_error=build_replay_error,native_prefix_error=prefix_error,
        full_only_methods=sorted(LP_FULL_ONLY),tokens=payload['tokens'])

def lp_run():
    samples=CD_MANIFEST[f'server{CD_SERVER}']
    if LP_LIMIT: samples=samples[:LP_LIMIT]
    output=LP_DIR/f'results_s{CD_SERVER}.jsonl'
    done={json.loads(line)['idx'] for line in output.read_text().splitlines()} if output.exists() else set()
    for sample in samples:
        idx=sample['idx']
        if idx in done: continue
        started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
        totals={m:np.zeros(k) for m in LP_METHODS};audits=[]
        for rotation,perm in enumerate(rotations(k)):
            path=LP_DIR/f'{idx}_rot{rotation}.json'
            if path.exists(): result=json.loads(path.read_text())
            else:
                print('LATE_PATH_ROT_START',CD_SERVER,idx,rotation,flush=True)
                payload=torch.load(CD_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
                reference=json.loads((CD_DIR/f'{idx}_rot{rotation}.json').read_text())
                result=lp_rotation(payload,k,stem,reference)
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=CD_SHA,runtime_sha256=LP_SHA)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                del payload;gc.collect();torch.cuda.empty_cache()
            assert result['runtime_sha256']==LP_SHA and result['manifest_sha256']==CD_SHA
            for method,values in result['scores'].items():
                for slot,option in enumerate(perm): totals[method][option]+=values[slot]
            audits.append({key:value for key,value in result.items() if key!='scores'})
        record=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=CD_SHA,
            runtime_sha256=LP_SHA,scores={m:values.tolist() for m,values in totals.items()},
            predictions={m:chr(65+int(values.argmax())) for m,values in totals.items()},audit=audits,
            seconds=round(time.time()-started,2))
        with output.open('a') as handle: handle.write(json.dumps(record)+'\n')
        done.add(idx);print('LATE_PATH_SAMPLE',len(done),'/',len(samples),json.dumps({key:value for key,value in record.items() if key!='audit'}),flush=True)
    return dict(server=CD_SERVER,completed=len(done),expected=len(samples))

try: RESULT=lp_run()
except Exception:
    with (LP_DIR/f'errors_s{CD_SERVER}.jsonl').open('a') as handle: handle.write(json.dumps(dict(error=traceback.format_exc()))+'\n')
    raise
print('LATE_PATH_COMPLETE',json.dumps(RESULT),flush=True)

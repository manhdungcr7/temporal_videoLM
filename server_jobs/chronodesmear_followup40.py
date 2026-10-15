"""Clean factor ladder, source-symmetric innovations, and native depth diagnostics."""
import gc,hashlib,json,time,traceback
import numpy as np
import torch
import torch.nn.functional as F

CD_SERVER=globals().get('CF_SERVER',1)
CF_LIMIT=globals().get('CF_LIMIT')
_source=(OUT/'uploads/chronodesmear40.py').read_text().split('try: RESULT=cd_run()',1)[0]
exec(compile(_source,'chronodesmear_definitions_only','exec'),globals())
CF_DIR=OUT/'chronodesmear_followup40';CF_DIR.mkdir(exist_ok=True)
CF_SHA=hashlib.sha256((OUT/'uploads/chronodesmear_followup40.py').read_bytes()
    +(OUT/'uploads/chronodesmear40.py').read_bytes()+(OUT/'uploads/chronodesmear_math.py').read_bytes()).hexdigest()
CF_LAYERS=(4,8,12,16,18,22,26,30,34,35)
CF_METHODS=['N0_native','N1_desmear_reference','N2_pure_delta','N3_pure_delta_centered',
    'N4_centered_plus_desmear','N5_centered_desmear_thought','N6_random_stratified',
    'N7_source_symmetric','N8_symmetric_same_layer_stem','N9_fixed_depth_ensemble']

def cf_native(payload):
    lens={};geometry={};handles=[]
    def post(li):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            with torch.inference_mode():
                norm_device=next(MODEL_.model.norm.parameters()).device
                normed=MODEL_.model.norm(h[:,-1:,:].to(norm_device))
                lg=MODEL_.lm_head(normed.to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
                lens[str(li)]=lg[CD_LABELS].tolist()
                source_metrics={}
                for name,source in [('visual',payload['is_v']),('audio',payload['is_a'])]:
                    indices=torch.where(source)[0].to(h.device)
                    if not len(indices): continue
                    values=h[0,indices].float();mean=values.mean(0)
                    source_metrics[name]=dict(mean_to_mean_token_norm=float(mean.norm()/values.norm(dim=-1).mean().clamp_min(1e-8)))
                geometry[str(li)]=source_metrics
            return out
        return hook
    try:
        for li in CF_LAYERS: handles.append(MODEL_.model.layers[li].register_forward_hook(post(li)))
        native,cap=cd_forward(payload,capture=True)
    finally:
        for handle in handles: handle.remove()
    parity=float((native[CD_LABELS]-torch.tensor(lens['35'])).abs().max())
    assert parity<=.03,('Last layer lens parity',parity)
    return native,cap,lens,geometry,parity

def cf_random_static(payload,selected):
    indices=torch.where(payload['is_v'])[0];times=payload['sec'][indices].float()
    lower,upper=float(times.min()),float(times.max());span=max(1e-3,upper-lower)
    mask=torch.zeros_like(selected);generator=torch.Generator().manual_seed(2027)
    for b in range(5):
        left=lower+b/5*span;right=lower+(b+1)/5*span+(1e-4 if b==4 else 0)
        members=indices[(times>=left)&(times<right)]
        count=int(selected[members].sum())
        if count: mask[members[torch.randperm(len(members),generator=generator)[:count]]]=True
    assert int(mask.sum())==int(selected.sum())
    return mask

def cf_bounded_stem_forward(payload,delta,span):
    handles=[]
    def hook(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;h=h.clone()
        start,end=span['lexical_stem_start'],span['lexical_stem_end']
        q=h[0,start:end].float()
        d=delta[start-payload['last_media']-1:end-payload['last_media']-1].to(h.device)
        scale=(.15*q.norm(dim=-1,keepdim=True)/d.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        h[0,start:end]=(q+scale*d).to(h.dtype)
        return (h,)+out[1:] if isinstance(out,tuple) else h
    try:
        handles.append(MODEL_.model.layers[22].register_forward_hook(hook))
        lg,_=cd_forward(payload)
    finally:
        for handle in handles: handle.remove()
    return lg

def cf_rotation(payload,k,stem,reference):
    native,cap,lens,geometry,parity=cf_native(payload)
    last=payload['last_media'];base=cap[22];q=base[last+1:]
    dtype=MODEL_.get_input_embeddings().weight.dtype
    span=cd_find_span(payload,stem)
    v,a=payload['is_v'],payload['is_a'];sec=payload['sec']
    duration=float(sec[v|a].max())
    static,_,_,_=static_visual_mask(base,v,sec,duration)
    random_static=cf_random_static(payload,static)
    audio_static,_,_,_=static_visual_mask(base,a,sec,duration)
    def coverage(source,selected):
        times=torch.unique(sec[source].float(),sorted=True)
        counts=[int((selected&(sec==t)).sum()) for t in times]
        covered=times[torch.tensor(counts)>0]
        return dict(timestamps=len(times),covered_timestamps=len(covered),
            largest_covered_timestamp_gap=float((covered[1:]-covered[:-1]).max()) if len(covered)>1 else None,
            kept_tokens_per_timestamp=counts)
    selection_coverage=dict(visual_static=coverage(v,static),audio_static=coverage(a,audio_static),
        random_visual_static=coverage(v,random_static))
    branches={}
    for name,layout,mask_static in [('visual_static',payload,static),('random_static',payload,random_static),
            ('audio_static',dict(payload,is_v=a,is_a=v),audio_static)]:
        mask=counterfactual_mask(layout,'static_plus_aud',dtype,static=mask_static)
        _,state=cd_forward(payload,mask=mask,capture=True)
        branches[name]=state[22][last+1:]
        del mask,state;torch.cuda.empty_cache()
    pure=q-branches['visual_static']
    centered=center_option_slice(pure,span['stem_len'])
    randomized=center_option_slice(q-branches['random_static'],span['stem_len'])
    symmetric=center_option_slice(.5*((q-branches['visual_static'])+(q-branches['audio_static'])),span['stem_len'])
    thought,_=compute_av_resonance_and_shift_diagnostics(base,v,a,static,sec,duration)
    plans=[interval_plan(torch.where(source)[0],sec) for source in (v,a)]
    logits={'N0_native':native}
    logits['N2_pure_delta'],_=cd_forward(payload,delta=pure)
    logits['N3_pure_delta_centered'],_=cd_forward(payload,delta=centered)
    logits['N4_centered_plus_desmear'],_=cd_forward(payload,delta=centered,desmear=True,plans=plans)
    logits['N5_centered_desmear_thought'],_=cd_forward(payload,delta=centered,desmear=True,plans=plans,thought=thought)
    logits['N6_random_stratified'],_=cd_forward(payload,delta=randomized)
    logits['N7_source_symmetric'],_=cd_forward(payload,delta=symmetric)
    logits['N8_symmetric_same_layer_stem']=cf_bounded_stem_forward(payload,symmetric,span)
    scores={m:torch.log_softmax(lg[CD_LABELS[:k]],dim=-1).numpy().tolist() for m,lg in logits.items()}
    scores['N1_desmear_reference']=reference['scores']['R5_C1_causal_desmear_1p']
    lens_scores={layer:torch.log_softmax(torch.tensor(values[:k]),dim=-1).numpy().tolist() for layer,values in lens.items()}
    scores['N9_fixed_depth_ensemble']=np.mean([lens_scores[str(li)] for li in (22,26,30,34,35)],axis=0).tolist()
    return dict(scores={m:scores[m] for m in CF_METHODS},layer_scores=lens_scores,geometry=geometry,
        last_layer_parity=parity,span=span,tokens=payload['tokens'],
        reference_error=float(np.max(np.abs(np.array(scores['N0_native'])-np.array(reference['scores']['R5_C0_baseline'])))),
        visual_static=int(static.sum()),audio_static=int(audio_static.sum()),
        selection_coverage=selection_coverage,
        random_static_overlap=int((static&random_static).sum()))

def cf_run():
    samples=CD_MANIFEST[f'server{CD_SERVER}']
    if CF_LIMIT: samples=samples[:CF_LIMIT]
    output=CF_DIR/f'results_s{CD_SERVER}.jsonl'
    done={json.loads(line)['idx'] for line in output.read_text().splitlines()} if output.exists() else set()
    for sample in samples:
        idx=sample['idx']
        if idx in done: continue
        started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
        totals={m:np.zeros(k) for m in CF_METHODS};layers={str(li):np.zeros(k) for li in CF_LAYERS};audits=[]
        for rotation,perm in enumerate(rotations(k)):
            path=CF_DIR/f'{idx}_rot{rotation}.json'
            if path.exists(): result=json.loads(path.read_text())
            else:
                print('FOLLOWUP_ROT_START',CD_SERVER,idx,rotation,flush=True)
                payload=torch.load(CD_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
                reference=json.loads((CD_DIR/f'{idx}_rot{rotation}.json').read_text())
                result=cf_rotation(payload,k,stem,reference)
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=CD_SHA,runtime_sha256=CF_SHA)
                assert result['reference_error']<=.03,result['reference_error']
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                del payload;gc.collect();torch.cuda.empty_cache()
            assert result['runtime_sha256']==CF_SHA and result['manifest_sha256']==CD_SHA
            for m,values in result['scores'].items():
                for slot,option in enumerate(perm): totals[m][option]+=values[slot]
            for li,values in result['layer_scores'].items():
                for slot,option in enumerate(perm): layers[li][option]+=values[slot]
            audits.append({key:value for key,value in result.items() if key not in ('scores','layer_scores')})
        record=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=CD_SHA,
            runtime_sha256=CF_SHA,scores={m:values.tolist() for m,values in totals.items()},
            predictions={m:chr(65+int(values.argmax())) for m,values in totals.items()},
            layer_scores={li:values.tolist() for li,values in layers.items()},
            layer_predictions={li:chr(65+int(values.argmax())) for li,values in layers.items()},
            audit=audits,seconds=round(time.time()-started,2))
        with output.open('a') as handle: handle.write(json.dumps(record)+'\n')
        done.add(idx)
        print('FOLLOWUP_SAMPLE',len(done),'/',len(samples),json.dumps({key:value for key,value in record.items() if key!='audit'}),flush=True)
    return dict(server=CD_SERVER,completed=len(done),expected=len(samples))

try: RESULT=cf_run()
except Exception:
    with (CF_DIR/f'errors_s{CD_SERVER}.jsonl').open('a') as handle: handle.write(json.dumps(dict(error=traceback.format_exc()))+'\n')
    raise
print('FOLLOWUP_COMPLETE',json.dumps(RESULT),flush=True)

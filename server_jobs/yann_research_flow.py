"""R1: isolate media serialization topology; no task or answer-dependent routing."""
import gc,json,time,torch
from pathlib import Path
YR_DIR=OUT/'yann_temporal_research';YR_DIR.mkdir(exist_ok=True)
yr_manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
yr_labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC']
yr_rows=[]
def yr_flow_mask(payload,mode):
    n=payload['tokens'];dt=payload['decoder']['inputs_embeds'].dtype
    mask=torch.zeros((n,n),dtype=dt)
    mask.masked_fill_(torch.ones((n,n),dtype=torch.bool).triu_(1),torch.finfo(dt).min)
    original=mask.clone();ix=torch.where(payload['is_v']|payload['is_a'])[0]
    v=payload['is_v'][ix];a=payload['is_a'][ix];t=payload['sec'][ix]
    sub=mask[ix[:,None],ix[None,:]]
    if mode=='bidir_all':sub[:]=0
    elif mode=='bidir_cross':sub[(v[:,None]&a[None,:])|(a[:,None]&v[None,:])]=0
    elif mode=='clock_causal':
        # This clock is native TMRoPE timing metadata; it is not independent event ground truth.
        sub[:]=torch.where(t[:,None]+1e-5>=t[None,:],0.,torch.finfo(dt).min)
    elif mode=='native':pass
    else:raise ValueError(mode)
    mask[ix[:,None],ix[None,:]]=sub
    lm=int(payload['last_media'])
    assert (mask[:lm+1,lm+1:]==torch.finfo(dt).min).all(), 'Question/options leaked into media'
    assert torch.equal(mask[lm+1:],original[lm+1:]),'Tail decoding changed'
    stats=dict(opened_edges=int(((mask==0)&(original<0)).sum()),removed_edges=int(((mask<0)&(original==0)).sum()),native_rope_preserved=True)
    return mask[None,None],stats
def yr_flow_forward(payload,mask=None,layers=(),capture=False):
    d=MODEL_.get_input_embeddings().weight.device
    kw={k:v.to(d) for k,v in payload['decoder'].items()};handles=[];masks={};box={}
    def pre(module,args,kwargs):
        h=kwargs.get('hidden_states',args[0] if args else None);key=str(h.device)
        if key not in masks:masks[key]=mask.to(h.device)
        kwargs['attention_mask']=masks[key];return args,kwargs
    def post(module,args,out):
        h=out[0] if isinstance(out,tuple) else out
        box['last']=h[0,-1].float().cpu()
    try:
        if mask is not None:
            for li in layers:handles.append(MODEL_.model.layers[li].register_forward_pre_hook(pre,with_kwargs=True))
        if capture:handles.append(MODEL_.model.layers[22].register_forward_hook(post))
        with torch.inference_mode():
            out=MODEL_.model(**kw,use_cache=False,return_dict=True)
            logits=MODEL_.lm_head(out.last_hidden_state[:,-1:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    finally:
        for h in handles:h.remove()
        masks.clear()
    assert torch.isfinite(logits).all()
    return logits,box
yr_specs=[('native',None,()),('explicit_native_mid','native',tuple(range(14,26))),
    ('bidir_all_mid','bidir_all',tuple(range(14,26))),('bidir_cross_mid','bidir_cross',tuple(range(14,26))),
    ('clock_causal_mid','clock_causal',tuple(range(14,26))),('clock_causal_early','clock_causal',tuple(range(0,12)))]
for sample in yr_manifest[f'server{YR_SERVER}'][:4]:
    p=torch.load(OUT/'argon_latent40/decoder_cache'/f"{sample['idx']}_rot0.pt",map_location='cpu',weights_only=True)
    r=dict(idx=sample['idx'],task=sample['task'],gold=sample['gold'],conditions={});base=None
    for name,mode,layers in yr_specs:
        started=time.time();mask,stats=yr_flow_mask(p,mode) if mode else (None,{})
        lg,_=yr_flow_forward(p,mask,layers);sl=lg[yr_labels[:len(p['native_logits'])]];prob=sl.softmax(0)
        if base is None:base=sl
        replay=float((sl-p['native_logits']).abs().max())
        if name in ('native','explicit_native_mid'):assert replay<1e-5,(name,replay)
        pred='ABC'[int(sl.argmax())]
        r['conditions'][name]=dict(prediction=pred,correct=pred==sample['gold'],logits=sl.tolist(),probabilities=prob.tolist(),seconds=time.time()-started,
            native_logit_max_diff=replay,tv=float(.5*(prob-base.softmax(0)).abs().sum()),**stats)
        del mask,lg;gc.collect();torch.cuda.empty_cache()
    yr_rows.append(r);(YR_DIR/'flow_partial.json').write_text(json.dumps(yr_rows,indent=2),encoding='utf-8')
    print('YR_FLOW',json.dumps(r),flush=True)
RESULT=dict(server=YR_SERVER,rows=yr_rows,hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()),
    counts={k:sum(r['conditions'][k]['correct'] for r in yr_rows) for k,_,_ in yr_specs})
assert RESULT['hooks']==0
(YR_DIR/'flow.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('YR_FLOW_DONE',json.dumps(RESULT['counts']),flush=True)

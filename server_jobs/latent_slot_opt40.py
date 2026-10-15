"""Per-recording source/time slot reconstruction with a frozen AV backbone.

Native/source-only teachers are label blind. Optimize input slots, export their
KV memories, then let the unchanged answer query read raw AV plus the memories.
This is an adaptation of latent optimization, not a claimed first such method.
"""
import gc,hashlib,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

LO_SERVER=globals().get('LO_SERVER',1)
LO_LIMIT=globals().get('LO_LIMIT')
SL_SERVER=LO_SERVER
exec(compile((OUT/'uploads/symmetric_latent40.py').read_text().split('RESULT=sl_run()',1)[0],
    'source_cache_helpers','exec'),globals())
LO_DIR=OUT/'latent_slot_opt40';LO_DIR.mkdir(exist_ok=True)
LO_SHA=hashlib.sha256((OUT/'uploads/latent_slot_opt40.py').read_bytes()).hexdigest()
LO_METHODS=['L0_native','L1_unoptimized_memory','L2_optimized_memory','L3_time_shuffled_target']
LO_STEPS=32;LO_LR=.05

def lo_mask(queries,prefix,extra,dtype,device,forbidden=None,memory_bias=None,memory_enabled=True):
    keys=prefix+extra+queries
    mask=torch.zeros((1,1,queries,keys),dtype=dtype,device=device)
    future=torch.arange(keys,device=device)[None,:]>(prefix+extra+torch.arange(queries,device=device))[:,None]
    mask.masked_fill_(future[None,None],torch.finfo(dtype).min)
    if forbidden is not None:mask[...,torch.where(forbidden.to(device))[0]]=torch.finfo(dtype).min
    if extra:
        mask[...,prefix:prefix+extra]=memory_bias if memory_enabled else torch.finfo(dtype).min
    return mask

def lo_slots(payload,source,h22):
    member=payload['is_v'] if source=='video' else payload['is_a']
    groups=torch.floor(payload['sec']/2).long();items=[]
    for group in torch.unique(groups[member],sorted=True):
        ix=torch.where(member&(groups==group))[0]
        items.append(dict(source=source,time=float(payload['sec'][ix].mean()),
            interval=[float(group*2),float(group*2+2)],
            init=payload['decoder']['inputs_embeds'][0,ix].float().mean(0),
            target=h22[ix].mean(0),
            position=payload['decoder']['position_ids'][:,:,ix].float().mean(-1).round().long()))
    assert items,'Missing source slots'
    return items

def lo_teacher(payload,source=None):
    captured={}
    def take(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;captured['h']=h[0].detach().float().cpu()
    handle=MODEL_.model.layers[22].register_forward_hook(take)
    mask=None if source is None else sl_mask(payload,('audio',) if source=='video' else ('visual',))
    try:logits,state=sl_forward(payload,mask=mask,build_bank=True)
    finally:handle.remove()
    with torch.no_grad():
        bank=tuple((k.detach().clone(),v.detach().clone()) for k,v in state.pop('_prefix_bank'))
    return logits,bank,captured['h']

def lo_slot_forward(payload,bank,source,z,positions,stop22=True):
    device=MODEL_.get_input_embeddings().weight.device;dtype=MODEL_.get_input_embeddings().weight.dtype
    prefix=payload['last_media']+1;n=len(z)
    absent=(payload['is_a'] if source=='video' else payload['is_v'])[:prefix]
    forbidden=torch.cat((absent,torch.zeros(n,dtype=torch.bool)))
    mask=lo_mask(n,prefix,0,dtype,device,forbidden=forbidden)
    hooks=[];masks={};state={}
    class StopAt22(Exception):pass
    def pre(module,args,kwargs):
        h=kwargs.get('hidden_states',args[0] if args else None);key=str(h.device)
        if key not in masks:masks[key]=mask.to(h.device)
        kwargs['attention_mask']=masks[key]
        return args,kwargs
    def take(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;state['h']=h[0]
        if stop22:raise StopAt22()
    try:
        for layer in MODEL_.model.layers:hooks.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
        hooks.append(MODEL_.model.layers[22].register_forward_hook(take))
        cache=DynamicCache(ddp_cache_data=bank,config=MODEL_.model.config)
        try:
            out=MODEL_.model(inputs_embeds=z.to(device,dtype)[None],position_ids=positions.to(device),
                cache_position=torch.arange(prefix,prefix+n,device=device),
                attention_mask=torch.ones(1,prefix+n,dtype=torch.long,device=device),
                past_key_values=cache,use_cache=not stop22,return_dict=True)
        except StopAt22:out=None
        if stop22:return state['h']
        slots=tuple((k[:,:,prefix:].detach().cpu(),v[:,:,prefix:].detach().cpu())
                    for k,v in out.past_key_values.to_legacy_cache())
        return slots,state['h'].detach().float().cpu()
    finally:
        for hook in hooks:hook.remove()
        masks.clear()

def lo_optimize(payload,banks,items,shuffled=False):
    device=MODEL_.get_input_embeddings().weight.device
    initial={s:torch.stack([x['init'] for x in items[s]]).to(device) for s in items}
    parameters={s:torch.nn.Parameter(z.clone()) for s,z in initial.items()}
    positions={s:torch.stack([x['position'] for x in items[s]],dim=-1) for s in items}
    targets={s:torch.stack([x['target'] for x in items[s]]) for s in items}
    generator=torch.Generator().manual_seed(42)
    if shuffled:targets={s:t[torch.randperm(len(t),generator=generator)] for s,t in targets.items()}
    optimizer=torch.optim.Adam(list(parameters.values()),lr=LO_LR);curve=[]
    for step in range(LO_STEPS):
        optimizer.zero_grad(set_to_none=True);stats={}
        for s,z in parameters.items():
            with torch.enable_grad():
                h=lo_slot_forward(payload,banks[s],s,z,positions[s]).float()
                target=targets[s].to(h.device)
                hn=F.normalize(h,dim=-1);tn=F.normalize(target,dim=-1)
                cosine=(1-(hn*tn).sum(-1)).mean()
                contrast=F.cross_entropy(hn@tn.T/.1,torch.arange(len(h),device=h.device))
                regularizer=(z-initial[s]).square().mean()/initial[s].square().mean().clamp_min(1e-5)
                loss=cosine+.1*contrast+.001*regularizer.to(h.device)
                assert torch.isfinite(loss)
                loss.backward();stats[s]=dict(loss=float(loss.detach()),cosine=float(cosine.detach()))
            del h,target,hn,tn,loss
        assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in parameters.values())
        torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.)
        optimizer.step()
        with torch.no_grad():
            for s,z in parameters.items():
                d=z-initial[s];radius=.5*initial[s].norm(dim=-1,keepdim=True).clamp_min(1.)
                z.copy_(initial[s]+d*(radius/d.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.))
        if step%8==0 or step==LO_STEPS-1:curve.append(dict(step=step+1,sources=stats))
    return {s:z.detach().cpu() for s,z in parameters.items()},curve

def lo_memory(payload,banks,items,parameters):
    memories={};positions={s:torch.stack([x['position'] for x in items[s]],dim=-1) for s in items}
    diagnostics={}
    for s,z in parameters.items():
        with torch.inference_mode():
            memories[s],h=lo_slot_forward(payload,banks[s],s,z,positions[s],stop22=False)
            target=F.normalize(torch.stack([x['target'] for x in items[s]]),dim=-1)
            similarities=F.normalize(h,dim=-1)@target.T
            diagnostics[s]=dict(cosine=float(similarities.diag().mean()),
                retrieval_correct=int((similarities.argmax(-1)==torch.arange(len(h))).sum()),n=len(h))
    order=sorted([(x['time'],s,j) for s in items for j,x in enumerate(items[s])])
    combined=[]
    for li in range(len(MODEL_.model.layers)):
        keys=torch.cat([memories[s][li][0][:,:,j:j+1] for _,s,j in order],dim=-2)
        values=torch.cat([memories[s][li][1][:,:,j:j+1] for _,s,j in order],dim=-2)
        combined.append((keys,values))
    return tuple(combined),diagnostics

def lo_answer(payload,bank,memory,k,enabled=True):
    prefix=payload['last_media']+1;device=MODEL_.get_input_embeddings().weight.device
    dtype=MODEL_.get_input_embeddings().weight.dtype
    extra=memory[0][0].shape[-2] if memory is not None else 0
    tail=payload['tokens']-prefix;merged=[]
    for li,(key,value) in enumerate(bank):
        if memory is not None:
            km,vm=memory[li];key=torch.cat((key,km.to(key.device,key.dtype)),dim=-2)
            value=torch.cat((value,vm.to(value.device,value.dtype)),dim=-2)
        merged.append((key,value))
    density=math.log(max(1.,float((payload['is_v']|payload['is_a']).sum())/max(1,extra)))
    early=lo_mask(tail,prefix,extra,dtype,device,memory_enabled=False)
    late=lo_mask(tail,prefix,extra,dtype,device,memory_bias=density,memory_enabled=enabled)
    hooks=[];masks={}
    def pre(li):
        def hook(module,args,kwargs):
            h=kwargs.get('hidden_states',args[0] if args else None);key=(str(h.device),li>=23)
            if key not in masks:masks[key]=(late if li>=23 else early).to(h.device)
            kwargs['attention_mask']=masks[key]
            return args,kwargs
        return hook
    try:
        for li,layer in enumerate(MODEL_.model.layers):hooks.append(layer.register_forward_pre_hook(pre(li),with_kwargs=True))
        with torch.inference_mode():
            out=MODEL_.model(inputs_embeds=payload['decoder']['inputs_embeds'][:,prefix:].to(device),
                position_ids=payload['decoder']['position_ids'][:,:,prefix:].to(device),
                cache_position=torch.arange(prefix+extra,prefix+extra+tail,device=device),
                attention_mask=torch.ones(1,prefix+extra+tail,dtype=torch.long,device=device),
                past_key_values=DynamicCache(ddp_cache_data=tuple(merged),config=MODEL_.model.config),
                use_cache=True,return_dict=True)
            logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    finally:
        for hook in hooks:hook.remove()
        masks.clear()
    assert torch.isfinite(logits).all()
    return logits[SL_LABELS[:k]]

def lo_run():
    output=LO_DIR/f'results_s{LO_SERVER}.jsonl'
    done={json.loads(x)['idx'] for x in output.read_text().splitlines()} if output.exists() else set()
    samples=SL_SAMPLES[:LO_LIMIT] if LO_LIMIT else SL_SAMPLES
    for sample in samples:
        idx=sample['idx']
        if idx in done:continue
        started=time.time();_,texts=parse_question(sample['question']);k=len(texts)
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        payload=torch.load(SL_CACHE/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
        native,bank,_=lo_teacher(payload)
        assert float((native[SL_LABELS[:k]]-payload['native_logits']).abs().max())<=.12
        banks={};items={}
        for s in ('video','audio'):
            _,banks[s],h=lo_teacher(payload,s);items[s]=lo_slots(payload,s,h);del h
        optfile=LO_DIR/f'{idx}_optimized.pt'
        if optfile.exists():
            saved=torch.load(optfile,map_location='cpu',weights_only=False);assert saved['runtime_sha256']==LO_SHA
            variants=saved['parameters'];curves=saved['curves']
        else:
            variants={'L1_unoptimized_memory':{s:torch.stack([x['init'] for x in items[s]]) for s in items}}
            curves={}
            for method,shuffle in [('L2_optimized_memory',False),('L3_time_shuffled_target',True)]:
                variants[method],curves[method]=lo_optimize(payload,banks,items,shuffled=shuffle)
                print('SLOT_OPTIMIZED',idx,method,json.dumps(curves[method]),flush=True)
            torch.save(dict(runtime_sha256=LO_SHA,parameters=variants,curves=curves),optfile)
        memories={};diagnostics={}
        for method,z in variants.items():memories[method],diagnostics[method]=lo_memory(payload,banks,items,z)
        del banks;gc.collect();torch.cuda.empty_cache()
        totals={m:np.zeros(k) for m in LO_METHODS};audits=[]
        source_prefix=payload['last_media']+1;source_embeds=payload['decoder']['inputs_embeds'][:,:source_prefix]
        for rotation,perm in enumerate(rotations(k)):
            p=torch.load(SL_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
            assert p['manifest_sha256']==SL_SHA
            assert torch.equal(source_embeds,p['decoder']['inputs_embeds'][:,:source_prefix]),'Media prefix differs across choices'
            raw=lo_answer(p,bank,None,k)
            error=float((raw-p['native_logits']).abs().max());assert error<=.06,('native cache replay',error)
            scores={'L0_native':torch.log_softmax(p['native_logits'],dim=-1).numpy().tolist()}
            neutral=lo_answer(p,bank,memories['L1_unoptimized_memory'],k,enabled=False)
            zero_error=float((neutral-raw).abs().max());assert zero_error<=.06,('zero memory identity',zero_error)
            for method,memory in memories.items():
                scores[method]=torch.log_softmax(lo_answer(p,bank,memory,k),dim=-1).numpy().tolist()
            result=dict(idx=idx,rotation=rotation,perm=list(perm),scores=scores,
                manifest_sha256=SL_SHA,runtime_sha256=LO_SHA,replay_error=error,zero_memory_error=zero_error)
            (LO_DIR/f'{idx}_rot{rotation}.json').write_text(json.dumps(result))
            for method,score in scores.items():
                for slot,opt in enumerate(perm):totals[method][opt]+=score[slot]
            audits.append({a:b for a,b in result.items() if a!='scores'})
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=SL_SHA,
            runtime_sha256=LO_SHA,scores={m:v.tolist() for m,v in totals.items()},
            predictions={m:chr(65+int(v.argmax())) for m,v in totals.items()},audit=audits,
            slots={s:len(v) for s,v in items.items()},curves=curves,reconstruction=diagnostics,
            seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done.add(idx);print('SLOT_SAMPLE',len(done),'/',len(samples),json.dumps(rec),flush=True)
        del payload,bank,items,memories;gc.collect();torch.cuda.empty_cache()
    return dict(server=LO_SERVER,completed=len(done),expected=len(samples))

_flags=[p.requires_grad for p in MODEL_.parameters()]
try:
    for p in MODEL_.parameters():p.requires_grad_(False)
    RESULT=lo_run()
finally:
    for p,flag in zip(MODEL_.parameters(),_flags):p.requires_grad_(flag)
print('SLOT_OPT_COMPLETE',json.dumps(RESULT),flush=True)

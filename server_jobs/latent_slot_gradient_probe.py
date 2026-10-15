"""Technical feasibility probe, not a QA method result: optimize input slots only."""
import gc,json,time
import torch
import torch.nn.functional as F
from transformers import DynamicCache

SL_SERVER=1
_defs=(OUT/'uploads/symmetric_latent40.py').read_text().split('RESULT=sl_run()',1)[0]
exec(compile(_defs,'native_cached_helpers','exec'),globals())

def gradient_probe():
    started=time.time();row=SL_SAMPLES[0]
    payload=torch.load(SL_CACHE/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=False)
    full={}
    def capture(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;full['h']=h[0].detach().float().cpu()
    handle=MODEL_.model.layers[22].register_forward_hook(capture)
    try:_,states=sl_forward(payload,build_bank=True)
    finally:handle.remove()
    with torch.no_grad():
        bank=tuple((k.detach().clone(),v.detach().clone()) for k,v in states.pop('_prefix_bank'))
    v=payload['is_v'];sec=payload['sec'];groups=torch.floor(sec/2).long()
    indices=[torch.where(v&(groups==g))[0] for g in torch.unique(groups[v])[:8]]
    positions=torch.stack([payload['decoder']['position_ids'][:,:,ix].float().mean(-1).round().long()
                           for ix in indices],dim=-1)
    initial=torch.stack([payload['decoder']['inputs_embeds'][0,ix].float().mean(0) for ix in indices])
    target=torch.stack([full['h'][ix].mean(0) for ix in indices]);del states,full
    device=MODEL_.get_input_embeddings().weight.device;dtype=MODEL_.get_input_embeddings().weight.dtype
    slots=torch.nn.Parameter(initial.to(device));optimizer=torch.optim.Adam([slots],lr=.05)
    saved=[p.requires_grad for p in MODEL_.parameters()]
    prefix=payload['last_media']+1;curve=[];grad_norms=[]
    class StopAt22(Exception):pass
    try:
        for p in MODEL_.parameters():p.requires_grad_(False)
        for step in range(8):
            state={}
            def take(module,args,out):
                h=out[0] if isinstance(out,tuple) else out;state['h']=h[0];raise StopAt22()
            hook=MODEL_.model.layers[22].register_forward_hook(take)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.enable_grad():
                    cache=DynamicCache(ddp_cache_data=bank,config=MODEL_.model.config)
                    try:
                        MODEL_.model(inputs_embeds=slots.to(dtype)[None],position_ids=positions.to(device),
                            cache_position=torch.arange(prefix,prefix+len(slots),device=device),
                            attention_mask=torch.ones(1,prefix+len(slots),dtype=torch.long,device=device),
                            past_key_values=cache,use_cache=False,return_dict=True)
                    except StopAt22:pass
                    h=state['h'].float();loss=(1-F.cosine_similarity(h,target.to(h.device),dim=-1)).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    grad=float(slots.grad.norm());assert grad>0 and torch.isfinite(slots.grad).all()
                    grad_norms.append(grad);curve.append(float(loss.detach()))
                    torch.nn.utils.clip_grad_norm_([slots],1.);optimizer.step()
            finally:hook.remove()
            del cache,state;gc.collect();torch.cuda.empty_cache()
        return dict(idx=row['idx'],slots=len(slots),steps=8,loss=curve,gradient_norm=grad_norms,
            seconds=round(time.time()-started,2),only_input_slots_optimized=True,
            max_memory_GB=[round(torch.cuda.max_memory_allocated(i)/2**30,3) for i in range(torch.cuda.device_count())],
            qa_evaluated=False)
    finally:
        for p,flag in zip(MODEL_.parameters(),saved):p.requires_grad_(flag)

RESULT=gradient_probe()
(OUT/'latent_slot_gradient_probe.json').write_text(json.dumps(RESULT,indent=2))
print('LATENT_GRADIENT_PROBE',json.dumps(RESULT),flush=True)

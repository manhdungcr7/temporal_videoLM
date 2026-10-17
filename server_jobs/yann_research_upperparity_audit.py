"""R10 feasibility/parity only: native media prefix, trainable text-tail upper decoder."""
import gc,json,math,torch
from torch import nn
from torch.nn import functional as F
YR_DIR=OUT/'yann_temporal_research';manifest=json.loads((OUT/'uploads/perception48_manifest.json').read_text())
row=manifest['rows'][YR_SERVER-1];c=torch.load(YR_DIR/'perception48'/f"{row['video_id']}_q0.pt",map_location='cpu',weights_only=True)
media=prepare_video(YR_DIR/'perception48'/f"{row['video_id']}.mp4",max_pixels=int(96*min(1.,32/row['duration']))*28*28)
inp,_=qwen_inputs(PROC,c['question']+'\nAnswer the question directly.',media,'av');plen=inp['input_ids'].shape[1]
target=c['target'];inp['input_ids']=torch.cat((inp['input_ids'],target[None]),1);inp['attention_mask']=torch.ones_like(inp['input_ids']);inp=move_inputs(inp,MODEL_)
cap={}
def decoder_inputs(module,args,kw):
    cap['position_ids']=kw['position_ids'].detach().cpu();cap['embeds']=kw['inputs_embeds'].detach().cpu()
def layer27(module,args,out):cap['h27']=(out[0] if isinstance(out,tuple) else out).detach().cpu()
def before_head(module,args):cap['final']=args[0].detach().cpu();return (args[0][:,-1:],)
hooks=[MODEL_.model.register_forward_pre_hook(decoder_inputs,with_kwargs=True),MODEL_.model.layers[27].register_forward_hook(layer27),MODEL_.lm_head.register_forward_pre_hook(before_head)]
try:
    with torch.inference_mode():out=MODEL_(**inp,use_audio_in_video=True,use_cache=True,return_dict=True)
finally:
    for h in hooks:h.remove()
vid=PROC.tokenizer.convert_tokens_to_ids(PROC.video_token);aud=PROC.tokenizer.convert_tokens_to_ids(PROC.audio_token)
prefix=int(torch.where((inp['input_ids'][0]==vid)|(inp['input_ids'][0]==aud))[0].max())+1
past=out.past_key_values
legacy=past.to_legacy_cache() if hasattr(past,'to_legacy_cache') else tuple((l.keys,l.values) for l in past.layers)
keys={i:(legacy[i][0][:,:,:prefix].detach().clone(),legacy[i][1][:,:,:prefix].detach().clone()) for i in range(28,36)}
dev=next(MODEL_.model.layers[28].parameters()).device;hd=MODEL_.lm_head.weight.device
class PrefixCache:
    def update(self,k,v,i,cache_kwargs=None):
        pk,pv=keys[i];return torch.cat((pk.to(k.device),k),2),torch.cat((pv.to(v.device),v),2)
def upper():
    h=cap['h27'][:,prefix:].to(dev);pos=cap['position_ids'][:,:,prefix:].to(dev);q=h.shape[1]
    mask=torch.zeros((1,1,q,prefix+q),device=dev,dtype=h.dtype)
    mask[:,:,:,prefix:]=torch.where(torch.ones((q,q),device=dev,dtype=torch.bool).triu(1),torch.finfo(h.dtype).min,0)
    pe=MODEL_.model.rotary_emb(h,pos)
    for block in MODEL_.model.layers[28:]:
        h=block(h,attention_mask=mask,past_key_values=PrefixCache(),position_embeddings=pe,use_cache=False,cache_position=torch.arange(prefix,prefix+q,device=dev))[0]
    return MODEL_.model.norm(h)
with torch.no_grad():replayed=upper();reference=cap['final'][:,prefix:].to(dev)
err=float((replayed-reference).abs().max());meanerr=float((replayed-reference).abs().mean())
ref_logits=MODEL_.lm_head(reference[:,plen-1-prefix:plen+len(target)-1-prefix].to(hd)).float()
replay_logits=MODEL_.lm_head(replayed[:,plen-1-prefix:plen+len(target)-1-prefix].to(hd)).float()
logerr=float((ref_logits-replay_logits).abs().max())
strict_absolute_pass=err<.02 and logerr<.25
relative_l2=float((replayed.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
token_agreement=float((ref_logits.argmax(-1)==replay_logits.argmax(-1)).float().mean())
reference_ce=float(F.cross_entropy(ref_logits[0],target.to(hd)))
replayed_ce=float(F.cross_entropy(replay_logits[0],target.to(hd)))
# Audit extension after the preregistered absolute threshold failed. Preserve
# that failure, report scale-sensitive and functional differences separately.
print('YR_UPPER_PARITY_AUDIT',json.dumps(dict(strict_absolute_pass=strict_absolute_pass,max_hidden=err,mean_hidden=meanerr,relative_l2=relative_l2,
    max_logit=logerr,teacher_token_agreement=token_agreement,native_ce=reference_ce,replayed_ce=replayed_ce)),flush=True)
versions=[p._version for p in MODEL_.parameters()];req=[p.requires_grad for p in MODEL_.parameters()]
for p in MODEL_.parameters():p.requires_grad_(False)
adapters=nn.ModuleList().to(dev);handles=[];torch.manual_seed(20261020)
for block in MODEL_.model.layers[28:]:
    for name in ('q_proj','v_proj'):
        projection=getattr(block.self_attn,name);a=nn.Linear(projection.in_features,8,bias=False,device=dev);b=nn.Linear(8,projection.out_features,bias=False,device=dev);nn.init.zeros_(b.weight)
        adapters.extend((a,b))
        def hook(module,args,output,a=a,b=b):return output+b(a(args[0].float())).to(output.dtype)
        handles.append(projection.register_forward_hook(hook))
try:
    with torch.enable_grad():
        adjusted=upper();lg=MODEL_.lm_head(adjusted[:,plen-1-prefix:plen+len(target)-1-prefix].to(hd)).float()
        loss=F.cross_entropy(lg[0],target.to(hd));loss.backward()
    finite=all(torch.isfinite(p.grad).all().item() for p in adapters.parameters() if p.grad is not None)
    grad=sum(float(p.grad.norm()) for p in adapters.parameters() if p.grad is not None)
    assert finite and grad>0
    RESULT=dict(server=YR_SERVER,video_id=row['video_id'],prefix_tokens=prefix,tail_tokens=cap['h27'].shape[1]-prefix,
        hidden_max_error=err,hidden_mean_error=meanerr,logit_max_error=logerr,strict_absolute_parity_pass=strict_absolute_pass,
        relative_hidden_l2=relative_l2,teacher_token_agreement=token_agreement,native_ce=reference_ce,replayed_ce=replayed_ce,gradient_finite=finite,gradient_norm_sum=grad,
        adapter_parameters=sum(p.numel() for p in adapters.parameters()),prefix_kv_bytes=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in keys.values()),
        backbone_versions_unchanged=versions==[p._version for p in MODEL_.parameters()],peak_allocated=[torch.cuda.max_memory_allocated(i) for i in (0,1)])
    (YR_DIR/'upperparity.json').write_text(json.dumps(RESULT,indent=2));print(json.dumps(RESULT),flush=True)
finally:
    for h in handles:h.remove()
    for p,r in zip(MODEL_.parameters(),req):p.requires_grad_(r)
    del out,past,legacy,keys,cap,inp,adapters;gc.collect();torch.cuda.empty_cache()

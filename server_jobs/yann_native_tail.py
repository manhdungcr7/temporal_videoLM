"""Native frozen-prefix/text-tail functions shared by data, training and inference."""
import torch
from torch import nn
from torch.nn import functional as F

class PrefixCache:
    def __init__(self,keys):self.keys=keys
    def update(self,k,v,i,cache_kwargs=None):
        pk,pv=self.keys[i]
        return torch.cat((pk.to(device=k.device,dtype=k.dtype),k),2),torch.cat((pv.to(device=v.device,dtype=v.dtype),v),2)

def run_blocks(model,h,pos,prefix,keys,start=0,stop=36):
    """All tail tokens in one forward, causal tail and unchanged media prefix."""
    n=h.shape[1];cache=PrefixCache(keys)
    for i in range(start,stop):
        block=model.model.layers[i];dev=next(block.parameters()).device;h=h.to(dev);p=pos.to(dev)
        mask=torch.zeros((1,1,n,prefix+n),device=dev,dtype=h.dtype)
        mask[:,:,:,prefix:]=torch.where(torch.ones((n,n),device=dev,dtype=torch.bool).triu(1),torch.finfo(h.dtype).min,0)
        pe=model.model.rotary_emb(h,p)
        h=block(h,attention_mask=mask,past_key_values=cache,position_embeddings=pe,use_cache=False,
            cache_position=torch.arange(prefix,prefix+n,device=dev))[0]
    return h

def capture_native(model,proc,inp):
    cap={}
    def decoder(module,args,kw):
        cap['position_ids']=kw['position_ids'].detach().cpu()
    def post27(module,args,out):cap['h27']=(out[0] if isinstance(out,tuple) else out).detach().cpu()
    def head(module,args):cap['final']=args[0].detach().cpu();return (args[0][:,-1:],)
    hooks=[model.model.register_forward_pre_hook(decoder,with_kwargs=True),model.model.layers[27].register_forward_hook(post27),model.lm_head.register_forward_pre_hook(head)]
    try:
        with torch.inference_mode():out=model(**inp,use_audio_in_video=True,use_cache=True,return_dict=True)
    finally:
        for hook in hooks:hook.remove()
    ids=inp['input_ids'][0];vid=proc.tokenizer.convert_tokens_to_ids(proc.video_token);aud=proc.tokenizer.convert_tokens_to_ids(proc.audio_token)
    lastmedia=int(torch.where((ids==vid)|(ids==aud))[0].max())
    # Audio/video EOS can share a rotary position. Keep that native delimiter
    # boundary in the frozen prefix; begin adaptation at a scalar text run.
    pos=cap['position_ids'];prefix=None
    for candidate in range(lastmedia+1,min(lastmedia+17,pos.shape[-1])):
        start=int(pos[0,0,candidate]);expected=start+torch.arange(pos.shape[-1]-candidate)
        if torch.equal(pos[:,:,candidate:],expected.view(1,1,-1).expand(3,1,-1)):
            prefix=candidate;break
    assert prefix is not None,'No scalar native text run after media delimiters'
    past=out.past_key_values
    legacy=past.to_legacy_cache() if hasattr(past,'to_legacy_cache') else tuple((l.keys,l.values) for l in past.layers)
    keys={i:(legacy[i][0][:,:,:prefix].detach(),legacy[i][1][:,:,:prefix].detach()) for i in range(36)}
    start=int(pos[0,0,prefix]);expected=start+torch.arange(pos.shape[-1]-prefix)
    assert torch.equal(pos[:,:,prefix:],expected.view(1,1,-1).expand(3,1,-1)),'Non-scalar native text-tail positions'
    return dict(prefix=prefix,last_media=lastmedia,keys=keys,start_position=start,tail=proc.tokenizer.decode(ids[prefix:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False),
        h27=cap['h27'][:,prefix:],final=cap['final'][:,prefix:])

def make_tail(model,proc,base,old_question,question,answer,other_answer=None):
    old=old_question+'\nAnswer the question directly.';new=question+'\nAnswer the question directly.'
    assert base['tail'].count(old)==1
    tail=base['tail'].replace(old,new);prompt=proc.tokenizer(tail,add_special_tokens=False)['input_ids']
    target=proc.tokenizer(answer,add_special_tokens=False)['input_ids'];assert target
    tokens=prompt+target[:-1];dev=model.get_input_embeddings().weight.device
    ids=torch.tensor([tokens],device=dev);pos=(base['start_position']+torch.arange(len(tokens))).view(1,1,-1).expand(3,1,-1)
    with torch.inference_mode():
        h=model.get_input_embeddings()(ids);h27=run_blocks(model,h,pos,base['prefix'],base['keys'],stop=28).cpu()
    other=None;diverge=None
    if other_answer is not None:
        other=proc.tokenizer(other_answer,add_special_tokens=False)['input_ids'];diverge=0
        while diverge<min(len(target),len(other)) and target[diverge]==other[diverge]:diverge+=1
        assert diverge<min(len(target),len(other)),(answer,other_answer)
    return dict(h27=h27,positions=pos,target=torch.tensor(target),prompt_tail_length=len(prompt),prefix=base['prefix'],
        question=question,answer=answer,other_answer=other_answer,other_target=torch.tensor(other) if other is not None else None,
        divergence=diverge,token_ids=torch.tensor(tokens),start_position=base['start_position'])

class TailLoRA(nn.Module):
    def __init__(self,model,rank=8):
        super().__init__();self.adapters=nn.ModuleList();self.handles=[]
        for block in model.model.layers[28:]:
            for name in ('q_proj','v_proj'):
                projection=getattr(block.self_attn,name);dev=projection.weight.device
                a=nn.Linear(projection.in_features,rank,bias=False,device=dev);b=nn.Linear(rank,projection.out_features,bias=False,device=dev);nn.init.zeros_(b.weight)
                self.adapters.extend((a,b))
                def hook(module,args,output,a=a,b=b):return output+b(a(args[0].float())).to(output.dtype)
                self.handles.append(projection.register_forward_hook(hook))
    def remove(self):
        for h in self.handles:h.remove()
        self.handles=[]

def tail_logits(model,case,keys):
    h=run_blocks(model,case['h27'],case['positions'],case['prefix'],keys,start=28)
    h=model.model.norm(h);start=case['prompt_tail_length']-1;n=len(case['target'])
    return model.lm_head(h[:,start:start+n].to(model.lm_head.weight.device)).float()[0]

"""Late native attention reads from independently encoded raw AV slot addresses.

No generated memory text, question rewrite, relation-to-option rule or media
reduction. Supports are learned without benchmark labels. This prototype's
reader was trained on text features; raw-feature transfer is not established.
"""
import contextlib
import torch
import torch.nn.functional as F

def addressed_key_bias(nodes,support,gate,sec,is_video,is_audio,gain=2.0):
    """Allocate supported address mass per token, retaining the null gate.

    Each address gets its share independent of its number of native tokens.
    Source masks exclude all textual keys. No nearest/best event is forced.
    """
    if gain<0:raise ValueError('gain must be nonnegative')
    support=support.float();gate=gate.float()
    membership=torch.stack([((is_video if n['source']=='video' else is_audio)&
        (sec.clamp_min(0)>=n['interval'][0])&(sec<n['interval'][1])) for n in nodes]) if nodes else torch.zeros(0,len(sec),dtype=torch.bool,device=sec.device)
    count=membership.sum(-1).float();has_tokens=count>0
    normalizer=(support*has_tokens[None]).sum().clamp_min(1e-8)
    weights=(support*gate[:,None]).sum(0)/normalizer
    density=((weights/count.clamp_min(1))[:,None]*membership).sum(0)
    return torch.log1p(gain*(is_video|is_audio).sum()*density)

def segmented_sdpa(query,key,value,selected,key_bias,attention_mask=None,
                   causal=True,scale=None):
    """Exact causal attention split at selected rows; no full LxL mask.

    Supports full prefill and single-query cached decode, standard GQA and
    existing additive/boolean masks. Zero bias preserves ordinary attention.
    """
    if query.shape[-3]!=key.shape[-3]:
        groups=query.shape[-3]//key.shape[-3]
        key=key.repeat_interleave(groups,dim=-3);value=value.repeat_interleave(groups,dim=-3)
    qlen,klen=query.shape[-2],key.shape[-2]
    offset=klen-qlen
    if offset<0:raise ValueError('Query cannot exceed the available keys')
    selected=selected.to(query.device).bool()
    if selected.shape!=(qlen,):raise ValueError('Wrong query selection length')
    bias=F.pad(key_bias.to(query.device,query.dtype),(0,max(0,klen-len(key_bias))))[:klen]
    boundaries=[0]+(torch.where(selected[1:]!=selected[:-1])[0]+1).tolist()+[qlen]
    outputs=[]
    for start,end in zip(boundaries,boundaries[1:]):
        part=query[...,start:end,:];boost=bool(selected[start])
        # Large unselected prefix can use the fused causal kernel directly.
        if start==0 and offset==0 and not boost and attention_mask is None and causal:
            result=F.scaled_dot_product_attention(part,key[...,:end,:],value[...,:end,:],is_causal=True,scale=scale)
        else:
            # Pad the row stride for CUTLASS's efficient bias alignment.
            padded=(klen+7)//8*8
            mask=torch.zeros((1,1,end-start,padded),device=query.device,dtype=query.dtype)[...,:klen]
            if attention_mask is not None:
                source=attention_mask[...,:klen]
                if source.shape[-2]!=1:source=source[...,start:end,:]
                if source.dtype==torch.bool:mask=mask.masked_fill(~source,float('-inf'))
                else:mask=mask+source.to(query.dtype)
            elif causal:
                allowed=(torch.arange(start+offset,end+offset,device=query.device)[:,None]>=torch.arange(klen,device=query.device)[None,:])
                mask.masked_fill_(~allowed,float('-inf'))
            if boost:mask=mask+bias
            result=F.scaled_dot_product_attention(part,key,value,attn_mask=mask,scale=scale)
        outputs.append(result)
    return torch.cat(outputs,dim=-2)

@contextlib.contextmanager
def native_addressed_attention(reader,bank,nodes,query_positions,layout,state,gain=2.0):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    late={id(x.self_attn) for x in MODEL_.model.layers[BIND_LAYER:]}
    _pos,seconds,video,audio,_last=layout
    selection=torch.zeros(len(seconds),dtype=torch.bool,device=seconds.device)
    selection[query_positions]=True;selection[-1]=True
    def capture(_module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        if hidden.shape[1]>max(query_positions):
            query=F.layer_norm(hidden[:,query_positions].float().mean(1,keepdim=True),(hidden.shape[-1],))
            result=reader(query,*bank);state['audit']=result
            support=torch.stack([hop.support[0,0] for hop in result.hops])
            gates=torch.stack([hop.null_gate[0,0] for hop in result.hops])
            state['bias']=addressed_key_bias(nodes,support,gates,seconds.to(query.device),video.to(query.device),audio.to(query.device),gain)
        # Do not add a hidden-state residual. Only subsequent attention reads.
        return output
    def dispatch(module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
        if id(module) not in late or 'bias' not in state or gain==0:
            return original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
        if dropout:raise ValueError('Native attention prototype is inference-only')
        qlen=query.shape[-2]
        chosen=selection.to(query.device) if qlen==len(selection) else torch.ones(qlen,dtype=torch.bool,device=query.device)
        causal=(qlen>1 and attention_mask is None and getattr(module,'is_causal',True)) if is_causal is None else bool(is_causal)
        output=segmented_sdpa(query,key,value,chosen,state['bias'],attention_mask,causal,scaling)
        return output.transpose(1,2).contiguous(),None
    handle=MODEL_.model.layers[BIND_LAYER-1].register_forward_hook(capture)
    ALL_ATTENTION_FUNCTIONS.register('sdpa',dispatch)
    try:yield
    finally:
        handle.remove();ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

def avbind_attention_answer(video_path,question,answer_options=None,slot_cache=None,gain=2.0):
    src=Path(video_path);started=time.time();duration=len(decode_audio(src))/16000
    nodes,features=latent_sensor_bank(src,duration,slot_cache)
    device=next(BIND_READER.parameters()).device;n=len(nodes)
    bank=(features.unsqueeze(0).to(device),torch.tensor([[0 if x['source']=='video' else 1 for x in nodes]],device=device,dtype=torch.long),
          torch.tensor([[x['interval'] for x in nodes]],device=device,dtype=torch.float32).reshape(1,n,2),torch.ones(1,n,device=device,dtype=torch.bool))
    media=prepare_video(src);audits=[]
    if answer_options is None:
        inp,qpos=latent_question_inputs(media,question);state={};layout=media_layout(MODEL_,PROC,inp)
        with native_addressed_attention(BIND_READER,bank,nodes,qpos,layout,state,gain),torch.no_grad():
            sequence=MODEL_.generate(**inp,use_audio_in_video=True,max_new_tokens=128,do_sample=False,use_cache=True)
        answer=clean_observation(TOK.decode(sequence[0,inp['input_ids'].shape[1]:],skip_special_tokens=True));scores=prediction=None
    else:
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp,qpos=latent_question_inputs(media,question,answer_options,perm);state={};layout=media_layout(MODEL_,PROC,inp)
            head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
            try:
                with native_addressed_attention(BIND_READER,bank,nodes,qpos,layout,state,gain),torch.no_grad():
                    logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,j in enumerate(perm):totals[j]+=lp[position]
            audits.append(dict(max_key_bias=round(float(state['bias'].max()),4),
                hops=[dict(support=h.support.detach().cpu().tolist(),null=h.null_support.detach().cpu().tolist()) for h in state['audit'].hops]))
            del inp
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,prediction=prediction,scores=scores,nodes=nodes,audits=audits,
                metadata=BIND_METADATA,gain=gain,layers=list(range(BIND_LAYER+1,len(MODEL_.model.layers)+1)),
                caption_text_in_context=False,raw_AV_retained=True,latency_sec=round(time.time()-started,2))

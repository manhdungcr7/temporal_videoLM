"""Independent raw content keys and source/clock-aware raw values for native QA.

The question/answer stream is untouched. External rows see only their modality
crop; neither textual captions nor options enter their construction. Content
keys omit the clock. Value states encode the actual observation address.
"""
import contextlib
import torch
import torch.nn.functional as F

def inverse_text_rope(query,cos,sin):
    half=query.shape[-1]//2
    rotated=torch.cat((-query[...,half:],query[...,:half]),dim=-1)
    return (query.float()*cos.float()-rotated.float()*sin.float())/(cos.float().square()+sin.float().square()).clamp_min(1e-8)

def joint_memory_read(query,key,value,memory_query,memory_key,memory_value,
                      bank_log_mass,raw_mask=None,scale=None):
    """Selected rows only; native and independent-bank logits share softmax."""
    heads=query.shape[-3]
    if key.shape[-3]!=heads:
        groups=heads//key.shape[-3];key=key.repeat_interleave(groups,dim=-3);value=value.repeat_interleave(groups,dim=-3)
    if memory_key.shape[-3]!=heads:
        groups=heads//memory_key.shape[-3]
        memory_key=memory_key.repeat_interleave(groups,dim=-3);memory_value=memory_value.repeat_interleave(groups,dim=-3)
    scale=query.shape[-1]**-.5 if scale is None else scale
    raw_logits=torch.matmul(query.float(),key.float().transpose(-2,-1))*scale
    if raw_mask is not None:
        if raw_mask.dtype==torch.bool:raw_logits=raw_logits.masked_fill(~raw_mask,float('-inf'))
        else:raw_logits=raw_logits+raw_mask.float()
    memory_logits=torch.matmul(memory_query.float(),memory_key.float().transpose(-2,-1))*scale+bank_log_mass
    weights=torch.softmax(torch.cat((raw_logits,memory_logits),dim=-1),dim=-1)
    n=key.shape[-2]
    output=torch.matmul(weights[...,:n],value.float())+torch.matmul(weights[...,n:],memory_value.float())
    return output.to(query.dtype)

def latent_address_value(media,mode,source,start,end):
    prompt=(f'This {source} observation is from {start:g} to {end:g} seconds of the original recording. '
            'The interval bounds this observation, not necessarily a complete event. '
            'Describe only the directly available content of this observation.')
    inp,_=qwen_inputs(PROC,prompt,media,mode);inp=move_inputs(inp,MODEL_);captured={}
    class ValueReady(Exception):pass
    def capture(_module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['value']=F.layer_norm(hidden[0,-1].float(),(hidden.shape[-1],)).detach().cpu()
        raise ValueReady()
    handle=MODEL_.model.layers[BIND_LAYER-1].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_audio_in_video=False,use_cache=False)
    except ValueReady:pass
    finally:handle.remove()
    return captured['value']

def addressed_value_bank(src,nodes,cache_path=None):
    cache=Path(cache_path) if cache_path else None
    metadata=dict(revision=BIND_METADATA['revision'],layer=BIND_LAYER,
                  media_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),
                  addresses=nodes,representation='isolated raw states with factual source/observation-clock prompt; no generation')
    if cache and cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        if saved['metadata']!=metadata:raise ValueError('Addressed raw value cache mismatch')
        return saved['values']
    values=[];previous=None;media=None
    for node in nodes:
        start,end=node['interval']
        if previous!=(start,end):
            crop=cut(src,start,end,f'avbind_kv_{start:g}')
            media=prepare_video(crop);previous=(start,end)
        mode='vision' if node['source']=='video' else 'audio'
        values.append(latent_address_value(media,mode,node['source'],start,end))
    values=torch.stack(values)
    if cache:torch.save(dict(values=values,metadata=metadata),cache)
    return values

@contextlib.contextmanager
def independent_kv_attention(reader,bank,address_values,nodes,qpositions,layout,state,gain=1.0):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    late={id(x.self_attn) for x in MODEL_.model.layers[BIND_LAYER:]}
    positions,seconds,video,audio,_last=layout
    selected=torch.zeros(len(seconds),dtype=torch.bool,device=seconds.device);selected[qpositions]=True;selected[-1]=True
    def capture(_module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        if hidden.shape[1]>max(qpositions):
            query=F.layer_norm(hidden[:,qpositions].float().mean(1,keepdim=True),(hidden.shape[-1],))
            result=reader(query,*bank);state['audit']=result
            supports=torch.stack([x.support[0,0] for x in result.hops]);gates=torch.stack([x.null_gate[0,0] for x in result.hops])
            state['mass']=(supports*gates[:,None]).sum(0)/supports.sum().clamp_min(1e-8)
        return output
    def dispatch(module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
        ordinary,_=original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
        if id(module) not in late or 'mass' not in state or gain==0:return ordinary,None
        if dropout:raise ValueError('Independent KV prototype is inference-only')
        length,klen=query.shape[-2],key.shape[-2]
        rows=torch.where(selected.to(query.device))[0] if length==len(selected) else torch.arange(length,device=query.device)
        if not len(rows):return ordinary,None
        global_rows=rows+(klen-length)
        pos=positions.to(query.device)[:,:,rows] if length==len(selected) else positions[:,:,-1:].to(query.device)+(klen-len(selected))
        if not (torch.equal(pos[0],pos[1]) and torch.equal(pos[0],pos[2])):raise ValueError('Expected ordinary text RoPE coordinates')
        dummy=torch.zeros(1,len(rows),reader.dim,device=query.device,dtype=query.dtype)
        cos,sin=MODEL_.model.rotary_emb(dummy,pos)
        content_query=inverse_text_rope(query[...,rows,:],cos[0].unsqueeze(1).to(query.device),sin[0].unsqueeze(1).to(query.device))
        dtype=module.k_proj.weight.dtype
        keys=module.k_proj(bank[0].to(query.device,dtype));vals=module.v_proj(address_values.unsqueeze(0).to(query.device,dtype))
        dim=query.shape[-1]
        keys=keys.reshape(1,len(nodes),-1,dim).transpose(1,2);vals=vals.reshape(1,len(nodes),-1,dim).transpose(1,2)
        mass=state['mass'].to(query.device)*gain*(video|audio).sum().to(query.device)
        log_mass=torch.where(mass>0,torch.log(mass),torch.full_like(mass,float('-inf'))).view(1,1,1,-1)
        mask=None
        if attention_mask is not None:
            mask=attention_mask[...,:klen]
            if mask.shape[-2]!=1:mask=mask[...,rows,:]
        else:
            causal=(length>1 and getattr(module,'is_causal',True)) if is_causal is None else bool(is_causal)
            if causal:mask=global_rows[:,None]>=torch.arange(klen,device=query.device)[None]
        replacement=joint_memory_read(query[...,rows,:],key,value,content_query,keys,vals,log_mass,mask,scaling)
        result=ordinary.clone();result[:,rows]=replacement.transpose(1,2)
        return result,None
    handle=MODEL_.model.layers[BIND_LAYER-1].register_forward_hook(capture)
    ALL_ATTENTION_FUNCTIONS.register('sdpa',dispatch)
    try:yield
    finally:handle.remove();ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

def avbind_kv_answer(video_path,question,answer_options=None,slot_cache=None,value_cache=None,gain=1.0):
    src=Path(video_path);started=time.time();duration=len(decode_audio(src))/16000
    nodes,features=latent_sensor_bank(src,duration,slot_cache);address_values=addressed_value_bank(src,nodes,value_cache)
    device=next(BIND_READER.parameters()).device;n=len(nodes)
    bank=(features.unsqueeze(0).to(device),torch.tensor([[0 if x['source']=='video' else 1 for x in nodes]],device=device,dtype=torch.long),
          torch.tensor([[x['interval'] for x in nodes]],device=device,dtype=torch.float32).reshape(1,n,2),torch.ones(1,n,device=device,dtype=torch.bool))
    media=prepare_video(src);audits=[]
    if answer_options is None:
        inp,qpos=latent_question_inputs(media,question);layout=media_layout(MODEL_,PROC,inp);state={}
        with independent_kv_attention(BIND_READER,bank,address_values,nodes,qpos,layout,state,gain),torch.no_grad():
            sequence=MODEL_.generate(**inp,use_audio_in_video=True,max_new_tokens=128,do_sample=False,use_cache=True)
        answer=clean_observation(TOK.decode(sequence[0,inp['input_ids'].shape[1]:],skip_special_tokens=True));scores=prediction=None
    else:
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp,qpos=latent_question_inputs(media,question,answer_options,perm);layout=media_layout(MODEL_,PROC,inp);state={}
            head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
            try:
                with independent_kv_attention(BIND_READER,bank,address_values,nodes,qpos,layout,state,gain),torch.no_grad():
                    logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,j in enumerate(perm):totals[j]+=lp[position]
            audits.append(dict(memory_mass=state['mass'].detach().cpu().tolist()))
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,prediction=prediction,scores=scores,nodes=nodes,audits=audits,
                metadata=BIND_METADATA,gain=gain,caption_text_in_context=False,external_values_modality_isolated=True,
                latency_sec=round(time.time()-started,2))

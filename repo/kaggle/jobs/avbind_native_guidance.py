"""R16: learned occurrence priors without replacing the native AV layout.

No prefix isolation, record serialization, caption, draft or added instruction.
Use the exact native benchmark input; only selected late Q/answer attention
rows receive a soft prior over existing physical media tokens. This isolates
the layout confound in R14, not a standalone novelty claim.
"""
import contextlib
import time
import numpy as np
import torch
import torch.nn.functional as F


def indexed_occurrence_bias(query,projected,indices,total_tokens):
    spans=[];cursor=0
    for index in indices:
        spans.append((cursor,cursor+len(index)));cursor+=len(index)
    packed,trace=occurrence_token_bias(query,projected,spans,cursor)
    bias=torch.zeros(len(query),total_tokens,device=packed.device)
    for (a,b),index in zip(spans,indices):bias[:,index.to(bias.device)]=packed[:,a:b]
    return bias,trace


def native_guidance_inputs(media,question,options=None,perm=None):
    body=question
    if options is not None:
        perm=tuple(range(len(options))) if perm is None else perm
        body+=' '+' '.join(f'({letter}) {options[index]}' for letter,index in zip('ABCD',perm))
    inp,_=qwen_inputs(PROC,body,media,'av');inp=move_inputs(inp,MODEL_)
    positions,_=MODEL_.get_rope_index(inp['input_ids'],None,inp.get('video_grid_thw'),
        inp['attention_mask'],True,inp['feature_attention_mask'].sum(-1),inp['video_second_per_grid'])
    inp['position_ids']=positions
    return inp


def native_guidance_bias(inp,media,question,query,offsets,nodes,raw):
    tokens={'video':TOK.convert_tokens_to_ids(PROC.video_token),'audio':TOK.convert_tokens_to_ids(PROC.audio_token)}
    ids=inp['input_ids'][0]
    records,_=source_record_indices(ids,inp['position_ids'],tokens,len(media['audio'][0])/16000)
    selected=[];retained=[]
    for record in records:
        matches=[i for i,node in enumerate(nodes) if node['source']==record['source'] and abs(node['start']-record['start'])<.01]
        if matches:selected.append(matches[0]);retained.append(record)
    device=next(GUIDE_ALIGNMENT.parameters()).device
    sources=torch.tensor([0 if nodes[i]['source']=='video' else 1 for i in selected],device=device)
    with torch.no_grad():projected=GUIDE_ALIGNMENT(raw[selected].to(device),sources)
    last=int(torch.where((ids==tokens['video'])|(ids==tokens['audio']))[0][-1])
    tail_ids=ids[last+1:].cpu().tolist();tail=TOK.decode(tail_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=TOK(tail,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==tail_ids
    begin=tail.index(question);end=begin+len(question);rows=[];mapped=[]
    for i,(a,b) in enumerate(encoded['offset_mapping']):
        if b<=begin or a>=end:continue
        a=max(0,a-begin);b=min(len(question),b-begin)
        weight=torch.tensor([max(0,min(b,y)-max(a,x)) for x,y in offsets])
        assert weight.sum()>0
        mapped.append((query*weight[:,None]).sum(0)/weight.sum());rows.append(last+1+i)
    mapped=torch.stack(mapped).to(device);indices=[r['indices'] for r in retained]
    bias,trace=indexed_occurrence_bias(mapped,projected,indices,len(ids))
    answer_bias=indexed_occurrence_bias(mapped.mean(0,keepdim=True),projected,indices,len(ids))[0]
    return rows,bias,answer_bias,indices,dict(gate=trace['gate'].cpu().tolist(),record_weights=trace['record_weights'].cpu().tolist(),
        record_addresses=[dict(source=r['source'],start=r['start'],end=r['end']) for r in retained])


@contextlib.contextmanager
def native_late_guidance(modules,rows,bias,answer_bias,indices,audit,gain=1):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa'];selected_modules={id(m):i for i,m in enumerate(modules)}
    total=bias.shape[-1]
    def dispatch(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
        ordinary,weights=original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
        if id(module) not in selected_modules or gain==0:return ordinary,weights
        if dropout:raise ValueError('Native guidance is inference-only')
        length,klen=query.shape[-2],key.shape[-2]
        if length==klen:
            chosen=torch.tensor(rows+[total-1],device=query.device)
            added=torch.cat((bias,answer_bias),dim=0).to(query.device,query.dtype)*gain
            global_rows=chosen
        elif length==1:
            chosen=torch.tensor([0],device=query.device);global_rows=torch.tensor([klen-1],device=query.device)
            added=F.pad(answer_bias,(0,klen-total)).to(query.device,query.dtype)*gain
        else:raise ValueError('Only full prefill or single-token decode supported')
        heads=query.shape[-3]
        if heads!=key.shape[-3]:
            groups=heads//key.shape[-3];key=key.repeat_interleave(groups,-3);value=value.repeat_interleave(groups,-3)
        allowed=torch.arange(klen,device=query.device)[None,:]<=global_rows[:,None]
        supplied=None
        if attention_mask is not None:
            supplied=attention_mask[...,:klen]
            if supplied.shape[-2]!=1:supplied=supplied[...,chosen,:]
        mask=added.masked_fill(~allowed,float('-inf'))
        if supplied is not None:
            mask=mask.masked_fill(~supplied,float('-inf')) if supplied.dtype==torch.bool else mask+supplied
        replacement=F.scaled_dot_product_attention(query[...,chosen,:],key,value,attn_mask=mask,scale=scaling,dropout_p=0)
        result=ordinary.clone();result[:,chosen]=replacement.transpose(1,2)
        # Measure actual last-row source-record attention, not just the prior.
        if module is modules[-1]:
            with torch.no_grad():
                logits=(query[0,:,-1:].float()@key[0].float().transpose(-1,-2)).squeeze(-2)*(scaling or query.shape[-1]**-.5)
                causal=torch.arange(klen,device=query.device)<=global_rows[-1]
                logits=logits.masked_fill(~causal,float('-inf'))
                if supplied is not None:
                    s=supplied[..., -1, :].reshape(-1,klen)
                    logits=logits.masked_fill(~s,float('-inf')) if s.dtype==torch.bool else logits+s
                before=logits.softmax(-1).mean(0);after=(logits+added[-1]).softmax(-1).mean(0)
                audit.append(dict(layer=36,actual_native_attention=True,
                    before=[float(before[index.to(before.device)].sum()) for index in indices],
                    after=[float(after[index.to(after.device)].sum()) for index in indices]))
        return result,None
    ALL_ATTENTION_FUNCTIONS.register('sdpa',dispatch)
    try:yield
    finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


def avbind_native_guidance_answer(video,question,options=None,slot_cache=None):
    started=time.time();nodes,raw=guided_raw_bank(video,slot_cache);query,offsets=question_content_tokens(question)
    media=prepare_video(video);audits=[];totals=np.zeros(len(options)) if options is not None else None
    modules=[layer.self_attn for layer in MODEL_.model.layers[28:]]
    permutations=rotations(len(options)) if options is not None else [None]
    for perm in permutations:
        inp=native_guidance_inputs(media,question,options,perm)
        rows,bias,answer_bias,indices,trace=native_guidance_bias(inp,media,question,query,offsets,nodes,raw);reads=[]
        ctx=native_late_guidance(modules,rows,bias,answer_bias,indices,reads)
        if options is None:answer,decode=witness_decode(inp,ctx)
        else:
            head=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
            try:
                with ctx,torch.no_grad():logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(options)]]
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):totals[index]+=lp[position]
        audits.append(dict(guidance=trace,measured_reads=reads))
    scores=totals.round(3).tolist() if totals is not None else None
    prediction=int(np.argmax(scores)) if scores is not None else None
    if options is not None:answer=options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,scores=scores,prediction=prediction,audits=audits,
        original_native_layout=True,no_prefix_isolation=True,question_type_routing=False,
        model_revision=GUIDE_METADATA['model_revision'],alignment_metadata=GUIDE_METADATA,
        latency_sec=round(time.time()-started,2),layers=[29,36],gain=2,caption_context=False)

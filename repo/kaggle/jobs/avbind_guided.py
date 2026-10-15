"""Native-trained semantic occurrence guidance for arbitrary AV questions.

Content keys are query/clock blind. Text query tokens retrieve soft occurrences
in the learned native alignment space; late decoder rows read the full native
source/time records, preserving the original media encoders. No generated
caption enters the inference context and no question/answer schema is parsed.
"""
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def guided_raw_bank(src,cache=None):
    duration=len(decode_audio(src))/16000
    metadata=dict(media_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),
        model_revision=GUIDE_METADATA['model_revision'],pair_source_sha256=GUIDE_METADATA['pair_source_sha256'],
        layer=28,question_used=False,clock_in_key=False)
    cache=Path(cache) if cache else None
    if cache and cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        assert saved['metadata']==metadata,'Grounding cache provenance mismatch'
        return saved['nodes'],saved['raw']
    nodes=[];features=[]
    for index in range(int(math.ceil(duration/2))):
        start,end=2*index,min(2*index+2,duration)
        if end-start<.2:continue
        media=prepare_video(cut(src,start,end,f'avbind_latent_{index}'))
        for source,mode in (('video','vision'),('audio','audio')):
            features.append(native_pair_feature(media,mode));nodes.append(dict(source=source,start=start,end=end))
    features=torch.stack(features)
    if cache:
        tmp=cache.with_suffix('.tmp');torch.save(dict(metadata=metadata,nodes=nodes,raw=features),tmp);tmp.replace(cache)
    return nodes,features


def question_content_tokens(question):
    messages=[dict(role='system',content=[dict(type='text',text=SYSTEM)]),
        dict(role='user',content=[dict(type='text',text='An independently observed fact: '+question)])]
    rendered=PROC.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    inp=PROC(text=rendered,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    encoded=TOK(rendered,return_offsets_mapping=True)
    assert encoded['input_ids']==inp['input_ids'][0].tolist()
    begin=rendered.index('An independently observed fact: ')+len('An independently observed fact: ');end=begin+len(question)
    chosen=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>begin and a<end]
    offsets=[(max(0,encoded['offset_mapping'][i][0]-begin),min(len(question),encoded['offset_mapping'][i][1]-begin)) for i in chosen]
    captured={}
    class Ready(Exception):pass
    def capture(module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['features']=F.layer_norm(hidden[0,chosen].float(),(hidden.shape[-1],)).detach().cpu();raise Ready()
    handle=MODEL_.model.layers[27].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_cache=False)
    except Ready:pass
    finally:handle.remove()
    return captured['features'],offsets


def map_query_features(inp,segments,question,features,offsets):
    tail_start=segments[-1][1]
    ids=inp['input_ids'][0,tail_start:].cpu().tolist()
    text=TOK.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=TOK(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids
    begin=text.index(question,text.index('[USER QUESTION]')+len('[USER QUESTION]'));end=begin+len(question)
    positions=[];mapped=[]
    for index,(a,b) in enumerate(encoded['offset_mapping']):
        if b<=begin or a>=end:continue
        a=max(0,a-begin);b=min(len(question),b-begin)
        weight=torch.tensor([max(0,min(b,y)-max(a,x)) for x,y in offsets],dtype=torch.float32)
        if weight.sum()==0:raise ValueError('Question token character correspondence failed')
        mapped.append((features*weight[:,None]).sum(0)/weight.sum());positions.append(tail_start+index)
    return positions,torch.stack(mapped)


def guided_tail_provider(inp,segments,audit,question,query,offsets,nodes,raw):
    device=next(GUIDE_ALIGNMENT.parameters()).device
    selected=[];spans=[]
    for (start,end),address in zip(segments,audit['record_addresses']):
        matches=[i for i,node in enumerate(nodes) if node['source']==address['source'] and abs(node['start']-address['start'])<.01]
        if matches:selected.append(matches[0]);spans.append((start,end))
    if len(spans)<2:raise ValueError('Too few source-record correspondences')
    sources=torch.tensor([0 if nodes[i]['source']=='video' else 1 for i in selected],device=device)
    with torch.no_grad():projected=GUIDE_ALIGNMENT(raw[selected].to(device),sources)
    qpositions,mapped=map_query_features(inp,segments,question,query,offsets)
    bias,trace=occurrence_token_bias(mapped.to(device),projected,spans,inp['input_ids'].shape[1])
    answer_bias=occurrence_token_bias(mapped.mean(0,keepdim=True).to(device),projected,spans,inp['input_ids'].shape[1])[0][0]
    late={id(layer.self_attn) for layer in MODEL_.model.layers[28:]}
    dictionary={position:index for index,position in enumerate(qpositions)}
    total=inp['input_ids'].shape[1]
    def provider(module,rows,keylen):
        if id(module) not in late:return None
        result=torch.zeros(len(rows),keylen,device=bias.device)
        for index,row in enumerate(rows.cpu().tolist()):
            if row in dictionary:result[index,:total]=bias[dictionary[row]]
            elif row>=total-1:result[index,:total]=answer_bias
        return result
    return provider,dict(question_token_gate=trace['gate'].cpu().tolist(),record_weights=trace['record_weights'].cpu().tolist(),
        mapped_records=[audit['record_addresses'][i] for i,span in enumerate(segments) if span in spans])


def avbind_guided_answer(video_path,question,answer_options=None,slot_cache=None):
    started=time.time();nodes,raw=guided_raw_bank(video_path,slot_cache)
    query,offsets=question_content_tokens(question);media=prepare_video(video_path);audits=[]
    decoder=[layer.self_attn for layer in MODEL_.model.layers]
    if answer_options is None:
        inp,segments,audit=context_clock_inputs(media,question,physical_clock=False)
        provider,trace=guided_tail_provider(inp,segments,audit,question,query,offsets,nodes,raw)
        length=inp['input_ids'].shape[1];mask=inp['attention_mask'];tokens=[];stops={TOK.eos_token_id,TOK.convert_tokens_to_ids('<|im_end|>')}
        head=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
        try:
            with timeline_isolation(segments,audit['prefix_end'],decoder_modules=decoder,tail_bias=provider),torch.no_grad():
                output=MODEL_(**inp,use_audio_in_video=True,use_cache=True);cache=output.past_key_values;position=inp['position_ids'][:,:,-1:]+1
                for step in range(128):
                    token=int(output.logits[0,-1].argmax())
                    if token in stops:break
                    tokens.append(token);mask=torch.cat((mask,torch.ones_like(mask[:,-1:])),dim=1)
                    output=MODEL_(input_ids=torch.tensor([[token]],device=inp['input_ids'].device),attention_mask=mask,position_ids=position,
                        past_key_values=cache,cache_position=torch.tensor([length+step],device=inp['input_ids'].device),use_audio_in_video=True,use_cache=True)
                    cache=output.past_key_values;position=position+1
        finally:head.remove()
        answer=TOK.decode(tokens,skip_special_tokens=True).strip();scores=prediction=None;audits.append(dict(layout=audit,guidance=trace))
    else:
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp,segments,audit=context_clock_inputs(media,question,answer_options,perm,False)
            provider,trace=guided_tail_provider(inp,segments,audit,question,query,offsets,nodes,raw)
            head=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
            try:
                with timeline_isolation(segments,audit['prefix_end'],decoder_modules=decoder,tail_bias=provider),torch.no_grad():
                    logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):totals[index]+=lp[position]
            audits.append(dict(layout=audit,guidance=trace))
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,scores=scores,prediction=prediction,audits=audits,metadata=GUIDE_METADATA,
        latency_sec=round(time.time()-started,2),question_type_routing=False,generated_caption_context=False,layers=[29,36],gain=2)

"""Context-preserving source encoders, occurrence-indexed native decoder records.

Encode each original modality once, without cutting speech into short files.
Place all native media tokens into source/clock-addressed records; retain native
within-source order and all feature rows. Decoder records are locally isolated,
while the query can access every record. Encoder states may carry context from
other times of their OWN modality: this is not strict temporal-crop isolation.
Matched serial-clock / physical-clock ablations have identical input features,
token IDs, headers and masks. No benchmark schema or question routing is used.
"""
import math
import time
import numpy as np
import torch


def source_record_indices(ids, positions, source_tokens, duration, interval=2):
    if duration <= 0 or interval <= 0:
        raise ValueError('Positive duration/interval required')
    media=torch.zeros_like(ids,dtype=torch.bool)
    for token_id in source_tokens.values():media |= ids==token_id
    if not media.any():raise ValueError('No native media')
    origin=positions[0,0,media].min()
    seconds=(positions[0,0].float()-origin)/25
    buckets=torch.floor(seconds/interval).long()
    records=[]
    for bucket in range(int(buckets[media].max())+1):
        start=bucket*interval;end=min(start+interval,duration)
        if end <= start:raise ValueError('Native tokens exceed physical recording duration')
        for source,token_id in source_tokens.items():
            indices=torch.where((ids==token_id)&(buckets==bucket))[0]
            if len(indices):records.append(dict(source=source,start=start,end=end,indices=indices))
    for source,token_id in source_tokens.items():
        gathered=torch.cat([r['indices'] for r in records if r['source']==source])
        if not torch.equal(gathered,torch.where(ids==token_id)[0]):
            raise ValueError('Native feature order or coverage changed')
    return records,origin


def serialize_record_positions(positions,segments,prefix_end):
    result=positions.clone();cursor=prefix_end
    for start,end in segments:
        result[:,:,start:end]+=cursor-int(result[:,:,start:end].min())
        cursor=int(result[:,:,start:end].max())+1
    tail=segments[-1][1]
    result[:,:,tail:]=(cursor+torch.arange(result.shape[-1]-tail,device=result.device))[None,None,:]
    return result


def context_clock_inputs(media,question,options=None,perm=None,physical_clock=True,auxiliary_text=None):
    native,_=qwen_inputs(PROC,'',media,'av')
    original_ids=native['input_ids'][0]
    positions,_=MODEL_.get_rope_index(native['input_ids'],None,native.get('video_grid_thw'),
        native['attention_mask'],True,native['feature_attention_mask'].sum(-1),native['video_second_per_grid'])
    duration=len(media['audio'][0])/16000
    tokens={'video':TOK.convert_tokens_to_ids(PROC.video_token),'audio':TOK.convert_tokens_to_ids(PROC.audio_token)}
    records,old_origin=source_record_indices(original_ids,positions,tokens,duration)
    marker='[NATIVE_SOURCE_RECORDS]'
    system=SYSTEM+' Source observations have original-recording clocks, not necessarily full event boundaries. Heard mentions are not direct visual evidence.'
    body=('\n[MODEL-GENERATED DRAFT: fallible, not an additional observation]\n'+auxiliary_text if auxiliary_text else '')
    body+='\n[USER QUESTION]\n'+question
    if options is not None:
        perm=tuple(range(len(options))) if perm is None else perm
        body+=' '+' '.join(f'({letter}) {options[i]}' for letter,i in zip('ABCD',perm))
    rendered=PROC.apply_chat_template([dict(role='system',content=[dict(type='text',text=system)]),
        dict(role='user',content=[dict(type='text',text=marker+body)])],add_generation_prompt=True,tokenize=False)
    prefix,tail=rendered.split(marker,1)
    prefix_ids=TOK.encode(prefix,add_special_tokens=False);tail_ids=TOK.encode(tail,add_special_tokens=False)
    parsed=[]
    for index,record in enumerate(records):
        source,start,end=record['source'],record['start'],record['end']
        bos=PROC.vision_bos_token if source=='video' else PROC.audio_bos_token
        eos=PROC.vision_eos_token if source=='video' else PROC.audio_eos_token
        header=TOK.encode(f'\n[Observation {source[0].upper()}{index:03d}: {source}, {start:g} to {end:g} seconds]\n'+bos,add_special_tokens=False)
        suffix=TOK.encode(eos+f'\n[End of {source} observation {start:g} to {end:g} seconds]\n',add_special_tokens=False)
        parsed.append((record,header,suffix))
    origin=len(prefix_ids)+max(len(header) for _,header,_ in parsed)+1
    id_parts=[torch.tensor(prefix_ids,dtype=torch.long)]
    position_parts=[torch.arange(len(prefix_ids)).view(1,-1).expand(3,-1)]
    segments=[];cursor=len(prefix_ids)
    for record,header,suffix in parsed:
        start,end=record['start'],record['end'];indices=record['indices']
        raw_positions=positions[:,0,indices]-old_origin+origin
        first=int(raw_positions[0,0])
        hp=(first-len(header)+torch.arange(len(header))).view(1,-1).expand(3,-1)
        ep=(int(raw_positions.max())+1+torch.arange(len(suffix))).view(1,-1).expand(3,-1)
        id_parts.extend((torch.tensor(header),original_ids[indices],torch.tensor(suffix)))
        position_parts.extend((hp,raw_positions,ep))
        count=len(header)+len(indices)+len(suffix);segments.append((cursor,cursor+count));cursor+=count
    query_origin=max(int(part.max()) for part in position_parts)+1
    id_parts.append(torch.tensor(tail_ids,dtype=torch.long))
    position_parts.append((query_origin+torch.arange(len(tail_ids))).view(1,-1).expand(3,-1))
    ids=torch.cat(id_parts)[None,:]
    clock=torch.cat(position_parts,dim=-1)[:,None,:]
    if not physical_clock:clock=serialize_record_positions(clock,segments,len(prefix_ids))
    for token_id in tokens.values():
        assert int((ids==token_id).sum())==int((original_ids==token_id).sum())
    native['input_ids']=ids;native['attention_mask']=torch.ones_like(ids);native['position_ids']=clock
    audit=dict(tokens=ids.shape[-1],records=len(segments),video_tokens=int((ids==tokens['video']).sum()),
        audio_tokens=int((ids==tokens['audio']).sum()),prefix_end=len(prefix_ids),physical_clock=physical_clock,
        source_encoder_context='original full source',modality_feature_order_preserved=True,
        record_addresses=[dict(source=r['source'],start=r['start'],end=r['end']) for r in records])
    return move_inputs(native,MODEL_),segments,audit


def avbind_context_clock_answer(video_path,question,answer_options=None,physical_clock=True):
    started=time.time();media=prepare_video(video_path);audits=[];decoder=[layer.self_attn for layer in MODEL_.model.layers]
    if answer_options is None:
        inp,segments,audit=context_clock_inputs(media,question,physical_clock=physical_clock)
        length=inp['input_ids'].shape[1];mask=inp['attention_mask'];tokens=[]
        stops={TOK.eos_token_id,TOK.convert_tokens_to_ids('<|im_end|>')}
        head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
        try:
            with timeline_isolation(segments,audit['prefix_end'],decoder_modules=decoder),torch.no_grad():
                output=MODEL_(**inp,use_audio_in_video=True,use_cache=True);cache=output.past_key_values
                next_position=inp['position_ids'][:,:,-1:]+1
                for step in range(128):
                    token=int(output.logits[0,-1].argmax())
                    if token in stops:break
                    tokens.append(token);mask=torch.cat((mask,torch.ones_like(mask[:,-1:])),dim=1)
                    output=MODEL_(input_ids=torch.tensor([[token]],device=inp['input_ids'].device),
                        attention_mask=mask,position_ids=next_position,past_key_values=cache,
                        cache_position=torch.tensor([length+step],device=inp['input_ids'].device),use_audio_in_video=True,use_cache=True)
                    cache=output.past_key_values;next_position=next_position+1
        finally:head.remove()
        answer=TOK.decode(tokens,skip_special_tokens=True).strip();scores=prediction=None;audits.append(audit)
    else:
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp,segments,audit=context_clock_inputs(media,question,answer_options,perm,physical_clock)
            head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
            try:
                with timeline_isolation(segments,audit['prefix_end'],decoder_modules=decoder),torch.no_grad():
                    logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):totals[index]+=lp[position]
            audits.append(audit)
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,scores=scores,prediction=prediction,audits=audits,physical_clock=physical_clock,
        latency_sec=round(time.time()-started,2),representation='context-preserved native source records',question_type_routing=False)

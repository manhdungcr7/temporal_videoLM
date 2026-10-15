"""R17: native-context grounded, per-token occurrence addressing.

Unlike R14/R16 text-only retrieval, each query state has already read BOTH
original sources. Capture layer28 states, route to independently encoded
content keys, then bias native media reads at layers29-36. Generated answer
tokens update their own route. No question taxonomy, answer parser, forced
source-order planner, observation text or record-layout replacement.
"""
import contextlib
import time
import numpy as np
import torch
import torch.nn.functional as F


def grounded_record_bank(inp,media,nodes,raw):
    ids=inp['input_ids'][0]
    tokens={'video':TOK.convert_tokens_to_ids(PROC.video_token),'audio':TOK.convert_tokens_to_ids(PROC.audio_token)}
    records,_=source_record_indices(ids,inp['position_ids'],tokens,len(media['audio'][0])/16000)
    retained=[];selected=[]
    for record in records:
        matches=[i for i,node in enumerate(nodes) if node['source']==record['source'] and abs(node['start']-record['start'])<.01]
        if matches:retained.append(record);selected.append(matches[0])
    sources=torch.tensor([0 if nodes[i]['source']=='video' else 1 for i in selected],device=raw.device)
    device=next(GUIDE_ALIGNMENT.parameters()).device
    with torch.no_grad():projected=GUIDE_ALIGNMENT(raw[selected].to(device),sources.to(device))
    return projected,[r['indices'] for r in retained],[dict(source=r['source'],start=r['start'],end=r['end']) for r in retained]


def grounded_question_rows(inp,question):
    ids=inp['input_ids'][0]
    audio=TOK.convert_tokens_to_ids(PROC.audio_token);video=TOK.convert_tokens_to_ids(PROC.video_token)
    last=int(torch.where((ids==audio)|(ids==video))[0][-1])
    tail_ids=ids[last+1:].cpu().tolist();tail=TOK.decode(tail_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=TOK(tail,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==tail_ids
    begin=tail.index(question);end=begin+len(question)
    return [last+1+i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>begin and a<end]


@contextlib.contextmanager
def grounded_occurrence_reads(inp,question,projected,indices,trace,reads):
    rows=grounded_question_rows(inp,question);total=inp['input_ids'].shape[-1]
    device=projected.device;bias=torch.zeros(len(rows),total,device=device);answer_bias=torch.zeros(1,total,device=device)
    def capture(module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        if hidden.shape[1]==total:
            features=F.layer_norm(hidden[0,rows].float(),(hidden.shape[-1],)).to(device)
            new_bias,query_audit=indexed_occurrence_bias(features,projected,indices,total)
            bias.copy_(new_bias)
            trace.append(dict(kind='native_multimodal_question',gate=query_audit['gate'].cpu().tolist(),
                record_weights=query_audit['record_weights'].cpu().tolist()))
        elif hidden.shape[1]!=1:raise ValueError('Unexpected cached state length')
        answer=F.layer_norm(hidden[0,-1:].float(),(hidden.shape[-1],)).to(device)
        new_answer,answer_audit=indexed_occurrence_bias(answer,projected,indices,total)
        answer_bias.copy_(new_answer)
        trace.append(dict(kind='current_answer_state',gate=answer_audit['gate'].cpu().tolist(),
            record_weights=answer_audit['record_weights'].cpu().tolist()))
        return output
    handle=MODEL_.model.layers[27].register_forward_hook(capture)
    modules=[layer.self_attn for layer in MODEL_.model.layers[28:]]
    try:
        with native_late_guidance(modules,rows,bias,answer_bias,indices,reads):yield
    finally:handle.remove()


def avbind_grounded_guidance_answer(video,question,options=None,slot_cache=None):
    started=time.time();nodes,raw=guided_raw_bank(video,slot_cache);media=prepare_video(video)
    totals=np.zeros(len(options)) if options is not None else None;audits=[]
    permutations=rotations(len(options)) if options is not None else [None]
    for perm in permutations:
        inp=native_guidance_inputs(media,question,options,perm)
        projected,indices,addresses=grounded_record_bank(inp,media,nodes,raw);trace=[];reads=[]
        ctx=grounded_occurrence_reads(inp,question,projected,indices,trace,reads)
        if options is None:answer,decode=witness_decode(inp,ctx)
        else:
            head=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
            try:
                with ctx,torch.no_grad():logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(options)]]
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):totals[index]+=lp[position]
        audits.append(dict(record_addresses=addresses,grounded_routes=trace,measured_reads=reads))
    scores=totals.round(3).tolist() if totals is not None else None
    prediction=int(np.argmax(scores)) if scores is not None else None
    if options is not None:answer=options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,scores=scores,prediction=prediction,audits=audits,
        original_native_layout=True,native_multimodal_query=True,current_answer_token_routing=True,
        question_type_routing=False,caption_context=False,layers=[29,36],gain=2,
        alignment_metadata=GUIDE_METADATA,latency_sec=round(time.time()-started,2))

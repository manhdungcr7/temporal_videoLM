"""Raw isolated modality slots, with no generated caption in the answer context.

Each content key sees one physical crop and one modality, a constant instruction,
and neither question, answer options nor clock. Source/clock are separate fields.
The answer model always retains original AV. This is an experimental transfer of
the existing reader, not a claim that textual supervision learns AV alignment.
"""
def latent_sensor_feature(media,mode):
    inp,_=qwen_inputs(PROC,'Observe this recording carefully and describe its directly available content.',media,mode)
    inp=move_inputs(inp,MODEL_);captured={}
    class SlotReady(Exception):pass
    def capture(_module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['feature']=F.layer_norm(hidden[0,-1].float(),(hidden.shape[-1],)).detach().cpu()
        raise SlotReady()
    handle=MODEL_.model.layers[BIND_LAYER-1].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_audio_in_video=False,use_cache=False)
    except SlotReady:pass
    finally:handle.remove()
    return captured['feature']

def latent_sensor_bank(src,duration,cache_path=None):
    cache=Path(cache_path) if cache_path else None
    expected=dict(revision=BIND_METADATA['revision'],layer=BIND_LAYER,
                  representation='isolated raw modality final prompt hidden state; no generation',
                  interval_sec=2,question_used=False,clock_in_content_key=False)
    if cache and cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        if saved['metadata']!=expected:raise ValueError('Raw slot cache metadata mismatch')
        return saved['nodes'],saved['features']
    nodes=[];features=[]
    for index in range(int(math.ceil(duration/2))):
        start,end=2*index,min(2*index+2,duration)
        if end-start<.2:continue
        crop=cut(src,start,end,f'avbind_latent_{index}');media=prepare_video(crop)
        for source,mode in (('video','vision'),('audio','audio')):
            features.append(latent_sensor_feature(media,mode))
            nodes.append(dict(id=f'{source[0]}{index}',source=source,interval=[start,end]))
        del media;torch.cuda.empty_cache()
        print('RAW_SLOT',json.dumps(dict(start=start,end=end,modality_isolated=True)),flush=True)
    features=torch.stack(features) if features else torch.zeros(0,MODEL_.get_input_embeddings().embedding_dim)
    if cache:
        temporary=cache.with_suffix('.tmp')
        torch.save(dict(nodes=nodes,features=features,metadata=expected),temporary);temporary.replace(cache)
    return nodes,features

def native_query_span(inp,question):
    """Offsets on the short decoded text tail, preserving multibyte Unicode."""
    ids=inp['input_ids'][0].tolist()
    media_ids={TOK.convert_tokens_to_ids(PROC.video_token),TOK.convert_tokens_to_ids(PROC.audio_token)}
    last=max(i for i,value in enumerate(ids) if value in media_ids)
    tail_ids=ids[last+1:]
    tail=TOK.decode(tail_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=TOK(tail,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=tail_ids:raise ValueError('Native text tail did not round-trip exactly')
    begin=tail.rindex(question);end=begin+len(question)
    result=[last+1+i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>begin and a<end]
    if not result:raise ValueError('Question feature span is empty')
    return result

def latent_question_inputs(media,question,options=None,perm=None):
    body=question
    if options is not None:
        perm=tuple(range(len(options))) if perm is None else perm
        body+=' '+ ' '.join(f'({letter}) {options[i]}' for letter,i in zip('ABCD',perm))
        # Match the native benchmark prompt; no extra answer instruction.
    inp,_=qwen_inputs(PROC,body,media,'av');inp=move_inputs(inp,MODEL_)
    return inp,native_query_span(inp,question)

def avbind_latent_answer(video_path,question,answer_options=None,slot_cache=None,enabled=True):
    src=Path(video_path);started=time.time();wave=decode_audio(src);duration=len(wave)/16000
    nodes,features=latent_sensor_bank(src,duration,slot_cache)
    device=next(BIND_READER.parameters()).device;n=len(nodes)
    bank=(features.unsqueeze(0).to(device),torch.tensor([[0 if x['source']=='video' else 1 for x in nodes]],device=device,dtype=torch.long),
          torch.tensor([[x['interval'] for x in nodes]],device=device,dtype=torch.float32).reshape(1,n,2),
          torch.ones(1,n,device=device,dtype=torch.bool))
    media=prepare_video(src);audits=[]
    if answer_options is None:
        inp,qpos=latent_question_inputs(media,question);state={}
        manager=binding_hook(BIND_READER,bank,qpos,state) if enabled else contextlib.nullcontext()
        with manager,torch.no_grad():
            seq=MODEL_.generate(**inp,use_audio_in_video=True,max_new_tokens=128,do_sample=False,use_cache=True)
        answer=clean_observation(TOK.decode(seq[0,inp['input_ids'].shape[1]:],skip_special_tokens=True))
        scores=prediction=None
    else:
        if not 2<=len(answer_options)<=4:raise ValueError('MC supports 2-4 options')
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp,qpos=latent_question_inputs(media,question,answer_options,perm);state={}
            manager=binding_hook(BIND_READER,bank,qpos,state) if enabled else contextlib.nullcontext()
            head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
            try:
                with manager,torch.no_grad():logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,j in enumerate(perm):totals[j]+=lp[position]
            if enabled:audits.append([dict(support=x.support.detach().cpu().tolist(),null=x.null_support.detach().cpu().tolist()) for x in state['audit'].hops])
            del inp
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,prediction=prediction,scores=scores,nodes=nodes,audits=audits,
                metadata=BIND_METADATA,source_sha256=globals().get('LATENT_SOURCE_SHA256'),enabled=enabled,raw_AV_retained=True,caption_text_in_context=False,
                inference_memory_features='isolated raw-modality hidden states',
                duration=round(duration,3),latency_sec=round(time.time()-started,2))

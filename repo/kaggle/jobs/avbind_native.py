"""Native raw-AV answer path augmented by the same generic binding reader.

No controller, task type, answer interpretation, or audio-first path. Raw
audio/video remain available to the native model for uncaptured details.
Local captions still supply an imperfect auxiliary memory in this prototype.
"""
import contextlib

def native_binding_inputs(media,nodes,question,options=None,perm=None):
    # Use exactly the same auxiliary evidence for enabled/disabled ablations.
    memory,addresses=binding_views(nodes)
    body=('The original audio and video are supplied. The following independent short observations '
          'are retrieval hints, not a replacement for the original recording. '
          'Each interval bounds an observation, not necessarily a complete event.\n'+memory+
          '\nUSER QUESTION:\n'+question+'\nAttached observation addresses:\n'+addresses+'\nAnswer choices:\n')
    if options is not None:
        perm=tuple(range(len(options))) if perm is None else perm
        body+=' '.join(f'({letter}) {options[i]}' for letter,i in zip('ABCD',perm))
        body+='\nGive only the letter of the best answer.'
    inp,_=qwen_inputs(PROC,body,media,'av');inp=move_inputs(inp,MODEL_)
    pieces=[TOK.decode([token]) for token in inp['input_ids'][0].tolist()]
    full=''.join(pieces);starts=np.cumsum([0]+[len(x) for x in pieces[:-1]])
    begin=full.rindex('\nUSER QUESTION:\n')+len('\nUSER QUESTION:\n')
    end=full.index('\nAttached observation addresses:\n',begin)
    positions=[i for i,a in enumerate(starts) if begin<=a<end]
    if not positions:raise RuntimeError('Native question span could not be located')
    return inp,positions

def native_binding_scores(media,nodes,question,options,enabled=True):
    if not 2<=len(options)<=4:raise ValueError('MC interface supports 2-4 options')
    bank=binding_bank(nodes,next(BIND_READER.parameters()).device)
    totals=np.zeros(len(options));audits=[]
    labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(options)]]
    for perm in rotations(len(options)):
        inp,qpos=native_binding_inputs(media,nodes,question,options,perm);state={}
        # Last-position projection only; no attention or position changes.
        head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
        manager=binding_hook(BIND_READER,bank,qpos,state) if enabled else contextlib.nullcontext()
        try:
            with manager,torch.no_grad():
                logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
        finally:head.remove()
        lp=torch.log_softmax(logits[labels],-1).numpy()
        for slot,j in enumerate(perm):totals[j]+=lp[slot]
        if enabled:
            audits.append([dict(support=h.support.detach().cpu().tolist(),
                                null=h.null_support.detach().cpu().tolist()) for h in state['audit'].hops])
        del inp
        torch.cuda.empty_cache()
    return totals.round(3).tolist(),audits

def avbind_native_answer(video_path,question,answer_options=None,observation_cache=None,
                         utterance_cache=None,enabled=True):
    started=time.time();src=Path(video_path);wave=decode_audio(src);duration=len(wave)/16000
    cache=Path(observation_cache) if observation_cache else None
    observations=json.loads(cache.read_text()) if cache and cache.exists() else extraction_memory(src,duration)
    if cache and not cache.exists():cache.write_text(json.dumps(observations),encoding='utf-8')
    ucache=Path(utterance_cache) if utterance_cache else None
    utterances=json.loads(ucache.read_text()) if ucache and ucache.exists() else utterance_nodes(src,wave,duration)
    if ucache and not ucache.exists():ucache.write_text(json.dumps(utterances),encoding='utf-8')
    nodes=avb_nodes(observations,utterances);media=prepare_video(src)
    if answer_options is None:
        inp,qpos=native_binding_inputs(media,nodes,question);state={}
        bank=binding_bank(nodes,next(BIND_READER.parameters()).device)
        manager=binding_hook(BIND_READER,bank,qpos,state) if enabled else contextlib.nullcontext()
        with manager,torch.no_grad():
            sequence=MODEL_.generate(**inp,use_audio_in_video=True,max_new_tokens=128,do_sample=False,use_cache=True)
        answer=clean_observation(TOK.decode(sequence[0,inp['input_ids'].shape[1]:],skip_special_tokens=True))
        scores=prediction=None;audits=[]
    else:
        scores,audits=native_binding_scores(media,nodes,question,answer_options,enabled)
        prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,prediction=prediction,scores=scores,nodes=nodes,audits=audits,
                enabled=enabled,raw_AV_retained=True,latency_sec=round(time.time()-started,2),
                duration=round(duration,3),metadata=BIND_METADATA,
                options_visible_to_evidence_reader=False,dataset_or_task_used=False)

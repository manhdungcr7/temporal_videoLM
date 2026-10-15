"""Experimental model-side binding reader over perceptual text observations.

This first integration uses frozen token-embedding features of independent
observations, NOT retained raw media-token states. It does not establish
raw-event grounding, multi-speaker binding, or universal reasoning.
"""
import contextlib,hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as F

def binding_bank(nodes,device):
    embedding=MODEL_.get_input_embeddings()
    vectors=[]
    with torch.no_grad():
        for node in nodes:
            ids=TOK.encode(node['content'],add_special_tokens=False) or [TOK.eos_token_id]
            tensor=torch.tensor(ids,device=embedding.weight.device)
            vectors.append(F.layer_norm(embedding(tensor).float().mean(0),(embedding.embedding_dim,)))
    dim=embedding.embedding_dim
    content=torch.stack(vectors).unsqueeze(0).to(device) if vectors else torch.zeros(1,0,dim,device=device)
    source=torch.tensor([[0 if n['source']=='video' else 1 for n in nodes]],device=device,dtype=torch.long)
    intervals=torch.tensor([[n['interval'] for n in nodes]],device=device,dtype=torch.float32).reshape(1,len(nodes),2)
    valid=torch.ones(1,len(nodes),device=device,dtype=torch.bool)
    return content,source,intervals,valid

def binding_prompt(nodes,question,options=None,perm=None):
    memory=avb_ledger(nodes,[],relations=False)
    body='Original-recording observations, each with its own source and time:\n'+memory+'\nUSER QUESTION:\n'+question
    marker='\nEVIDENCE_QUERY_BOUNDARY\n'
    body+=marker
    if options is not None:
        perm=tuple(range(len(options))) if perm is None else perm
        body+=' '.join(f'({letter}) {options[i]}' for letter,i in zip('ABCD',perm))
    messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
              {'role':'user','content':[{'type':'text','text':body}]}]
    chat=PROC.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    inp=PROC(text=chat,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    tokenized=TOK(chat,return_offsets_mapping=True)
    assert tokenized['input_ids']==inp['input_ids'][0].tolist()
    boundary=chat.index(marker)
    query_position=max(i for i,(a,b) in enumerate(tokenized['offset_mapping']) if b and b<=boundary)
    return inp,query_position

@contextlib.contextmanager
def binding_hook(reader,bank,query_position,state):
    def hook(_module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        if hidden.shape[1]>query_position:
            query=F.layer_norm(hidden[:,query_position:query_position+1].float(),(hidden.shape[-1],))
            state['query']=query
        else:
            query=state['query']
        result=reader(query,*bank)
        state['audit']=result
        # Query position is before options: evidence selection cannot inspect
        # their wording, order or label. Later decoder layers choose the answer.
        delta=result.residual.to(hidden.dtype)*hidden[:,-1:,:].detach().float().std(-1,keepdim=True).clamp_min(0.01).to(hidden.dtype)
        updated=hidden.clone();updated[:,-1:,:]=updated[:,-1:,:]+delta
        return (updated,*output[1:]) if isinstance(output,tuple) else updated
    handle=MODEL_.model.layers[BIND_LAYER-1].register_forward_hook(hook)
    try:yield
    finally:handle.remove()

def binding_scores(nodes,question,options,reader=None):
    if not 2<=len(options)<=4:raise ValueError('Current MC scorer supports 2-4 choices; free-form uses options=None')
    totals=np.zeros(len(options));audits=[]
    device=next(BIND_READER.parameters()).device
    bank=binding_bank(nodes,device)
    ids=torch.tensor([TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(options)]],device='cpu')
    for perm in rotations(len(options)):
        inp,qpos=binding_prompt(nodes,question,options,perm)
        state={}
        manager=binding_hook(reader,bank,qpos,state) if reader is not None else contextlib.nullcontext()
        with manager,torch.no_grad():
            logits=MODEL_(**inp,use_cache=False).logits[0,-1].float().cpu()
        lp=torch.log_softmax(logits[ids],-1).numpy()
        for slot,j in enumerate(perm):totals[j]+=lp[slot]
        if reader is not None:
            audits.append([dict(support=hop.support.detach().cpu().tolist(),
                                null=hop.null_support.detach().cpu().tolist()) for hop in state['audit'].hops])
        del inp
    return totals.round(3).tolist(),audits

def avbind_neural_answer(video_path,question,answer_options=None,observation_cache=None,
                         utterance_cache=None,enabled=True):
    started=time.time();src=Path(video_path);wave=decode_audio(src);duration=len(wave)/16000
    cache=Path(observation_cache) if observation_cache else None
    observations=json.loads(cache.read_text()) if cache and cache.exists() else extraction_memory(src,duration)
    if cache and not cache.exists():cache.write_text(json.dumps(observations),encoding='utf-8')
    ucache=Path(utterance_cache) if utterance_cache else None
    utterances=json.loads(ucache.read_text()) if ucache and ucache.exists() else utterance_nodes(src,wave,duration)
    if ucache and not ucache.exists():ucache.write_text(json.dumps(utterances),encoding='utf-8')
    nodes=avb_nodes(observations,utterances)
    reader=BIND_READER if enabled else None
    if answer_options is None:
        inp,qpos=binding_prompt(nodes,question);state={}
        bank=binding_bank(nodes,next(BIND_READER.parameters()).device)
        manager=binding_hook(reader,bank,qpos,state) if reader is not None else contextlib.nullcontext()
        with manager,torch.no_grad():
            sequence=MODEL_.generate(**inp,max_new_tokens=128,do_sample=False,use_cache=True)
        answer=clean_observation(TOK.decode(sequence[0,inp['input_ids'].shape[1]:],skip_special_tokens=True))
        scores=prediction=None;audits=[]
    else:
        scores,audits=binding_scores(nodes,question,answer_options,reader)
        prediction=int(np.argmax(scores));answer=answer_options[prediction]
    return dict(answer=answer,prediction=prediction,scores=scores,nodes=nodes,audits=audits,
                enabled=enabled,latency_sec=round(time.time()-started,2),
                duration=round(duration,3),metadata=BIND_METADATA,
                options_visible_to_evidence_reader=False,dataset_or_task_used=False)

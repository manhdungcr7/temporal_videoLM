"""Source-Time Counterfactual Adapter: a small, explicitly trained control.

Only query-tail Q/V projections are adapted. Native encoders/media-prefix KV
remain unchanged. Supervision is synthetic source/time evidence, NOT raw AV
alignment, benchmark questions, development clips, or development labels.
This pilot alone is not a CVPR novelty/generalization claim.

Server1 trains once. Server2 requires the exact exported checkpoint; never trains
a different checkpoint on its own shard. Original model weights remain frozen.
"""
import json,time,gc,random,hashlib,contextlib,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

R40_STCA_PATH=OUT/'research40_stca.pt'
if R40_SERVER==2 and not R40_STCA_PATH.exists():
    R40_STCA_PATH=OUT/'uploads/research40_stca.pt'
R40_STCA_FILE=OUT/f'research40_stca_s{R40_SERVER}.jsonl'
R40_STCA_CONFIG=dict(version=1,rank=8,scale=2.0,layers=list(range(18,35,2)),
                     seed=43117,train_pairs=64,epochs=2,lr=.0003)

def r40_synthetic_pairs(seed,count):
    rng=random.Random(seed);pairs=[]
    vocabulary=['opens a wooden drawer','sweeps the floor','closes a window',
        'waves a blue flag','pushes a shopping cart','picks up a red ball',
        'switches on a lamp','folds a towel','turns a metal handle',
        'ties a shoelace','lifts a suitcase','waters a plant','plays a drum',
        'places a book on a desk','opens an umbrella','pours water into a glass']
    for index in range(count):
        e0,e1,e2=rng.sample(vocabulary,3);family=index%8
        t0=rng.randint(0,9);d0=rng.randint(2,5);gap=rng.randint(7,12)
        t1=t0+d0+gap;d1=d0+1;t2=t1+d1+gap
        # Two counterfactual recordings: identical event identity/text, different
        # physical occurrence clocks. Speech tense deliberately stays unchanged.
        voice=rng.choice([t0,t1]) if family in (2,3) else (t0+t1)//2
        duration=3
        utterance=rng.choice(['Next, someone '+e0+'.','Earlier, someone '+e0+'.',
                             'I am describing how someone '+e0+'.'])
        contexts=[];truth=[];requests=[]
        for swap in (False,True):
            bounds=[(t1,t1+d1),(t0,t0+d0),(t2,t2+2)] if swap else [(t0,t0+d0),(t1,t1+d1),(t2,t2+2)]
            facts=[f"V{i} silent video [{a},{b}]s: A person {event}."
                   for i,((a,b),event) in enumerate(zip(bounds,[e0,e1,e2]))]
            facts+=[f'A0 audio speech [{voice},{voice+duration}]s: "{utterance}"',
                    f'A1 audio sound [{t2+4},{t2+6}]s: a short alarm tone.']
            rng.shuffle(facts)
            contexts.append('Independent source observations. Speech is a reported statement, '
                            'not evidence that its described action was visible at that time.\n'+'\n'.join(facts)+'\n\n')
            a,b=bounds[0]
            if family==0:
                question='Which of the visible actions starts first?';response='A person '+e0+'.'
                value=a==min(x[0] for x in bounds)
            elif family==1:
                question=f'Does the visible action where a person {e0} finish before the spoken statement begins?'
                response='Yes';value=b<voice
            elif family==2:
                question=f'What visible action is underway when the words "{utterance}" begin?'
                response='A person '+e0+'.';value=a<=voice<b
            elif family==3:
                question=f'What spoken words are heard while a person {e0}?'
                response=utterance;value=max(a,voice)<min(b,voice+duration)
            elif family==4:
                question=f'For how many seconds is the action "a person {e0}" observed?'
                response=f'{d0} seconds';value=b-a==d0
            elif family==5:
                question=f'Was the visible action "a person {e0}" already over when the statement was heard?'
                response='No';value=not(b<voice)
            elif family==6:
                question=f'Which visible action finishes later, "{e0}" or "{e1}"?'
                response=e0;value=b>bounds[1][1]
            else:
                # Retention control: content/source verification, not a clock
                # query. Both true and false answers occur with the same prompt.
                question='Which source records the short alarm tone?'
                response='audio' if not swap else 'silent video';value=not swap
            truth.append(value)
            requests.append((question,response))
        # Keep real counterfactual flips, not zero-information duplicate pairs.
        # Voice for simultaneity families coincides with one swapped event by
        # construction; all other temporal families flip under the clock swap.
        assert truth[0]!=truth[1],(family,truth,t0,t1,voice)
        pair=[]
        for context,value,(question,response) in zip(contexts,truth,requests):
            pair.append(dict(context=context,question=question,response=response,label=int(value),family=family))
        pairs.append(pair)
    return pairs

class R40LowRankQuery(nn.Module):
    def __init__(self,module):
        super().__init__();device=module.weight.device
        self.a=nn.Parameter(torch.empty(8,module.in_features,device=device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(module.out_features,8,device=device,dtype=torch.float32))
        nn.init.normal_(self.a,std=.01)
    def delta(self,x):return F.linear(F.linear(x.float(),self.a),self.b).to(x.dtype)*2

# Fail before touching model state if a synthetic counterfactual does not flip.
for r40_validation_seed in (43117,94109,100003):
    r40_synthetic_pairs(r40_validation_seed,256)
print('STCA_SYNTHETIC_INVARIANTS_OK',flush=True)
R40_STCA_ADAPTERS={};R40_STCA_GATE=None;R40_STCA_ENABLED=False
torch.manual_seed(R40_STCA_CONFIG['seed'])
for layer in R40_STCA_CONFIG['layers']:
    for projection in ('q_proj','v_proj'):
        module=getattr(MODEL_.model.layers[layer].self_attn,projection)
        R40_STCA_ADAPTERS[f'{layer}.{projection}']=R40LowRankQuery(module)

@contextlib.contextmanager
def r40_stca_active(gate):
    global R40_STCA_GATE,R40_STCA_ENABLED
    assert not R40_STCA_ENABLED
    handles=[];R40_STCA_GATE=gate;R40_STCA_ENABLED=True
    def make_hook(adapter):
        def hook(module,args,output):
            delta=adapter.delta(args[0]);g=R40_STCA_GATE
            if g is not None:
                assert g.shape[1]==delta.shape[1],(g.shape,delta.shape)
                delta=delta*g.to(delta.device,dtype=delta.dtype)
            return output+delta
        return hook
    for key,adapter in R40_STCA_ADAPTERS.items():
        layer,projection=key.split('.')
        handles.append(getattr(MODEL_.model.layers[int(layer)].self_attn,projection).register_forward_hook(make_hook(adapter)))
    try:yield
    finally:
        for h in handles:h.remove()
        R40_STCA_GATE=None;R40_STCA_ENABLED=False

def r40_stca_tokens(example):
    prompt=example['context']+r40_critic_request(example['question'],example['response'])
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[{'type':'text','text':prompt}]}]
    rendered=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    ids=PROC.tokenizer.encode(rendered,add_special_tokens=False)
    prefix=len(PROC.tokenizer.encode(rendered[:rendered.index('Original user question:')],add_special_tokens=False))
    return ids,prefix

def r40_stca_synthetic_logits(example,adapted):
    ids,prefix=r40_stca_tokens(example);device=MODEL_.get_input_embeddings().weight.device
    input_ids=torch.tensor([ids],device=device);positions=torch.arange(len(ids),device=device)[None,None,:].repeat(3,1,1)
    mask=torch.ones((1,len(ids)),dtype=torch.long,device=device)
    gate=torch.zeros((1,len(ids),1),device=device);gate[:,prefix:]=1
    ctx=r40_stca_active(gate) if adapted else contextlib.nullcontext()
    with ctx:
        output=MODEL_.model(inputs_embeds=MODEL_.get_input_embeddings()(input_ids),position_ids=positions,
                            attention_mask=mask,use_cache=False,return_dict=True)
        last=output.last_hidden_state[:,-1:].to(MODEL_.lm_head.weight.device)
        # Restrict projection to the two generic judgement tokens, not the full
        # vocabulary over every position. Original head weights remain frozen.
        weights=MODEL_.lm_head.weight[[R40_NO[0],R40_YES[0]]]
        logits=F.linear(last,weights)[0,-1].float()
    return logits

def r40_stca_synthetic_eval(pairs,adapted):
    right=0;total=0;by_family={}
    with torch.no_grad():
        for pair in pairs:
            for ex in pair:
                ok=int(r40_stca_synthetic_logits(ex,adapted).argmax())==ex['label']
                right+=ok;total+=1
                cell=by_family.setdefault(str(ex['family']),dict(right=0,total=0));cell['right']+=ok;cell['total']+=1
    return dict(right=right,total=total,by_family=by_family)

if not R40_STCA_PATH.exists():
    assert R40_SERVER==1,'Server2 must use the exact checkpoint trained on server1.'
    train=r40_synthetic_pairs(43117,64);validation=r40_synthetic_pairs(94109,16)
    encoded=json.dumps(dict(train=train,validation=validation),sort_keys=True).encode()
    data_sha=hashlib.sha256(encoded).hexdigest()
    (OUT/'research40_stca_synthetic.json').write_bytes(encoded)
    print('STCA_DATA',data_sha,'synthetic only; no development labels',flush=True)
    base_flags=[p.requires_grad for p in MODEL_.parameters()]
    for p in MODEL_.parameters():p.requires_grad_(False)
    MODEL_.eval()
    params=[p for adapter in R40_STCA_ADAPTERS.values() for p in adapter.parameters()]
    optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=.01)
    history=[];started=time.time()
    try:
        before=r40_stca_synthetic_eval(validation,False)
        print('STCA_SYNTHETIC_BEFORE',json.dumps(before),flush=True)
        for epoch in range(2):
            ordering=list(range(len(train)));random.Random(43117+epoch).shuffle(ordering)
            for index in ordering:
                optimizer.zero_grad(set_to_none=True);loss_sum=0
                for example in train[index]:
                    logits=r40_stca_synthetic_logits(example,True)
                    target=torch.tensor([example['label']],device=logits.device)
                    loss=F.cross_entropy(logits[None],target)/2
                    assert bool(torch.isfinite(loss)),loss
                    loss.backward();loss_sum+=float(loss.detach())
                # Params span two GPUs; use the non-foreach clipping path.
                torch.nn.utils.clip_grad_norm_(params,1.,foreach=False,error_if_nonfinite=True)
                optimizer.step();history.append(loss_sum)
                if len(history)%8==0:
                    print('STCA_TRAIN',len(history),'/128','loss',round(float(np.mean(history[-8:])),4),flush=True)
        after=r40_stca_synthetic_eval(validation,True)
        state=dict(config=R40_STCA_CONFIG,data_sha256=data_sha,synthetic_validation_before=before,
                   synthetic_validation_after=after,losses=history,seconds=round(time.time()-started,2),
                   weights={k:{name:v.detach().cpu() for name,v in adapter.state_dict().items()}
                            for k,adapter in R40_STCA_ADAPTERS.items()})
        torch.save(state,R40_STCA_PATH)
        (OUT/'research40_stca_training.json').write_text(json.dumps({k:v for k,v in state.items() if k!='weights'},indent=2))
        print('STCA_SYNTHETIC_AFTER',json.dumps(after),flush=True)
    finally:
        for p,flag in zip(MODEL_.parameters(),base_flags):p.requires_grad_(flag)
        del optimizer;gc.collect();torch.cuda.empty_cache()
else:
    state=torch.load(R40_STCA_PATH,map_location='cpu',weights_only=False)
    assert state['config']==R40_STCA_CONFIG
    for key,adapter in R40_STCA_ADAPTERS.items():adapter.load_state_dict(state['weights'][key])

def r40_stca_media_score(payload,question,response,supplement,adapted):
    prompt=r40_critic_request(question,response)
    # Native media-prefix computation is explicitly outside the adapter context.
    bank,prefix=r40_prefix(payload)
    tail=PROC.tokenizer.decode(payload['input_ids'][0,prefix:].tolist(),skip_special_tokens=False)
    beginning=tail[:tail.index(question)];ending=tail[tail.rindex('<|im_end|>'):]
    lead=beginning+supplement;ids=PROC.tokenizer.encode(lead+prompt+ending,add_special_tokens=False)
    cut=len(PROC.tokenizer.encode(lead,add_special_tokens=False))
    gate=torch.zeros((1,len(ids),1));gate[:,cut:]=1
    cache=DynamicCache(ddp_cache_data=r40_gpu_bank(bank),config=MODEL_.model.config)
    pos=payload['decoder']['position_ids'][:,:,prefix:prefix+1]
    with r40_stca_active(gate) if adapted else contextlib.nullcontext():
        logits,_,_,_=r40_cached_text(ids,cache,pos,prefix)
    return float(logits[R40_YES[0]]-logits[R40_NO[0]])

evidence_by_id={r['idx']:r for r in r40_json_rows(OUT/f'research40_verified_graph_v2_s{R40_SERVER}.jsonl')}
assert len(evidence_by_id)==len(R40_SAMPLES)
base_by_id={r['idx']:r for r in r40_json_rows(R40_BASE_FILE)}
records=r40_json_rows(R40_STCA_FILE);done={r['idx'] for r in records}
for row in R40_SAMPLES:
    if row['idx'] in done:continue
    started=time.time();question,responses=parse_question(row['question']);e=evidence_by_id[row['idx']]
    supplement=r40_compact_evidence(e['observations'],e['links'])
    payload=torch.load(R40_CACHE/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=row['idx']
    raw=[];text=[];frozen=[]
    for response in responses:
        raw.append(r40_stca_media_score(payload,question,response,supplement,True))
        example=dict(context=supplement,question=question,response=response)
        with torch.no_grad():
            lg=r40_stca_synthetic_logits(example,True);text.append(float(lg[1]-lg[0]))
            lg=r40_stca_synthetic_logits(example,False);frozen.append(float(lg[1]-lg[0]))
    rec=dict(idx=row['idx'],task=row['task'],gold=row['gold'],status='ok',
             manifest_sha256=R40_MANIFEST['manifest_sha256'],data_sha256=state['data_sha256'],
             scores=dict(stca_raw_verifier=raw,stca_text_verifier=text,frozen_text_verifier=frozen),
             seconds=round(time.time()-started,2))
    with R40_STCA_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(row['idx'])
    merged=[dict(r,scores=dict(base_by_id[r['idx']]['scores'],**r['scores'])) for r in records]
    r40_snapshot(merged,'stca');print('STCA_SAMPLE',json.dumps(rec),flush=True)
    del payload;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'stca_done');print('STCA_SUMMARY',json.dumps(RESULT),flush=True)

"""R4: cache frozen features, train address ranking before answer adaptation.

Uses only the existing constructed training corpus, never benchmark labels.
Its textual holdout is diagnostic only, not evidence of raw AV grounding.
"""
import gc,random,traceback
reader=NEURAL_NS['BIND_READER'];device=next(reader.parameters()).device
cache_path=OUT/'avbind_address_feature_cache.pt'
feature_rows=[];started=time.time()
for i,row in enumerate(train):
    inp,qpos=NEURAL_NS['binding_prompt'](row['nodes'],row['question'])
    captured={}
    class QueryReady(Exception):pass
    def capture(_module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['query']=torch.nn.functional.layer_norm(hidden[:,qpos].float().mean(1,keepdim=True),(hidden.shape[-1],)).detach().cpu()
        raise QueryReady()
    handle=MODEL_.model.layers[NEURAL_NS['BIND_LAYER']-1].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_cache=False)
    except QueryReady:pass
    finally:handle.remove()
    bank=NEURAL_NS['binding_bank'](row['nodes'],device)
    labels=[]
    for hop in range(2):
        ids=row['evidence'][f'hop{hop+1}']
        if hop==0 and ids:
            refs={(n['source'],n['content']) for n in row['nodes'] if n['id'] in ids}
            ids=[n['id'] for n in row['nodes'] if (n['source'],n['content']) in refs]
        labels.append(torch.tensor([float(n['id'] in ids) for n in row['nodes']]))
    feature_rows.append(dict(query=captured['query'],bank=tuple(x.detach().cpu() for x in bank),labels=torch.stack(labels)))
    if (i+1)%32==0:
        print('ADDRESS_CACHE',json.dumps(dict(n=i+1,total=len(train),elapsed=round(time.time()-started,1))),flush=True)
        torch.save(dict(rows=feature_rows,metadata=NEURAL_NS['BIND_METADATA']),cache_path)
    del inp;gc.collect()
torch.save(dict(rows=feature_rows,metadata=NEURAL_NS['BIND_METADATA']),cache_path)

def batch_rows(indices):
    selected=[feature_rows[i] for i in indices];n=max(x['bank'][0].shape[1] for x in selected);b=len(selected)
    dim=reader.dim
    queries=torch.cat([x['query'] for x in selected]).to(device)
    content=torch.zeros(b,n,dim,device=device);sources=torch.zeros(b,n,dtype=torch.long,device=device)
    intervals=torch.zeros(b,n,2,device=device);valid=torch.zeros(b,n,dtype=torch.bool,device=device)
    labels=torch.zeros(b,2,n,device=device)
    for j,row in enumerate(selected):
        length=row['bank'][0].shape[1]
        content[j,:length]=row['bank'][0][0].to(device);sources[j,:length]=row['bank'][1][0].to(device)
        intervals[j,:length]=row['bank'][2][0].to(device);valid[j,:length]=True
        labels[j,:,:length]=row['labels'].to(device)
    return queries,(content,sources,intervals,valid),labels

optimizer=torch.optim.AdamW(reader.parameters(),lr=3e-4,weight_decay=.01)
reader.train();rng=random.Random(2026101406);history=[]
try:
    for step in range(1500):
        q,bank,labels=batch_rows(rng.sample(range(len(feature_rows)),16));optimizer.zero_grad(set_to_none=True)
        result=reader(q,*bank);loss=torch.zeros((),device=device)
        for hop,audit in enumerate(result.hops):
            target=labels[:,hop];valid=bank[-1];positive=target.sum(-1);has_positive=positive>0
            # Ranking is auxiliary supervision, not a mutually exclusive
            # inference choice. Multiple true occurrences remain valid.
            logits=audit.support_logits[:,0].masked_fill(~valid,-1e4)
            distribution=target/positive[:,None].clamp_min(1)
            rank=-(distribution*torch.log_softmax(logits,-1)).sum(-1)
            bce=torch.nn.functional.binary_cross_entropy_with_logits(audit.support_logits[:,0],target,reduction='none')
            bce=(bce*valid).sum(-1)/valid.sum(-1).clamp_min(1)
            null=torch.nn.functional.binary_cross_entropy_with_logits(audit.null_logits[:,0],(~has_positive).float(),reduction='none')
            loss+=((rank*has_positive)+bce+null).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite address pretraining loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(reader.parameters(),1);optimizer.step()
        if (step+1)%50==0:
            rec=dict(step=step+1,loss=round(float(loss.detach()),4),elapsed=round(time.time()-started,1))
            history.append(rec)
            with (OUT/'avbind_address_pretraining_r4.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n');f.flush()
            print('ADDRESS_PRETRAIN',json.dumps(rec),flush=True)
        if (step+1)%250==0:
            torch.save(dict(state_dict=reader.state_dict(),step=step+1,metadata=NEURAL_NS['BIND_METADATA']),OUT/'avbind_address_latest_r4.pt')
    NEURAL_NS['BIND_METADATA']=dict(NEURAL_NS['BIND_METADATA'],candidate='r4 address pretraining',address_pretraining_steps=1500,address_pretraining_seed=2026101406)
    torch.save(dict(state_dict=reader.state_dict(),metadata=NEURAL_NS['BIND_METADATA']),OUT/'avbind_reader_checkpoint_r4_address.pt')
    RESULT=dict(status='address_pretrained',last=history[-1],metadata=NEURAL_NS['BIND_METADATA'])
    print('ADDRESS_PRETRAIN_FINISHED',json.dumps(RESULT),flush=True)
finally:
    reader.eval();torch.cuda.empty_cache()

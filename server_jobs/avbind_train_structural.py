"""Train only the binding adapter, with exact synthetic scene supervision.

No benchmark samples, task labels, answers, or teacher-model labels are read.
Synthetic success is NOT audiovisual QA success. Backbone weights stay frozen.
"""
import types,sys,random,traceback,gc
module=types.ModuleType('avbind_synthetic_runtime');sys.modules[module.__name__]=module
exec(compile((OUT/'uploads/avbind_synthetic.py').read_text(),'avbind_synthetic.py','exec'),module.__dict__)
TRAIN_SEED=2026101403
TRAIN_SIZE=256
STEPS=256
held=module.generate_corpus(38,TRAIN_SEED+1)
train=module.generate_corpus(TRAIN_SIZE,TRAIN_SEED)

def short_addresses(row):
    mapping={n['id']:f'n{j}' for j,n in enumerate(row['nodes'])}
    for n in row['nodes']:n['id']=mapping[n['id']]
    for hop,ids in row['evidence'].items():row['evidence'][hop]=[mapping[x] for x in ids]
    return row
train=[short_addresses(x) for x in train];held=[short_addresses(x) for x in held]
reader=NEURAL_NS['BIND_READER'];reader.train()
device=next(reader.parameters()).device
flags=[p.requires_grad for p in MODEL_.parameters()]
for p in MODEL_.parameters():p.requires_grad_(False)
optimizer=torch.optim.AdamW(reader.parameters(),lr=3e-4,weight_decay=0.01)
history=[]
path=OUT/'avbind_structural_training.jsonl'
started=time.time()
head_hook=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
try:
    for step in range(STEPS):
        row=train[step%len(train)]
        random_gen=random.Random(TRAIN_SEED+step)
        perm=list(range(len(row['answer_options'])));random_gen.shuffle(perm)
        # Inputs do not contain family, oracle query specs, gold or pointers.
        inp,qpos=NEURAL_NS['binding_prompt'](row['nodes'],row['question'],row['answer_options'],perm)
        bank=NEURAL_NS['binding_bank'](row['nodes'],device)
        state={};optimizer.zero_grad(set_to_none=True)
        with NEURAL_NS['binding_hook'](reader,bank,qpos,state):
            logits=MODEL_(**inp,use_cache=False).logits[0,-1].float()
        letters=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(perm)]]
        target=perm.index(row['gold'])
        answer_loss=torch.nn.functional.cross_entropy(logits[letters].unsqueeze(0),torch.tensor([target],device=logits.device))
        address_loss=torch.zeros((),device=device);null_loss=torch.zeros((),device=device)
        for index,audit in enumerate(state['audit'].hops):
            ids=row['evidence'][f'hop{index+1}']
            labels=torch.tensor([[[float(n['id'] in ids) for n in row['nodes']]]],device=device)
            positive=max(1,len(ids));weight=min(20,(len(row['nodes'])-len(ids))/positive)
            address_loss+=torch.nn.functional.binary_cross_entropy_with_logits(audit.support_logits,labels,
                                                                                 pos_weight=torch.tensor(weight,device=device))
            null_target=torch.tensor([[float(not ids)]],device=device)
            null_loss+=torch.nn.functional.binary_cross_entropy_with_logits(audit.null_logits,null_target)
        loss=answer_loss.to(device)+0.25*address_loss+0.1*null_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),1.0)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite adapter gradient')
        optimizer.step()
        rec=dict(step=step+1,loss=round(float(loss.detach()),4),answer_loss=round(float(answer_loss.detach()),4),
                 address_loss=round(float(address_loss.detach()),4),null_loss=round(float(null_loss.detach()),4),
                 family=row['metadata']['family'],elapsed_sec=round(time.time()-started,1))
        with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(rec)+'\n');handle.flush()
        history.append(rec)
        if (step+1)%16==0:print('BIND_TRAIN',json.dumps(rec),flush=True)
        if (step+1)%64==0:
            # Recoverable training state; the monitor downloads this off Kaggle.
            checkpoint=OUT/'avbind_training_latest.pt'
            temporary=checkpoint.with_suffix('.tmp')
            torch.save(dict(state_dict=reader.state_dict(),optimizer=optimizer.state_dict(),
                            step=step+1,seed=TRAIN_SEED,metadata=NEURAL_NS['BIND_METADATA']),temporary)
            temporary.replace(checkpoint)
        del inp,logits,loss,state;gc.collect()
    reader.eval()
    NEURAL_NS['BIND_METADATA'].update(trained=True,training='exact synthetic textual structure only',
                                    training_seed=TRAIN_SEED,training_size=TRAIN_SIZE,steps=STEPS,
                                    native_AV_supervision=False,benchmark_labels_used=False)
    torch.save(dict(state_dict=reader.state_dict(),metadata=NEURAL_NS['BIND_METADATA']),OUT/'avbind_reader_checkpoint.pt')
    (OUT/'avbind_structural_holdout.json').write_text(json.dumps(held,indent=2),encoding='utf-8')
    (OUT/'avbind_reader_metadata.json').write_text(json.dumps(NEURAL_NS['BIND_METADATA'],indent=2),encoding='utf-8')
    RESULT=dict(status='trained',steps=STEPS,elapsed_sec=round(time.time()-started,1),
                last_loss=history[-1],metadata=NEURAL_NS['BIND_METADATA'])
    print('BIND_TRAIN_FINISHED',json.dumps(RESULT),flush=True)
finally:
    head_hook.remove()
    for p,flag in zip(MODEL_.parameters(),flags):p.requires_grad_(flag)
    reader.eval();torch.cuda.empty_cache()

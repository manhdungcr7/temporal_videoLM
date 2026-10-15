"""Matched 512-step calibration for new pooled media keys, no QA supervision."""
import hashlib,types,sys,random
checkpoint=OUT/'avbind_native_alignment_r18.pt'
if checkpoint.exists():
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['metadata']['pair_source_sha256']==hashlib.sha256((OUT/'uploads/avbind_pooled_pairs.py').read_text().encode()).hexdigest()
    assert saved['metadata']['model_revision']==MREV
    RESULT=dict(file=checkpoint.name,metadata=saved['metadata'],already_completed=True)
    print('NATIVE_POOL_ALIGN_REUSED',json.dumps(RESULT),flush=True)
else:
    source=(OUT/'uploads/avbind_alignment.py').read_text();module=types.ModuleType('native_pooled_alignment_runtime');sys.modules[module.__name__]=module
    exec(compile(source,'avbind_alignment.py','exec'),module.__dict__)
    parts=[torch.load(OUT/'uploads'/f'avbind_pooled_pairs_s{s}.pt',map_location='cpu',weights_only=True) for s in (1,2)]
    assert parts[0]['metadata']['source_sha256']==parts[1]['metadata']['source_sha256']
    examples=[x for part in parts for x in part['examples']]
    train=[x for x in examples if x['split']=='train'];held=[x for x in examples if x['split']=='heldout']
    assert len(train)==202 and len(held)==50,'Incomplete new representation; finish extraction first'
    assert {x['recording_id'] for x in train}.isdisjoint({x['recording_id'] for x in held})
    excluded=set().union(*(set(part['metadata']['selection']['excluded_recording_ids']) for part in parts))
    assert not {x['recording_id'] for x in examples}&excluded
    torch.manual_seed(2026101411);rng=random.Random(2026101411)
    alignment=module.NativeSemanticAlignment().to('cuda:0')
    raw=torch.stack([x['raw'] for x in train]).to('cuda:0');text=torch.stack([x['text'] for x in train]).to('cuda:0')
    sources=torch.tensor([0 if x['source']=='video' else 1 for x in train],device='cuda:0')
    normalized=[x['content'].strip().lower() for x in train]
    with torch.no_grad():
        for s in (0,1):alignment.bias[s].copy_(text[sources==s].mean(0)-raw[sources==s].mean(0))
    optimizer=torch.optim.AdamW(alignment.parameters(),lr=1e-3,weight_decay=1e-3)
    for step in range(512):
        indices=rng.sample(range(len(train)),min(48,len(train)))
        positive=torch.tensor([[normalized[i]==normalized[j] for j in indices] for i in indices],device='cuda:0')
        loss=module.semantic_pair_loss(alignment(raw[indices],sources[indices]),text[indices],positive)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(alignment.parameters(),1);optimizer.step()
        if (step+1)%64==0:print('NATIVE_POOL_ALIGN_TRAIN',json.dumps(dict(step=step+1,loss=float(loss))),flush=True)
    alignment.eval()
    with torch.no_grad():
        hr=torch.stack([x['raw'] for x in held]).to('cuda:0');ht=torch.stack([x['text'] for x in held]).to('cuda:0')
        hs=torch.tensor([0 if x['source']=='video' else 1 for x in held],device='cuda:0');correct={}
        for name,features in (('unadapted',hr),('aligned',alignment(hr,hs))):
            match=(torch.nn.functional.normalize(ht,dim=-1)@torch.nn.functional.normalize(features,dim=-1).T).argmax(-1).cpu().tolist()
            correct[name]=sum(held[i]['content'].strip().lower()==held[j]['content'].strip().lower() for i,j in enumerate(match))
    metadata=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),pair_source_sha256=parts[0]['metadata']['source_sha256'],
        model_revision=MREV,dim=2048,rank=64,seed=2026101411,steps=512,benchmark_QA_used=False,train_pairs=202,held_pairs=50,
        representation='mean native media-token hidden states',pseudo_description_match_correct=correct,
        excluded_recording_ids=sorted(excluded),train_recording_ids=sorted({x['recording_id'] for x in train}),
        held_recording_ids=sorted({x['recording_id'] for x in held}),warning='Pseudo-description retrieval, NOT temporal QA or accurate event timing')
    temporary=checkpoint.with_suffix('.tmp');torch.save(dict(state_dict=alignment.state_dict(),metadata=metadata),temporary);temporary.replace(checkpoint)
    (OUT/'avbind_native_alignment_r18.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    RESULT=dict(file=checkpoint.name,metadata=metadata)
    print('NATIVE_POOL_ALIGN_DONE',json.dumps(RESULT),flush=True)

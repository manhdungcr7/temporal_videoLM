"""Fixed-budget native semantic calibration; held recording groups never train."""
import hashlib,types,sys,random
source=(OUT/'uploads/avbind_alignment.py').read_text()
module=types.ModuleType('native_alignment_runtime');sys.modules[module.__name__]=module
exec(compile(source,'avbind_alignment.py','exec'),module.__dict__)
parts=[torch.load(OUT/'uploads'/f'avbind_native_pairs_s{s}.pt',map_location='cpu',weights_only=True) for s in (1,2)]
assert parts[0]['metadata']['source_sha256']==parts[1]['metadata']['source_sha256']
examples=[x for part in parts for x in part['examples']]
train=[x for x in examples if x['split']=='train'];held=[x for x in examples if x['split']=='heldout']
assert {x['recording_id'] for x in train}.isdisjoint({x['recording_id'] for x in held})
excluded=set(parts[0]['metadata']['selection']['excluded_recording_ids'])|set(parts[1]['metadata']['selection']['excluded_recording_ids'])
assert not {x['recording_id'] for x in examples}&excluded
torch.manual_seed(2026101411);rng=random.Random(2026101411)
alignment=module.NativeSemanticAlignment().to('cuda:0')
raw=torch.stack([x['raw'] for x in train]).to('cuda:0');text=torch.stack([x['text'] for x in train]).to('cuda:0')
sources=torch.tensor([0 if x['source']=='video' else 1 for x in train],device='cuda:0')
normalized=[x['content'].strip().lower() for x in train]
with torch.no_grad():
    for s in (0,1):alignment.bias[s].copy_(text[sources==s].mean(0)-raw[sources==s].mean(0))
optimizer=torch.optim.AdamW(alignment.parameters(),lr=1e-3,weight_decay=1e-3)
losses=[]
for step in range(512):
    indices=rng.sample(range(len(train)),min(48,len(train)))
    positive=torch.tensor([[normalized[i]==normalized[j] for j in indices] for i in indices],device='cuda:0')
    prediction=alignment(raw[indices],sources[indices])
    loss=module.semantic_pair_loss(prediction,text[indices],positive)
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(alignment.parameters(),1);optimizer.step()
    losses.append(float(loss.detach()))
    if (step+1)%64==0:print('NATIVE_ALIGN_TRAIN',json.dumps(dict(step=step+1,loss=round(float(loss),4),pairs=len(train))),flush=True)
alignment.eval()
with torch.no_grad():
    hr=torch.stack([x['raw'] for x in held]).to('cuda:0');ht=torch.stack([x['text'] for x in held]).to('cuda:0')
    hs=torch.tensor([0 if x['source']=='video' else 1 for x in held],device='cuda:0')
    correct={}
    for name,features in (('unadapted',hr),('aligned',alignment(hr,hs))):
        match=(torch.nn.functional.normalize(ht,dim=-1)@torch.nn.functional.normalize(features,dim=-1).T).argmax(-1).cpu().tolist()
        correct[name]=sum(held[i]['content'].strip().lower()==held[j]['content'].strip().lower() for i,j in enumerate(match))
metadata=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),pair_source_sha256=parts[0]['metadata']['source_sha256'],
    model_revision=parts[0]['metadata']['model_revision'],dim=2048,rank=64,seed=2026101411,steps=512,
    benchmark_QA_used=False,train_pairs=len(train),held_pairs=len(held),pseudo_description_match_correct=correct,
    warning='This is pseudo-description retrieval, NOT accurate event timing or benchmark QA',
    excluded_recording_ids=sorted(excluded),train_recording_ids=sorted({x['recording_id'] for x in train}),
    held_recording_ids=sorted({x['recording_id'] for x in held}))
checkpoint=OUT/'avbind_native_alignment_r14.pt'
torch.save(dict(state_dict=alignment.state_dict(),metadata=metadata),checkpoint)
(OUT/'avbind_native_alignment_r14.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
RESULT=dict(file=checkpoint.name,metadata=metadata)
print('NATIVE_ALIGN_DONE',json.dumps(RESULT),flush=True)

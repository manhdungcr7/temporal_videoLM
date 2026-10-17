"""R6: frozen oracle-trained reader, fit retrieval separately with/without answer distillation."""
import json,torch,gc
YR_DIR=OUT/'yann_temporal_research'
source=(OUT/'uploads/yann_research_trainbinding_ampfix.py').read_text()
setup=source.split('try:\n    for step in range(401):')[0].replace("'trainbinding_config.json'","'distillretrieval_setup_config.json'")
setup=setup.replace('center=contrast=YR_SERVER==2','center=contrast=False')
setup=setup.replace("else p[j] for j in range(4)","else torch.full_like(p[j],1/p.shape[-1]) for j in range(4)")
exec(compile(setup,'r6_shared_setup','exec'),globals())
teacher=torch.load(OUT/'uploads/r5_oracle_s1.pt',map_location=dev,weights_only=True)
probe.load_state_dict(teacher['state_dict'])
warm=torch.load(OUT/'uploads/r4_probe_s1.pt',map_location=dev,weights_only=True)['state_dict']
with torch.no_grad():
    for key in ('query.weight','key.weight','mod_key.weight'):probe.state_dict()[key].copy_(warm[key])
for name,p in probe.named_parameters():p.requires_grad_(name.startswith(('query.','key.','mod_key.')))
optimizer=torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad],lr=.0003,weight_decay=.03)
scaler=torch.amp.GradScaler('cuda',init_scale=1024)
config.update(name='reader_distilled_retrieval' if YR_SERVER==2 else 'grounding_only_retrieval',steps=1600,lr=.0003,
    selection='minimum validation endpoint cross-entropy, recording averaged, includes step0',
    frozen_oracle_reader=True,reader_step=teacher['step'],warm_retriever='R4 ordinary selected step200',
    objective='endpoint CE'+(' + .5 binary teacher-probability distillation' if YR_SERVER==2 else ''),
    backbone_trainable=False,trainable_parameters=sum(p.numel() for p in probe.parameters() if p.requires_grad))
history=[];best=math.inf;beststep=None
def retrieval_validation():
    probe.eval();values=[];tops=[0,0];count=0
    with torch.no_grad():
        byrec={}
        for g in val_groups:
            for c in g.values():
                p=attention(c,c);ds=[density(c,e) for e in c['endpoints']]
                losses=[-(d*p[j].clamp_min(1e-8).log()).sum() for j,d in enumerate(ds)]
                byrec.setdefault(c['video_id'],[]).append(float(torch.stack(losses).mean()))
                for j,d in enumerate(ds):tops[j]+=bool(d[int(p[j].argmax())]>0)
                count+=1
        return dict(endpoint_ce=sum(sum(x)/len(x) for x in byrec.values())/len(byrec),endpoint_top1=tops,total=count)
try:
    for step in range(1601):
        if step in (0,200,400,800,1600):
            val=retrieval_validation();metric=val['endpoint_ce']
            ck=dict(state_dict=probe.state_dict(),config=config,step=step)
            torch.save(ck,YR_DIR/f'distillretrieval_step{step}.pt')
            if metric<best:best=metric;beststep=step;torch.save(ck,YR_DIR/'distillretrieval_selected.pt')
            history.append(dict(step=step,validation=val,selected_step=beststep))
            (YR_DIR/'distillretrieval_partial.json').write_text(json.dumps(dict(config=config,history=history),indent=2))
            print('YR_RETRIEVAL_VAL',json.dumps(history[-1]),flush=True)
        if step==1600:break
        probe.train();optimizer.zero_grad(set_to_none=True);g=rng.choice(train_groups)
        for c in g.values():
            context=g['O'] if step%2 else c
            with torch.enable_grad():
                if YR_SERVER==1:
                    p=attention(c,context);ds=[density(c,e) for e in c['endpoints']]
                    loss=torch.stack([-(d*p[j].clamp_min(1e-8).log()).sum() for j,d in enumerate(ds)]).mean()
                else:
                    lm,rank,ground,_,_,other=binding_forward(c,context)
                    with torch.no_grad():tlm,_,_,_,_,tother=binding_forward(c,context,oracle=True)
                    target_prob=torch.sigmoid(tother-tlm)
                    loss=ground.to(hd)+.5*F.binary_cross_entropy_with_logits(other-lm,target_prob)
                scaler.scale(loss/4).backward()
        scaler.unscale_(optimizer)
        finite=all(torch.isfinite(p.grad).all().item() for p in probe.parameters() if p.grad is not None)
        if finite:torch.nn.utils.clip_grad_norm_([p for p in probe.parameters() if p.requires_grad],1.)
        else:skipped_steps.append(dict(step=step,scale=scaler.get_scale()))
        scaler.step(optimizer);scaler.update()
    ck=torch.load(YR_DIR/'distillretrieval_selected.pt',map_location=dev,weights_only=True);probe.load_state_dict(ck['state_dict'])
    RESULT=dict(server=YR_SERVER,config=config,selected_step=beststep,history=history,amp_skipped_steps=skipped_steps,
        validation=evaluate(val_groups,human_val),diagnostic=evaluate(diag_groups),train=evaluate(train_groups),
        diagnostic_oracle=evaluate(diag_groups,oracle=True),diagnostic_address_shuffle=evaluate(diag_groups,shuffle=True),
        diagnostic_interchange=evaluate(diag_groups,interchange=True),
        backbone_versions_unchanged=versions==[p._version for p in MODEL_.parameters()])
    assert RESULT['backbone_versions_unchanged']
    (YR_DIR/'distillretrieval.json').write_text(json.dumps(RESULT,indent=2))
    print('YR_RETRIEVAL_DONE',json.dumps({k:RESULT[k]['metrics'] for k in ('train','validation','diagnostic','diagnostic_oracle')}),flush=True)
finally:
    for p,r in zip(MODEL_.parameters(),req):p.requires_grad_(r)
    optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()

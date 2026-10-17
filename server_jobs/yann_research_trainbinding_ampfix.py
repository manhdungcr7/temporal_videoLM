"""R4: matched physical-pair/interchange-supervised readout with explicit endpoint losses."""
import collections,gc,hashlib,importlib.util,json,math,random,tarfile,torch
from pathlib import Path
from torch.nn import functional as F
YR_DIR=OUT/'yann_temporal_research';peer=3-YR_SERVER
ix=json.loads((OUT/f'uploads/bindingallpack_s{peer}.json').read_text());archive=OUT/f'uploads/binding_all_features_s{peer}.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==ix['sha256']
with tarfile.open(archive,'r:gz') as z:
    for rec in ix['files']:
        assert rec['name'].startswith(('binding48_v2/','binding_evalextra/')) and '..' not in Path(rec['name']).parts
        b=z.extractfile(rec['name']).read();assert hashlib.sha256(b).hexdigest()==rec['sha256'];p=YR_DIR/rec['name'];p.parent.mkdir(exist_ok=True);p.write_bytes(b)
spec=importlib.util.spec_from_file_location('tam_binding',OUT/'uploads/yann_tam_probe_model.py');tm=importlib.util.module_from_spec(spec);spec.loader.exec_module(tm)
all_cases=[torch.load(p,map_location='cpu',weights_only=True) for p in sorted((YR_DIR/'binding48_v2').glob('*.pt')) if not p.stem.endswith('_other')]
groups={}
for c in all_cases:
    assert c['bank_support_version'].startswith('native visual tubelet')
    assert all(not(e['source']=='audio' and e['label'].strip()=='Muted') for e in c['endpoints'])
    groups.setdefault(c['video_id'],{})[c['variant']]=c
for vid,g in groups.items():
    assert set(g)=={'O','A','V','AV'}
    assert len({c['input_prompt_len'] for c in g.values()})==1,(vid,[c['input_prompt_len'] for c in g.values()])
    assert g['O']['answer']==g['AV']['answer']
train_groups=[g for g in groups.values() if g['O']['split']=='train'];val_groups=[g for g in groups.values() if g['O']['split']=='validation'];diag_groups=[g for g in groups.values() if g['O']['split']=='diagnostic']
original_val_groups=val_groups;original_diag_groups=diag_groups
extra={}
for p in sorted((YR_DIR/'binding_evalextra').glob('*.pt')):
    if p.stem.endswith('_other'):continue
    c=torch.load(p,map_location='cpu',weights_only=True);extra.setdefault(c['video_id'],{})[c['variant']]=c
for g in extra.values():
    assert set(g)=={'O','A','V','AV'} and len({c['input_prompt_len'] for c in g.values()})==1
    assert any(c['answer']!=g['O']['answer'] for c in g.values())
# Extra questions are annotation-selected before fitting, same recordings, not additional independent data.
val_groups=val_groups+[g for vid,g in extra.items() if g['O']['split']=='validation' and not any(o['O']['video_id']==vid and o['O']['question']==g['O']['question'] for o in original_val_groups)]
diag_groups=diag_groups+[g for vid,g in extra.items() if g['O']['split']=='diagnostic' and not any(o['O']['video_id']==vid and o['O']['question']==g['O']['question'] for o in original_diag_groups)]
human=[tm.restore_cached_video_support(torch.load(p,map_location='cpu',weights_only=True)) for p in sorted((YR_DIR/'perception48').glob('*_q0.pt'))]
human_train=[c for c in human if c['split']=='train'];human_val=[c for c in human if c['split']=='validation']
assert len(human_train)==32 and len(human_val)==8
seed=20261017;torch.manual_seed(seed);rng=random.Random(seed)
dev=MODEL_.get_input_embeddings().weight.device;hd=MODEL_.lm_head.weight.device
# Full and matched ordinary residual; the full factorial is a later labeled extension.
center=contrast=YR_SERVER==2
probe=tm.TAMBindingProbe(center_time=center,contrast_path=contrast).to(dev)
name='TAM_binding_dependent' if center else 'matched_ordinary_temporal_memory'
config=dict(name=name,seed=seed,steps=400,lr=.0008,weight_decay=.01,ground_weight=1.,rank_weight=.5,
    oracle_curriculum_steps=100,amp_initial_scale=1024,interchange='every other step; source memory with O native context/query, target matched by answer text',
    selection='min(validation recording-averaged binary event-choice NLL + .25*validation human QA token NLL), includes step0; original+informative questions jointly',
    center_time=center,contrast_path=contrast,parameters=sum(p.numel() for p in probe.parameters()),
    groups={split:len(gs) for split,gs in [('train',train_groups),('validation',val_groups),('diagnostic',diag_groups)]},
    independent_recordings={split:len({g['O']['video_id'] for g in gs}) for split,gs in [('train',train_groups),('validation',val_groups),('diagnostic',diag_groups)]},
    flip_groups={split:sum(any(c['answer']!=g['O']['answer'] for c in g.values()) for g in gs) for split,gs in [('train',train_groups),('validation',val_groups),('diagnostic',diag_groups)]})
(YR_DIR/'trainbinding_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print('YR_BINDING_CONFIG',json.dumps(config),flush=True)
versions=[p._version for p in MODEL_.parameters()];req=[p.requires_grad for p in MODEL_.parameters()]
for p in MODEL_.parameters():p.requires_grad_(False)
optimizer=torch.optim.AdamW(probe.parameters(),lr=.0008,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=1024);history=[];best=math.inf;beststep=None;skipped_steps=[]
def density(c,e):
    return tm.target_density(dict(spans=c['spans'],mods=c['mods'],intervals=[e['interval']],source=e['source']),dev)
def attention(c,context):
    b=F.layer_norm(c['bank'].to(dev).float(),(2048,));q=F.layer_norm(context['q'].to(dev).float(),(2048,))
    with torch.autocast('cuda',dtype=torch.float16):
        queries=probe.query(q).reshape(4,64);keys=probe.key(b)+probe.mod_key(c['mods'].to(dev));p=F.softmax(((queries@keys.T)/8).float(),dim=-1)
    return p
def hidden_for(c,context,correct):
    text=c['answer'] if correct else c['other_answer'];target=c['target'] if correct else c['target_other']
    if text==context['answer']:h=context['answer_h']
    else:assert text==context['other_answer'];h=context['answer_h_other']
    assert len(h)==len(target)
    return h,target
def binding_forward(c,context=None,teacher=0.,oracle=False,shuffle=False):
    context=c if context is None else context;p=attention(c,context);dens=[density(c,e) for e in c['endpoints']]
    grounds=[-(d*p[j].clamp_min(1e-8).log()).sum() for j,d in enumerate(dens) if d is not None]
    ground=torch.stack(grounds).mean() if grounds else p.sum()*0
    used=p
    if teacher>0 or oracle:
        rows=[dens[j] if j<2 and dens[j] is not None else p[j] for j in range(4)]
        exact=torch.stack(rows);used=exact if oracle else teacher*exact+(1-teacher)*p
    spans=c['spans'].to(dev)
    if shuffle:
        rg=torch.Generator().manual_seed(20261017+int(c['video_id'].split('_')[-1]));per=torch.randperm(len(spans),generator=rg).to(dev);spans=spans[per]
    lms=[];di=None
    for correct in (True,False):
        h,target=hidden_for(c,context,correct)
        with torch.autocast('cuda',dtype=torch.float16):
            adjusted,_,di=probe(c['bank'].to(dev),context['q'].to(dev),spans,c['mods'].to(dev),c['duration'],h.to(dev),force_attention=used)
            lg=MODEL_.lm_head(adjusted.to(device=hd,dtype=MODEL_.lm_head.weight.dtype)).float()
            lms.append(F.cross_entropy(lg,target.to(hd)))
    rank=F.softplus(lms[0]-lms[1]);return lms[0],rank,ground,p,di,lms[1]
def human_forward(c):
    with torch.autocast('cuda',dtype=torch.float16):
        h,_,_=probe(c['bank'].to(dev),c['q'].to(dev),c['spans'].to(dev),c['mods'].to(dev),c['duration'],c['answer_h'].to(dev))
        lg=MODEL_.lm_head(h.to(device=hd,dtype=MODEL_.lm_head.weight.dtype)).float();loss=F.cross_entropy(lg,c['target'].to(hd))
    return loss
def evaluate(gs,human_cases=(),oracle=False,shuffle=False,interchange=False):
    probe.eval();rows=[];human_losses=[]
    with torch.no_grad():
        for g in gs:
            choices=[];rr=[]
            for variant,c in g.items():
                lm,rank,gr,p,di,other=binding_forward(c,context=g['O'] if interchange else None,oracle=oracle,shuffle=shuffle);correct=float(lm)<float(other);choices.append(correct)
                dens=[density(c,e) for e in c['endpoints']]
                rr.append(dict(variant=variant,correct=correct,lm=float(lm),other_lm=float(other),rank=float(rank),ground=float(gr),
                    endpoint_top1=[bool(d[int(p[j].argmax())]>0) if d is not None else None for j,d in enumerate(dens)],
                    predicted_choice=c['answer'] if correct else c['other_answer'],gold=c['answer'],delta_norm=float(di['delta_norm'].mean())))
            flip=any(c['answer']!=g['O']['answer'] for c in g.values());rows.append(dict(video_id=g['O']['video_id'],question=g['O']['question'],direction=g['O']['actual_direction'],flip=flip,all4=all(choices),variants=rr,
                flip_pattern=[g[v]['answer']!=g['O']['answer'] for v in ('O','A','V','AV')]))
        for c in human_cases:human_losses.append(float(human_forward(c)))
    recs={}
    for r in rows:recs.setdefault(r['video_id'],[]).append(r)
    mixed=[rs for rs in recs.values() if len({tuple(r['flip_pattern']) for r in rs})>1]
    metrics=dict(n_groups=len(rows),n_recordings=len(recs),all4=sum(r['all4'] for r in rows),flip_groups=sum(r['flip'] for r in rows),
        flip_all4=sum(r['all4'] and r['flip'] for r in rows),correct=sum(v['correct'] for r in rows for v in r['variants']),total=4*len(rows),
        joint_recording_correct=sum(all(r['all4'] for r in rs) for rs in recs.values()),
        mixed_flip_pattern_recordings=len(mixed),mixed_pattern_joint_correct=sum(all(r['all4'] for r in rs) for rs in mixed),
        mean_rank=sum(sum(v['rank'] for r in rs for v in r['variants'])/(4*len(rs)) for rs in recs.values())/len(recs),
        mean_human_qa_lm=sum(human_losses)/len(human_losses) if human_losses else None)
    return dict(metrics=metrics,rows=rows)
try:
    for step in range(401):
        if step in (0,50,100,200,400):
            val=evaluate(val_groups,human_val);metric=val['metrics']['mean_rank']+.25*val['metrics']['mean_human_qa_lm']
            ck=dict(state_dict=probe.state_dict(),config=config,step=step,validation=val['metrics']);torch.save(ck,YR_DIR/f'binding_probe_step{step}.pt')
            (YR_DIR/f'binding_validation_step{step}.json').write_text(json.dumps(val,indent=2),encoding='utf-8')
            if metric<best:best=metric;beststep=step;torch.save(ck,YR_DIR/'binding_probe_selected.pt')
            history.append(dict(step=step,selection_loss=metric,validation=val['metrics'],selected_step=beststep))
            (YR_DIR/'trainbinding_partial.json').write_text(json.dumps(dict(config=config,history=history),indent=2),encoding='utf-8')
            print('YR_BINDING_VAL',json.dumps(history[-1]),flush=True)
        if step==400:break
        probe.train();optimizer.zero_grad(set_to_none=True);g=rng.choice(train_groups);teacher=max(0.,1-step/100)
        losses=[]
        for c in g.values():
            context=g['O'] if step%2 else c
            with torch.enable_grad():
                lm,rank,ground,_,_,_=binding_forward(c,context,teacher=teacher);loss=lm+.5*rank+ground.to(hd)
                scaler.scale(loss/5).backward();losses.append(float(loss.detach()))
        with torch.enable_grad():loss=human_forward(rng.choice(human_train));scaler.scale(loss/5).backward();losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        finite=all(torch.isfinite(p.grad).all().item() for p in probe.parameters() if p.grad is not None)
        if finite:norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
        else:
            norm=torch.tensor(float('inf'));skipped_steps.append(dict(step=step,scale=scaler.get_scale()));print('YR_AMP_SKIP',step,scaler.get_scale(),flush=True)
        scaler.step(optimizer);scaler.update()
        if (step+1)%25==0:print('YR_BINDING_STEP',step+1,sum(losses)/5,float(norm),flush=True)
    ck=torch.load(YR_DIR/'binding_probe_selected.pt',map_location=dev,weights_only=True);probe.load_state_dict(ck['state_dict'])
    RESULT=dict(server=YR_SERVER,config=config,history=history,selected_step=beststep,amp_skipped_steps=skipped_steps,
        validation=evaluate(val_groups,human_val),diagnostic=evaluate(diag_groups),
        original_validation=evaluate(original_val_groups),original_diagnostic=evaluate(original_diag_groups),
        diagnostic_oracle=evaluate(diag_groups,oracle=True),diagnostic_address_shuffle=evaluate(diag_groups,shuffle=True),
        diagnostic_interchange=evaluate(diag_groups,interchange=True),
        train=evaluate(train_groups),backbone_versions_unchanged=versions==[p._version for p in MODEL_.parameters()])
    assert RESULT['backbone_versions_unchanged']
    (YR_DIR/'trainbinding.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
    print('YR_BINDING_DONE',json.dumps(dict(selected_step=beststep,diagnostic=RESULT['diagnostic']['metrics'],oracle=RESULT['diagnostic_oracle']['metrics'])),flush=True)
finally:
    for p,r in zip(MODEL_.parameters(),req):p.requires_grad_(r)
    optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()

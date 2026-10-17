"""R3 matched training: learned final readout, human QA+multi-positive event grounding."""
import copy,gc,hashlib,importlib.util,json,math,random,tarfile,time,torch
from pathlib import Path
from torch.nn import functional as F
YR_DIR=OUT/'yann_temporal_research'
yr_peer=3-YR_SERVER;yr_ix=json.loads((OUT/f'uploads/pack_s{yr_peer}.json').read_text())
yr_tar=OUT/f'uploads/supervision_features_s{yr_peer}.tar.gz'
assert hashlib.sha256(yr_tar.read_bytes()).hexdigest()==yr_ix['sha256']
with tarfile.open(yr_tar,'r:gz') as z:
    for rec in yr_ix['files']:
        assert rec['name'].startswith('perception48/') and '..' not in Path(rec['name']).parts
        blob=z.extractfile(rec['name']).read();assert hashlib.sha256(blob).hexdigest()==rec['sha256']
        p=YR_DIR/rec['name'];p.parent.mkdir(exist_ok=True);p.write_bytes(blob)
yr_spec=importlib.util.spec_from_file_location('yann_tam_probe_model',OUT/'uploads/yann_tam_probe_model.py')
yr_mod=importlib.util.module_from_spec(yr_spec);yr_spec.loader.exec_module(yr_mod)
yr_cases=[torch.load(p,map_location='cpu',weights_only=True) for p in sorted((YR_DIR/'perception48').glob('*.pt'))]
assert len(yr_cases)==144,len(yr_cases)
yr_train=[c for c in yr_cases if c['split']=='train'];yr_val=[c for c in yr_cases if c['split']=='validation'];yr_diag=[c for c in yr_cases if c['split']=='diagnostic']
assert (len(yr_train),len(yr_val),len(yr_diag))==(96,24,24)
assert not {c['video_id'] for c in yr_train}&{c['video_id'] for c in yr_val+yr_diag}
yr_device=MODEL_.get_input_embeddings().weight.device;yr_hdevice=MODEL_.lm_head.weight.device
yr_seed=20261016;torch.manual_seed(yr_seed);random.seed(yr_seed)
yr_probe=yr_mod.TAMProbe(use_time=YR_SERVER==2).to(yr_device)
yr_cfg=dict(name='TAM_Lite_time' if YR_SERVER==2 else 'matched_content_memory',seed=yr_seed,steps=400,batch=4,lr=.001,
    weight_decay=.01,ground_weight=.10,residual_cap=.10,checkpoints=[0,50,100,200,400],selection='lowest validation human_qa answer-token NLL (includes step0)',
    parameters=sum(p.numel() for p in yr_probe.parameters()),data_videos=dict(train=32,validation=8,diagnostic=8),hook_index=18,
    readout='learned residual before frozen lm_head; no backbone gradient or update',query_convention='last token of option-free direct-answer prompt')
(YR_DIR/'trainprobe_config.json').write_text(json.dumps(yr_cfg,indent=2),encoding='utf-8')
yr_versions=[p._version for p in MODEL_.parameters()];yr_req=[p.requires_grad for p in MODEL_.parameters()]
for p in MODEL_.parameters():p.requires_grad_(False)
yr_opt=torch.optim.AdamW(yr_probe.parameters(),lr=.001,weight_decay=.01)
yr_scaler=torch.amp.GradScaler('cuda');yr_rng=random.Random(yr_seed);yr_history=[];yr_best=math.inf;yr_best_step=None
def yr_case_forward(c,model=yr_probe):
    bank=c['bank'].to(yr_device);q=c['q'].to(yr_device);h=c['answer_h'].to(yr_device)
    with torch.autocast('cuda',dtype=torch.float16):
        adjusted,attn,diag=model(bank,q,c['spans'].to(yr_device),c['mods'].to(yr_device),c['duration'],h)
        logits=MODEL_.lm_head(adjusted.to(device=yr_hdevice,dtype=MODEL_.lm_head.weight.dtype)).float()
        target=c['target'].to(yr_hdevice);lm=F.cross_entropy(logits,target)
        density=yr_mod.target_density(c,yr_device)
        gp=attn.mean(0).clamp_min(1e-8)
        ground=-(density*gp.log()).sum() if density is not None else gp.sum()*0
    return lm,ground,logits,attn,diag
def yr_evaluate(cases):
    bykind={};preds=[];yr_probe.eval()
    with torch.no_grad():
        for c in cases:
            lm,g,lg,at,diag=yr_case_forward(c);kind=c['kind'];b=bykind.setdefault(kind,dict(n=0,lm_sum=0.,ground_sum=0.,token_correct=0,tokens=0,ground_top1=0,ground_n=0))
            b['n']+=1;b['lm_sum']+=float(lm);b['ground_sum']+=float(g)
            target=c['target'].to(lg.device);tokcorrect=(lg.argmax(-1)==target);b['token_correct']+=int(tokcorrect.sum());b['tokens']+=len(target)
            density=yr_mod.target_density(c,yr_device)
            if density is not None:b['ground_top1']+=int(density[int(at.mean(0).argmax())]>0);b['ground_n']+=1
            preds.append(dict(video_id=c['video_id'],kind=kind,source=c.get('source'),question=c['question'],target_ids=c['target'].tolist(),predicted_teacher_forced_ids=lg.argmax(-1).cpu().tolist(),
                lm=float(lm),ground=float(g),attention_mean=at.mean(0).cpu().tolist(),delta_norm=float(diag['delta_norm'].mean())))
    for b in bykind.values():b['mean_lm']=b['lm_sum']/b['n'];b['mean_ground']=b['ground_sum']/b['n']
    return dict(metrics=bykind,predictions=preds)
try:
    # Nontrivial active tuple-permutation check, not just zero-init equality.
    c=yr_train[0];test=copy.deepcopy(yr_probe)
    with torch.no_grad():test.out.weight.normal_(std=.01)
    gen=torch.Generator().manual_seed(yr_seed);perm=torch.randperm(len(c['bank']),generator=gen).to(yr_device)
    with torch.no_grad():
        args=(c['bank'].to(yr_device),c['q'].to(yr_device),c['spans'].to(yr_device),c['mods'].to(yr_device),c['duration'],c['answer_h'].to(yr_device))
        x,_,_=test(*args);y,_,_=test(args[0][perm],args[1],args[2][perm],args[3][perm],args[4],args[5])
        perm_error=float((x-y).abs().max());assert perm_error<1e-4,perm_error
        native,_,_=yr_probe(*args);identity=float((native-args[5].float()).abs().max());assert identity==0
    del test,args,x,y,native
    for step in range(401):
        if step in yr_cfg['checkpoints']:
            val=yr_evaluate(yr_val);vl=val['metrics']['human_qa']['mean_lm']
            torch.save(dict(state_dict=yr_probe.state_dict(),config=yr_cfg,step=step,validation=val['metrics']),YR_DIR/f'probe_step{step}.pt')
            (YR_DIR/f'validation_step{step}.json').write_text(json.dumps(val,indent=2),encoding='utf-8')
            if vl<yr_best:yr_best=vl;yr_best_step=step;torch.save(dict(state_dict=yr_probe.state_dict(),config=yr_cfg,step=step,validation=val['metrics']),YR_DIR/'probe_selected.pt')
            yr_history.append(dict(step=step,validation=val['metrics'],selected_step=yr_best_step))
            (YR_DIR/'trainprobe_partial.json').write_text(json.dumps(dict(config=yr_cfg,history=yr_history),indent=2),encoding='utf-8')
            print('YR_TRAINPROBE_VAL',json.dumps(yr_history[-1]),flush=True)
        if step==400:break
        yr_probe.train();yr_opt.zero_grad(set_to_none=True);losses=[]
        for c in yr_rng.sample(yr_train,4):
            with torch.enable_grad():
                lm,g,_,_,_=yr_case_forward(c);loss=lm+.10*g.to(lm.device);yr_scaler.scale(loss/4).backward();losses.append(float(loss.detach()))
        yr_scaler.unscale_(yr_opt);norm=torch.nn.utils.clip_grad_norm_(yr_probe.parameters(),1.)
        assert torch.isfinite(norm),'Nonfinite probe gradient'
        yr_scaler.step(yr_opt);yr_scaler.update()
        if (step+1)%25==0:print('YR_TRAINPROBE_STEP',step+1,sum(losses)/4,float(norm),flush=True)
    selected=torch.load(YR_DIR/'probe_selected.pt',map_location=yr_device,weights_only=True);yr_probe.load_state_dict(selected['state_dict'])
    diag=yr_evaluate(yr_diag);train=yr_evaluate(yr_train)
    RESULT=dict(server=YR_SERVER,config=yr_cfg,selected_step=yr_best_step,history=yr_history,diagnostic=diag,train=train,
        unit_checks=dict(zero_init_max_error=identity,active_tuple_permutation_max_error=perm_error),
        backbone_versions_unchanged=yr_versions==[p._version for p in MODEL_.parameters()])
    assert RESULT['backbone_versions_unchanged']
    (YR_DIR/'trainprobe.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
    print('YR_TRAINPROBE_DONE',json.dumps(dict(server=YR_SERVER,selected_step=yr_best_step,diagnostic=diag['metrics'])),flush=True)
finally:
    for p,r in zip(MODEL_.parameters(),yr_req):p.requires_grad_(r)
    yr_opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()

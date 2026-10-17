"""R5 privileged readout diagnostic: always-correct endpoint retrieval, not a method score.

Both conditions use O answer-prefix context while reading physically transformed
variant memory. The control permutes addresses. Same model/objective/budget.
"""
import json,torch,gc
YR_DIR=OUT/'yann_temporal_research'
source=(OUT/'uploads/yann_research_trainbinding_ampfix.py').read_text()
setup=source.split('try:\n    for step in range(401):')[0]
assert setup!=source
setup=setup.replace("'trainbinding_config.json'","'oraclelearn_setup_config.json'")
setup=setup.replace('center=contrast=YR_SERVER==2','center=contrast=False')
setup=setup.replace("else p[j] for j in range(4)","else torch.full_like(p[j],1/p.shape[-1]) for j in range(4)")
exec(compile(setup,'r5_shared_readout_setup','exec'),globals())
config.update(name='oracle_address_shuffle' if YR_SERVER==2 else 'oracle_true_address',steps=800,
    selection='minimum recording-averaged validation binary NLL under oracle retrieval and O context; includes step0',
    objective='mean over four physical variants of binary NLL + .05 correct-answer token NLL; all use O answer context',
    diagnostic_only=True,oracle_attention=True,control_shuffle=YR_SERVER==2)
(YR_DIR/'oraclelearn_config.json').write_text(json.dumps(config,indent=2))
history=[];best=math.inf;beststep=None
try:
    for step in range(801):
        if step in (0,100,200,400,800):
            val=evaluate(val_groups,oracle=True,shuffle=YR_SERVER==2,interchange=True)
            metric=val['metrics']['mean_rank']
            ck=dict(state_dict=probe.state_dict(),config=config,step=step)
            torch.save(ck,YR_DIR/f'oraclelearn_step{step}.pt')
            if metric<best:
                best=metric;beststep=step;torch.save(ck,YR_DIR/'oraclelearn_selected.pt')
            history.append(dict(step=step,validation=val['metrics'],selected_step=beststep))
            (YR_DIR/'oraclelearn_partial.json').write_text(json.dumps(dict(config=config,history=history),indent=2))
            print('YR_ORACLE_VAL',json.dumps(history[-1]),flush=True)
        if step==800:break
        probe.train();optimizer.zero_grad(set_to_none=True);g=rng.choice(train_groups)
        for c in g.values():
            with torch.enable_grad():
                lm,rank,_,_,_,_=binding_forward(c,g['O'],oracle=True,shuffle=YR_SERVER==2)
                loss=rank+.05*lm;scaler.scale(loss/4).backward()
        scaler.unscale_(optimizer)
        finite=all(torch.isfinite(p.grad).all().item() for p in probe.parameters() if p.grad is not None)
        if finite:torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
        else:skipped_steps.append(dict(step=step,scale=scaler.get_scale()))
        scaler.step(optimizer);scaler.update()
    ck=torch.load(YR_DIR/'oraclelearn_selected.pt',map_location=dev,weights_only=True);probe.load_state_dict(ck['state_dict'])
    selected_state={k:v.detach().clone() for k,v in probe.state_dict().items()}
    outputs={k:evaluate(gs,oracle=True,shuffle=YR_SERVER==2,interchange=True) for k,gs in [('train',train_groups),('validation',val_groups),('diagnostic',diag_groups)]}
    step0=torch.load(YR_DIR/'oraclelearn_step0.pt',map_location=dev,weights_only=True);probe.load_state_dict(step0['state_dict'])
    outputs['native_diagnostic']=evaluate(diag_groups,oracle=True,shuffle=YR_SERVER==2,interchange=True)
    probe.load_state_dict(selected_state)
    RESULT=dict(server=YR_SERVER,config=config,selected_step=beststep,history=history,amp_skipped_steps=skipped_steps,**outputs,
        backbone_versions_unchanged=versions==[p._version for p in MODEL_.parameters()])
    assert RESULT['backbone_versions_unchanged']
    (YR_DIR/'oraclelearn.json').write_text(json.dumps(RESULT,indent=2))
    print('YR_ORACLE_DONE',json.dumps({k:v['metrics'] for k,v in outputs.items()}),flush=True)
finally:
    for p,r in zip(MODEL_.parameters(),req):p.requires_grad_(r)
    optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()

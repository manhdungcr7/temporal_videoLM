import ast,gc,importlib.util,json,torch
YR_DIR=OUT/'yann_temporal_research'
sp=importlib.util.spec_from_file_location('tam_bind_eval',OUT/'uploads/yann_tam_probe_model.py');tm=importlib.util.module_from_spec(sp);sp.loader.exec_module(tm)
dev=MODEL_.get_input_embeddings().weight.device;hd=MODEL_.lm_head.weight.device;yr_probes={}
for s in (1,2):
    ck=torch.load(OUT/f'uploads/r4_probe_s{s}.pt',map_location=dev,weights_only=True);cfg=ck['config']
    model=tm.TAMBindingProbe(center_time=cfg['center_time'],contrast_path=cfg['contrast_path']).to(dev);model.load_state_dict(ck['state_dict']);model.eval();yr_probes[cfg['name']]=model
tree=ast.parse((OUT/'uploads/yann_research_probe_eval.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'r3_shared_generation_parity','exec'),globals())
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text());old={r['idx']:r for r in json.loads((YR_DIR/'probe_eval.json').read_text())['rows']};rows=[]
for row in manifest[f'server{YR_SERVER}'][:4]:
    c=torch.load(YR_DIR/'microfeatures'/f"{row['idx']}.pt",map_location='cpu',weights_only=True)
    original=torch.load(OUT/'argon_latent40/decoder_cache'/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=True)
    # The archived native clock can exceed container duration (known decoder-clock
    # mismatch). Preserve it as native, not a claim of physical/PTS calibration.
    c=dict(c);c['spans']=c['spans'].clone();v=c['mods']==1
    spg=float(original['spg']);assert .1<spg<5
    c['spans'][v,0]=c['times'][v];c['spans'][v,1]=c['times'][v]+spg
    c['duration']=max(float(c['duration']),float(c['spans'].max()))
    payload=torch.load(YR_DIR/'microfeatures'/f"{row['idx']}_optionfree_decoder.pt",map_location='cpu',weights_only=True)
    _,options=parse_question(row['question']);labels=[PROC.tokenizer.convert_tokens_to_ids(x) for x in 'ABC'][:len(options)]
    result=dict(idx=row['idx'],task=row['task'],gold=row['gold'],mc={},generated={'native':old[row['idx']]['generated']['native']},
        time_metadata=dict(mode='archived native clock, visual support=cache spg, horizon covers all cells; not physical PTS calibration',
            container_duration=row['duration'],native_horizon=c['duration'],visual_support=spg))
    full=yr_probes['TAM_binding_dependent'];rng=torch.Generator().manual_seed(20261017+row['idx']);perm=torch.randperm(len(c['bank']),generator=rng)
    conditions=[('native',None,None,None)]+[(k,v,None,None) for k,v in yr_probes.items()]+[
        ('binding_address_shuffle',full,c['spans'][perm],None),('binding_uniform_attention',full,None,torch.full((4,len(c['bank'])),1/len(c['bank'])))]
    native_lg=None
    with torch.no_grad():
        for name,probe,spans,force in conditions:
            h=c['mc_h'].to(dev)
            if probe is not None:
                with torch.autocast('cuda',dtype=torch.float16):h,_,di=probe(c['bank'].to(dev),c['q'].to(dev),(c['spans'] if spans is None else spans).to(dev),c['mods'].to(dev),c['duration'],h,force_attention=force)
            lg=MODEL_.lm_head(h.to(device=hd,dtype=MODEL_.lm_head.weight.dtype))[0,labels].float().cpu()
            if native_lg is None:native_lg=lg
            pred='ABC'[int(lg.argmax())];result['mc'][name]=dict(prediction=pred,correct=pred==row['gold'],logits=lg.tolist(),probabilities=lg.softmax(0).tolist(),max_logit_change=float((lg-native_lg).abs().max()))
    for name,probe in yr_probes.items():result['generated'][name]=yr_generate(payload,c,probe)
    rows.append(result);(YR_DIR/'binding_probe_eval_partial.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('YR_BINDING_MICRO',json.dumps(dict(idx=row['idx'],predictions={k:v['prediction'] for k,v in result['mc'].items()},texts={k:v['text'] for k,v in result['generated'].items()})),flush=True)
RESULT=dict(server=YR_SERVER,rows=rows,counts={k:sum(r['mc'][k]['correct'] for r in rows) for k in rows[0]['mc']},hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
assert RESULT['hooks']==0
(YR_DIR/'binding_probe_eval.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print('YR_BINDING_MICRO_DONE',json.dumps(RESULT['counts']),flush=True)

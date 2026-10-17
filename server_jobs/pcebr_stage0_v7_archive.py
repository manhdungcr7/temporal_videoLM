"""Recompute all stored predictions; audit actual couplings and archive evidence."""
import hashlib
import json
import tarfile
import torch

files=[];probes=[];nquery=0;couplings=[]
for path in sorted((V7_DIR/'artifacts').glob('*.pt')):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if '_retrieval_' in path.name:
        assert v7_orig_eval(obj['sim'],obj['examples'])==obj['metrics'],path.name
        nquery+=len(obj['examples'])
    if path.name.startswith('m2_probe_'):
        y=obj['y_he'];p=obj['logits_he'].argmax(dim=-1);mask=y<3
        assert torch.equal(p,obj['preds'])
        p3=obj['logits_he'][mask,:3].argmax(dim=-1)
        assert torch.equal(p3,obj['preds_3way'])
        byclass={str(c):dict(correct=int(((y==c)&(p==c)).sum()),total=int((y==c).sum())) for c in range(4)}
        fm,rm={},{}
        for i,e in enumerate(obj['heldout_metadata']):
            if e['pair_key'] is not None:
                key,side=e['pair_key'];(fm if side=='fwd' else rm)[key]=(e['label'],int(obj['logits_he'][i,:3].argmax()))
        paired=set(fm)&set(rm)
        hits=sum(fm[k][0]==fm[k][1] and rm[k][0]==rm[k][1] for k in paired)
        probes.append(dict(file=path.name,correct4=int((p==y).sum()),total=len(y),correct3=int((p3==y[mask]).sum()),
            total3=int(mask.sum()),by_class=byclass,paired_correct=hits,paired_total=len(paired),pred_counts=torch.bincount(p,minlength=4).tolist()))
    if path.name.startswith('m3_'):
        assert float((obj['legacy_archived_source_prob']-obj['p_legacy_v5']).abs().max())<1e-6
        p1=obj['aud_fwd']['p_a_hat'];p2=obj['aud_fwd']['p_v_hat'];K=len(p1)
        gates=[]
        for lg in (obj['lg_a'],obj['lg_v']):
            g=torch.relu(torch.softmax(lg,dim=-1)-1./K).sum();gates.append(g*g/(g*g+0.05**2))
        gate=gates[0]*gates[1]
        omega=obj['slot_meta']['pair_omega']
        score=V7_NS['compute_dustbin_raw_score'](obj['lg_a']*.07,obj['lg_v']*.07,obj['dustbin_theta'])
        J,C,P=V7_NS['compute_sinkhorn_joint_event_pair_posterior'](p1,p2,omega,score,gate)
        J0,C0,P0=V7_NS['compute_sinkhorn_joint_event_pair_posterior'](p1,p2,torch.zeros_like(omega),score,gate)
        act=(gate*torch.sigmoid(-score)).clamp(1e-7,1.-1e-7);null=1.-act
        r=torch.cat([act*p1,null.view(1)]);c=torch.cat([act*p2,null.view(1)])
        # The dual-calibrated construction has analytic block solution [a^2*C, a(1-a)*p; ..., (1-a)^2].
        # Test its implied scalar P_null and lack of omega-dependence at fixed marginals/score.
        phi,_=V7_NS['compute_joint_lag_distribution_features'](C,obj['iv_t'],obj['iv_t'],gate)
        phif,_=V7_NS['compute_joint_lag_distribution_features'](p1[:,None]*p2[None,:],obj['iv_t'],obj['iv_t'],gate)
        audit=dict(file=path.name,K=K,joint_gate=float(gate),dustbin_score=float(score),act_target=float(act),
            P_null=float(P),analytic_one_minus_act_squared=float(1.-act.square()),
            analytic_pnull_error=float((P-(1.-act.square())).abs()),
            P_null_zero_pair_omega=float(P0),pnull_change_with_fixed_inputs_zero_omega=float((P-P0).abs()),
            J_aug_row_error=float((J.sum(1)-r).abs().max()),J_aug_col_error=float((J.sum(0)-c).abs().max()),
            J_cond_row_error=float((C.sum(1)-p1).abs().max()),J_cond_col_error=float((C.sum(0)-p2).abs().max()),
            cond_distance_from_outer_L1=float((C-p1[:,None]*p2[None,:]).abs().sum()),
            marginal_only_phi_channels_9_10_11_12_max_change=float((phi[[9,10,11,12]]-phif[[9,10,11,12]]).abs().max()))
        couplings.append(audit)
        torch.save(dict(J_aug=J,J_cond=C,P_null=P,p1=p1,p2=p2,omega=omega,gate=gate,score=score,phi=phi,phi_factorized=phif),V7_DIR/'artifacts'/('coupling_'+path.stem+'.pt'))

v7_write('coupling_actual_input_audit.json',couplings)
base=torch.load(V7_DIR/'artifacts/m2_episodes_21.pt',map_location='cpu',weights_only=False)['episodes']
hop_comparisons=[]
for n,kind in [(31,'teacher_forced'),(33,'oracle_anchor_and_location'),(35,'no_bg_subtract'),(37,'factorized'),(39,'legacy_6d')]:
    other=torch.load(V7_DIR/'artifacts'/f'm2_episodes_{n:02d}.pt',map_location='cpu',weights_only=False)['episodes']
    assert [(e['label'],e['dir_tag'],e['pair_key']) for e in base]==[(e['label'],e['dir_tag'],e['pair_key']) for e in other]
    row=dict(kind=kind,n=len(base),same_hop1_flags=all(a['hop1_correct']==b['hop1_correct'] for a,b in zip(base,other)),
        same_hop2_flags=all(a['hop2_correct']==b['hop2_correct'] for a,b in zip(base,other)),
        hop2_flags_changed=sum(a['hop2_correct']!=b['hop2_correct'] for a,b in zip(base,other)))
    if base[0]['full_feat'].shape==other[0]['full_feat'].shape:
        dd=torch.stack([e['full_feat'] for e in base])-torch.stack([e['full_feat'] for e in other])
        row.update(feature_mean_abs_difference=float(dd.abs().mean()),feature_max_abs_difference=float(dd.abs().max()))
    hop_comparisons.append(row)
v7_write('hop_condition_comparison.json',hop_comparisons)
probe_comparisons=[]
baseprobe=torch.load(V7_DIR/'artifacts/m2_probe_18.pt',map_location='cpu',weights_only=False)
for n,kind in [(26,'no_psi'),(29,'no_bg_subtract'),(30,'factorized'),(31,'legacy_6d')]:
    pp=torch.load(V7_DIR/'artifacts'/f'm2_probe_{n:02d}.pt',map_location='cpu',weights_only=False)
    y=baseprobe['y_he'];assert torch.equal(y,pp['y_he'])
    p=baseprobe['preds'];q=pp['preds']
    probe_comparisons.append(dict(kind=kind,n=len(y),prediction_changes=int((p!=q).sum()),
        full_rescues_vs_control=int(((p==y)&(q!=y)).sum()),full_harms_vs_control=int(((p!=y)&(q==y)).sum()),
        logits_max_abs_difference=float((baseprobe['logits_he']-pp['logits_he']).abs().max())))
for src in ('stage0_pcebr_v7_final_original.py','stage0_pcebr_v5_original.py','pcebr_stage0_v7_run.py','pcebr_stage0_v7_prepare.py','pcebr_stage0_v7_continue.py','pcebr_stage0_v7_diagnose_m0.py','pcebr_stage0_v7_controls.py','pcebr_stage0_v7_archive.py'):
    (V7_DIR/src).write_bytes((OUT/'uploads'/src).read_bytes())
for path in sorted(V7_DIR.rglob('*')):
    if not path.is_file() or path.name in ('raw_artifacts.tar.gz','archive_index.json','archive_audit.json'):continue
    files.append(dict(name=str(path.relative_to(V7_DIR)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
v7_write('archive_audit.json',dict(server=PC_SERVER,files=files,raw_files=len(list((V7_DIR/'artifacts').glob('*.pt'))),
    retrieval_queries=nquery,retrieval_metrics_recomputed_exact=True,probes=probes,couplings=couplings,hop_comparisons=hop_comparisons,roll_probe_comparisons=probe_comparisons))
archive=V7_DIR/'raw_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for item in files:tar.add(V7_DIR/item['name'],arcname=item['name'])
RESULT=dict(server=PC_SERVER,file='pcebr_stage0_v7/raw_artifacts.tar.gz',bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),raw_files=len(list((V7_DIR/'artifacts').glob('*.pt'))),retrieval_queries=nquery,probes=len(probes))
v7_write('archive_index.json',RESULT)
print('PCV7_ARCHIVE',json.dumps(RESULT),flush=True)

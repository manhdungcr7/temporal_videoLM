"""Recompute counts from saved scores/logits and package run evidence."""
import hashlib
import json
import tarfile
import torch

v6_files, v6_probes, v6_nquery = [], [], 0
v6_moment_checks=[]
for path in sorted((V6_DIR/'artifacts').glob('*.pt')):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if '_retrieval_' in path.name:
        assert v6_orig_eval(obj['sim'],obj['examples'])==obj['metrics'],path.name
        v6_nquery+=len(obj['examples'])
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
        v6_probes.append(dict(file=path.name,correct4=int((p==y).sum()),total=len(y),correct3=int((p3==y[mask]).sum()),
            total3=int(mask.sum()),by_class=byclass,paired_correct=hits,paired_total=len(paired),pred_counts=torch.bincount(p,minlength=4).tolist()))
    if path.name.startswith('m3_'):
        assert torch.equal(obj['legacy_archived_source_prob'],obj['p_legacy_v5']) or float((obj['legacy_archived_source_prob']-obj['p_legacy_v5']).abs().max())<1e-6
        pv=obj['aud_fwd']['p_v_hat'];pa=obj['aud_fwd']['p_a_hat'];K=len(pv)
        centers01=(torch.arange(K,dtype=torch.float32)+0.5)/K
        gates=[]
        for lg in (obj['lg_v'],obj['lg_a']):
            g=torch.relu(torch.softmax(lg,dim=-1)-1./K).sum()
            gates.append(g*g/(g*g+0.05**2))
        expected=(gates[0]*gates[1])*((pv-pa)*centers01).sum()
        actual=obj['phi_fwd'][3]-obj['phi_fwd'][4]
        v6_moment_checks.append(dict(file=path.name,phi_w_plus_minus=float(actual),gated_normalized_mean_lag=float(expected),absolute_error=float((actual-expected).abs())))

v6_write('wasserstein_signed_mean_identity.json',v6_moment_checks)
roll_comparison=[]
base_eps=torch.load(V6_DIR/'artifacts/m2_episodes_21.pt',map_location='cpu',weights_only=False)['episodes']
for fname,kind in [('m2_episodes_31.pt','teacher_forced_bridge'),('m2_episodes_33.pt','oracle_anchor_vector')]:
    eps=torch.load(V6_DIR/'artifacts'/fname,map_location='cpu',weights_only=False)['episodes']
    assert [(e['label'],e['dir_tag'],e['pair_key']) for e in eps]==[(e['label'],e['dir_tag'],e['pair_key']) for e in base_eps]
    delta=torch.stack([e['full_feat'] for e in eps])-torch.stack([e['full_feat'] for e in base_eps])
    roll_comparison.append(dict(kind=kind,episodes=len(eps),mean_abs_feature_difference=float(delta.abs().mean()),
        max_abs_feature_difference=float(delta.abs().max()),same_hop1_flags=all(a['hop1_correct']==b['hop1_correct'] for a,b in zip(eps,base_eps)),
        same_hop2_flags=all(a['hop2_correct']==b['hop2_correct'] for a,b in zip(eps,base_eps))))
v6_write('hop_condition_comparison.json',roll_comparison)

for src in ('stage0_pcebr_v6_final_original.py','stage0_pcebr_v5_original.py','pcebr_stage0_v6_run.py','pcebr_stage0_v6_prepare.py','pcebr_stage0_v6_continue.py','pcebr_stage0_v6_controls.py'):
    (V6_DIR/src).write_bytes((OUT/'uploads'/src).read_bytes())
for path in sorted(V6_DIR.rglob('*')):
    if not path.is_file() or path.name in ('raw_artifacts.tar.gz','archive_index.json','archive_audit.json'):continue
    v6_files.append(dict(name=str(path.relative_to(V6_DIR)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
v6_write('archive_audit.json',dict(server=PC_SERVER,files=v6_files,raw_files=len(list((V6_DIR/'artifacts').glob('*.pt'))),
    retrieval_queries=v6_nquery,retrieval_metrics_recomputed_exact=True,probes=v6_probes))
archive=V6_DIR/'raw_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for item in v6_files:tar.add(V6_DIR/item['name'],arcname=item['name'])
RESULT=dict(server=PC_SERVER,file='pcebr_stage0_v6/raw_artifacts.tar.gz',bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),raw_files=len(list((V6_DIR/'artifacts').glob('*.pt'))),retrieval_queries=v6_nquery)
v6_write('archive_index.json',RESULT)
print('PCV6_ARCHIVE',json.dumps(RESULT),flush=True)

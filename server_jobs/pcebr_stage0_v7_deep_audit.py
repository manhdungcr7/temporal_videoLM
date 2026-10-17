"""Read stored tensors from archives without extraction; diagnose inactive joint path."""
import io
import json
import tarfile
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v7'
rows=[]
for server in (1,2):
    with tarfile.open(D/f'raw_artifacts_s{server}.tar.gz','r:gz') as tar:
        names={m.name for m in tar.getmembers()}
        for name in sorted(names):
            if not(name.startswith('artifacts/m3_') and name.endswith('.pt')):continue
            obj=torch.load(io.BytesIO(tar.extractfile(name).read()),map_location='cpu',weights_only=False)
            cname='artifacts/coupling_'+Path(name).stem+'.pt'
            cc=torch.load(io.BytesIO(tar.extractfile(cname).read()),map_location='cpu',weights_only=False)
            J=cc['J_aug'];C=cc['J_cond'];p1=cc['p1'];p2=cc['p2'];K1,K2=C.shape
            phi=obj['phi_fwd'];gate=cc['gate'];actual_mass=J[:K1,:K2].sum()
            # Correct normalization is an audit only; no additional QA forward or mainrun change.
            normalized=J[:K1,:K2]/actual_mass
            rows.append(dict(server=server,idx=int(Path(name).stem.split('_')[-1]),
                lg_a_max_abs=float(obj['lg_a'].abs().max()),lg_v_max_abs=float(obj['lg_v'].abs().max()),
                alpha_a=obj['aud_fwd']['alpha_a'],alpha_v=obj['aud_fwd']['alpha_v'],
                audio_content_excess=obj['slot_meta']['a_content_excess'],visual_content_excess=obj['slot_meta']['v_content_excess'],
                outer_gamma=obj['aud_fwd']['gamma_gate'],joint_gate=float(gate),
                effective_pair_interaction_max_abs=float((gate*cc['omega']).abs().max()),
                psi_max_abs=float(phi.abs().max()),P_null=float(cc['P_null']),
                real_block_mass=float(actual_mass),returned_J_cond_sum=float(C.sum()),
                returned_J_cond_row_error=float((C.sum(1)-p1).abs().max()),returned_J_cond_col_error=float((C.sum(0)-p2).abs().max()),
                audit_only_normalization_without_floor_sum=float(normalized.sum()),
                audit_only_row_error_without_floor=float((normalized.sum(1)-p1).abs().max()),
                audit_only_col_error_without_floor=float((normalized.sum(0)-p2).abs().max())))
out=dict(n=len(rows),all_joint_gate_zero=all(r['joint_gate']==0 for r in rows),all_psi_zero=all(r['psi_max_abs']==0 for r in rows),
    all_audio_logits_zero=all(r['lg_a_max_abs']==0 for r in rows),
    note='Original tensors are unmodified. Direct division by actual real mass shown only as a numerical diagnostic, not a new QA variant.',rows=rows)
(D/'deep_mechanism_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

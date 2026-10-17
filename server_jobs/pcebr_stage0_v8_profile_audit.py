import ast,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
def load(name):
    t=ast.parse((D/name).read_text(encoding='utf-8'))
    t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RESULT' for x in n.targets))]
    ns={};exec(compile(t,name,'exec'),ns);return ns
orig=load('stage0_pcebr_v8_original.py');fixed=load('stage0_pcebr_v8_logicfix.py')
clock=dict(v_indices=torch.arange(12),a_indices=torch.arange(12,24))
aud=dict(p_a_hat=torch.tensor([.85,.05,.05,.05]),p_v_hat=torch.tensor([.05,.05,.05,.85]),
    gamma_gate=1.,bhatt_sync=.5,rel_scalar=.7,w_l1_transport=.3,lag_dispersion_std=.1,
    alpha_v=.9,alpha_a=.9,regime_masses=dict(m_pre=.05,m_sync=.1,m_post=.85))
meta=dict(v_token_slot_ids=torch.arange(4).repeat_interleave(3),a_token_slot_ids=torch.arange(4).repeat_interleave(3),
    token_innov_v=torch.linspace(-1.,1.,12),token_innov_a=torch.linspace(-1.,1.,12))
rows=[]
for regime,rho in [('A->V',[1.,0.]),('V->A',[0.,1.]),('A->A',[1.,1.]),('V->V',[0.,0.])]:
    m=dict(meta,endpoint_modality_weights_A=rho)
    a=orig['build_caor_token_bias_profiles'](24,clock,m,aud)
    b=fixed['build_caor_token_bias_profiles'](24,clock,m,aud)
    rows.append(dict(regime=regime,original_v_l2=float(a['tok_bias_v'].norm()),original_a_l2=float(a['tok_bias_a'].norm()),
        fixed_v_l2=float(b['tok_bias_v'].norm()),fixed_a_l2=float(b['tok_bias_a'].norm()),
        v_change=float((a['tok_bias_v']-b['tok_bias_v']).abs().max()),a_change=float((a['tok_bias_a']-b['tok_bias_a']).abs().max())))
assert rows[0]['v_change']==0 and rows[0]['a_change']==0
assert rows[2]['fixed_v_l2']==0 and rows[3]['fixed_a_l2']==0
(D/'endpoint_profile_unit_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))

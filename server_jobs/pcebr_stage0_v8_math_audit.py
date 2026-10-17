"""Independent CPU checks of mechanistic claims, without QA labels or training."""
import ast, hashlib, json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
s=(D/'stage0_pcebr_v8_original.py').read_text(encoding='utf-8')
t=ast.parse(s)
t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RESULT' for x in n.targets))]
ns=dict(OUT=D);exec(compile(t,'v8_definitions','exec'),ns)
# Uniform marginals imply the bilinear expectation of any double-centered Omega
# is zero: a nontrivial interaction can change J_cond without changing P_null.
p=torch.ones(4)/4
om=4*torch.eye(4)
J,C,P=ns['compute_sinkhorn_joint_event_pair_posterior'](p,p,om,-.5,.8)
J0,C0,P0=ns['compute_sinkhorn_joint_event_pair_posterior'](p,p,torch.zeros_like(om),-.5,.8)
iv=torch.stack((torch.arange(4),torch.arange(1,5)),dim=1).float()
psi,_=ns['compute_joint_lag_distribution_features'](C,iv,iv,torch.tensor(1.))
psi0,_=ns['compute_joint_lag_distribution_features'](C0,iv,iv,torch.tensor(1.))
# Two query tokens are supported at disjoint times. This can still open the
# purported conjunctive gate because token support uses max over slots first.
disjoint=torch.tensor([[.8,0.],[0.,.8]])
gate=ns['compute_unified_gate_from_token_matrix'](disjoint,torch.zeros(2),'conjunctive')
result=dict(source_lf_sha256=hashlib.sha256(s.encode()).hexdigest(),
    uniform_marginal_null_probe=dict(pnull_interaction=float(P),pnull_zero=float(P0),conditional_coupling_l1=float((C-C0).abs().sum()),
        conclusion='Double-centered compatibility can change event pairing without changing the heuristic null score under uniform marginals'),
    lag_mean_fixed_marginals=dict(interaction=float(psi[12]),factorized=float(psi0[12]),
        conclusion='Mean lag is determined by the marginals, not a genuine joint-only functional'),
    disjoint_token_gate=dict(matrix=disjoint.tolist(),gate=gate,
        conclusion='Conjunctive gate does not guarantee tokens coexist in one event instance or one slot'))
(D/'independent_math_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

"""CPU-only independent algebra audit of the original coupling function."""
import ast
import hashlib
import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v7'
s=(D/'stage0_pcebr_v7_final_original.py').read_text(encoding='utf-8')
tree=ast.parse(s);tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RESULT' for x in n.targets))]
ns=dict(OUT=D);exec(compile(tree,'original_v7_definitions_only','exec'),ns)
rows=[]
for K,seed in [(5,101),(6,202),(9,303)]:
    g=torch.Generator().manual_seed(seed)
    p=torch.softmax(torch.randn(K,generator=g),0);q=torch.softmax(torch.randn(K,generator=g),0)
    omega=torch.randn((K,K),generator=g)*.5;gate=torch.tensor(.8);z=torch.tensor(-.7)
    J,C,P=ns['compute_sinkhorn_joint_event_pair_posterior'](p,q,omega,z,gate)
    J0,C0,P0=ns['compute_sinkhorn_joint_event_pair_posterior'](p,q,torch.zeros_like(omega),z,gate)
    a=(gate*torch.sigmoid(-z)).clamp(1e-7,1-1e-7)
    rows.append(dict(K=K,seed=seed,P_null=float(P),expected_one_minus_a_squared=float(1-a*a),
        analytical_error=float((P-(1-a*a)).abs()),change_null_zero_omega=float((P-P0).abs()),
        conditional_coupling_change_L1=float((C-C0).abs().sum())))
bridges=[]
for path in sorted(D.glob('bridge_*_s*.pt')):
    b=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
    bridges.append(dict(file=path.name,scalars={k:v.tolist() for k,v in b.items() if v.numel()<=6},buffer_shapes={k:list(v.shape) for k,v in b.items() if v.numel()>6}))
out=dict(note='CPU diagnostic only; not QA evaluation or an alternative training run',source_sha256=hashlib.sha256(s.encode()).hexdigest(),coupling=rows,available_bridge_parameters=bridges)
(D/'independent_cpu_math.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

import ast,json,torch
from pathlib import Path
from yann_tam_probe_model import TAMProbe,target_density
R=Path(__file__).resolve().parents[1];D=R/'results/yann_temporal_research'
torch.manual_seed(20261016);m=TAMProbe(hidden=16,width=8,roles=2)
b=torch.randn(7,16);q=torch.randn(16);spans=torch.stack((torch.arange(7).float(),torch.arange(7).float()+.5),1);mods=torch.tensor([0,0,0,1,1,1,1]);h=torch.randn(3,16)
y,p,_=m(b,q,spans,mods,7.,h);identity=float((y-h).abs().max());assert identity==0
with torch.no_grad():m.out.weight.normal_(std=.1)
x,px,_=m(b,q,spans,mods,7.,h);ix=torch.randperm(7)
y,py,_=m(b[ix],q,spans[ix],mods[ix],7.,h);perr=float((x-y).abs().max());assert perr<1e-5
z,pz,_=m(b,q,spans.flip(0),mods,7.,h)
address_effect=float((x-z).abs().max());score_invariance=float((px-pz).abs().max());assert address_effect>0 and score_invariance==0
loss=(x-torch.randn_like(x)).square().mean();loss.backward();assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
files=list((R/'server_jobs').glob('yann_*.py'))
for f in files:ast.parse(f.read_text(encoding='utf-8'))
result=dict(zero_init_error=identity,active_tuple_permutation_error=perr,changed_address_output_difference=address_effect,
    address_only_change_leaves_retrieval_scores_unchanged=score_invariance,finite_gradient=True,parsed_source_files=len(files),note='CPU synthetic implementation checks; no QA performance claims')
(D/'tam_probe_unit_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

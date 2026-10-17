import json,torch
from pathlib import Path
from yann_tam_probe_model import TAMBindingProbe
D=Path(__file__).resolve().parents[1]/'results/yann_temporal_research'
torch.manual_seed(20261017);m=TAMBindingProbe(hidden=16,width=8,roles=2)
b=torch.randn(7,16);q=torch.randn(16);spans=torch.stack((torch.arange(7).float(),torch.arange(7).float()+.5),1);mods=torch.tensor([0,0,0,1,1,1,1]);h=torch.randn(3,16)
with torch.no_grad():m.out.weight.normal_(std=.1)
uniform=torch.full((2,7),1/7)
y,_,_=m(b,q,spans,mods,7.,h,force_attention=uniform);uniform_err=float((y-h).abs().max().detach());assert uniform_err<1e-6,uniform_err
z,_,_=m(b,q,torch.zeros_like(spans),mods,7.,h);null_err=float((z-h).abs().max().detach());assert null_err<1e-6,null_err
x,_,_=m(b,q,spans,mods,7.,h);ix=torch.randperm(7);p,_,_=m(b[ix],q,spans[ix],mods[ix],7.,h)
perm_err=float((x-p).abs().max().detach());assert perm_err<1e-6
source_uniform=uniform.clone();source_uniform[:]=0;source_uniform[:,:3]=1/3
s,_,_=m(b,q,spans,mods,7.,h,force_attention=source_uniform);source_uniform_effect=float((s-h).abs().max().detach())
loss=(x-torch.randn_like(x)).square().mean();loss.backward();assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
result=dict(active_global_uniform_native_error=uniform_err,active_identical_addresses_native_error=null_err,tuple_permutation_error=perm_err,
    uniform_within_one_modality_effect=source_uniform_effect,finite_gradients=True,note='Module identity only; no claim that nonuniform retrieval is semantically correct or calibrated.')
(D/'tam_binding_unit_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

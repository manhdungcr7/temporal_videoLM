"""Check scientific equivalence and causal isolation of the optimized Round5 math."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('chronodesmear_math',Path(__file__).parents[1]/'server_jobs/chronodesmear_math.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_prefix_echo_matches_dense_and_preserves_norm():
    torch.manual_seed(17)
    times=torch.arange(28).repeat_interleave(3).float()*.5
    indices=torch.arange(len(times))
    values=torch.randn(len(times),19)
    plan=m.interval_plan(indices,times)
    dt=times[:,None]-times[None,:]
    weights=((dt>=1.8)&(dt<=12.)).float()
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1)
    torch.testing.assert_close(m.interval_mean(values,plan),weights@values,atol=2e-6,rtol=2e-5)
    cleaned=m.desmear_values(values,plan,.26)
    torch.testing.assert_close(cleaned.norm(dim=-1),values.norm(dim=-1))
    assert torch.equal(cleaned[~plan['has_previous']],values[~plan['has_previous']])
    assert torch.equal(m.desmear_values(values,plan,0),values)
    # Alter future values: no earlier output may change.
    changed=values.clone();changed[times>=9.]+=100
    torch.testing.assert_close(m.desmear_values(changed,plan,.26)[times<9.],cleaned[times<9.])

def test_stratified_mask_matches_original_and_covers_time_bins():
    source=Path(__file__).parents[1]/'results/chronodesmear40/proposal_original.py'
    import ast
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_stratified_static_visual_mask')
    namespace={'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original_selector','exec'),namespace)
    torch.manual_seed(19)
    times=torch.arange(30).repeat_interleave(4).float()*.5
    is_v=torch.ones(len(times),dtype=torch.bool)
    values=torch.randn(len(times),17)
    original,scores,indices=namespace[node.name](values,is_v,times,15.)
    actual,actual_scores,actual_indices,bins=m.static_visual_mask(values,is_v,times,15.)
    torch.testing.assert_close(actual_scores,scores,atol=2e-6,rtol=2e-5)
    assert torch.equal(original,actual)
    assert torch.equal(indices,actual_indices)
    assert all(b['static']>0 and b['static']<b['total'] for b in bins)

def test_masks_preserve_causality_and_block_indirect_source_paths():
    payload=dict(tokens=9,is_v=torch.tensor([0,1,1,0,1,1,0,0,0],dtype=torch.bool),
        is_a=torch.tensor([0,0,0,1,0,0,1,0,0],dtype=torch.bool),
        sec=torch.tensor([0.,0.,0.,.2,2.,2.,2.2,0.,0.]),spg=1.)
    static=torch.tensor([0,1,0,0,1,0,0,0,0],dtype=torch.bool)
    for mode in ('static_plus_aud','v_all_only','hard_sync','anti_sync'):
        mask=m.counterfactual_mask(payload,mode,torch.float16,static,(.1,1.))
        neg=torch.finfo(mask.dtype).min
        assert bool((mask[0,0][torch.ones(9,9,dtype=torch.bool).triu(1)]==neg).all())
        if mode=='static_plus_aud':
            blocked=payload['is_v']&~static
            assert bool((mask[0,0][(~blocked)[:,None]&blocked[None,:]]==neg).all())

if __name__=='__main__':
    for test in (test_prefix_echo_matches_dense_and_preserves_norm,
                 test_stratified_mask_matches_original_and_covers_time_bins,
                 test_masks_preserve_causality_and_block_indirect_source_paths):
        test();print('PASS',test.__name__,flush=True)

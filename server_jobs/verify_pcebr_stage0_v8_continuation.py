"""Local tensor/formula and artifact audit for the two completed diagnostic rounds."""
import hashlib,io,json,tarfile
from pathlib import Path
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1]
L=R/'results/pcebr_stage0_v8_layerlocal';A=R/'results/pcebr_stage0_v8_readout'
ls=json.loads((L/'summary.json').read_text(encoding='utf-8'))
rs=json.loads((A/'summary.json').read_text(encoding='utf-8'))
assert len(ls['rows'])==len(rs['rows'])==40
assert {r['idx'] for r in ls['rows']}=={r['idx'] for r in rs['rows']}
tensor_audits=[];errors=[];random_max_error=0.;random_all_bitwise=True
for s in (1,2):
    for d in (L,A):
        idx=json.loads((d/f'archive_index_s{s}.json').read_text())
        assert hashlib.sha256((d/f'results_s{s}.tar.gz').read_bytes()).hexdigest()==idx['sha256']
    with tarfile.open(L/f'results_s{s}.tar.gz') as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.pt'):continue
            raw=torch.load(io.BytesIO(tar.extractfile(member).read()),map_location='cpu',weights_only=False)
            ds=raw['directions'];idx=raw['idx']
            for ll in (14,16,18,20,22):
                ev=F.normalize(raw['counterfactual_states']['supp'][ll]-raw['counterfactual_states']['foil'][ll],dim=-1)
                od=F.normalize(raw['counterfactual_states']['supp'][ll]-raw['counterfactual_states']['rev'][ll],dim=-1)
                assert torch.equal(raw['ev'][ll],ev) and torch.equal(raw['od'][ll],od)
                expected=F.normalize(.45*ev+.35*od,dim=-1)
                assert torch.equal(ds['layerlocal_native'][ll],expected)
                assert torch.equal(ds['layerlocal_negated'][ll],-expected)
                assert torch.equal(ds['layerlocal_ev'][ll],ev)
            assert torch.equal(ds['shared22_native'],ds['layerlocal_native'][22])
            for seed in (101,202,303):
                rng=torch.Generator().manual_seed(idx+seed)
                for ll in (14,16,18,20,22):
                    expected=F.normalize(torch.randn(raw['ev'][ll].shape,generator=rng),dim=-1)
                    actual=ds[f'random_{seed}'][ll]
                    random_all_bitwise=random_all_bitwise and torch.equal(actual,expected)
                    err=float((actual-expected).abs().max())
                    random_max_error=max(random_max_error,err)
                    assert err<=1e-6,('cross_environment_random_replay',idx,seed,ll,err)
            tensor_audits.append(idx)
    with tarfile.open(A/f'results_s{s}.tar.gz') as tar:
        e=json.load(tar.extractfile('initial_replay_error.json'))
        assert 'replay' in e['error']
        errors.append(dict(server=s,initial_error_retained=True))
assert len(tensor_audits)==40 and len(set(tensor_audits))==40
record=dict(n=40,n_origins=ls['n_origins'],all_counterfactual_differences_and_directions_recomputed_bitwise=True,
    all_three_random_seeds_recomputed_within_1e_minus6=True,random_replay_all_bitwise=random_all_bitwise,random_replay_max_abs_error=random_max_error,
    local_torch=torch.__version__,server_torch='2.11.0+cu128',
    random_replay_note='Cross-environment bitwise check failed; report actual numerical error separately. Exact stored tensor bytes remain SHA-verified.',
    all_archive_hashes_valid=True,initial_wrapper_errors_retained=errors,
    layerlocal_prior_replay_max_error=max(max(r['prior40_replay_errors'].values()) for r in ls['rows']),
    readout_prior_replay_max_error=max(max(r['prior_replay_errors'].values()) for r in rs['rows']),
    goal_status='active',qa_improvement_requirement='not_achieved',independent_validation_requirement='not_achieved',novelty_requirement='not_proven',
    next_step='Independent same-recording interval verification diagnostic; no additional coefficient search on frozen40')
(L/'continuation_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))

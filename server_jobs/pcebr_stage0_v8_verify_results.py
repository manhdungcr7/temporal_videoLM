import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
checks=[]
for server in (1,2):
    p=D/f'all_runs_artifacts_s{server}.tar.gz'
    ix=json.loads((D/f'all_runs_archive_index_s{server}.json').read_text())
    assert p.stat().st_size==ix['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==ix['sha256']
    with tarfile.open(p,'r:gz') as tar:
        names=tar.getnames()
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names)
        audit=json.load(tar.extractfile('archive_audit.json'))
        for rec in audit['artifacts']:
            name=f"{rec['variant']}/compact_artifacts/{rec['file']}"
            data=tar.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest()==rec['compact_sha256']
            assert rec['all_tensor_values_bitwise_equal']
        for tag in ('original','logicfix','corrected'):
            archived=json.load(tar.extractfile(tag+'/samples.json'))
            local=json.loads((D/f'{tag}_samples_s{server}.json').read_text(encoding='utf-8-sig'))
            assert archived==local
            complete=json.load(tar.extractfile(tag+'/completion.json'))
            assert complete['backbone_parameters_unchanged'] and complete['n_samples']==4
        for tag in ('logicfix','corrected'):
            spec=json.load(tar.extractfile(tag+'/runtime_patch_spec.json'))
            data=tar.extractfile(tag+'/effective_source.py').read()
            assert hashlib.sha256(data).hexdigest()==spec['effective_source_sha256']
            (D/f'effective_{tag}_s{server}.py').write_bytes(data)
            (D/f'runtime_patch_spec_{tag}_s{server}.json').write_text(json.dumps(spec,indent=2))
        checks.append(dict(server=server,archive_sha256=ix['sha256'],n_compact_artifacts=len(audit['artifacts']),all_recorded_tensor_values_equal=True,
            all_backbone_parameter_versions_unchanged=True))
for tag in ('logicfix','corrected'):
    a=(D/f'effective_{tag}_s1.py').read_bytes();b=(D/f'effective_{tag}_s2.py').read_bytes();assert a==b
component=[r for s in (1,2) for r in json.loads((D/f'component_ablation_s{s}.json').read_text())]
assert len(component)==8
assert max(r['direct_replay_max_probability_error'] for r in component)==0.0
summary=dict(archives=checks,effective_algorithm_source_equal_across_shards=True,component_replay_max_probability_error=0.,
    component_scores={k:sum(r['conditions'][k]['pred']==r['gold'] for r in component) for k in component[0]['conditions']})
(D/'verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

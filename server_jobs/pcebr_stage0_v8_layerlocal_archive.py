"""Audit inputs, completed predictions and compact directions; archive current round only."""
import hashlib,json,tarfile
from pathlib import Path
import torch
D=OUT/'pcebr_stage0_v8_layerlocal'
completion=json.loads((D/'completion.json').read_text())
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
rows=completion['samples']
assert {r['idx'] for r in rows}=={r['idx'] for r in manifest[f'server{PC_SERVER}']}
assert completion['backbone_versions_unchanged']
audit=[]
for row in rows:
    idx=row['idx']
    art=D/'artifacts'/f'{idx}.pt'
    assert hashlib.sha256(art.read_bytes()).hexdigest()==row['artifact_sha256']
    raw=torch.load(art,map_location='cpu',weights_only=False)
    p=OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['input_cache_sha256']
    assert max(row['prior40_replay_errors'].values())<=1e-6
    assert row['kernel_noop_err']<1e-5
    assert len(raw['directions']['layerlocal_native'])==5
    for ll,d in raw['directions']['layerlocal_native'].items():
        assert abs(float(d.norm())-1)<1e-5
        assert torch.equal(raw['directions']['layerlocal_negated'][ll],-d)
    audit.append(dict(idx=idx,input_cache_sha256=row['input_cache_sha256'],artifact_sha256=row['artifact_sha256']))
(D/'audit.json').write_text(json.dumps(dict(server=PC_SERVER,n=20,all_inputs_and_artifacts_verified=True,records=audit),indent=2))
tarpath=D/'results.tar.gz'
with tarfile.open(tarpath,'w:gz') as tar:
    for p in sorted(D.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json','.pt'):
            tar.add(p,arcname=str(p.relative_to(D)))
RESULT=dict(server=PC_SERVER,bytes=tarpath.stat().st_size,sha256=hashlib.sha256(tarpath.read_bytes()).hexdigest(),n=20)
(D/'archive_index.json').write_text(json.dumps(RESULT,indent=2))
print('LAYERLOCAL_ARCHIVE',json.dumps(RESULT),flush=True)

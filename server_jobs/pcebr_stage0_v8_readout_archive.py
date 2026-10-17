import hashlib,json,tarfile
D=OUT/'pcebr_stage0_v8_readout'
C=json.loads((D/'completion.json').read_text())
parent=json.loads((OUT/'pcebr_stage0_v8_layerlocal/completion.json').read_text())
P={r['idx']:r for r in parent['samples']}
assert C['backbone_versions_unchanged'] and len(C['samples'])==20
assert {r['idx'] for r in C['samples']}==set(P)
assert C['parent_source_sha256']==parent['effective_source_sha256']
assert C['route_source_sha256']==hashlib.sha256((D/'route_source.py').read_bytes()).hexdigest()
for r in C['samples']:
    assert r['source_artifact_sha256']==P[r['idx']]['artifact_sha256']
    p=OUT/'argon_latent40/decoder_cache'/f"{r['idx']}_rot0.pt"
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['input_cache_sha256']
    assert max(r['prior_replay_errors'].values())<=1e-6
(D/'audit.json').write_text(json.dumps(dict(n=20,all_inputs_sources_replays_verified=True,server=PC_SERVER),indent=2))
archive=D/'results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for f in sorted(D.iterdir()):
        if f.is_file() and f.suffix in ('.json','.py'):tar.add(f,arcname=f.name)
RESULT=dict(server=PC_SERVER,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),n=20)
(D/'archive_index.json').write_text(json.dumps(RESULT,indent=2))
print('READOUT_ARCHIVE',json.dumps(RESULT),flush=True)

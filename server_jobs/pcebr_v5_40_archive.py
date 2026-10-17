"""Recompute every rotation aggregation and archive raw evidence without decoder caches."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import torch

d=OUT/'pcebr_v5_40'
rows=json.loads((d/'results.json').read_text(encoding='utf-8'))
assert len(rows)==20
files=[];n_rot=0;replay_errors=[];parity=[]
for row in rows:
    k=len(row['scores']['native'])
    totals={name:np.zeros(k,dtype=np.float64) for name in ('native','v5')}
    for rot in range(row['rotation_count']):
        p=d/'rotations'/f"{row['idx']}_rot{rot}.json"
        record=json.loads(p.read_text(encoding='utf-8'))
        state=torch.load(d/'states'/f"{row['idx']}_rot{rot}.pt",map_location='cpu',weights_only=True)
        assert state['hook_rows'] and not set(state['hook_rows'])&set(state['option_rows'])
        for name,key in [('native','native_logits'),('v5','routed_logits')]:
            assert torch.equal(state[key],torch.tensor(record['logits'][name],dtype=state[key].dtype))
            assert torch.allclose(torch.log_softmax(state[key],dim=-1),torch.tensor(record['logprobs'][name]),rtol=0,atol=0)
            for slot,original in enumerate(record['perm']):totals[name][original]+=record['logprobs'][name][slot]
        replay_errors.append(record['native_replay_max_error']);parity.append(record['parity_checks'])
        n_rot+=1
    for name in totals:
        assert np.array_equal(totals[name],row['scores'][name])
        assert 'ABCD'[int(totals[name].argmax())]==row['predictions'][name]
for pattern in ('rotations/*.json','states/*','*.json'):
    for p in sorted(d.glob(pattern)):
        if p.name in ('archive_audit.json','archive_index.json'):continue
        files.append(dict(name=str(p.relative_to(d)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
audit=dict(server=O40_SERVER,samples=len(rows),rotations=n_rot,max_replay_error=max(replay_errors),
           all_rotations_reaggregated_exact=True,all_raw_logits_logprobs_verified=True,
           parity_failures=sum(not x['original_micro_asserts_pass'] for x in parity),
           zero_gate_rotations=sum(x['zero_gate'] for x in parity),files=files,
           hooks=sum(len(l._forward_hooks) for l in MODEL_.model.layers),
           cuda_ok=bool(torch.isfinite(torch.ones(1,device='cuda:0')).all()))
(d/'archive_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
archive=d/'raw_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for item in files:tar.add(d/item['name'],arcname=item['name'])
RESULT=dict(server=O40_SERVER,file='pcebr_v5_40/raw_artifacts.tar.gz',bytes=archive.stat().st_size,
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),samples=len(rows),rotations=n_rot)
(d/'archive_index.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('OBTVOR40_ARCHIVE',json.dumps(RESULT),flush=True)

"""Verify completed interval cases and archive crops, scores and small hidden tensors."""
import hashlib,inspect,json,tarfile
from pathlib import Path
D=OUT/'native_interval_verification';c=json.loads((D/'completion.json').read_text())
M=json.loads((OUT/'uploads/native_interval_manifest.json').read_text())
pairs=[p for p in M['pairs'] if p['server']==PC_SERVER]
expected={(p['pair_id'],w,mode) for p in pairs for w in ('positive','negative') for mode in ('av','audio','vision')}
actual={(r['pair_id'],r['window'],r['mode']) for r in c['records']}
assert expected==actual and len(c['records'])==len(expected)
assert c['backbone_versions_unchanged'] and c['remaining_hooks']==0
for r in c['records']:
    crop=D/'crops'/f"{r['pair_id']}_{r['window']}.mp4"
    art=D/'artifacts'/f"{r['pair_id']}_{r['window']}_{r['mode']}.pt"
    assert hashlib.sha256(crop.read_bytes()).hexdigest()==r['crop_sha256']
    assert hashlib.sha256(art.read_bytes()).hexdigest()==r['artifact_sha256']
    assert r['presence_input']['prefix_ids_sha256']==r['description_input']['prefix_ids_sha256']
    cnt=r['presence_input']['media_token_counts']
    assert (cnt['audio']>0)==(r['mode'] in ('audio','av'))
    assert (cnt['vision']>0)==(r['mode'] in ('vision','av'))
(D/'audit.json').write_text(json.dumps(dict(n_pairs=len(pairs),n_cases=len(c['records']),all_crop_artifact_input_contract_checks_passed=True),indent=2))
(D/'manifest.json').write_bytes((OUT/'uploads/native_interval_manifest.json').read_bytes())
(D/'helpers.json').write_text(json.dumps(dict(helpers={n:inspect.getsource(globals()[n]) for n in ('prepare_video','qwen_inputs','move_inputs','conversation_for')},
    model=MODEL_.config._name_or_path,revision=getattr(MODEL_.config,'_commit_hash',None)),indent=2))
archive=D/'results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for f in sorted(D.rglob('*')):
        if f.is_file() and f.suffix in ('.json','.py','.md','.pt','.mp4') and 'originals' not in f.relative_to(D).parts:
            tar.add(f,arcname=str(f.relative_to(D)))
index=dict(server=PC_SERVER,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),n_pairs=len(pairs),n_cases=len(c['records']))
(D/'archive_index.json').write_text(json.dumps(index,indent=2))
RESULT=index;print('INTERVAL_ARCHIVE',json.dumps(index),flush=True)

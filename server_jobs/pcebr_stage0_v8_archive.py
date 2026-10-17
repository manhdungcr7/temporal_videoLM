"""Compact view-backed tensors without changing values, hash and archive all v8 artifacts."""
import hashlib, io, json, tarfile, time
from pathlib import Path
import torch
D=OUT/'pcebr_stage0_v8'
def compact(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
    if isinstance(x,list):return [compact(v) for v in x]
    if isinstance(x,tuple):return tuple(compact(v) for v in x)
    return x
def same(a,b):
    if torch.is_tensor(a):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
records=[]
for tag in ('original','logicfix'):
    folder=D/tag
    if not folder.exists():continue
    compact_dir=folder/'compact_artifacts';compact_dir.mkdir(exist_ok=True)
    for f in sorted((folder/'artifacts').glob('*.pt')):
        x=torch.load(f,map_location='cpu',weights_only=False)
        y=compact(x);assert same(x,y)
        out=compact_dir/f.name;torch.save(y,out)
        records.append(dict(variant=tag,file=f.name,original_bytes=f.stat().st_size,compact_bytes=out.stat().st_size,
            original_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),compact_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),all_tensor_values_bitwise_equal=True))
input_records=[]
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
for row in manifest[f'server{PC_SERVER}'][:4]:
    path=OUT/'argon_latent40/decoder_cache'/f"{row['idx']}_rot0.pt"
    input_records.append(dict(idx=row['idx'],bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(D/'archive_audit.json').write_text(json.dumps(dict(server=PC_SERVER,artifacts=records,input_cache_hashes=input_records),indent=2))
archive=D/'raw_compact_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for f in D.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.py','.pt') and 'artifacts' not in f.relative_to(D).parts:
            tar.add(f,arcname=str(f.relative_to(D)))
    for f in (OUT/'uploads').glob('*'):
        if f.name in ('stage0_pcebr_v8_original.py','stage0_pcebr_v8_logicfix.py','pcebr_stage0_v8_token_interface.py','pcebr_stage0_v8_run.py','research40_manifest.json'):
            tar.add(f,arcname='sources/'+f.name)
index=dict(server=PC_SERVER,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),n_artifacts=len(records))
(D/'archive_index.json').write_text(json.dumps(index,indent=2))
RESULT=index
print('V8_ARCHIVE',json.dumps(index),flush=True)

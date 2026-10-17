"""Audit all40 exact prompt roundtrips and persist output/caches provenance."""
import hashlib,json,tarfile
import torch
D=OUT/'pcebr_stage0_v8/clc40'
m=json.loads((OUT/'uploads/research40_manifest.json').read_text())
records=[]
for r in m[f'server{PC_SERVER}']:
    pth=OUT/'argon_latent40/decoder_cache'/f"{r['idx']}_rot0.pt"
    p=torch.load(pth,map_location='cpu',weights_only=False)
    ids=p['input_ids'][0,p['last_media']+1:].tolist()
    txt=PROC.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    ok=PROC.tokenizer(txt,add_special_tokens=False)['input_ids']==ids
    assert ok,r['idx']
    records.append(dict(idx=r['idx'],tail_tokenizer_roundtrip=ok,input_cache_sha256=hashlib.sha256(pth.read_bytes()).hexdigest()))
(D/'input_audit.json').write_text(json.dumps(records,indent=2))
out=D/'results_archive.tar.gz'
with tarfile.open(out,'w:gz') as tar:
    for f in D.glob('*.json'):tar.add(f,arcname=f.name)
RESULT=dict(n=len(records),all_roundtrip_ok=True,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(D/'archive_index.json').write_text(json.dumps(RESULT,indent=2))
print('V8_40_AUDIT',json.dumps(RESULT),flush=True)

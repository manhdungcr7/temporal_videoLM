"""Preserve actual per-recording fitted inputs, not just optimizer statistics."""
import concurrent.futures,hashlib,json,time,urllib.error
import torch
from argon_remote import ROOT,call

FOLDER=ROOT/'results/latent_slot_opt40'

def archive(item):
    server,row=item;idx=row['idx'];path=FOLDER/'fits'/f'{idx}_optimized.pt'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                blob=call(server,f'/file?path=latent_slot_opt40/{idx}_optimized.pt',raw=True,timeout=60);break
            except (urllib.error.URLError,TimeoutError,ConnectionResetError):
                if attempt==3:raise
                time.sleep(.5*2**attempt)
        path.write_bytes(blob)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['runtime_sha256']==row['runtime_sha256']
    for variants in saved['parameters'].values():
        assert all(torch.isfinite(z).all() for z in variants.values())
    return dict(idx=idx,server=server,bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),runtime_sha256=saved['runtime_sha256'])

if __name__=='__main__':
    items=[(s,json.loads(x)) for s in (1,2) for x in (FOLDER/f'results_s{s}.jsonl').read_text().splitlines() if x.strip()]
    assert len(items)==40
    with concurrent.futures.ThreadPoolExecutor(3) as pool:records=list(pool.map(archive,items))
    result=dict(n=len(records),bytes=sum(x['bytes'] for x in records),files=records)
    (FOLDER/'fit_archive_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))

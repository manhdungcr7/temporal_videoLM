import json
from urllib.error import HTTPError
from argon_remote import ROOT,call
rows=[]
for s in (1,2):
    for n in range(4):
        try:
            data=call(s,f'/file?path=pcebr_stage0_v7/artifacts/hop2_bridge_{n:02d}.pt',raw=True,timeout=10)
            (ROOT/f'results/pcebr_stage0_v7/bridge_{n:02d}_s{s}.pt').write_bytes(data)
            rows.append(dict(server=s,bridge=n,bytes=len(data)))
        except HTTPError as e:
            rows.append(dict(server=s,bridge=n,status=e.code))
print(json.dumps(rows))

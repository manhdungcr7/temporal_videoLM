import concurrent.futures,hashlib,json,time
from argon_remote import ROOT,call,upload
D=ROOT/'results/yann_temporal_research'
def get(s):
    b=call(s,'/file?path=yann_temporal_research/probe_selected.pt',raw=True);p=D/f'r3_probe_s{s}.pt';p.write_bytes(b)
    return s,p,hashlib.sha256(b).hexdigest()
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(get,(1,2)))
out=[]
def retry_upload(target,p):
    for attempt in range(3):
        try:return upload(target,p)
        except Exception:
            if attempt==2:raise
            time.sleep(2)
for s,p,h in rows:
    for target in (1,2):out.append(dict(source=s,target=target,sha256=h,upload=retry_upload(target,p)))
(D/'probe_exchange.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

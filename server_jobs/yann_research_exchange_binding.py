import concurrent.futures,hashlib,json,time
from argon_remote import ROOT,call,upload
D=ROOT/'results/yann_temporal_research'
def get(s):
    ix=json.loads(call(s,'/file?path=yann_temporal_research/bindingallpack.json',raw=True));p=D/f'binding_all_features_s{s}.tar.gz'
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=ix['sha256']:p.write_bytes(call(s,'/file?path='+ix['archive'],raw=True,timeout=180))
    assert len(p.read_bytes())==ix['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==ix['sha256']
    ip=D/f'bindingallpack_s{s}.json';ip.write_text(json.dumps(ix,indent=2),encoding='utf-8');return s,p,ip,ix
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(get,(1,2)))
out=[]
for s,p,ip,ix in rows:
    for f in (p,ip):
        for attempt in range(3):
            try:r=upload(3-s,f);break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
        out.append(dict(source=s,target=3-s,file=f.name,upload=r))
(D/'binding_exchange.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

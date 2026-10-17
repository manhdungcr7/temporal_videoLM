"""Relay only frozen public-data feature archives between the two authorized runtimes."""
import concurrent.futures,hashlib,json
from argon_remote import ROOT,call,upload
D=ROOT/'results/yann_temporal_research';D.mkdir(exist_ok=True)
def get(s):
    ix=json.loads(call(s,'/file?path=yann_temporal_research/pack.json',raw=True))
    p=D/f'supervision_features_s{s}.tar.gz'
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=ix['sha256']:
        p.write_bytes(call(s,'/file?path='+ix['archive'],raw=True,timeout=180))
    assert p.stat().st_size==ix['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==ix['sha256']
    (D/f'pack_s{s}.json').write_text(json.dumps(ix,indent=2),encoding='utf-8')
    return s,p,ix
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(get,(1,2)))
def relay(row):
    s,p,ix=row;peer=3-s
    a=upload(peer,p);b=upload(peer,D/f'pack_s{s}.json')
    return dict(from_server=s,to_server=peer,archive_sha256=ix['sha256'],bytes=ix['bytes'],upload=a,index_upload=b)
with concurrent.futures.ThreadPoolExecutor(2) as pool:result=list(pool.map(relay,rows))
(D/'exchange.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

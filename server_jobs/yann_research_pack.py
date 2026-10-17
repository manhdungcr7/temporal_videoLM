"""Package public-data frozen features for identical matched training on both servers."""
import hashlib,json,tarfile
YR_DIR=OUT/'yann_temporal_research'
files=sorted((YR_DIR/'perception48').glob('*.pt'))
index=[dict(name=str(p.relative_to(YR_DIR)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
p=YR_DIR/f'supervision_features_s{YR_SERVER}.tar.gz'
with tarfile.open(p,'w:gz',compresslevel=1) as z:
    for f in files:z.add(f,arcname=str(f.relative_to(YR_DIR)),recursive=False)
RESULT=dict(server=YR_SERVER,files=index,archive=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(YR_DIR/'pack.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in RESULT.items() if k!='files'}),flush=True)

"""Inspect public human annotations and a remote ZIP index without downloading a corpus."""
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/yann_temporal_research/data';D.mkdir(exist_ok=True)
BASE='https://storage.googleapis.com/dm-perception-test/'
def fetchsmall(url,limit=12_000_000):
    with urllib.request.urlopen(url,timeout=60) as r:
        data=r.read(limit+1)
    if len(data)>limit:raise ValueError('metadata download exceeded bound')
    return data
record={}
for name in ('mc_question_train_annotations.zip','sound_localisation_train_annotations.zip','action_localisation_train_annotations.zip'):
    p=D/name
    if not p.exists():p.write_bytes(fetchsmall(BASE+'zip_data/'+name))
    with zipfile.ZipFile(p) as z:
        members={}
        for info in z.infolist():
            if info.filename.endswith('.json'):
                obj=json.loads(z.read(info.filename));out=D/Path(info.filename).name;out.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
                if isinstance(obj,dict):sample={k:obj[k] for k in list(obj)[:1]}
                else:sample=obj[:1]
                members[info.filename]=dict(type=type(obj).__name__,length=len(obj),sample=sample)
        record[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),members=members)
p=D/'cut_frame_mapping_train.json'
if not p.exists():p.write_bytes(fetchsmall(BASE+'misc/cut_frame_mapping_train.json'))
class RemoteZipFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:self.length=int(r.headers['Content-Length'])
        self.pos=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset;return self.pos
    def read(self,n=-1):
        n=self.length-self.pos if n<0 else min(n,self.length-self.pos)
        if n==0:return b''
        if n>5_000_000:raise ValueError('index read too large')
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        with urllib.request.urlopen(req,timeout=60) as r:
            if r.status!=206:raise ValueError('server did not honor byte range')
            data=r.read(n+1)
        if len(data)!=n:raise ValueError((len(data),n))
        self.pos+=n;self.transferred+=n;return data
try:
    rf=RemoteZipFile(BASE+'zip_data/train_videos.zip')
    with zipfile.ZipFile(rf) as z:index=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size,offset=i.header_offset) for i in z.infolist()]
    (D/'train_video_zip_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    record['video_index']=dict(archive_bytes=rf.length,index_bytes_transferred=rf.transferred,members=len(index),examples=index[:3])
except Exception as e:record['video_index_error']=str(e)
(D/'annotation_inventory.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2)[:16000])

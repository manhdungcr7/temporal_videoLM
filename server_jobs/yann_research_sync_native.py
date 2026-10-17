"""Incremental verified transport of completed public cache files, no credentials in artifacts."""
import concurrent.futures,hashlib,json,time
from urllib.parse import quote
from argon_remote import ROOT,call,upload
D=ROOT/'results/yann_temporal_research';temp=D/'native_transport';temp.mkdir(exist_ok=True)
indexpath=D/'native_transport_index.json'
done=json.loads(indexpath.read_text()) if indexpath.exists() else []
known={(x['source'],x['file']) for x in done}
todo=[]
for s in (1,2):
    try:meta=call(s,'/file?path=yann_temporal_research/nativecache_partial.json',raw=True);rows=json.loads(meta)['rows']
    except Exception:continue
    for r in rows:
        for v in (['N','O','A','V','AV'] if r['paired_questions'] else ['N']):
            for suffix in ('prefix','cases'):
                name=f"{r['video_id']}_{v}_{suffix}.pt"
                if (s,name) not in known:todo.append((s,name))
def transfer(item):
    s,name=item;b=call(s,'/file?path='+quote('yann_temporal_research/native_tail48/'+name),raw=True,timeout=180)
    path=temp/name;path.write_bytes(b);sha=hashlib.sha256(b).hexdigest()
    try:
        for attempt in range(3):
            try:u=upload(3-s,path,'peer_native_'+name);break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
        return dict(source=s,target=3-s,file=name,bytes=len(b),sha256=sha,upload=u)
    finally:path.unlink(missing_ok=True)
with concurrent.futures.ThreadPoolExecutor(3) as pool:
    for x in pool.map(transfer,todo):
        done.append(x);indexpath.write_text(json.dumps(done,indent=2));print(json.dumps({k:x[k] for k in ('source','file','bytes')}),flush=True)
print(json.dumps(dict(total_files=len(done),new_files=len(todo),total_bytes=sum(x['bytes'] for x in done))),flush=True)

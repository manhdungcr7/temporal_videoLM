"""Launch, checkpoint and download the frozen40 v5 evaluation."""
import argparse
import concurrent.futures
import json
from urllib.parse import quote
from argon_remote import ROOT,call,upload,submit

D=ROOT/'results/pcebr_v5_40';D.mkdir(exist_ok=True)
def launch(s,phase):
    name='pcebr_v5_40.py' if phase=='run' else 'pcebr_v5_40_archive.py'
    path=ROOT/'server_jobs'/name
    upload(s,path)
    result=submit(s,f'O40_SERVER={s}\n'+path.read_text(encoding='utf-8'),'pcebr_v5_40_'+phase)
    (D/f'launch_{phase}_s{s}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(server=s,phase=phase,**result)

def checkpoint(s,phase):
    job=json.loads((D/f'launch_{phase}_s{s}.json').read_text(encoding='utf-8'))
    status=call(s,'/job/'+job['job_id'])
    (D/f'job_{phase}_s{s}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    names=['protocol.json','results.json','completion.json','hook_audit.json','error.json'] if phase=='run' else ['archive_audit.json','archive_index.json']
    for name in names:
        try:(D/f'{name[:-5]}_s{s}.json').write_bytes(call(s,'/file?path='+quote('pcebr_v5_40/'+name),raw=True))
        except Exception:pass
    if phase=='archive' and status.get('status')=='done':
        idx=json.loads((D/f'archive_index_s{s}.json').read_text(encoding='utf-8'))
        path=D/f'raw_artifacts_s{s}.tar.gz'
        if not path.exists() or path.stat().st_size!=idx['bytes']:
            path.write_bytes(call(s,'/file?path=pcebr_v5_40/raw_artifacts.tar.gz',raw=True,timeout=180))
    return dict(server=s,phase=phase,status=status.get('status'),seconds=status.get('seconds'),error=status.get('error'),tail=status.get('out','')[-1600:])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','checkpoint']);p.add_argument('phase',choices=['run','archive']);a=p.parse_args()
    fn=launch if a.action=='launch' else checkpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda s:fn(s,a.phase),(1,2)):print(json.dumps(result,ensure_ascii=False,indent=2))

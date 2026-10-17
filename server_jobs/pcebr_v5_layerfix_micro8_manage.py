"""Two-server launch and local evidence collection for the matched micro8 fix."""
import argparse
import concurrent.futures
import json
from urllib.parse import quote
from argon_remote import ROOT,call,upload,submit

D=ROOT/'results/pcebr_v5_layerfix_micro8';D.mkdir(exist_ok=True)
def launch(s,phase):
    name='pcebr_v5_layerfix_micro8.py' if phase=='run' else 'pcebr_v5_layerfix_micro8_archive.py'
    path=ROOT/'server_jobs'/name;upload(s,path)
    r=submit(s,f'LF_SERVER={s}\n'+path.read_text(encoding='utf-8'),'layerfix_micro8_'+phase)
    (D/f'launch_{phase}_s{s}.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    return dict(server=s,**r)
def checkpoint(s,phase):
    launch=json.loads((D/f'launch_{phase}_s{s}.json').read_text(encoding='utf-8'))
    r=call(s,'/job/'+launch['job_id'])
    (D/f'job_{phase}_s{s}.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    names=['results','protocol','runtime_source','completion','hook_audit','error'] if phase=='run' else ['archive_audit','archive_index']
    for name in names:
        try:(D/f'{name}_s{s}.json').write_bytes(call(s,'/file?path='+quote(f'pcebr_v5_layerfix_micro8/{name}.json'),raw=True))
        except Exception:pass
    if phase=='archive' and r.get('status')=='done':
        idx=json.loads((D/f'archive_index_s{s}.json').read_text(encoding='utf-8'))
        dest=D/f'raw_artifacts_s{s}.tar.gz'
        if not dest.exists() or dest.stat().st_size!=idx['bytes']:
            dest.write_bytes(call(s,'/file?path=pcebr_v5_layerfix_micro8/raw_artifacts.tar.gz',raw=True,timeout=180))
    return dict(server=s,status=r.get('status'),seconds=r.get('seconds'),error=r.get('error'),tail=r.get('out','')[-2200:])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','checkpoint']);p.add_argument('phase',choices=['run','archive']);a=p.parse_args()
    fn=launch if a.action=='launch' else checkpoint
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for r in pool.map(lambda s:fn(s,a.phase),(1,2)):print(json.dumps(r,ensure_ascii=False,indent=2))

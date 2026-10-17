"""Transport for autonomous research; no credentials are written to results."""
import argparse, concurrent.futures, json
from urllib.parse import quote
from argon_remote import ROOT,call,submit,upload
D=ROOT/'results/yann_temporal_research';D.mkdir(exist_ok=True)
def launch(s,phase):
    if phase.startswith(('trainprobe','trainbinding')):upload(s,ROOT/'server_jobs/yann_tam_probe_model.py')
    if phase in ('nativecache','nativecache_boundaryfix','traintail','tail_eval'):upload(s,ROOT/'server_jobs/yann_native_tail.py')
    p=ROOT/'server_jobs'/f'yann_research_{phase}.py'
    upload(s,p)
    result=submit(s,f'YR_SERVER={s}\n'+p.read_text(encoding='utf-8'),f'yann_temporal_{phase}')
    (D/f'launch_{phase}_s{s}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(server=s,**result)
def status(s,phase):
    job=json.loads((D/f'launch_{phase}_s{s}.json').read_text())
    r=call(s,'/job/'+job['job_id'])
    (D/f'job_{phase}_s{s}.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    filephase={'supervision_resume':'supervision','bindingdata_resume':'bindingdata','trainbinding_ampfix':'trainbinding','binding_probe_eval_clockfix':'binding_probe_eval','denseprobe_unionfix':'denseprobe','upperparity_audit':'upperparity','nativecache_boundaryfix':'nativecache'}.get(phase,phase)
    for name in (filephase,f'{filephase}_partial'):
        try:(D/f'{name}_s{s}.json').write_bytes(call(s,'/file?path='+quote(f'yann_temporal_research/{name}.json'),raw=True))
        except Exception:pass
    return dict(server=s,status=r.get('status'),seconds=r.get('seconds'),error=r.get('error'),out=r.get('out','')[-4500:])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status']);p.add_argument('phase');p.add_argument('--server',type=int,choices=[1,2]);a=p.parse_args()
    fn=launch if a.action=='launch' else status
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for r in pool.map(lambda s:fn(s,a.phase),[a.server] if a.server else [1,2]):print(json.dumps(r,ensure_ascii=False,indent=2))

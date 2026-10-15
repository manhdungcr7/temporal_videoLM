"""Launch both shards, or fetch a compact checkpoint without busy-waiting."""
import argparse, concurrent.futures, json
from pathlib import Path
from argon_remote import ROOT, call, submit, upload

FOLDER=ROOT/'results/argon_latent40'
STATE=FOLDER/'jobs.json'

def launch(server):
    upload(server,ROOT/'server_jobs/argon_latent40.py')
    code='ARGON_SERVER = %d\nARGON_LIMIT = None\n' % server
    code+="exec(compile((OUT/'uploads/argon_latent40.py').read_text(),'argon_latent40.py','exec'),globals())"
    return {'server':server,**submit(server,code,'argon_latent40_full')}

def checkpoint(job):
    server=job['server'];state=call(server,'/job/'+job['job_id'])
    (FOLDER/f'job_s{server}.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        data=call(server,f'/file?path=argon_latent40/results_s{server}.jsonl',raw=True)
        (FOLDER/f'results_s{server}.jsonl').write_bytes(data)
        count=len(data.splitlines())
    except Exception:count=0
    tail=state.get('out','').splitlines()[-3:]
    return dict(server=server,status=state['status'],seconds=state['seconds'],completed=count,
                error=state.get('error'),tail=tail)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint'])
    args=parser.parse_args()
    if args.action=='launch':
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:jobs=list(pool.map(launch,[1,2]))
        STATE.write_text(json.dumps(jobs,indent=2),encoding='utf-8');print(json.dumps(jobs))
    else:
        jobs=json.loads(STATE.read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:status=list(pool.map(checkpoint,jobs))
        print(json.dumps(status,ensure_ascii=False,indent=2))

"""Archive progress from all submitted jobs every45s; report only changed counts."""
import concurrent.futures,json,sys,time,urllib.error
from pathlib import Path
from argon_remote import ROOT,call

PHASES=sys.argv[1:] or ['latent_reopen40','argon_ccld40','symmetric_latent40']
SNAPSHOT=f'{PHASES[0]}_live_status.json' if len(PHASES)==1 else 'latent_research_live_status.json'

def fetch_job(item):
    phase,job=item;s=job['server'];folder=ROOT/'results'/phase
    state=call(s,'/job/'+job['job_id'])
    (folder/f'job_s{s}.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    path=folder/f'results_s{s}.jsonl'
    try:
        remote='symmetric_latent40_v2' if phase=='symmetric_latent40' else phase
        blob=call(s,f'/file?path={remote}/results_s{s}.jsonl',raw=True,timeout=30)
        parsed=[json.loads(x) for x in blob.splitlines() if x.strip()]
        if not path.exists() or len(parsed)>=len(path.read_text().splitlines()):path.write_bytes(blob)
    except urllib.error.HTTPError as exc:
        if exc.code!=404:raise
    rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    return dict(phase=phase,server=s,status=state['status'],n=len(rows),seconds=state['seconds'],error=state.get('error'))

if __name__=='__main__':
    seen=None
    while True:
        items=[(phase,job) for phase in PHASES for job in json.loads((ROOT/'results'/phase/'jobs.json').read_text())]
        try:
            with concurrent.futures.ThreadPoolExecutor(6) as pool:states=list(pool.map(fetch_job,items))
            fingerprint=[(s['phase'],s['server'],s['status'],s['n']) for s in states]
            if fingerprint!=seen:
                print(json.dumps(states),flush=True);seen=fingerprint
                (ROOT/'results'/SNAPSHOT).write_text(json.dumps(states,indent=2))
            if any(s['status']=='error' for s in states):break
            if all(s['status']=='done' for s in states):break
        except Exception as exc:print('MONITOR_TRANSPORT_RETRY',type(exc).__name__,str(exc)[:160],flush=True)
        time.sleep(45)

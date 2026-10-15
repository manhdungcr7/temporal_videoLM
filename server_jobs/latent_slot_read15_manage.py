"""Queue a no-refit readout-stage ablation immediately after slot optimization."""
import argparse,concurrent.futures,json
from argon_remote import ROOT,call,upload,submit
import argon_ccld_manage as report
FOLDER=ROOT/'results/latent_slot_read15';FOLDER.mkdir(exist_ok=True)
report.FOLDER=FOLDER

def launch(s):
    upload(s,ROOT/'server_jobs/latent_slot_read15.py')
    code=f'REPLAY_SERVER={s}\n'
    code+="exec(compile((OUT/'uploads/latent_slot_read15.py').read_text(),'latent_slot_read15.py','exec'),globals())"
    return dict(server=s,**submit(s,code,'latent_slot_read15_no_refit'))

def checkpoint(job):
    s=job['server'];state=call(s,'/job/'+job['job_id'])
    (FOLDER/f'job_s{s}.json').write_text(json.dumps(state,indent=2))
    try:
        blob=call(s,f'/file?path=latent_slot_read15/results_s{s}.jsonl',raw=True)
        rows=[json.loads(x) for x in blob.splitlines() if x.strip()]
        local=FOLDER/f'results_s{s}.jsonl'
        if not local.exists() or len(rows)>=len(local.read_text().splitlines()):local.write_bytes(blob)
    except Exception as exc:
        if state['status'] not in ('queued','running'):print('DOWNLOAD_ERROR',s,repr(exc))
    return dict(server=s,status=state['status'],seconds=state['seconds'],error=state.get('error'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint','summary']);args=parser.parse_args()
    if args.action=='launch':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:jobs=list(pool.map(launch,[1,2]))
        (FOLDER/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
    elif args.action=='checkpoint':
        jobs=json.loads((FOLDER/'jobs.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:print(json.dumps(list(pool.map(checkpoint,jobs))))
        report.summary('E0_native');print((FOLDER/'table.md').read_text())
    else:report.summary('E0_native');print((FOLDER/'table.md').read_text())

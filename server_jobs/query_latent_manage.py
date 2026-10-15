"""Schedule query-conditioned prototypes after same-layer controls."""
import argparse,concurrent.futures,json
from argon_remote import ROOT,call,upload,submit
import argon_ccld_manage as report
FOLDER=ROOT/'results/query_latent40';FOLDER.mkdir(exist_ok=True)
report.FOLDER=FOLDER

def launch(s):
    upload(s,ROOT/'server_jobs/query_latent40.py')
    code=f'QL_SERVER={s}\n'
    code+="exec(compile((OUT/'uploads/query_latent40.py').read_text(),'query_latent40.py','exec'),globals())"
    return dict(server=s,**submit(s,code,'query_latent40_after_controls'))

def checkpoint(job):
    s=job['server'];state=call(s,'/job/'+job['job_id'])
    (FOLDER/f'job_s{s}.json').write_text(json.dumps(state,indent=2))
    try:
        blob=call(s,f'/file?path=query_latent40/results_s{s}.jsonl',raw=True)
        records=[json.loads(x) for x in blob.splitlines() if x.strip()]
        local=FOLDER/f'results_s{s}.jsonl'
        if not local.exists() or len(records)>=len(local.read_text().splitlines()):local.write_bytes(blob)
    except Exception as exc:
        if state['status'] not in ('running','queued'):print('DOWNLOAD_ERROR',s,repr(exc))
    return dict(server=s,status=state['status'],seconds=state['seconds'],error=state.get('error'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint','summary'])
    args=parser.parse_args()
    if args.action=='launch':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:jobs=list(pool.map(launch,[1,2]))
        (FOLDER/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
    elif args.action=='checkpoint':
        jobs=json.loads((FOLDER/'jobs.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:print(json.dumps(list(pool.map(checkpoint,jobs))))
        report.summary('Q0_native');print((FOLDER/'table.md').read_text())
    else:report.summary('Q0_native');print((FOLDER/'table.md').read_text())

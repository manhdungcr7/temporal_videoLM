"""Run the previously pending presentation-clock correctness controls."""
import argparse,concurrent.futures,json
from argon_remote import ROOT,call,upload,submit
import argon_ccld_manage as report
FOLDER=ROOT/'results/presentation_resample40';FOLDER.mkdir(exist_ok=True)
report.FOLDER=FOLDER

def launch(s,limit=None):
    upload(s,ROOT/'server_jobs/presentation_resample40.py')
    code=f'PC_SERVER={s}\nPC_LIMIT={limit!r}\n'
    code+="exec(compile((OUT/'uploads/presentation_resample40.py').read_text(),'presentation_resample40.py','exec'),globals())"
    return dict(server=s,**submit(s,code,'presentation_resample40'))

def checkpoint(job):
    s=job['server'];state=call(s,'/job/'+job['job_id'])
    (FOLDER/f'job_s{s}.json').write_text(json.dumps(state,indent=2))
    try:
        blob=call(s,f'/file?path=presentation_resample40/results_s{s}.jsonl',raw=True)
        rows=[json.loads(x) for x in blob.splitlines() if x.strip()];local=FOLDER/f'results_s{s}.jsonl'
        if not local.exists() or len(rows)>=len(local.read_text().splitlines()):local.write_bytes(blob)
    except Exception as exc:
        if state['status'] not in ('queued','running'):print('DOWNLOAD_ERROR',s,repr(exc))
    return dict(server=s,status=state['status'],seconds=state['seconds'],error=state.get('error'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint','summary'])
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.action=='launch':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:jobs=list(pool.map(lambda s:launch(s,args.limit),[1,2]))
        (FOLDER/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
    elif args.action=='checkpoint':
        jobs=json.loads((FOLDER/'jobs.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:print(json.dumps(list(pool.map(checkpoint,jobs))))
        if all((FOLDER/f'results_s{s}.jsonl').exists() for s in (1,2)):
            report.summary('P0_original_native_reference');print((FOLDER/'table.md').read_text())
    else:report.summary('P0_original_native_reference');print((FOLDER/'table.md').read_text())

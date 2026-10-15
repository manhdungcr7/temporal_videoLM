"""Transport and matched reporting for the reversible evidence experiments."""
import argparse, concurrent.futures, json
from pathlib import Path
from argon_remote import ROOT,call,upload,submit

FOLDER=ROOT/'results/latent_reopen40'
FOLDER.mkdir(exist_ok=True)

def launch(server,limit=None):
    upload(server,ROOT/'server_jobs/latent_reopen40.py')
    code=f'LR_SERVER={server}\nLR_LIMIT={limit!r}\n'
    code+="exec(compile((OUT/'uploads/latent_reopen40.py').read_text(),'latent_reopen40.py','exec'),globals())"
    return dict(server=server,**submit(server,code,'latent_reopen40'))

def prepare(server):
    for path in ['results/research40_manifest.json','server_jobs/argon_latent40.py',
                 'results/argon_latent40/proposal_original.py','server_jobs/avbind_sdpa_compat.py']:
        upload(server,ROOT/path)
    return {'prepared':server}

def checkpoint(job):
    server=job['server'];state=call(server,'/job/'+job['job_id'])
    (FOLDER/f'job_s{server}.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    try:
        blob=call(server,f'/file?path=latent_reopen40/results_s{server}.jsonl',raw=True)
        (FOLDER/f'results_s{server}.jsonl').write_bytes(blob)
    except Exception:pass
    lines=state.get('out','').splitlines()
    sample_lines=[x for x in lines if x.startswith('REOPEN_SAMPLE')]
    last=json.loads(sample_lines[-1][sample_lines[-1].index('{'):]) if sample_lines else None
    return dict(server=server,status=state['status'],seconds=state['seconds'],
        error=state.get('error'),completed=len(sample_lines),
        last=None if last is None else {k:last[k] for k in ('idx','task','gold','predictions','seconds')})

def summary():
    rows=[]
    for server in (1,2):
        p=FOLDER/f'results_s{server}.jsonl'
        if p.exists():rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert len({x['idx'] for x in rows})==len(rows)
    cells=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']
    historical={}
    for server in (1,2):
        p=ROOT/f'results/argon_latent40/results_s{server}.jsonl'
        historical.update({r['idx']:r for r in [json.loads(x) for x in p.read_text().splitlines() if x.strip()]})
    for r in rows:
        refs=r.get('reference_scores',historical[r['idx']]['scores'])
        for m in ('R3_M0_base','R3_M2_true_gate_audio'):
            r['scores'][m]=refs[m]
            r['predictions'][m]=chr(65+max(range(len(refs[m])),key=lambda i:refs[m][i]))
    methods=sorted({m for r in rows for m in r['scores']})
    report={m:dict(total=sum(r['predictions'][m]==r['gold'] for r in rows),n=len(rows),
        cells={c:[sum(r['predictions'][m]==r['gold'] for r in rows if r['task']+'/'+r['gold']==c),
                  sum(r['task']+'/'+r['gold']==c for r in rows)] for c in cells},
        tf_predictions={p:sum(r['task']=='TF' and r['predictions'][m]==p for r in rows) for p in 'ABC'})
        for m in methods}
    (FOLDER/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint','summary'])
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.action=='launch':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs=list(pool.map(lambda s:launch(s,args.limit),[1,2]))
        (FOLDER/'jobs.json').write_text(json.dumps(jobs,indent=2))
        print(json.dumps(jobs))
    elif args.action=='checkpoint':
        jobs=json.loads((FOLDER/'jobs.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(2) as pool: print(json.dumps(list(pool.map(checkpoint,jobs))))
        print(json.dumps(summary()))
    else:print(json.dumps(summary()))

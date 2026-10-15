"""Launch/checkpoint/summarize frozen40 Round5; preserve raw outputs locally."""
import argparse,concurrent.futures,csv,json,math
from pathlib import Path
from argon_remote import ROOT,call,upload,submit

FOLDER=ROOT/'results/chronodesmear40'
CELLS=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']

def launch(server,limit=None):
    existing=FOLDER/'jobs.json'
    if existing.exists() and any(j['server']==server for j in json.loads(existing.read_text())):
        raise RuntimeError('Full suite already submitted on this server; resume explicitly')
    code=f'CD_SERVER={server}\nCD_LIMIT={limit!r}\n'
    code+="exec(compile((OUT/'uploads/chronodesmear40.py').read_text(),'chronodesmear40.py','exec'),globals())"
    return dict(server=server,**submit(server,code,'chronodesmear_full40'))

def checkpoint(job):
    server=job['server'];state=call(server,'/job/'+job['job_id'])
    (FOLDER/f'job_s{server}.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    error=None
    try:
        blob=call(server,f'/file?path=chronodesmear40/results_s{server}.jsonl',raw=True)
        parsed=[json.loads(line) for line in blob.splitlines() if line.strip()]
        local=FOLDER/f'results_s{server}.jsonl'
        old=[json.loads(line) for line in local.read_text().splitlines()] if local.exists() else []
        assert len(parsed)>=len(old),'Refuse shrinking checkpoint'
        assert len({r['idx'] for r in parsed})==len(parsed)
        local.write_bytes(blob)
    except Exception as exc: error=repr(exc)
    return dict(server=server,status=state['status'],seconds=state['seconds'],
        error=state.get('error'),download_error=error,tail=state.get('out','')[-350:])

def summary(folder=FOLDER,baseline='R5_C0_baseline'):
    rows=[]
    for server in (1,2):
        path=folder/f'results_s{server}.jsonl'
        if path.exists(): rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    assert len({r['idx'] for r in rows})==len(rows)
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    by_id={s['idx']:s for s in manifest['selected']}
    for row in rows:
        assert row['manifest_sha256']==manifest['manifest_sha256']
        assert (row['task'],row['gold'])==(by_id[row['idx']]['task'],by_id[row['idx']]['gold'])
    methods=list(rows[0]['scores']) if rows else []
    predictions={m:{r['idx']:r['predictions'][m] for r in rows} for m in methods}
    if rows and 'shuffled_seed_scores' in rows[0]:
        for seed in ('42','1337','2027'):
            method='R5_C6_seed_'+seed;methods.append(method)
            predictions[method]={r['idx']:chr(65+max(range(len(r['shuffled_seed_scores'][seed])),key=lambda i:r['shuffled_seed_scores'][seed][i])) for r in rows}
    report={}
    for method in methods:
        yes=lambda r:predictions[method][r['idx']]==r['gold']
        rescued=sum(yes(r) and predictions[baseline][r['idx']]!=r['gold'] for r in rows)
        damaged=sum(not yes(r) and predictions[baseline][r['idx']]==r['gold'] for r in rows)
        n=rescued+damaged
        p=min(1.,2*sum(math.comb(n,i) for i in range(min(rescued,damaged)+1))/2**n) if n else 1.
        report[method]=dict(correct=sum(yes(r) for r in rows),wrong=sum(not yes(r) for r in rows),n=len(rows),
            cells={c:dict(correct=sum(yes(r) for r in rows if r['task']+'/'+r['gold']==c),
                wrong=sum(not yes(r) for r in rows if r['task']+'/'+r['gold']==c),
                n=sum(r['task']+'/'+r['gold']==c for r in rows)) for c in CELLS},
            tf_pred_ABC=[sum(r['task']=='TF' and predictions[method][r['idx']]==choice for r in rows) for choice in 'ABC'],
            rescued=rescued,damaged=damaged,paired_p=p)
    tests=sorted((v['paired_p'],m) for m,v in report.items() if m!=baseline)
    running=0.
    for rank,(p,m) in enumerate(tests):
        running=max(running,min(1.,(len(tests)-rank)*p));report[m]['holm_p']=running
    folder.mkdir(exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for method,entry in report.items():
        values=[method]+[f"{entry['cells'][c]['correct']}/{entry['cells'][c]['wrong']}" for c in CELLS]
        values.extend([f"{entry['correct']}/{entry['wrong']}",'/'.join(map(str,entry['tf_pred_ABC']))])
        lines.append('|'+ '|'.join(values)+'|')
    (folder/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (folder/'per_sample.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['idx','task','gold','method','prediction','correct'])
        for row in rows:
            for method in methods:
                pred=predictions[method][row['idx']]
                writer.writerow([row['idx'],row['task'],row['gold'],method,pred,int(pred==row['gold'])])
    return dict(completed=len(rows),expected=40,methods={m:v['correct'] for m,v in report.items()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['checkpoint','summary','launch'])
    parser.add_argument('--server',type=int);args=parser.parse_args()
    if args.action=='launch':
        job=launch(args.server)
        path=FOLDER/'jobs.json';jobs=json.loads(path.read_text()) if path.exists() else []
        jobs.append(job);path.write_text(json.dumps(jobs,indent=2));print(json.dumps(job))
    elif args.action=='checkpoint':
        jobs=json.loads((FOLDER/'jobs.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(2) as pool: print(json.dumps(list(pool.map(checkpoint,jobs))))
        print(json.dumps(summary()))
    else:
        print(json.dumps(summary()));print((FOLDER/'table.md').read_text())

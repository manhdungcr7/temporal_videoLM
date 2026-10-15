"""Queue CCLD immediately after current jobs; archive every checkpoint locally."""
import argparse,concurrent.futures,csv,json,math
from pathlib import Path
from argon_remote import ROOT,call,upload,submit

FOLDER=ROOT/'results/argon_ccld40';FOLDER.mkdir(exist_ok=True)
CELLS=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']

def markdown_lines(lines):
    output=[]
    for line in lines:
        if not line:
            if output and output[-1]:output.append('')
        else:
            if output and output[-1] and not (line.startswith('|') and output[-1].startswith('|')):
                output.append('')
            output.append(line)
    return '\n'.join(output).strip()+'\n'

def launch(server,limit=None):
    upload(server,ROOT/'results/argon_ccld40/proposal_original.py','ccld_proposal_original.py')
    upload(server,ROOT/'server_jobs/argon_ccld40.py')
    code=f'CC_SERVER={server}\nCC_LIMIT={limit!r}\n'
    code+="exec(compile((OUT/'uploads/argon_ccld40.py').read_text(),'argon_ccld40.py','exec'),globals())"
    return dict(server=server,**submit(server,code,'argon_ccld40'))

def checkpoint(job):
    s=job['server'];state=call(s,'/job/'+job['job_id'])
    (FOLDER/f'job_s{s}.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    try:
        blob=call(s,f'/file?path=argon_ccld40/results_s{s}.jsonl',raw=True)
        parsed=[json.loads(x) for x in blob.splitlines() if x.strip()]
        local=FOLDER/f'results_s{s}.jsonl'
        if not local.exists() or len(parsed)>=len(local.read_text().splitlines()):local.write_bytes(blob)
    except Exception as exc:
        if state['status'] not in ('queued','running'):print('DOWNLOAD_ERROR',s,repr(exc))
    return dict(server=s,status=state['status'],seconds=state['seconds'],error=state.get('error'),
        progress=state.get('out','').splitlines()[-1:][0][:160] if state.get('out') else '')

def summary(baseline='R4_C0_baseline'):
    rows=[]
    for server in (1,2):
        p=FOLDER/f'results_s{server}.jsonl'
        if p.exists():rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert len({x['idx'] for x in rows})==len(rows)
    methods=sorted({m for row in rows for m in row['scores']})
    report={}
    for m in methods:
        yes=lambda r:r['predictions'][m]==r['gold']
        rescued=sum(yes(r) and r['predictions'][baseline]!=r['gold'] for r in rows)
        damaged=sum(not yes(r) and r['predictions'][baseline]==r['gold'] for r in rows)
        n=rescued+damaged
        p=min(1.,2*sum(math.comb(n,i) for i in range(min(rescued,damaged)+1))/2**n) if n else 1.
        report[m]=dict(correct=sum(map(yes,rows)),wrong=sum(not yes(r) for r in rows),n=len(rows),
            cells={c:dict(correct=sum(yes(r) for r in rows if r['task']+'/'+r['gold']==c),
                wrong=sum(not yes(r) for r in rows if r['task']+'/'+r['gold']==c),
                n=sum(r['task']+'/'+r['gold']==c for r in rows)) for c in CELLS},
            tf_pred_ABC=[sum(r['task']=='TF' and r['predictions'][m]==p for r in rows) for p in 'ABC'],
            rescued=rescued,damaged=damaged,paired_p=p)
    (FOLDER/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for m,v in report.items():
        line=[m]+[f"{v['cells'][c]['correct']}/{v['cells'][c]['wrong']}" for c in CELLS]
        line += [f"{v['correct']}/{v['wrong']}",'/'.join(map(str,v['tf_pred_ABC']))]
        lines.append('|'+ '|'.join(line)+'|')
    (FOLDER/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (FOLDER/'per_sample.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['idx','task','gold','method','prediction','correct'])
        for r in rows:
            for m in methods:writer.writerow([r['idx'],r['task'],r['gold'],m,r['predictions'][m],int(r['predictions'][m]==r['gold'])])
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint','summary'])
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.action=='launch':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:jobs=list(pool.map(lambda s:launch(s,args.limit),[1,2]))
        (FOLDER/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
    elif args.action=='checkpoint':
        jobs=json.loads((FOLDER/'jobs.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:print(json.dumps(list(pool.map(checkpoint,jobs))))
        summary();print((FOLDER/'table.md').read_text())
    else:summary();print((FOLDER/'table.md').read_text())

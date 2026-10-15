import argparse
import concurrent.futures
import json
from argon_remote import ROOT,call,upload,submit

folder=ROOT/'results/pcebr_stage0'

def launch(server):
    suffix='r14' if server==1 else 'r18'
    upload(server,ROOT/'repo/kaggle/jobs/avbind_alignment.py')
    upload(server,ROOT/'results'/f'avbind_native_alignment_{suffix}.pt')
    script=ROOT/'server_jobs/pcebr_stage0_baseline.py'
    upload(server,script)
    job=submit(server,'PC_SERVER = %d\n'%server+script.read_text(encoding='utf-8'),'pcebr_stage0_legacy_replay')
    (folder/f'launch_baseline_s{server}.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
    return dict(server=server,**job)

def checkpoint(server):
    job=json.loads((folder/f'launch_baseline_s{server}.json').read_text(encoding='utf-8'))
    status=call(server,'/job/'+job['job_id'])
    (folder/f'job_baseline_s{server}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if status.get('status')=='done':
        name=f'baseline_s{server}.json'
        (folder/name).write_bytes(call(server,'/file?path=pcebr_stage0/'+name,raw=True))
    return dict(server=server,status=status.get('status'),error=status.get('error'),result=status.get('result'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint']);args=parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(launch if args.action=='launch' else checkpoint,(1,2)):
            print(json.dumps(result,ensure_ascii=False,indent=2))

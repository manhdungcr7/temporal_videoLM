"""PC-EBR v2 preparation, launch, and artifact download; no credentials in logs."""
import argparse
import concurrent.futures
import json
import time
from urllib.parse import quote
from argon_remote import ROOT,call,upload,submit

FOLDER=ROOT/'results/pcebr_stage0_v2'
_upload_once=upload
def upload(server,path):
    for attempt in range(4):
        try:return _upload_once(server,path)
        except Exception:
            if attempt==3:raise
            time.sleep(1+attempt)

def launch(server,phase):
    if phase=='prepare':
        for prefix in ('avbind_native_pairs','avbind_pooled_pairs'):
            for shard in (1,2):upload(server,ROOT/'results'/f'{prefix}_s{shard}.pt')
        upload(server,ROOT/'results/research40_manifest.json')
        upload(server,ROOT/'server_jobs/avbind_sdpa_compat.py')
        upload(server,ROOT/'server_jobs/chronodesmear_cache40.py')
        upload(server,FOLDER/'stage0_pcebr_mechanism_check_v2_original.py')
    if phase=='m3fix':upload(server,FOLDER/'stage0_pcebr_mechanism_check_v2_devicefix.py')
    path=ROOT/'server_jobs'/f'pcebr_stage0_v2_{phase}.py'
    upload(server,path)
    code=f'PC_SERVER={server}\nLAB_SERVER={server}\n'+path.read_text(encoding='utf-8')
    result=submit(server,code,f'pcebr_stage0_v2_{phase}')
    (FOLDER/f'launch_{phase}_s{server}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(server=server,phase=phase,**result)

def checkpoint(server,phase):
    job=json.loads((FOLDER/f'launch_{phase}_s{server}.json').read_text(encoding='utf-8'))
    status=call(server,'/job/'+job['job_id'])
    (FOLDER/f'job_{phase}_s{server}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    files={'prepare':['prepare_audit.json'],'run':['progress.json','main_results.json','input_audit.json','instrumentation.json'],
        'm3fix':['m3_devicefix_results.json','m3_audit.json','main_results.json'],
        'controls':[f'controls_s{server}.json'],
        'archive':['archive_audit.json','archive_index.json','retrieval_per_query.json']}[phase]
    for name in files:
        try:
            data=call(server,'/file?path='+quote('pcebr_stage0_v2/'+name),raw=True)
            target=name if phase=='controls' else f'{name[:-5]}_s{server}.json'
            (FOLDER/target).write_bytes(data)
        except Exception:
            pass
    if phase=='archive' and status.get('status')=='done':
        target=FOLDER/f'raw_artifacts_s{server}.tar.gz'
        if not target.exists():target.write_bytes(call(server,'/file?path=pcebr_stage0_v2/raw_artifacts.tar.gz',raw=True,timeout=180))
    return dict(server=server,phase=phase,status=status.get('status'),seconds=status.get('seconds'),error=status.get('error'),tail=status.get('out','')[-1800:])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint']);parser.add_argument('phase',choices=['prepare','run','controls','m3fix','archive']);args=parser.parse_args()
    fn=launch if args.action=='launch' else checkpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda s:fn(s,args.phase),(1,2)):print(json.dumps(result,ensure_ascii=False,indent=2))

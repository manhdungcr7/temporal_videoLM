"""PC-EBR v3 transport and phase checkpoints."""
import argparse
import concurrent.futures
import json
import time
from urllib.parse import quote
from argon_remote import ROOT,call,upload as upload_once,submit

FOLDER=ROOT/'results/pcebr_stage0_v3'
def upload(server,path):
    for n in range(4):
        try:return upload_once(server,path)
        except Exception:
            if n==3:raise
            time.sleep(1+n)

def launch(server,phase):
    if phase=='prepare':
        for prefix in ('avbind_native_pairs','avbind_pooled_pairs'):
            for shard in (1,2):upload(server,ROOT/'results'/f'{prefix}_s{shard}.pt')
        for name in ('research40_manifest.json',):upload(server,ROOT/'results'/name)
        for name in ('avbind_sdpa_compat.py','chronodesmear_cache40.py'):upload(server,ROOT/'server_jobs'/name)
        upload(server,FOLDER/'stage0_pcebr_mechanism_check_v3_original.py')
    path=ROOT/'server_jobs'/f'pcebr_stage0_v3_{phase}.py'
    upload(server,path)
    result=submit(server,f'PC_SERVER={server}\nLAB_SERVER={server}\n'+path.read_text(encoding='utf-8'),f'pcebr_stage0_v3_{phase}')
    (FOLDER/f'launch_{phase}_s{server}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(server=server,phase=phase,**result)

def checkpoint(server,phase):
    job=json.loads((FOLDER/f'launch_{phase}_s{server}.json').read_text(encoding='utf-8'))
    status=call(server,'/job/'+job['job_id'])
    (FOLDER/f'job_{phase}_s{server}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    names={'prepare':['prepare_audit.json'],
        'run':['progress.json','instrumentation.json','main_results.json','input_audit.json','m3_audit.json'],
        'resume':['progress.json','instrumentation.json','main_results.json','m3_audit.json','freeze_repair_audit.json'],
        'archive':['archive_index.json','archive_audit.json','retrieval_per_query.json'],
        'controls':['controls.json']}[phase]
    for name in names:
        try:(FOLDER/f'{name[:-5]}_s{server}.json').write_bytes(call(server,'/file?path='+quote('pcebr_stage0_v3/'+name),raw=True))
        except Exception:pass
    if phase=='archive' and status.get('status')=='done':
        index=json.loads((FOLDER/f'archive_index_s{server}.json').read_text(encoding='utf-8'))
        path=FOLDER/f'raw_artifacts_s{server}.tar.gz'
        if not path.exists() or path.stat().st_size!=index['bytes']:
            path.write_bytes(call(server,'/file?path=pcebr_stage0_v3/raw_artifacts.tar.gz',raw=True,timeout=180))
    return dict(server=server,phase=phase,status=status.get('status'),seconds=status.get('seconds'),error=status.get('error'),tail=status.get('out','')[-2400:])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','checkpoint']);p.add_argument('phase',choices=['prepare','run','resume','archive','controls']);a=p.parse_args()
    fn=launch if a.action=='launch' else checkpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda s:fn(s,a.phase),(1,2)):print(json.dumps(result,ensure_ascii=False,indent=2))

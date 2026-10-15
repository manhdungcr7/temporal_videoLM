"""Complete logical-shard1 probes on server2 after server1 CUDA context error."""
import argparse
import json
from argon_remote import ROOT,call,upload,submit

folder=ROOT/'results/pcebr_stage0_v2'

def launch(phase):
    if phase=='m3':
        path=ROOT/'server_jobs/pcebr_stage0_v2_prepare.py'
        prep=path.read_text(encoding='utf-8')
        run=(ROOT/'server_jobs/pcebr_stage0_v2_m3fix.py').read_text(encoding='utf-8')
        code='PC_SERVER=1\nLAB_SERVER=1\n'+prep+'\n'+run
    else:
        code='PC_SERVER=1\nLAB_SERVER=1\n'+(ROOT/'server_jobs/pcebr_stage0_v2_controls.py').read_text(encoding='utf-8')
    result=submit(2,code,'pcv2_recover_logical1_'+phase)
    (folder/f'launch_recover_{phase}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def checkpoint(phase):
    job=json.loads((folder/f'launch_recover_{phase}.json').read_text(encoding='utf-8'))
    status=call(2,'/job/'+job['job_id'])
    (folder/f'job_recover_{phase}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if status.get('status')=='done':
        files=['m3_devicefix_results.json','m3_audit.json','main_results.json'] if phase=='m3' else ['controls_s1.json']
        for name in files:
            target=name if phase=='controls' else name[:-5]+'_s1.json'
            (folder/target).write_bytes(call(2,'/file?path=pcebr_stage0_v2/'+name,raw=True))
        if phase=='controls':
            (folder/'uniform_control_probes.pt').write_bytes(call(2,'/file?path=pcebr_stage0_v2/uniform_control_probes.pt',raw=True))
    return dict(status=status.get('status'),seconds=status.get('seconds'),error=status.get('error'),tail=status.get('out','')[-1600:])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','checkpoint']);parser.add_argument('phase',choices=['m3','controls']);args=parser.parse_args()
    print(json.dumps(launch(args.phase) if args.action=='launch' else checkpoint(args.phase),ensure_ascii=False,indent=2))

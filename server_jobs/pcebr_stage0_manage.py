"""Archive and execute the exact supplied Stage-0 script on authorized GPU servers."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from argon_remote import ROOT, call, submit, upload

FOLDER = ROOT / 'results/pcebr_stage0'
VARIANT = 'original'

def launch(server):
    paths = [FOLDER / f'stage0_pcebr_mechanism_check_{VARIANT}.py']
    paths += [] if VARIANT != 'original' else [ROOT / 'results' / f'{prefix}_s{s}.pt'
              for prefix in ('avbind_native_pairs', 'avbind_pooled_pairs') for s in (1, 2)]
    for path in paths:
        upload(server, path)
    source = """
import hashlib, json, platform, time, torch
from pathlib import Path
_pc_folder = OUT / 'pcebr_stage0'
_pc_folder.mkdir(exist_ok=True)
_pc_source = OUT / 'uploads/stage0_pcebr_mechanism_check_original.py'
_pc_audit = dict(model=MODEL_.config._name_or_path, torch=torch.__version__,
    source_sha256=hashlib.sha256(_pc_source.read_bytes()).hexdigest(), caches={})
for _pc_prefix in ('avbind_native_pairs', 'avbind_pooled_pairs'):
    _pc_parts = []
    for _pc_shard in (1, 2):
        _pc_path = OUT / 'uploads' / f'{_pc_prefix}_s{_pc_shard}.pt'
        _pc_data = torch.load(_pc_path, map_location='cpu', weights_only=True)
        _pc_parts.append(_pc_data)
        _pc_audit['caches'][_pc_path.name] = dict(
            sha256=hashlib.sha256(_pc_path.read_bytes()).hexdigest(),
            examples=len(_pc_data['examples']), metadata=_pc_data.get('metadata', {}),
            example_keys=sorted(_pc_data['examples'][0]))
    _pc_examples = [x for p in _pc_parts for x in p['examples']]
    _pc_tr = [x for x in _pc_examples if x['split']=='train']
    _pc_he = [x for x in _pc_examples if x['split']=='heldout']
    _pc_audit[_pc_prefix] = dict(train=len(_pc_tr),heldout=len(_pc_he),
        train_origins=sorted({x['recording_id'] for x in _pc_tr}),
        heldout_origins=sorted({x['recording_id'] for x in _pc_he}),
        overlap=sorted({x['recording_id'] for x in _pc_tr}&{x['recording_id'] for x in _pc_he}),
        raw_shapes=sorted({str(tuple(x['raw'].shape)) for x in _pc_examples}),
        text_shapes=sorted({str(tuple(x['text'].shape)) for x in _pc_examples}))
(_pc_folder/'input_audit.json').write_text(json.dumps(_pc_audit,indent=2),encoding='utf-8')
print('PCEBR_INPUT_AUDIT',json.dumps({k:v for k,v in _pc_audit.items() if k!='caches'}),flush=True)
_pc_ns = dict(OUT=OUT, __name__='pcebr_exact_original')
exec(compile(_pc_source.read_text(encoding='utf-8'),str(_pc_source),'exec'),_pc_ns)
_pc_result = _pc_ns['RESULT']
(_pc_folder/'original_results.json').write_text(json.dumps(_pc_result,indent=2),encoding='utf-8')
RESULT = _pc_result
"""
    if VARIANT != 'original':
        source = source.replace('stage0_pcebr_mechanism_check_original.py', f'stage0_pcebr_mechanism_check_{VARIANT}.py')
        source = source.replace('original_results.json', f'{VARIANT}_results.json')
    job = submit(server, source, f'pcebr_stage0_{VARIANT}')
    (FOLDER / f'launch_{VARIANT}_s{server}.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
    return dict(server=server, **job)

def checkpoint(server):
    job = json.loads((FOLDER / f'launch_{VARIANT}_s{server}.json').read_text(encoding='utf-8'))
    status = call(server, '/job/' + job['job_id'])
    (FOLDER / f'job_{VARIANT}_s{server}.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    for name in ('input_audit.json', f'{VARIANT}_results.json'):
        try:
            raw = call(server, '/file?path=pcebr_stage0/' + name, raw=True)
            (FOLDER / f'{Path(name).stem}_s{server}.json').write_bytes(raw)
        except Exception as error:
            print(json.dumps(dict(server=server,file=name,unavailable=str(error))))
    return dict(server=server,status=status.get('status'),seconds=status.get('seconds'),
                error=status.get('error'),tail=status.get('out','')[-2000:])

def audit_launch(server):
    path=ROOT/'server_jobs/pcebr_stage0_audit.py'
    upload(server,path)
    job=submit(server,'PC_SERVER = %d\n'%server+path.read_text(encoding='utf-8'),'pcebr_stage0_exploratory_audit')
    (FOLDER/f'launch_audit_s{server}.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
    return dict(server=server,**job)

def audit_checkpoint(server):
    job=json.loads((FOLDER/f'launch_audit_s{server}.json').read_text(encoding='utf-8'))
    status=call(server,'/job/'+job['job_id'])
    (FOLDER/f'job_audit_s{server}.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if status.get('status')=='done':
        for suffix in ('json','pt'):
            name=f'audit_results_s{server}.json' if suffix=='json' else f'audit_similarities_s{server}.pt'
            (FOLDER/name).write_bytes(call(server,'/file?path=pcebr_stage0/'+name,raw=True))
    return dict(server=server,status=status.get('status'),seconds=status.get('seconds'),error=status.get('error'),tail=status.get('out','')[-1600:])

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['launch','checkpoint','audit-launch','audit-checkpoint'])
    parser.add_argument('--variant',choices=['original','devicefix'],default='original')
    args=parser.parse_args()
    VARIANT=args.variant
    action={'launch':launch,'checkpoint':checkpoint,'audit-launch':audit_launch,'audit-checkpoint':audit_checkpoint}[args.action]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(action,(1,2)):
            print(json.dumps(result,ensure_ascii=False,indent=2))

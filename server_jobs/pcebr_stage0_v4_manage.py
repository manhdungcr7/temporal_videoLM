"""Two-server transport for the supplied Stage-0 v4 diagnostic."""
import argparse
import concurrent.futures
import json
import time
from urllib.parse import quote
from argon_remote import ROOT, call, upload as upload_once, submit

FOLDER = ROOT / 'results/pcebr_stage0_v4'

def upload(server, path):
    for attempt in range(4):
        try:
            return upload_once(server, path)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

def launch(server, phase):
    if phase == 'prepare':
        for prefix in ('avbind_native_pairs', 'avbind_pooled_pairs'):
            for shard in (1, 2):
                upload(server, ROOT / 'results' / f'{prefix}_s{shard}.pt')
        upload(server, ROOT / 'results/research40_manifest.json')
        for name in ('avbind_sdpa_compat.py', 'chronodesmear_cache40.py'):
            upload(server, ROOT / 'server_jobs' / name)
        upload(server, FOLDER / 'stage0_pcebr_v4_original.py')
    path = ROOT / 'server_jobs' / f'pcebr_stage0_v4_{phase}.py'
    upload(server, path)
    result = submit(server, f'PC_SERVER={server}\nLAB_SERVER={server}\n' + path.read_text(encoding='utf-8'), f'pcebr_stage0_v4_{phase}')
    (FOLDER / f'launch_{phase}_s{server}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return dict(server=server, phase=phase, **result)

def checkpoint(server, phase):
    launch_data = json.loads((FOLDER / f'launch_{phase}_s{server}.json').read_text(encoding='utf-8'))
    status = call(server, '/job/' + launch_data['job_id'])
    (FOLDER / f'job_{phase}_s{server}.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    names = {'prepare': ['prepare_audit.json'],
             'run': ['progress.json', 'instrumentation.json', 'main_results.json', 'input_audit.json', 'm3_audit.json'],
             'controls': ['controls.json'],
             'archive': ['archive_index.json', 'archive_audit.json']}[phase]
    for name in names:
        try:
            data = call(server, '/file?path=' + quote('pcebr_stage0_v4/' + name), raw=True)
            (FOLDER / f'{name[:-5]}_s{server}.json').write_bytes(data)
        except Exception:
            pass
    if phase == 'archive' and status.get('status') == 'done':
        index = json.loads((FOLDER / f'archive_index_s{server}.json').read_text(encoding='utf-8'))
        path = FOLDER / f'raw_artifacts_s{server}.tar.gz'
        if not path.exists() or path.stat().st_size != index['bytes']:
            path.write_bytes(call(server, '/file?path=pcebr_stage0_v4/raw_artifacts.tar.gz', raw=True, timeout=180))
    return dict(server=server, phase=phase, status=status.get('status'), seconds=status.get('seconds'), error=status.get('error'), tail=status.get('out', '')[-2800:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'checkpoint'])
    parser.add_argument('phase', choices=['prepare', 'run', 'controls', 'archive'])
    args = parser.parse_args()
    fn = launch if args.action == 'launch' else checkpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(lambda server: fn(server, args.phase), (1, 2)):
            print(json.dumps(row, ensure_ascii=False, indent=2))

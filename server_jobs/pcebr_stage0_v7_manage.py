"""Transport for the final supplied CAOR-WEMD source and observation audits."""
import argparse
import concurrent.futures
import json
import time
from urllib.parse import quote
from argon_remote import ROOT, call, upload, submit

D = ROOT / 'results/pcebr_stage0_v7'
D.mkdir(exist_ok=True)

def launch(server, phase):
    if phase == 'probe':
        code = """import json, torch
from pathlib import Path
RESULT = dict(out=str(OUT), model=str(MODEL_.config._name_or_path),
    helpers={k: k in globals() for k in ['PROC','prepare_video','fetch','qwen_inputs','move_inputs','media_layout','render','parse_question','rotations']},
    files=[str(p.relative_to(OUT)) for p in Path(OUT).glob('*')],
    dtype=str(next(MODEL_.parameters()).dtype), torch=torch.__version__)
print(json.dumps(RESULT),flush=True)
"""
    else:
        if phase == 'prepare':
            for prefix in ('avbind_native_pairs', 'avbind_pooled_pairs'):
                for shard in (1, 2):
                    upload(server, ROOT / 'results' / f'{prefix}_s{shard}.pt')
            upload(server, ROOT / 'results/research40_manifest.json')
            for name in ('avbind_sdpa_compat.py', 'chronodesmear_cache40.py'):
                upload(server, ROOT / 'server_jobs' / name)
            upload(server, ROOT / 'results/pcebr_stage0_v7/stage0_pcebr_v7_final_original.py')
            upload(server, ROOT / 'results/pcebr_stage0_v5/stage0_pcebr_v5_original.py')
        path = ROOT / 'server_jobs' / f'pcebr_stage0_v7_{phase}.py'
        upload(server, path)
        code = path.read_text(encoding='utf-8')
    result = submit(server, f'PC_SERVER={server}\nLAB_SERVER={server}\n' + code, f'pcebr_stage0_v7_{phase}')
    (D / f'launch_{phase}_s{server}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return dict(server=server, **result)

def checkpoint(server, phase):
    launch_data = json.loads((D / f'launch_{phase}_s{server}.json').read_text())
    status = call(server, '/job/' + launch_data['job_id'])
    (D / f'job_{phase}_s{server}.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    names = {'probe': [], 'prepare': ['prepare_audit'],
             'run': ['progress', 'instrumentation', 'main_results', 'input_audit', 'm3_audit', 'completion', 'error', 'extra_unit_audit'],
             'continue': ['progress', 'instrumentation', 'main_results', 'input_audit', 'm3_audit', 'completion', 'error', 'extra_unit_audit'],
             'diagnose_m0': ['m0_diagnostic'],
             'controls': ['control_accuracy'],
             'archive': ['archive_index', 'archive_audit', 'coupling_actual_input_audit', 'hop_condition_comparison']}[phase]
    for name in names:
        try:
            data = call(server, '/file?path=' + quote(f'pcebr_stage0_v7/{name}.json'), raw=True)
            (D / f'{name}_s{server}.json').write_bytes(data)
        except Exception:
            pass
    if phase == 'archive' and status.get('status') == 'done':
        index = json.loads((D / f'archive_index_s{server}.json').read_text())
        dest = D / f'raw_artifacts_s{server}.tar.gz'
        if not dest.exists() or dest.stat().st_size != index['bytes']:
            dest.write_bytes(call(server, '/file?path=pcebr_stage0_v7/raw_artifacts.tar.gz', raw=True, timeout=180))
    return dict(server=server, status=status.get('status'), seconds=status.get('seconds'),
                error=status.get('error'), tail=status.get('out', '')[-1200:])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['launch', 'checkpoint'])
    p.add_argument('phase', choices=['probe', 'prepare', 'run', 'continue', 'diagnose_m0', 'controls', 'archive'])
    args = p.parse_args()
    fn = launch if args.action == 'launch' else checkpoint
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for row in pool.map(lambda s: fn(s, args.phase), (1, 2)):
            print(json.dumps(row, ensure_ascii=False, indent=2))

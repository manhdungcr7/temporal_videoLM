from pathlib import Path
folder=Path(__file__).resolve().parent
s=(folder/'pcebr_stage0_v7_run.py').read_text(encoding='utf-8')
tail=s[s.index('    for V7_PREFIX in ('):]
pro='''import shutil
failed=V7_DIR/'original_m0_failure'
failed.mkdir(exist_ok=True)
for name in ('error.json','completion.json','progress.json','instrumentation.json'):
    if (V7_DIR/name).exists(): shutil.copy2(V7_DIR/name,failed/name)
V7_STATE['M0']=dict(status='failed_original_assertion',failed_lines=[x['line'] for x in v7_m0_checks if not x['passed']],diagnostic=v7_m0_output)
v7_start=time.time()
try:
'''
compile(pro+tail,'v7_continue','exec')
(folder/'pcebr_stage0_v7_continue.py').write_text(pro+tail,encoding='utf-8')
print('Original M1/M2/M3 phase continuation prepared; M0 remains failed.')

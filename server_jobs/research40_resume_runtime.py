"""Restore definitions only; do not rerun completed baseline or methods."""
import json
import shutil
from pathlib import Path
for filename in ('research40_manifest.json',f'research40_baseline_s{R40_SERVER}.jsonl'):
    target=OUT/filename
    if not target.exists():shutil.copyfile(OUT/'uploads'/filename,target)
sources=[('research40_baseline.py','records=r40_json_rows(R40_BASE_FILE)'),
         ('research40_attention.py','baseline=r40_json_rows(R40_BASE_FILE)'),
         ('research40_reasoning.py','baseline=r40_json_rows(R40_BASE_FILE)'),
         ('research40_memory.py','baseline=r40_json_rows(R40_BASE_FILE)')]
for filename,marker in sources:
    text=(OUT/'uploads'/filename).read_text(encoding='utf-8')
    head=text.split(marker,1)[0]
    assert len(head)<len(text),filename
    exec(compile(head,filename,'exec'),globals())
status={}
for name in ('baseline','attention_v2','reasoning','memory','raw_kv','faithful_memory'):
    path=OUT/f'research40_{name}_s{R40_SERVER}.jsonl'
    rows=r40_json_rows(path)
    status[name]=dict(completed=len(rows),missing=[r['idx'] for r in R40_SAMPLES if r['idx'] not in {x['idx'] for x in rows}])
cache_count=len(list(R40_CACHE.glob('*_rot*.pt')))
RESULT=dict(server=R40_SERVER,cache_count=cache_count,status=status,
            manifest_sha256=R40_MANIFEST['manifest_sha256'])
print('RESUMED_DEFINITIONS_ONLY',json.dumps(RESULT),flush=True)

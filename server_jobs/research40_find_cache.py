import json
from pathlib import Path
items=[]
for path in sorted(OUT.parent.glob('svhalluc_prompt_lab_*')):
    if not path.is_dir():continue
    cache=path/'research40_decoder_cache'
    manifest=path/'research40_manifest.json'
    if not manifest.exists():manifest=path/'uploads/research40_manifest.json'
    info=dict(path=str(path),cache_count=len(list(cache.glob('*_rot*.pt'))) if cache.exists() else 0,
              result_files=[p.name for p in path.glob('research40_*s2.jsonl')])
    if manifest.exists():
        m=json.loads(manifest.read_text());info['manifest_sha256']=m.get('manifest_sha256')
    items.append(info)
RESULT=items
print('CACHE_INVENTORY',json.dumps(items),flush=True)

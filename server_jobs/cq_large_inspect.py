"""Read dataset inventory and runtime identity without running inference."""
from collections import Counter
rows=json.loads(fetch('samples.json').read_text(encoding='utf-8'))
inventory=Counter((r['task_type_abbr'],r['answer']) for r in rows if r['data_type']=='temporal')
result=dict(server=LAB_SERVER,model=MODEL_.config._name_or_path,
            revision=globals().get('MREV'),layers=len(MODEL_.model.layers),
            cells=[dict(task=t,gold=g,n=n) for (t,g),n in sorted(inventory.items())],
            samples_sha256=__import__('hashlib').sha256(fetch('samples.json').read_bytes()).hexdigest())
(OUT/f'cq_large_inventory_s{LAB_SERVER}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
RESULT=result
print('LARGE_INVENTORY',json.dumps(result),flush=True)

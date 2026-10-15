records=[]
for p in OUT.glob('cq_physical_memory_*.json'):
    value=json.loads(p.read_text())
    records.append(dict(name=p.name,type=type(value).__name__,keys=list(value)[:3] if isinstance(value,dict) else None))
RESULT=dict(caches=records)
print('CACHE_TYPES',json.dumps(RESULT),flush=True)

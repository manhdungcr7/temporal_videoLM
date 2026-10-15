"""Unwrap PowerShell serialization artifacts; preserve bad copies for audit."""
repaired=[]
for p in list(OUT.glob('cq_physical_memory_*.json'))+list(OUT.glob('avbind_utterances_*.json')):
    value=json.loads(p.read_text())
    if isinstance(value,dict) and isinstance(value.get('value'),list):
        backup=p.with_suffix('.powershell-wrapper.json')
        if not backup.exists():backup.write_bytes(p.read_bytes())
        p.write_text(json.dumps(value['value']),encoding='utf-8');repaired.append(p.name)
    elif not isinstance(value,list):raise ValueError(f'Unexpected cache format: {p.name}')
RESULT=dict(repaired=repaired)
print('CACHE_WRAPPERS_REPAIRED',json.dumps(RESULT),flush=True)

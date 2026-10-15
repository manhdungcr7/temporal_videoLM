"""Restore only physically independent observations, never prior answers."""
bundle=json.loads((OUT/'uploads/avbind_restore_bundle.json').read_text())
restored=[]
for name,value in bundle.items():
    if not re.fullmatch(r'(cq_physical_memory|avbind_utterances)_\d+\.json',name):
        raise ValueError('Unexpected cache name')
    path=OUT/name
    if not path.exists():
        path.write_text(json.dumps(value),encoding='utf-8');restored.append(name)
RESULT=dict(restored=restored)
print('AVBIND_CACHES_RESTORED',json.dumps(RESULT),flush=True)

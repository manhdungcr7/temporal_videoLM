import hashlib
source=(OUT/'uploads/avbind_clocked.py').read_text()
exec(compile(source,'avbind_clocked.py','exec'),TIMELINE_NS)
RESULT=dict(api='avbind_clocked_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),question_type_routing=False)
print('CLOCKED_READY',json.dumps(RESULT),flush=True)

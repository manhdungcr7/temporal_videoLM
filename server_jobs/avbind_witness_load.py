import hashlib
exec(compile((OUT/'uploads/avbind_context_clock.py').read_text(),'avbind_context_clock.py','exec'),TIMELINE_NS)
source=(OUT/'uploads/avbind_witness.py').read_text()
TIMELINE_NS['WITNESS_SOURCE_SHA']=hashlib.sha256(source.encode()).hexdigest()
exec(compile(source,'avbind_witness.py','exec'),TIMELINE_NS)
RESULT=dict(api='avbind_witness_answer',source_sha256=TIMELINE_NS['WITNESS_SOURCE_SHA'],reuses_R14_alignment=True)
print('WITNESS_READY',json.dumps(RESULT),flush=True)

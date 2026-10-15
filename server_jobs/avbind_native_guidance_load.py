import hashlib
source=(OUT/'uploads/avbind_native_guidance.py').read_text()
exec(compile(source,'avbind_native_guidance.py','exec'),TIMELINE_NS)
RESULT=dict(api='avbind_native_guidance_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),no_retraining=True)
print('NATIVE_LAYOUT_GUIDANCE_READY',json.dumps(RESULT),flush=True)

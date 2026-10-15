import hashlib
source=(OUT/'uploads/avbind_grounded_guidance.py').read_text()
exec(compile(source,'avbind_grounded_guidance.py','exec'),TIMELINE_NS)
RESULT=dict(api='avbind_grounded_guidance_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),no_retraining=True)
print('GROUNDED_GUIDANCE_READY',json.dumps(RESULT),flush=True)

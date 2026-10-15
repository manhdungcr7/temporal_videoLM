import hashlib
TIMELINE_NS['qwen_inputs']=qwen_inputs
assert {'qwen_inputs','timeline_isolation','MODEL_','PROC','TOK','SYSTEM','rotations'} <= TIMELINE_NS.keys()
source=(OUT/'uploads/avbind_context_clock.py').read_text()
exec(compile(source,'avbind_context_clock.py','exec'),TIMELINE_NS)
RESULT=dict(api='avbind_context_clock_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    source_encoder_context='full original source; not strict temporal crop isolation',question_type_routing=False)
print('CONTEXT_CLOCK_READY',json.dumps(RESULT),flush=True)

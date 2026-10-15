source=(OUT/'uploads/avbind_native.py').read_text()
exec(compile(source,'avbind_native.py','exec'),NEURAL_NS)
RESULT=dict(raw_AV_retained=True,metadata=NEURAL_NS['BIND_METADATA'])
print('BIND_NATIVE_READY',json.dumps(RESULT),flush=True)

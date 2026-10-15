source=(OUT/'uploads/avbind_kv.py').read_text()
exec(compile(source,'avbind_kv.py','exec'),NEURAL_NS)
RESULT=dict(api='avbind_kv_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            gain=1.0,reader_checkpoint=NEURAL_NS['BIND_METADATA'],external_keys_clock_free=True,external_values_source_isolated=True)
print('INDEPENDENT_KV_READY',json.dumps(RESULT),flush=True)

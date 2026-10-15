NEURAL_NS['media_layout']=media_layout
source=(OUT/'uploads/avbind_attention.py').read_text()
exec(compile(source,'avbind_attention.py','exec'),NEURAL_NS)
RESULT=dict(api='avbind_attention_answer',layers=list(range(29,37)),gain=2.0,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),question_template_routing=False)
print('NATIVE_SLOT_ATTENTION_READY',json.dumps(RESULT),flush=True)

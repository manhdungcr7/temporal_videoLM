source=(OUT/'uploads/avbind_counterfactual.py').read_text()
exec(compile(source,'avbind_counterfactual.py','exec'),NEURAL_NS)
RESULT=dict(api='avbind_counterfactual_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            coefficient=1,question_template_routing=False,native_answer_media_unchanged=True)
print('DIFFERENTIAL_SLOT_API_READY',json.dumps(RESULT),flush=True)

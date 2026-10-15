source=(OUT/'uploads/avbind_latent.py').read_text()
NEURAL_NS['LATENT_SOURCE_SHA256']=hashlib.sha256(source.encode()).hexdigest()
exec(compile(source,'avbind_latent.py','exec'),NEURAL_NS)
RESULT=dict(api='avbind_latent_answer',generated_captions_used=False,question_template_router=False,
            supervision='Reader weights transferred from constructed textual training; not native AV supervision')
print('RAW_LATENT_API_READY',json.dumps(RESULT),flush=True)

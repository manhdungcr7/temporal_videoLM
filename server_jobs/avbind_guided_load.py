import hashlib,types,sys
# Update the existing clean namespace, preserving the context-clock functions.
exec(compile((OUT/'uploads/avbind_timeline.py').read_text(),'avbind_timeline.py','exec'),TIMELINE_NS)
exec(compile((OUT/'uploads/avbind_context_clock.py').read_text(),'avbind_context_clock.py','exec'),TIMELINE_NS)
alignment_source=(OUT/'uploads/avbind_alignment.py').read_text()
module=types.ModuleType('avbind_alignment_inference');sys.modules[module.__name__]=module
exec(compile(alignment_source,'avbind_alignment.py','exec'),module.__dict__)
checkpoint=torch.load(OUT/'uploads/avbind_native_alignment_r14.pt',map_location='cpu',weights_only=True)
assert checkpoint['metadata']['source_sha256']==hashlib.sha256(alignment_source.encode()).hexdigest()
assert checkpoint['metadata']['model_revision']==MREV
alignment=module.NativeSemanticAlignment().to('cuda:0');alignment.load_state_dict(checkpoint['state_dict']);alignment.eval()
pair_source=(OUT/'uploads/avbind_native_pairs.py').read_text()
assert checkpoint['metadata']['pair_source_sha256']==hashlib.sha256(pair_source.encode()).hexdigest()
pair_ns={name:globals()[name] for name in ('MODEL_','PROC','SYSTEM','qwen_inputs','move_inputs')};pair_ns['TOK']=PROC.tokenizer
exec(compile(pair_source,'avbind_native_pairs.py','exec'),pair_ns)
TIMELINE_NS.update(GUIDE_ALIGNMENT=alignment,GUIDE_METADATA=checkpoint['metadata'],
    occurrence_token_bias=module.occurrence_token_bias,native_pair_feature=pair_ns['native_pair_feature'])
source=(OUT/'uploads/avbind_guided.py').read_text()
exec(compile(source,'avbind_guided.py','exec'),TIMELINE_NS)
RESULT=dict(api='avbind_guided_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    alignment_metadata=checkpoint['metadata'],question_type_routing=False)
print('NATIVE_GUIDANCE_READY',json.dumps(RESULT),flush=True)

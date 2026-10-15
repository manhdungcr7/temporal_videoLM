import hashlib,types,sys
source=(OUT/'uploads/avbind_alignment.py').read_text();module=types.ModuleType('pooled_alignment_inference');sys.modules[module.__name__]=module
exec(compile(source,'avbind_alignment.py','exec'),module.__dict__)
saved=torch.load(OUT/'uploads/avbind_native_alignment_r18.pt',map_location='cpu',weights_only=True)
assert saved['metadata']['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
assert saved['metadata']['model_revision']==MREV
pool_source=(OUT/'uploads/avbind_pooled_pairs.py').read_text()
assert saved['metadata']['pair_source_sha256']==hashlib.sha256(pool_source.encode()).hexdigest()
alignment=module.NativeSemanticAlignment().to('cuda:0');alignment.load_state_dict(saved['state_dict']);alignment.eval()
TIMELINE_NS.update(GUIDE_ALIGNMENT=alignment,GUIDE_METADATA=saved['metadata'],POOL_SOURCE_SHA=hashlib.sha256(pool_source.encode()).hexdigest())
exec(compile(pool_source,'avbind_pooled_pairs.py','exec'),TIMELINE_NS)
TIMELINE_NS['guided_raw_bank']=TIMELINE_NS['pooled_raw_bank']
RESULT=dict(api='avbind_grounded_guidance_answer',representation=saved['metadata']['representation'],metadata=saved['metadata'])
print('POOLED_NATIVE_GUIDANCE_READY',json.dumps(RESULT),flush=True)

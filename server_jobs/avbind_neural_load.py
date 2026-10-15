"""Initialize a small reader without changing backbone parameters."""
import types,sys,hashlib
module=types.ModuleType('avbind_reader_runtime');sys.modules[module.__name__]=module
reader_source=(OUT/'uploads'/globals().get('BIND_READER_SOURCE','avbind_reader.py')).read_text()
exec(compile(reader_source,'avbind_reader.py','exec'),module.__dict__)
NEURAL_NS=dict(AVB_NS)
BIND_LAYER=28
dim=MODEL_.get_input_embeddings().embedding_dim
device=next(MODEL_.model.layers[BIND_LAYER-1].parameters()).device
torch.manual_seed(2026101404)
reader=module.AVBindReader(dim=dim,rank=64,num_sources=2,time_scale=50).to(device)
with torch.no_grad():
    for hop in reader.hops:hop.output.weight.zero_()
metadata=dict(model=CQ_INFERENCE_METADATA['model'],revision=CQ_INFERENCE_METADATA['model_revision'],
              reader_source_sha256=hashlib.sha256(reader_source.encode()).hexdigest(),
              layer=BIND_LAYER,dim=dim,rank=64,parameters=sum(p.numel() for p in reader.parameters()),
              trained=False,memory_features=globals().get('BIND_MEMORY_MODE','token_embedding'))
NEURAL_NS.update(BIND_LAYER=BIND_LAYER,BIND_READER=reader,BIND_METADATA=metadata,
                 BIND_MEMORY_MODE=globals().get('BIND_MEMORY_MODE','token_embedding'),BIND_FEATURE_CACHE={})
source=(OUT/'uploads'/globals().get('BIND_NEURAL_SOURCE','avbind_neural.py')).read_text()
metadata['integration_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
metadata['clock_origin']='minimum valid observation start'
exec(compile(source,'avbind_neural.py','exec'),NEURAL_NS)
assert not {'task','gold','DATA','ROWS','MANIFEST','interpret_choice','resolve_grounded_references'}&NEURAL_NS.keys()
RESULT=metadata
print('BIND_READER_READY',json.dumps(RESULT),flush=True)

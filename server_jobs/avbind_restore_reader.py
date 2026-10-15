"""Load only our own archived adapter, enforcing source/model compatibility."""
name=globals().get('BIND_CHECKPOINT','avbind_reader_checkpoint_r3.pt')
checkpoint=torch.load(OUT/'uploads'/name,map_location=next(NEURAL_NS['BIND_READER'].parameters()).device,weights_only=True)
metadata=checkpoint['metadata']
for key in ('reader_source_sha256','integration_source_sha256','revision','layer','dim','rank','memory_features'):
    if metadata.get(key)!=NEURAL_NS['BIND_METADATA'].get(key):
        raise ValueError(f'Checkpoint mismatch: {key}')
if not metadata.get('trained'):raise ValueError('Checkpoint is not a finished experiment')
NEURAL_NS['BIND_READER'].load_state_dict(checkpoint['state_dict'],strict=True)
NEURAL_NS['BIND_READER'].eval()
NEURAL_NS['BIND_METADATA']=metadata
RESULT=dict(restored=name,metadata=metadata)
print('AVBIND_CHECKPOINT_RESTORED',json.dumps(RESULT),flush=True)

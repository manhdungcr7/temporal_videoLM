"""Native media geometry, within vs across recordings; no answer intervention."""
import gc,hashlib,json,time
import numpy as np
import torch

CD_SERVER=globals().get('MG_SERVER',1)
_source=(OUT/'uploads/chronodesmear40.py').read_text().split('try: RESULT=cd_run()',1)[0]
exec(compile(_source,'chrono_definitions_only','exec'),globals())
MG_DIR=OUT/'media_geometry40';MG_DIR.mkdir(exist_ok=True)
MG_LAYERS=(0,8,16,22,30,35)
MG_SHA=hashlib.sha256((OUT/'uploads/media_geometry40.py').read_bytes()).hexdigest()
MG_ROWS=[]
for sample in CD_MANIFEST[f'server{CD_SERVER}']:
    idx=sample['idx'];path=MG_DIR/f'{idx}.npz';meta_path=MG_DIR/f'{idx}.json'
    if path.exists() and meta_path.exists():
        metadata=json.loads(meta_path.read_text());assert metadata['runtime_sha256']==MG_SHA
        MG_ROWS.append(metadata);continue
    payload=torch.load(CD_CACHE/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
    handles=[];features={}
    chunks=torch.floor(payload['sec'].float()/2).long()
    def capture(li):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            for name,source in [('video',payload['is_v']),('audio',payload['is_a'])]:
                pooled=[];times=[]
                for chunk in sorted(set(chunks[source].tolist())):
                    indices=torch.where(source&(chunks==chunk))[0].to(h.device)
                    pooled.append(h[0,indices].float().mean(0).detach().cpu().numpy())
                    times.append(2.*chunk)
                features[f'{name}_l{li}']=np.stack(pooled)
                features[f'{name}_times']=np.array(times)
            return out
        return hook
    try:
        for layer in MG_LAYERS: handles.append(MODEL_.model.layers[layer].register_forward_hook(capture(layer)))
        logits,_=cd_forward(payload)
    finally:
        for handle in handles: handle.remove()
    k=len(payload['native_logits'])
    replay=float((logits[CD_LABELS[:k]]-payload['native_logits']).abs().max())
    assert replay<=.12,replay
    np.savez_compressed(path,**features)
    metadata=dict(idx=idx,manifest_sha256=CD_SHA,runtime_sha256=MG_SHA,
        replay_error=replay,video_bins=len(features['video_times']),audio_bins=len(features['audio_times']))
    meta_path.write_text(json.dumps(metadata));MG_ROWS.append(metadata)
    del features,payload;gc.collect();torch.cuda.empty_cache()
    print('MEDIA_GEOMETRY',CD_SERVER,len(MG_ROWS),'/',20,idx,flush=True)
(MG_DIR/f'index_s{CD_SERVER}.json').write_text(json.dumps(MG_ROWS))
RESULT=dict(server=CD_SERVER,completed=len(MG_ROWS),expected=20,runtime_sha256=MG_SHA)
print('MEDIA_GEOMETRY_COMPLETE',json.dumps(RESULT),flush=True)

"""Native media-token content slots, no assistant/prompt state pooling.

Physical source isolation, question/clock blind keys. This is a representation
ablation, not proof that one averaged slot retains all fine-grained events.
"""
import hashlib
import math
from pathlib import Path
import torch
import torch.nn.functional as F


def pool_media_states(hidden,mask):
    if hidden.ndim!=3 or mask.shape!=hidden.shape[:2] or not mask.any():
        raise ValueError('Expected nonempty native media states')
    if hidden.shape[0]!=1:raise ValueError('One physical source per content slot')
    selected=hidden[0,mask[0]].float()
    return F.layer_norm(selected.mean(0),(hidden.shape[-1],))


def pooled_native_feature(media,mode):
    if mode not in ('vision','audio'):raise ValueError('Physical source isolation required')
    inp,_=qwen_inputs(PROC,'',media,mode);inp=move_inputs(inp,MODEL_)
    token=TOK.convert_tokens_to_ids(PROC.video_token if mode=='vision' else PROC.audio_token)
    mask=inp['input_ids']==token;captured={}
    class Ready(Exception):pass
    def capture(module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['raw']=pool_media_states(hidden,mask.to(hidden.device)).detach().cpu();raise Ready()
    handle=MODEL_.model.layers[27].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_audio_in_video=False,use_cache=False)
    except Ready:pass
    finally:handle.remove()
    return captured['raw']


def pooled_raw_bank(video,cache=None):
    metadata=dict(media_sha256=hashlib.sha256(Path(video).read_bytes()).hexdigest(),
        model_revision=GUIDE_METADATA['model_revision'],source_sha256=POOL_SOURCE_SHA,
        representation='mean native media-token hidden states',layer=28,question_used=False,clock_in_key=False)
    path=Path(cache) if cache else None
    if path and path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['metadata']==metadata,'Pooled cache provenance mismatch'
        return saved['nodes'],saved['raw']
    duration=len(decode_audio(video))/16000;nodes=[];raw=[]
    for index in range(int(math.ceil(duration/2))):
        start,end=2*index,min(2*index+2,duration)
        if end-start<.2:continue
        media=prepare_video(cut(video,start,end,f'avbind_latent_{index}'))
        for source,mode in (('video','vision'),('audio','audio')):
            raw.append(pooled_native_feature(media,mode));nodes.append(dict(source=source,start=start,end=end))
    raw=torch.stack(raw)
    if path:
        temporary=path.with_suffix('.tmp');torch.save(dict(metadata=metadata,nodes=nodes,raw=raw),temporary);temporary.replace(path)
    return nodes,raw

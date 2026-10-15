"""Encode the unchanged frozen40 inputs once; reusable by all decoder audits."""
import gc, hashlib, json, time
import numpy as np
import torch
from pathlib import Path

CD_SERVER = globals().get('CD_SERVER', globals().get('R40_SERVER', 1))
CD_MANIFEST = json.loads((OUT/'uploads/research40_manifest.json').read_text())
CD_SHA = CD_MANIFEST['manifest_sha256']
CD_CACHE = OUT/'argon_latent40/decoder_cache'
CD_CACHE.mkdir(parents=True, exist_ok=True)
CD_LABELS = [PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC']
exec(compile((OUT/'uploads/avbind_sdpa_compat.py').read_text(),'sdpa_compat','exec'),globals())

def cd_prompt(stem, texts, perm):
    return render(stem,texts,perm)+'\nAnswer with only the letter of the best answer.'

def cd_encode(inp, k):
    captured={}
    def capture(_module,args,kwargs):
        for key in ('inputs_embeds','position_ids','attention_mask','cache_position'):
            if kwargs.get(key) is not None: captured[key]=kwargs[key].detach().cpu()
    handles=[MODEL_.model.register_forward_pre_hook(capture,with_kwargs=True),
             MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))]
    try:
        with torch.inference_mode():
            logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
    finally:
        for handle in handles: handle.remove()
    _,sec,is_v,is_a,last=media_layout(MODEL_,PROC,inp)
    spg=inp.get('video_second_per_grid')
    assert torch.isfinite(logits).all()
    return dict(decoder=captured,native_logits=logits[CD_LABELS[:k]],sec=sec.cpu(),
        is_v=is_v.cpu(),is_a=is_a.cpu(),last_media=int(last),
        spg=float(spg.reshape(-1)[0]) if spg is not None else None,
        tokens=int(inp['input_ids'].shape[1]),manifest_sha256=CD_SHA,prompt='letter_only_v1')

def cd_build_cache():
    assert CD_MANIFEST['total']==40
    completed=0
    for sample in CD_MANIFEST[f'server{CD_SERVER}']:
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        stem,texts=parse_question(sample['question']);k=len(texts);media=None
        for rotation,perm in enumerate(rotations(k)):
            cache=CD_CACHE/f"{sample['idx']}_rot{rotation}.pt"
            if not cache.exists():
                if media is None:
                    source=fetch(sample['video_path'])
                    media=prepare_video(source,max_pixels=int(160*min(1.,32/sample['duration']))*28*28)
                inputs,_=qwen_inputs(PROC,cd_prompt(stem,texts,perm),media,'av')
                inputs=move_inputs(inputs,MODEL_)
                payload=cd_encode(inputs,k)
                tmp=cache.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(cache)
                del inputs,payload;gc.collect();torch.cuda.empty_cache()
            completed+=1
        del media
        print('CHRONO_CACHE',CD_SERVER,sample['idx'],completed,flush=True)
    return dict(server=CD_SERVER,rotations=completed,manifest_sha256=CD_SHA)

if globals().get('CD_CACHE_RUN',True):
    RESULT=cd_build_cache()
    print('CHRONO_CACHE_COMPLETE',json.dumps(RESULT),flush=True)

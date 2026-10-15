"""Rebuild lost encoder inputs after a new Kaggle session; preserve saved baseline scores."""
import json,time,gc
import numpy as np
import torch

class R40CapturedDecoderInput(Exception):pass

def r40_capture_encoder_only(inp):
    captured={}
    def hook(_module,args,kwargs):
        for key in ('inputs_embeds','position_ids','attention_mask','cache_position'):
            if kwargs.get(key) is not None:captured[key]=kwargs[key].detach().cpu()
        raise R40CapturedDecoderInput()
    handle=MODEL_.model.register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with torch.inference_mode():MODEL_(**inp,use_audio_in_video=True,use_cache=False)
    except R40CapturedDecoderInput:pass
    finally:handle.remove()
    assert 'inputs_embeds' in captured and 'position_ids' in captured
    return captured

baseline={r['idx']:r for r in r40_json_rows(R40_BASE_FILE)}
assert len(baseline)==len(R40_SAMPLES)==20
for r in R40_SAMPLES:
    started=time.time();stem,texts=parse_question(r['question']);k=len(texts)
    paths=[R40_CACHE/f"{r['idx']}_rot{i}.pt" for i in range(k)]
    if all(p.exists() for p in paths):continue
    pixels=baseline[r['idx']]['pixels'];media=prepare_video(fetch(r['video_path']),max_pixels=pixels)
    first_capture=None;first_ids=None;first_prefix=None
    for rotation,perm in enumerate(rotations(k)):
        cache=paths[rotation]
        if cache.exists():continue
        inp,_=qwen_inputs(PROC,r40_prompt(stem,texts,perm),media,'av');inp=move_inputs(inp,MODEL_)
        assert inp['input_ids'].shape[1]==baseline[r['idx']]['tokens'][rotation],'Input token count changed'
        pos,sec,is_v,is_a,last=media_layout(MODEL_,PROC,inp);prefix=last+1
        if first_capture is None:
            captured=r40_capture_encoder_only(inp);first_capture=captured;first_ids=inp['input_ids'].cpu();first_prefix=prefix
        else:
            assert prefix==first_prefix and torch.equal(inp['input_ids'].cpu()[:,:prefix],first_ids[:,:prefix])
            with torch.inference_mode():tail=MODEL_.get_input_embeddings()(inp['input_ids'][:,prefix:]).cpu()
            captured=dict(inputs_embeds=torch.cat((first_capture['inputs_embeds'][:,:prefix],tail),dim=1),
                          position_ids=pos.cpu(),attention_mask=inp['attention_mask'].cpu(),
                          cache_position=torch.arange(inp['input_ids'].shape[1]))
        payload=dict(decoder=captured,is_v=is_v.cpu(),is_a=is_a.cpu(),sec=sec.cpu(),last_media=last,
                     input_ids=inp['input_ids'].cpu(),num_options=k,fingerprint=R40_FINGERPRINT,
                     provenance='Encoder-only capture after GPU session reset. Original baseline scores preserved; no fabricated per-rotation logits.')
        torch.save(payload,cache);del inp
    print('ENCODER_INPUT_RECOVERED',r['idx'],k,'rotations',round(time.time()-started,2),'seconds; baseline not rescored',flush=True)
    del media,first_capture;gc.collect();torch.cuda.empty_cache()
RESULT=dict(cache_count=len(list(R40_CACHE.glob('*_rot*.pt'))),baseline_rows=len(baseline),baseline_unchanged=True)
print('INPUT_RECOVERY_DONE',json.dumps(RESULT),flush=True)

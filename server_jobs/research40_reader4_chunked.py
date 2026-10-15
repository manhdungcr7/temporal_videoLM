"""Causally equivalent bounded-memory prefill; no evidence/token truncation."""
import gc,json
import numpy as np
import torch
from transformers import DynamicCache

def r40_reader_prefill(inputs):
    ids=inputs['input_ids'];mask=inputs['attention_mask'];cache=DynamicCache(config=R40_READER_MODEL.config)
    offset=0;logits=None
    handle=R40_READER_MODEL.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():
            for lo in range(0,ids.shape[1],256):
                hi=min(ids.shape[1],lo+256)
                result=R40_READER_MODEL(input_ids=ids[:,lo:hi],attention_mask=mask[:,:hi],
                    past_key_values=cache,cache_position=torch.arange(lo,hi,device=ids.device),
                    use_cache=True,return_dict=True)
                logits=result.logits[0,-1].float();cache=result.past_key_values;offset=hi
    finally:handle.remove()
    return logits,cache,offset

def r40_reader_scores(messages):
    inputs=r40_reader_inputs(messages)
    logits,cache,offset=r40_reader_prefill(inputs)
    return logits[R40_READER_LABELS].cpu().numpy()

def r40_atomic_generate(messages,limit=256):
    inputs=r40_reader_inputs(messages,False)
    cfg=R40_READER_MODEL.generation_config
    # With these settings manual argmax is equivalent to the existing greedy
    # generation. Refuse to silently omit a configured logits processor.
    assert getattr(cfg,'repetition_penalty',1.)==1.
    assert getattr(cfg,'no_repeat_ngram_size',0)==0
    assert getattr(cfg,'forced_eos_token_id',None) is None
    assert getattr(cfg,'suppress_tokens',None) is None
    assert getattr(cfg,'min_length',0)==0
    eos=cfg.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
    eos.add(R40_READER_TOKENIZER.eos_token_id)
    logits,cache,offset=r40_reader_prefill(inputs);tokens=[]
    device=inputs['input_ids'].device
    handle=R40_READER_MODEL.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():
            for _ in range(limit):
                token=int(logits.argmax())
                if token in eos:break
                tokens.append(token)
                result=R40_READER_MODEL(input_ids=torch.tensor([[token]],device=device),
                    attention_mask=torch.ones((1,offset+1),dtype=torch.long,device=device),
                    past_key_values=cache,cache_position=torch.tensor([offset],device=device),
                    use_cache=True,return_dict=True)
                logits=result.logits[0,-1].float();cache=result.past_key_values;offset+=1
    finally:handle.remove()
    return R40_READER_TOKENIZER.decode(tokens,skip_special_tokens=True).strip()

# Audit on a synthetic text prompt, not a re-evaluation of development baseline.
audit_messages=[dict(role='user',content=(
    'Independent observations: video shows a person opening a drawer from2 to4s; '
    'audio contains a bell from8 to9s. ')*32+
    '\nWhich starts first? (A) drawer opening (B) bell (C) neither. Reply only a letter.')]
audit_inputs=r40_reader_inputs(audit_messages)
handle=R40_READER_MODEL.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
try:
    with torch.inference_mode():reference=R40_READER_MODEL(**audit_inputs,use_cache=False).logits[0,-1][R40_READER_LABELS].float().cpu().numpy()
finally:handle.remove()
chunked=r40_reader_scores(audit_messages)
error=float(np.max(np.abs(reference-chunked)))
assert error<=.1,(reference.tolist(),chunked.tolist(),error)
audit=dict(tokens=int(audit_inputs['input_ids'].shape[1]),chunk_tokens=256,
           full_scores=reference.tolist(),chunked_scores=chunked.tolist(),max_logit_error=error,
           same_argmax=bool(reference.argmax()==chunked.argmax()),evidence_truncated=False)
(OUT/f'research40_reader4_prefill_audit_s{R40_SERVER}.json').write_text(json.dumps(audit,indent=2))
print('READER4_CAUSAL_PREFILL_AUDIT',json.dumps(audit),flush=True)
del audit_inputs;gc.collect();torch.cuda.empty_cache()

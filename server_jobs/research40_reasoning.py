"""Question-conditioned, option-blind reasoning with the original AV always accessible.

One shared reasoning policy for every question; no task names, speech gates, label
semantics, modality order, source annotations, or ground truth in the inference path.
"""
import json, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import DynamicCache

R40_REASON_FILE=OUT/f'research40_reasoning_s{R40_SERVER}.jsonl'
R40_REASON_INSTRUCTION=(
    '\nExamine the original recording to answer this question. Give a concise analysis of the relevant '
    'observations and how they support the answer. Keep what is heard distinct from what is visibly observed. '
    'Use whichever evidence is needed, in whichever order is appropriate. '
    'For relationships over time, compare the actual occurrence intervals rather than assuming that related '
    'content is simultaneous. Do not invent an observation or a timestamp. Do not give the final answer yet.')

def r40_gpu_bank(bank):
    return tuple((k.to(layer.self_attn.q_proj.weight.device),v.to(layer.self_attn.q_proj.weight.device))
                 for layer,(k,v) in zip(MODEL_.model.layers,bank))

def r40_cached_text(ids,cache,next_pos,offset):
    device=MODEL_.get_input_embeddings().weight.device
    tokens=torch.tensor([ids],device=device,dtype=torch.long)
    embeds=MODEL_.get_input_embeddings()(tokens)
    positions=next_pos.to(device)+torch.arange(len(ids),device=device)[None,None,:]
    with torch.inference_mode():
        out=MODEL_.model(inputs_embeds=embeds,position_ids=positions,
            attention_mask=torch.ones((1,offset+len(ids)),dtype=torch.long,device=device),
            cache_position=torch.arange(offset,offset+len(ids),device=device),
            past_key_values=cache,use_cache=True,return_dict=True)
        logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float()
    return logits,out.past_key_values,positions[:,:,-1:]+1,offset+len(ids)

def r40_analyze(payload,question):
    bank,prefix=r40_prefix(payload);cache=DynamicCache(ddp_cache_data=r40_gpu_bank(bank),config=MODEL_.model.config)
    tail=PROC.tokenizer.decode(payload['input_ids'][0,prefix:].tolist(),skip_special_tokens=False)
    beginning=tail[:tail.index(question)]
    ending=tail[tail.rindex('<|im_end|>'):]
    prompt=beginning+question+R40_REASON_INSTRUCTION+ending
    ids=PROC.tokenizer.encode(prompt,add_special_tokens=False)
    next_pos=payload['decoder']['position_ids'][:,:,prefix:prefix+1]
    logits,cache,next_pos,offset=r40_cached_text(ids,cache,next_pos,prefix)
    generated=[];eos={PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')}
    for _ in range(80):
        token=int(logits.argmax())
        if token in eos:break
        generated.append(token)
        logits,cache,next_pos,offset=r40_cached_text([token],cache,next_pos,offset)
    return dict(bank=cache.to_legacy_cache(),next_pos=next_pos,offset=offset,
                text=PROC.tokenizer.decode(generated,skip_special_tokens=True),tokens=len(generated))

def r40_reason_answer(state,texts,perm):
    prompt=('<|im_end|>\n<|im_start|>user\nUse the original recording and the analysis above to select the best answer. '
            'Answer choices: '+ ' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))+
            '\nAnswer with only the letter of the best answer.<|im_end|>\n<|im_start|>assistant\n')
    ids=PROC.tokenizer.encode(prompt,add_special_tokens=False)
    cache=DynamicCache(ddp_cache_data=state['bank'],config=MODEL_.model.config)
    lg,_,_,_=r40_cached_text(ids,cache,state['next_pos'],state['offset'])
    return lg[R40_LABELS[:len(texts)]].cpu().numpy()

baseline=r40_json_rows(R40_BASE_FILE)
assert len(baseline)==len(R40_SAMPLES),'Baseline incomplete; keep fixed sample IDs.'
base_by_id={r['idx']:r for r in baseline}
records=r40_json_rows(R40_REASON_FILE);done={r['idx'] for r in records}
print('REASONING_START',R40_SERVER,len(done),'/',len(R40_SAMPLES),flush=True)
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time()
    try:
        stem,texts=parse_question(r['question']);k=len(texts)
        payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False)
        payload['sample_idx']=r['idx']
        state=r40_analyze(payload,stem);tot=np.zeros(k)
        for perm in rotations(k):
            lg=r40_reason_answer(state,texts,perm);lp=lg-np.logaddexp.reduce(lg)
            for slot,i in enumerate(perm):tot[i]+=lp[slot]
        prior=np.array(base_by_id[r['idx']]['prior_scores'])
        rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',
                 manifest_sha256=R40_MANIFEST['manifest_sha256'],
                 scores=dict(evidence_cot=tot.tolist(),evidence_cot_low_pmi=(tot-.25*prior).tolist()),
                 analysis=state['text'],reasoning_tokens=state['tokens'],seconds=round(time.time()-started,2))
        with R40_REASON_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(r['idx'])
        merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
        r40_snapshot(merged,'reasoning')
        print('REASONING_SAMPLE',json.dumps(rec),flush=True)
        del payload,state;gc.collect();torch.cuda.empty_cache()
    except Exception:
        import traceback
        print('REASONING_ERROR',r['idx'],traceback.format_exc(),flush=True)
        torch.cuda.empty_cache();raise
RESULT=r40_snapshot(merged,'reasoning_done')
print('REASONING_SUMMARY',json.dumps(RESULT),flush=True)

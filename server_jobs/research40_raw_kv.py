"""Source-preserving native KV: independent AV perception, unrestricted question reasoning.

Encoder inputs, rotary clocks, full recording, and native weights are unchanged.
Only media-query edges are source restricted; every question can read both streams.
No question classifier, speech window, relation-specific vector, or fixed handoff.
"""
import contextlib, json, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as r40_modeling

R40_RAW_FILE=OUT/f'research40_raw_kv_s{R40_SERVER}.jsonl'

@contextlib.contextmanager
def r40_source_prefix_attention(payload):
    impl=MODEL_.model.layers[0].self_attn.config._attn_implementation
    original=r40_modeling.ALL_ATTENTION_FUNCTIONS[impl]
    prefix=payload['last_media']+1
    a_cpu=payload['is_a'][:prefix];v_cpu=payload['is_v'][:prefix]
    first=int(torch.where(a_cpu|v_cpu)[0][0])
    def kernel(module,q,k,v,attention_mask,**kw):
        assert q.shape[-2]==prefix and k.shape[-2]==prefix,'Prefix-only source intervention'
        device=q.device;a=a_cpu.to(device);vi=v_cpu.to(device)
        indices=torch.arange(prefix,device=device);system=indices<first
        kg=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        vg=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
        out=torch.empty((q.shape[0],prefix,q.shape[1],q.shape[-1]),dtype=q.dtype,device=device)
        groups=((a,a|system),(vi,vi|system),(~(a|vi),torch.ones_like(a)))
        for rows_mask,keys_mask in groups:
            rows=torch.where(rows_mask)[0];keys=torch.where(keys_mask)[0]
            if rows.numel()==0:continue
            selected_k=kg[:,:,keys,:];selected_v=vg[:,:,keys,:]
            for start in range(0,len(rows),256):
                rr=rows[start:start+256]
                mask=torch.zeros((1,1,len(rr),len(keys)),dtype=q.dtype,device=device)
                future=keys[None,:]>rr[:,None]
                mask.masked_fill_(future[None,None],float('-inf'))
                if attention_mask is not None:
                    mask+=attention_mask[:,:,rr,:][:,:,:,keys].to(q.dtype)
                z=F.scaled_dot_product_attention(q[:,:,rr,:],selected_k,selected_v,
                    attn_mask=mask,dropout_p=0.0,is_causal=False,scale=float(kw.get('scaling',module.scaling)))
                out[:,rr,:,:]=z.transpose(1,2)
        return out,None
    r40_modeling.ALL_ATTENTION_FUNCTIONS.register(impl,kernel)
    try:yield
    finally:r40_modeling.ALL_ATTENTION_FUNCTIONS.register(impl,original)

def r40_source_bank(payload):
    path=R40_CACHE/f"{payload['sample_idx']}_source_kv_v1.pt"
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    device=MODEL_.get_input_embeddings().weight.device;prefix=payload['last_media']+1
    source=payload['decoder']
    kwargs=dict(inputs_embeds=source['inputs_embeds'][:,:prefix].to(device),
                position_ids=source['position_ids'][:,:,:prefix].to(device),
                attention_mask=source['attention_mask'][:,:prefix].to(device),
                cache_position=torch.arange(prefix,device=device))
    with r40_source_prefix_attention(payload),torch.inference_mode():
        output=MODEL_.model(**kwargs,use_cache=True,return_dict=True)
    bank=tuple((k.detach().cpu(),v.detach().cpu()) for k,v in output.past_key_values)
    torch.save(bank,path);del output;torch.cuda.empty_cache()
    return bank

@contextlib.contextmanager
def r40_av_boost(payload,question,strength=1.5):
    impl=MODEL_.model.layers[0].self_attn.config._attn_implementation
    original=r40_modeling.ALL_ATTENTION_FUNCTIONS[impl]
    prefix=payload['last_media']+1
    rows_cpu=torch.tensor(r40_reader_rows(payload,question),dtype=torch.long)-prefix
    def kernel(module,q,k,v,attention_mask,**kw):
        out,weights=original(module,q,k,v,attention_mask,**kw)
        if not 18<=module.layer_idx<33:return out,weights
        rows=rows_cpu.to(q.device);ma=(payload['is_a']|payload['is_v']).to(q.device)
        kg=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        vg=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
        logits=torch.matmul(q[:,:,rows,:].float(),kg.float().transpose(-2,-1))*float(kw.get('scaling',module.scaling))
        if attention_mask is not None:logits+=attention_mask[:,:,rows,:].float()
        else:
            future=torch.arange(k.shape[-2],device=q.device)[None,:]>(rows[:,None]+prefix)
            logits.masked_fill_(future[None,None],float('-inf'))
        logits[...,ma]+=strength
        z=torch.matmul(torch.softmax(logits,dim=-1).to(vg.dtype),vg).transpose(1,2)
        out=out.clone();out[:,rows,:,:]=z.to(out.dtype)
        return out,weights
    r40_modeling.ALL_ATTENTION_FUNCTIONS.register(impl,kernel)
    try:yield
    finally:r40_modeling.ALL_ATTENTION_FUNCTIONS.register(impl,original)

def r40_bank_scores(payload,bank):
    prefix=payload['last_media']+1
    cache=DynamicCache(ddp_cache_data=r40_gpu_bank(bank),config=MODEL_.model.config)
    # Reuse exact saved rotary positions, including native media closing tokens.
    source=payload['decoder'];device=MODEL_.get_input_embeddings().weight.device
    length=source['inputs_embeds'].shape[1]
    with torch.inference_mode():
        out=MODEL_.model(inputs_embeds=source['inputs_embeds'][:,prefix:].to(device),
            position_ids=source['position_ids'][:,:,prefix:].to(device),
            attention_mask=source['attention_mask'].to(device),
            cache_position=torch.arange(prefix,length,device=device),
            past_key_values=cache,use_cache=True,return_dict=True)
        logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    options=payload['num_options'] if 'num_options' in payload else len(payload['logits'])
    return logits[R40_LABELS[:options]].numpy()

baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
assert len(baseline)==len(R40_SAMPLES)
records=r40_json_rows(R40_RAW_FILE);done={r['idx'] for r in records}
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time()
    try:
        stem,texts=parse_question(r['question']);k=len(texts)
        totals={m:np.zeros(k) for m in ('native_av_boost','independent_raw_kv','independent_raw_kv_boost')}
        control_total=np.zeros(k)
        payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
        source_bank=r40_source_bank(payload)
        native_bank,_=r40_prefix(payload)
        for rotation,perm in enumerate(rotations(k)):
            payload=torch.load(R40_CACHE/f"{r['idx']}_rot{rotation}.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
            if rotation==0 or 'logits' not in payload:
                control=r40_bank_scores(payload,native_bank)
                if 'logits' in payload:
                    error=float(np.max(np.abs(control-payload['logits'].numpy())))
                    assert error<=.06,f'Native KV replay mismatch {error}'
                else:
                    lp_control=control-np.logaddexp.reduce(control)
                    for slot,i in enumerate(perm):control_total[i]+=lp_control[slot]
            with r40_av_boost(payload,stem):lg1=r40_bank_scores(payload,native_bank)
            lg2=r40_bank_scores(payload,source_bank)
            with r40_av_boost(payload,stem):lg3=r40_bank_scores(payload,source_bank)
            for method,lg in zip(totals,(lg1,lg2,lg3)):
                lp=lg-np.logaddexp.reduce(lg)
                for slot,i in enumerate(perm):totals[method][i]+=lp[slot]
        if 'logits' not in payload:
            error=float(np.max(np.abs(control_total-np.array(base_by_id[r['idx']]['scores']['base']))))
            assert error<=.2,f'Recovered aggregate native replay mismatch {error}'
        rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',manifest_sha256=R40_MANIFEST['manifest_sha256'],
                 scores={m:v.tolist() for m,v in totals.items()},seconds=round(time.time()-started,2),replay_error=error)
        with R40_RAW_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(r['idx'])
        merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
        r40_snapshot(merged,'source_preserving_raw_kv')
        print('RAW_KV_SAMPLE',json.dumps(rec),flush=True)
        del source_bank,native_bank,payload;gc.collect();torch.cuda.empty_cache()
    except Exception:
        import traceback
        print('RAW_KV_ERROR',r['idx'],traceback.format_exc(),flush=True)
        torch.cuda.empty_cache();raise
RESULT=r40_snapshot(merged,'raw_kv_done')
print('RAW_KV_SUMMARY',json.dumps(RESULT),flush=True)

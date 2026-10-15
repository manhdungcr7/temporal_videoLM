"""Additive, source-symmetric attention probes on frozen raw-AV decoder inputs.

No task/label/window router, answer-semantic vectors, modality deletion, or RoPE change.
All corrections preserve the combined original A+V softmax numerator per row/head.
"""
import contextlib, json, time, gc, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as r40_modeling

R40_ATTENTION_METHODS=('additive_count_balance','argon_symmetric_soft_handoff','count_and_handoff')
R40_ATTN_FILE=OUT/f'research40_attention_v2_s{R40_SERVER}.jsonl'
R40_ATTN_CACHE=OUT/'research40_decoder_cache'
R40_ATTN_AUDIT=[]
R40_PREFIX_CACHE={}

def r40_prefix(payload):
    sample=payload['sample_idx'];prefix=payload['last_media']+1
    if sample not in R40_PREFIX_CACHE:
        disk=R40_ATTN_CACHE/f'{sample}_media_kv.pt'
        if disk.exists():
            bank=torch.load(disk,map_location='cpu',weights_only=False)
        else:
            device=MODEL_.get_input_embeddings().weight.device
            source=payload['decoder']
            kwargs=dict(inputs_embeds=source['inputs_embeds'][:,:prefix].to(device),
                        position_ids=source['position_ids'][:,:,:prefix].to(device),
                        attention_mask=source['attention_mask'][:,:prefix].to(device),
                        cache_position=torch.arange(prefix,device=device))
            with torch.inference_mode():
                output=MODEL_.model(**kwargs,use_cache=True,return_dict=True)
            bank=tuple((k.detach().cpu(),v.detach().cpu()) for k,v in output.past_key_values)
            torch.save(bank,disk)
            del output;torch.cuda.empty_cache()
        # Raw media cannot see later question tokens under native causal attention.
        # Reuse its native KV for every question rotation and intervention.
        R40_PREFIX_CACHE.clear();R40_PREFIX_CACHE[sample]=bank
    return R40_PREFIX_CACHE[sample],prefix

def r40_decoder_logits(payload):
    device=MODEL_.get_input_embeddings().weight.device
    bank,prefix=r40_prefix(payload)
    gpu_bank=[]
    for layer,(key,value) in zip(MODEL_.model.layers,bank):
        target=layer.self_attn.q_proj.weight.device
        gpu_bank.append((key.to(target),value.to(target)))
    cache=DynamicCache(ddp_cache_data=gpu_bank,config=MODEL_.model.config)
    source=payload['decoder'];length=source['inputs_embeds'].shape[1]
    kwargs=dict(inputs_embeds=source['inputs_embeds'][:,prefix:].to(device),
                position_ids=source['position_ids'][:,:,prefix:].to(device),
                attention_mask=source['attention_mask'].to(device),
                cache_position=torch.arange(prefix,length,device=device),past_key_values=cache)
    with torch.inference_mode():
        out=MODEL_.model(**kwargs,use_cache=True,return_dict=True)
        last=out.last_hidden_state[:,-1:,:]
        logits=MODEL_.lm_head(last.to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    return logits[R40_LABELS[:len(payload['logits'])]].numpy()

def r40_reader_rows(payload,question):
    ids=payload['input_ids'][0].tolist()
    qids=PROC.tokenizer.encode(question,add_special_tokens=False)
    starts=[i for i in range(payload['last_media']+1,len(ids)-len(qids)+1) if ids[i:i+len(qids)]==qids]
    if len(starts)!=1:raise RuntimeError(f'Cannot uniquely identify question token span: {starts}')
    # Actual question and assistant readout; answer-option tokens remain unchanged.
    start=starts[0]
    answer_prefix=PROC.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
    ast=[i for i in range(start,len(ids)-len(answer_prefix)+1) if ids[i:i+len(answer_prefix)]==answer_prefix]
    if len(ast)!=1:raise RuntimeError('Cannot identify assistant answer prefix')
    return list(range(start,start+len(qids)))+list(range(ast[0],len(ids)))

@contextlib.contextmanager
def r40_attention_context(payload,question,mode):
    config=MODEL_.model.layers[0].self_attn.config
    original_impl=config._attn_implementation
    original_kernel=r40_modeling.ALL_ATTENTION_FUNCTIONS[original_impl]
    prefix=payload['last_media']+1
    rows_cpu=torch.tensor(r40_reader_rows(payload,question),dtype=torch.long)-prefix
    state=dict(audit=[])
    layer_count=len(MODEL_.model.layers)
    lo=layer_count//2;hi=layer_count-3
    def kernel(module,q,k,v,attention_mask,**kw):
        out,weights=original_kernel(module,q,k,v,attention_mask,**kw)
        if not lo<=module.layer_idx<hi:return out,weights
        rows=rows_cpu.to(q.device)
        # Native rotary Q/K, actual scaling, native causal mask: no approximation to signs.
        kg=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        vg=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
        logits=torch.matmul(q[:,:,rows,:].float(),kg.float().transpose(-2,-1))*float(kw.get('scaling',module.scaling))
        if attention_mask is not None:
            logits+=attention_mask[:,:,rows,:k.shape[-2]].float()
        else:
            future=torch.arange(k.shape[-2],device=q.device)[None,:]>(rows[:,None]+prefix)
            logits.masked_fill_(future[None,None,:,:],float('-inf'))
        ma=payload['is_a'].to(q.device);mv=payload['is_v'].to(q.device)
        la=torch.logsumexp(logits[...,ma],dim=-1,keepdim=True)
        lv=torch.logsumexp(logits[...,mv],dim=-1,keepdim=True)
        combined=torch.logaddexp(la,lv)
        ua,uv=la,lv
        if mode in ('additive_count_balance','count_and_handoff'):
            # Remove half the token-count contribution, keeping AV's total budget fixed.
            ua=ua-.5*math.log(int(ma.sum()));uv=uv-.5*math.log(int(mv.sum()))
        if mode in ('argon_symmetric_soft_handoff','count_and_handoff'):
            # Softly exchange25% of source mass; directions vary per query/head/layer.
            norm=torch.logaddexp(ua,uv);pa=torch.exp(ua-norm)
            pa=.75*pa+.25*(1-pa)
            ua=torch.log(pa);uv=torch.log1p(-pa)
        total=torch.logaddexp(ua,uv)
        ba=combined+ua-total-la;bv=combined+uv-total-lv
        logits[...,ma]+=ba;logits[...,mv]+=bv
        if module.layer_idx in (lo,hi-1):
            denom=torch.logsumexp(logits,dim=-1,keepdim=True)
            state['audit'].append(dict(layer=module.layer_idx,
                a_mass_before=float(torch.exp(la-denom).mean()),v_mass_before=float(torch.exp(lv-denom).mean()),
                audio_bias=float(ba.mean()),video_bias=float(bv.mean()),n_audio=int(ma.sum()),n_video=int(mv.sum())))
        p=torch.softmax(logits,dim=-1).to(vg.dtype)
        updated=torch.matmul(p,vg).transpose(1,2).to(out.dtype)
        out=out.clone();out[:,rows,:,:]=updated
        return out,weights
    # Keep native mask dispatch under 'sdpa'; a new implementation name can
    # cause the Transformers causal-mask registry to return no mask.
    r40_modeling.ALL_ATTENTION_FUNCTIONS.register(original_impl,kernel)
    try:yield state
    finally:r40_modeling.ALL_ATTENTION_FUNCTIONS.register(original_impl,original_kernel)

baseline=r40_json_rows(R40_BASE_FILE)
assert len(baseline)==len(R40_SAMPLES),'Complete fixed baseline first; retain missing IDs, do not replace.'
base_by_id={r['idx']:r for r in baseline}
records=r40_json_rows(R40_ATTN_FILE);done={r['idx'] for r in records}
print('ATTENTION_START',R40_SERVER,len(done),'/',len(R40_SAMPLES),flush=True)
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time()
    try:
        stem,texts=parse_question(r['question']);k=len(texts)
        totals={m:np.zeros(k) for m in R40_ATTENTION_METHODS};audits={m:[] for m in totals}
        for rotation,perm in enumerate(rotations(k)):
            payload=torch.load(R40_ATTN_CACHE/f"{r['idx']}_rot{rotation}.pt",map_location='cpu',weights_only=False)
            payload['sample_idx']=r['idx']
            # Numerical identity check guards cache/position drift before any intervention.
            if rotation==0:
                replay=r40_decoder_logits(payload)
                error=float(np.max(np.abs(replay-payload['logits'].numpy())))
                if error>.06:raise RuntimeError(f'Baseline decoder replay mismatch {error}')
            for method in totals:
                with r40_attention_context(payload,stem,method) as state:
                    lg=r40_decoder_logits(payload)
                lp=lg-np.logaddexp.reduce(lg)
                for slot,i in enumerate(perm):totals[method][i]+=lp[slot]
                audits[method].append(state['audit'])
            del payload;torch.cuda.empty_cache()
        rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',
                 manifest_sha256=R40_MANIFEST['manifest_sha256'],
                 scores={m:v.tolist() for m,v in totals.items()},audit=audits,
                 replay_error=error,seconds=round(time.time()-started,2))
        with R40_ATTN_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(r['idx'])
        merged=[]
        for rr in records:
            br=base_by_id[rr['idx']]
            merged.append(dict(rr,scores=dict(br['scores'],**rr['scores'])))
        r40_snapshot(merged,'attention')
        print('ATTENTION_SAMPLE',json.dumps({k:rec[k] for k in ('idx','task','gold','scores','seconds','replay_error')}),flush=True)
    except Exception as exc:
        import traceback
        print('ATTENTION_ERROR',r['idx'],traceback.format_exc(),flush=True)
        torch.cuda.empty_cache()
        # A systemic implementation fault stops this method instead of producing a biased subset.
        raise
RESULT=r40_snapshot(merged,'attention_done')
print('ATTENTION_SUMMARY',json.dumps(RESULT),flush=True)

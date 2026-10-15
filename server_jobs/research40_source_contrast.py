"""Source counterfactuals on independent native KV, with one rule for all questions.

Related control: RED/AVCD-style contrastive evidence. Not claimed as novel by itself.
Mask absent sources additively; do not zero Keys or inject answer vectors.
"""
import contextlib,json,time,gc
import numpy as np
import torch
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as r40_modeling

R40_CONTRAST_FILE=OUT/f'research40_source_contrast_s{R40_SERVER}.jsonl'

@contextlib.contextmanager
def r40_source_readout(payload,source):
    impl=MODEL_.model.layers[0].self_attn.config._attn_implementation
    original=r40_modeling.ALL_ATTENTION_FUNCTIONS[impl]
    prefix=payload['last_media']+1
    first=int(torch.where(payload['is_a']|payload['is_v'])[0][0])
    def kernel(module,q,k,v,attention_mask,**kw):
        # These forward calls contain only the query tail with the media in KV.
        positions=torch.arange(k.shape[-2],device=q.device)
        allowed=(positions<first)|(positions>=prefix)
        if source=='audio':allowed|=payload['is_a'].to(q.device)
        elif source=='visual':allowed|=payload['is_v'].to(q.device)
        if attention_mask is None:
            mask=torch.zeros((1,1,q.shape[-2],k.shape[-2]),dtype=q.dtype,device=q.device)
            future=positions[None,:]>torch.arange(prefix,prefix+q.shape[-2],device=q.device)[:,None]
            mask.masked_fill_(future[None,None],float('-inf'))
        else:mask=attention_mask.clone()
        mask[...,~allowed]=float('-inf')
        return original(module,q,k,v,mask,**kw)
    r40_modeling.ALL_ATTENTION_FUNCTIONS.register(impl,kernel)
    try:yield
    finally:r40_modeling.ALL_ATTENTION_FUNCTIONS.register(impl,original)

raw=r40_json_rows(OUT/f'research40_raw_kv_s{R40_SERVER}.jsonl')
assert len(raw)==len(R40_SAMPLES),'Finish source KV ablation for every fixed ID first.'
raw_by_id={r['idx']:r for r in raw}
baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
records=r40_json_rows(R40_CONTRAST_FILE);done={r['idx'] for r in records}
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time();stem,texts=parse_question(r['question']);k=len(texts)
    totals={source:np.zeros(k) for source in ('audio','visual','neither')}
    payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
    bank=r40_source_bank(payload)
    for rotation,perm in enumerate(rotations(k)):
        payload=torch.load(R40_CACHE/f"{r['idx']}_rot{rotation}.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
        for source in totals:
            with r40_source_readout(payload,source):lg=r40_bank_scores(payload,bank)
            lp=lg-np.logaddexp.reduce(lg)
            for slot,i in enumerate(perm):totals[source][i]+=lp[slot]
    joint_scores=np.array(raw_by_id[r['idx']]['scores']['independent_raw_kv'])
    interaction=joint_scores-totals['audio']-totals['visual']+totals['neither']
    rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',manifest_sha256=R40_MANIFEST['manifest_sha256'],
        scores=dict(source_kv_joint_contrast05=(joint_scores+.5*interaction).tolist(),
                    source_kv_joint_contrast1=(joint_scores+interaction).tolist(),
                    source_kv_grounding_margin=(joint_scores+.5*(joint_scores-totals['neither'])).tolist()),
        branches={s:v.tolist() for s,v in totals.items()},interaction=interaction.tolist(),seconds=round(time.time()-started,2))
    with R40_CONTRAST_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(r['idx'])
    merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
    r40_snapshot(merged,'source_contrast')
    print('SOURCE_CONTRAST_SAMPLE',json.dumps(rec),flush=True)
    del bank,payload;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'source_contrast_done')
print('SOURCE_CONTRAST_SUMMARY',json.dumps(RESULT),flush=True)

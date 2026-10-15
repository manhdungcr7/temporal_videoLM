"""Stronger evidence-reader diagnostic, not a same-backbone improvement claim.

Freeze the original Omni model, clips, and source evidence. Compare direct,
question-only and option-blind reasoning using Qwen3-4B. No benchmark task router.
"""
import json,time,gc
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import HfApi

R40_READER_ID='Qwen/Qwen3-4B'
R40_READER_REV=HfApi().model_info(R40_READER_ID).sha
R40_READER_TOKENIZER=AutoTokenizer.from_pretrained(R40_READER_ID,revision=R40_READER_REV)
gc.collect();torch.cuda.empty_cache()
budget={}
for d in range(torch.cuda.device_count()):
    free,_=torch.cuda.mem_get_info(d)
    budget[d]=min(5*1024**3,int(free)-2*1024**3)
assert sum(budget.values())>9*1024**3,('Insufficient safe headroom',budget)
print('READER4_LOADING',R40_READER_ID,R40_READER_REV,budget,flush=True)
R40_READER_MODEL=AutoModelForCausalLM.from_pretrained(R40_READER_ID,revision=R40_READER_REV,
    torch_dtype=torch.float16,device_map='auto',max_memory=budget,attn_implementation='sdpa').eval()
assert all(v not in ('cpu','disk') for v in R40_READER_MODEL.hf_device_map.values()),R40_READER_MODEL.hf_device_map
R40_READER_LABELS=R40_READER_TOKENIZER.convert_tokens_to_ids(list('ABC'))
R40_READER_SYSTEM=('Answer the user question using the supplied independent observations of a recording. '
    'Distinguish words heard from actions seen. A speaker can describe another time. '
    'Keep the question direction and entities unchanged. Use any needed source, in any order. '
    'Source observations may be imperfect; do not invent missing events or times.')

def r40_reader_inputs(messages,thinking=False):
    text=R40_READER_TOKENIZER.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=True,enable_thinking=thinking)
    return R40_READER_TOKENIZER(text,return_tensors='pt').to(R40_READER_MODEL.get_input_embeddings().weight.device)

def r40_reader_scores(messages):
    inp=r40_reader_inputs(messages)
    h=R40_READER_MODEL.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():lg=R40_READER_MODEL(**inp,use_cache=False).logits[0,-1].float()
    finally:h.remove()
    return lg[R40_READER_LABELS].cpu().numpy()

def r40_reader_analysis(context,question,idx):
    messages=[dict(role='system',content=R40_READER_SYSTEM),dict(role='user',content=context+
        'Original user question: '+question+'\nAnalyze the relevant source evidence and actual time bounds. '
        'Be concise, preserve the original referents, and do not select a final answer yet.')]
    inp=r40_reader_inputs(messages,True);length=inp['input_ids'].shape[1]
    torch.manual_seed(1731+idx);torch.cuda.manual_seed_all(1731+idx)
    h=R40_READER_MODEL.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():seq=R40_READER_MODEL.generate(**inp,max_new_tokens=512,
            do_sample=True,temperature=.6,top_p=.95,top_k=20,use_cache=True)
    finally:h.remove()
    tokens=seq[0,length:].tolist()
    return dict(text=R40_READER_TOKENIZER.decode(tokens,skip_special_tokens=True),tokens=len(tokens),
                hit_budget=len(tokens)==512)

evidence_by_id={r['idx']:r for r in r40_json_rows(OUT/f'research40_verified_graph_v2_s{R40_SERVER}.jsonl')}
assert len(evidence_by_id)==len(R40_SAMPLES)
base_by_id={r['idx']:r for r in r40_json_rows(R40_BASE_FILE)}
for phase in ('plain','reason'):
    file=OUT/f'research40_reader4_{phase}_s{R40_SERVER}.jsonl'
    records=r40_json_rows(file);done={r['idx'] for r in records}
    for row in R40_SAMPLES:
        if row['idx'] in done:continue
        started=time.time();question,texts=parse_question(row['question']);k=len(texts)
        e=evidence_by_id[row['idx']];context=r40_compact_evidence(e['observations'],e['links'])
        analysis=r40_reader_analysis(context,question,row['idx']) if phase=='reason' else None
        totals=np.zeros(k);prior=np.zeros(k)
        for perm in rotations(k):
            prompt=r40_prompt(question,texts,perm)
            messages=[dict(role='system',content=R40_READER_SYSTEM),dict(role='user',content=context+prompt)]
            if analysis:
                messages=[dict(role='system',content=R40_READER_SYSTEM),
                    dict(role='user',content=context+'Original user question: '+question),
                    dict(role='assistant',content=analysis['text']),dict(role='user',content=prompt)]
            lg=r40_reader_scores(messages)[:k];lp=lg-np.logaddexp.reduce(lg)
            for slot,i in enumerate(perm):totals[i]+=lp[slot]
            if phase=='plain':
                lg=r40_reader_scores([dict(role='system',content='Answer the user question.'),
                                       dict(role='user',content=prompt)])[:k]
                pp=lg-np.logaddexp.reduce(lg)
                for slot,i in enumerate(perm):prior[i]+=pp[slot]
        scores={f'reader4_{phase}':totals.tolist()}
        if phase=='plain':scores['reader4_question_only']=prior.tolist()
        rec=dict(idx=row['idx'],task=row['task'],gold=row['gold'],status='ok',
            manifest_sha256=R40_MANIFEST['manifest_sha256'],model=R40_READER_ID,revision=R40_READER_REV,
            scores=scores,analysis=analysis,seconds=round(time.time()-started,2),
            raw_av_visible=False,same_original_evidence=True)
        with file.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(row['idx'])
        merged=[dict(r,scores=dict(base_by_id[r['idx']]['scores'],**r['scores'])) for r in records]
        r40_snapshot(merged,f'reader4_{phase}');print('READER4_SAMPLE',json.dumps(rec),flush=True)
        gc.collect();torch.cuda.empty_cache()
    RESULT=r40_snapshot(merged,f'reader4_{phase}_done')
    print('READER4_SUMMARY',phase,json.dumps(RESULT),flush=True)
# Release the additional reader; retain the original Omni model/runtime.
del R40_READER_MODEL;gc.collect();torch.cuda.empty_cache()

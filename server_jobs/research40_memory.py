"""Independent, time-indexed source observations plus the untouched original AV.

Extraction is question/option/label blind. The reader has no task router, speech
anchor, prescribed modality order, or relation-specific answer rule.
"""
import json, math, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import DynamicCache

R40_MEMORY_FILE=OUT/f'research40_memory_s{R40_SERVER}.jsonl'
R40_MEMORY_DIR=OUT/'research40_source_memory';R40_MEMORY_DIR.mkdir(exist_ok=True)
R40_CHUNK_SECONDS=4.0

def r40_observation_inputs(media,start,end,kind):
    if kind=='visual':
        prompt=(f'Visual recording interval {start:.2f}s to {end:.2f}s. '
                'Describe the directly visible actions and changes in at most12 words. '
                'Begin with the action. Do not infer speech, intent, or unseen events.')
        content=dict(type='video',video=media['path'],max_pixels=media['max_pixels_per_frame'])
        frames=media['videos'][0]
        lo=max(0,int(math.floor(start*media['fps'])));hi=min(len(frames),int(math.ceil(end*media['fps'])))
        clip=frames[lo:hi]
        if len(clip)==0:clip=frames[-1:]
        if len(clip)%2:clip=torch.cat((clip,clip[-1:]),dim=0)
        audio=None;videos=[clip]
    else:
        prompt=(f'Audio recording interval {start:.2f}s to {end:.2f}s. '
                'Transcribe only the audible spoken words, without explanation or added words. '
                'If there is no intelligible speech, output [no intelligible speech].')
        content=dict(type='audio',audio=media['path'])
        raw=media['audio'][0]
        audio=[raw[int(start*16000):min(len(raw),int(end*16000))]];videos=None
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[content,{'type':'text','text':prompt}]}]
    text=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inputs=PROC(text=text,audio=audio,videos=videos,images=None,return_tensors='pt',padding=True,
                videos_kwargs={'fps':media['fps'],'use_audio_in_video':False})
    return move_inputs(inputs,MODEL_)

def r40_observe(media,start,end,kind):
    inputs=r40_observation_inputs(media,start,end,kind);length=inputs['input_ids'].shape[1]
    # Avoid projecting every media position onto the full vocabulary during prefill.
    hook=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():
            sequence=MODEL_.generate(**inputs,use_audio_in_video=False,max_new_tokens=32,do_sample=False,use_cache=True,
                eos_token_id=[PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')])
    finally:hook.remove()
    text=PROC.tokenizer.decode(sequence[0,length:],skip_special_tokens=True).strip()
    del inputs,sequence;torch.cuda.empty_cache()
    return text

def r40_extract_memory(path,duration,pixels,cache_path):
    if cache_path.exists():nodes=json.loads(cache_path.read_text())
    else:nodes=[]
    done={(n['start'],n['kind']) for n in nodes}
    media=prepare_video(path,max_pixels=pixels)
    for start in np.arange(0,duration,R40_CHUNK_SECONDS):
        start=float(start);end=min(duration,start+R40_CHUNK_SECONDS)
        if end-start<.05:continue
        for kind in ('visual','audio'):
            if (start,kind) in done:continue
            text=r40_observe(media,start,end,kind)
            node=dict(start=start,end=end,kind=kind,text=text)
            nodes.append(node)
            cache_path.write_text(json.dumps(nodes,indent=2),encoding='utf-8')
        print('SOURCE_MEMORY_CHUNK',path.stem,start,end,flush=True)
    del media;gc.collect();torch.cuda.empty_cache()
    return sorted(nodes,key=lambda n:(n['start'],n['kind']))

def r40_memory_text(nodes):
    intro=('Independent observations of the same original recording follow. Each was made using only the '
           'named source in that interval, without the question or answer choices. These are partial '
           'observations; consult the supplied original recording for omitted details. '
           'Audio words are statements heard, and need not describe events visible at that time. '
           'An interval bounds the observation, not necessarily the full duration of an event.\n')
    return intro+'\n'.join(f"[{n['kind']} {n['start']:.2f}-{n['end']:.2f}s] {n['text']}" for n in nodes)+'\n\n'

def r40_memory_context(payload,question,supplement,choices=None):
    bank,prefix=r40_prefix(payload);bank=r40_gpu_bank(bank)
    cache=DynamicCache(ddp_cache_data=bank,config=MODEL_.model.config)
    tail=PROC.tokenizer.decode(payload['input_ids'][0,prefix:].tolist(),skip_special_tokens=False)
    beginning=tail[:tail.index(question)];ending=tail[tail.rindex('<|im_end|>'):]
    body=supplement+'USER QUESTION: '+question
    if choices is None:body+=R40_REASON_INSTRUCTION
    else:body+=' '+choices+'\nAnswer with only the letter of the best answer.'
    ids=PROC.tokenizer.encode(beginning+body+ending,add_special_tokens=False)
    next_pos=payload['decoder']['position_ids'][:,:,prefix:prefix+1]
    return r40_cached_text(ids,cache,next_pos,prefix)

def r40_memory_analysis(payload,question,supplement):
    logits,cache,next_pos,offset=r40_memory_context(payload,question,supplement)
    generated=[];eos={PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')}
    for _ in range(80):
        token=int(logits.argmax())
        if token in eos:break
        generated.append(token)
        logits,cache,next_pos,offset=r40_cached_text([token],cache,next_pos,offset)
    return dict(bank=cache.to_legacy_cache(),next_pos=next_pos,offset=offset,
                text=PROC.tokenizer.decode(generated,skip_special_tokens=True),tokens=len(generated))

baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
assert len(baseline)==len(R40_SAMPLES)
records=r40_json_rows(R40_MEMORY_FILE);done={r['idx'] for r in records}
print('MEMORY_START',R40_SERVER,len(done),'/',len(R40_SAMPLES),flush=True)
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time()
    try:
        stem,texts=parse_question(r['question']);k=len(texts)
        nodes=r40_extract_memory(fetch(r['video_path']),r['decoded_audio_duration'],base_by_id[r['idx']]['pixels'],
                                 R40_MEMORY_DIR/f"{r['idx']}_4s.json")
        supplement=r40_memory_text(nodes)
        payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False)
        payload['sample_idx']=r['idx'];plain=np.zeros(k);cot=np.zeros(k)
        state=r40_memory_analysis(payload,stem,supplement)
        for perm in rotations(k):
            choices=' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))
            lg,_,_,_=r40_memory_context(payload,stem,supplement,choices)
            arr=lg[R40_LABELS[:k]].cpu().numpy();lp=arr-np.logaddexp.reduce(arr)
            lg2=r40_reason_answer(state,texts,perm);lp2=lg2-np.logaddexp.reduce(lg2)
            for slot,i in enumerate(perm):plain[i]+=lp[slot];cot[i]+=lp2[slot]
        rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',
                 manifest_sha256=R40_MANIFEST['manifest_sha256'],
                 scores=dict(independent_source_memory=plain.tolist(),source_memory_reasoning=cot.tolist()),
                 observations=nodes,analysis=state['text'],seconds=round(time.time()-started,2))
        with R40_MEMORY_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(r['idx'])
        merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
        r40_snapshot(merged,'independent_source_memory')
        print('MEMORY_SAMPLE',json.dumps({k:rec[k] for k in ('idx','task','gold','scores','seconds','analysis')}),flush=True)
        del payload,state;gc.collect();torch.cuda.empty_cache()
    except Exception:
        import traceback
        print('MEMORY_ERROR',r['idx'],traceback.format_exc(),flush=True)
        torch.cuda.empty_cache();raise
RESULT=r40_snapshot(merged,'memory_done')
print('MEMORY_SUMMARY',json.dumps(RESULT),flush=True)

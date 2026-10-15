"""One native baseline per rotation, cached at the decoder input for all later rounds."""
import json, time, math, gc, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

R40_MANIFEST_FILE=OUT/'research40_manifest.json'
if not R40_MANIFEST_FILE.exists():
    R40_MANIFEST_FILE=OUT/'uploads/research40_manifest.json'
R40_MANIFEST=json.loads(R40_MANIFEST_FILE.read_text())
R40_SAMPLES=R40_MANIFEST[f'server{R40_SERVER}']
R40_CACHE=OUT/'research40_decoder_cache';R40_CACHE.mkdir(exist_ok=True)
R40_BASE_FILE=OUT/f'research40_baseline_s{R40_SERVER}.jsonl'
R40_LABELS=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD']
R40_FINGERPRINT=dict(manifest=R40_MANIFEST['manifest_sha256'],model=MODEL_.config._name_or_path,
                     prompt_version='letter_only_v1',pixels='160*28*28*min(1,32/duration)',
                     fps='native_default',rotations='all cyclic content rotations')

def r40_prompt(stem,texts,perm):
    return stem+' '+ ' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))+ '\nAnswer with only the letter of the best answer.'

def r40_json_rows(path):
    if not path.exists():return []
    result={}
    for line in path.read_text().splitlines():
        try:
            r=json.loads(line)
            if r.get('status')=='ok':result[r['idx']]=r
        except json.JSONDecodeError:pass
    return list(result.values())

def r40_snapshot(records,phase):
    counts=defaultdict(dict)
    for r in records:
        cell=r['task']+'/'+r['gold']
        for method,scores in r.get('scores',{}).items():
            z=counts[cell].setdefault(method,dict(correct=0,wrong=0,total=0))
            correct=int(np.argmax(scores))==ord(r['gold'])-65
            z['correct']+=int(correct);z['wrong']+=int(not correct);z['total']+=1
    payload=dict(server=R40_SERVER,phase=phase,manifest_sha256=R40_MANIFEST['manifest_sha256'],
                 completed=len(records),expected=len(R40_SAMPLES),counts=dict(counts),unix_time=time.time())
    path=OUT/f'research40_progress_s{R40_SERVER}.json'
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(path)
    return payload

def r40_text_logits(prompt):
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[{'type':'text','text':prompt}]}]
    text=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inp=PROC(text=text,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    handle=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():out=MODEL_(**inp,use_cache=False).logits[0,-1].float().cpu()
    finally:handle.remove()
    return out[R40_LABELS].numpy()

def r40_native_and_cache(inp,path,k):
    captured={}
    def capture(_m,args,kw):
        for key in ('inputs_embeds','position_ids','attention_mask','cache_position'):
            val=kw.get(key)
            if val is not None:captured[key]=val.detach().cpu()
    h1=MODEL_.model.register_forward_pre_hook(capture,with_kwargs=True)
    h2=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():lg=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
    finally:h1.remove();h2.remove()
    assert 'inputs_embeds' in captured and 'position_ids' in captured,captured.keys()
    pos,sec,is_v,is_a,last=media_layout(MODEL_,PROC,inp)
    payload=dict(decoder=captured,logits=lg[R40_LABELS[:k]],is_v=is_v.cpu(),is_a=is_a.cpu(),
                 sec=sec.cpu(),last_media=last,input_ids=inp['input_ids'].cpu(),fingerprint=R40_FINGERPRINT)
    torch.save(payload,path)
    return payload

records=r40_json_rows(R40_BASE_FILE);done={r['idx'] for r in records}
print('BASELINE_START',R40_SERVER,len(done),'/',len(R40_SAMPLES),flush=True)
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time()
    try:
        assert r['duration']<=45 and r['decoded_audio_duration']<=45
        stem,texts=parse_question(r['question']);k=len(texts)
        gc.collect();torch.cuda.empty_cache()
        px=int(160*min(1,32/r['duration']))*28*28
        media=None;tot=np.zeros(k);prior=np.zeros(k);token_lengths=[]
        for rotation,perm in enumerate(rotations(k)):
            cache=R40_CACHE/f"{r['idx']}_rot{rotation}.pt"
            prompt=r40_prompt(stem,texts,perm)
            if cache.exists():
                payload=torch.load(cache,map_location='cpu',weights_only=False)
                assert payload['fingerprint']==R40_FINGERPRINT
            else:
                if media is None:media=prepare_video(fetch(r['video_path']),max_pixels=px)
                inp,_=qwen_inputs(PROC,prompt,media,'av');inp=move_inputs(inp,MODEL_)
                payload=r40_native_and_cache(inp,cache,k);del inp
            if 'prior_logits' not in payload:
                payload['prior_logits']=torch.from_numpy(r40_text_logits(prompt)[:k])
                torch.save(payload,cache)
            lg=payload['logits'].float().numpy();pr=payload['prior_logits'].float().numpy()
            lp=lg-np.logaddexp.reduce(lg);pp=pr-np.logaddexp.reduce(pr)
            for slot,i in enumerate(perm):tot[i]+=lp[slot];prior[i]+=pp[slot]
            token_lengths.append(payload['input_ids'].shape[1]);del payload
            torch.cuda.empty_cache()
        rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],duration=r['duration'],status='ok',
                 manifest_sha256=R40_MANIFEST['manifest_sha256'],scores=dict(base=tot.tolist(),
                 argon_text_pmi=(tot-.75*prior).tolist()),prior_scores=prior.tolist(),
                 tokens=token_lengths,pixels=px,seconds=round(time.time()-started,2))
        with R40_BASE_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(r['idx']);snapshot=r40_snapshot(records,'baseline')
        print('BASELINE_SAMPLE',json.dumps(rec),flush=True)
        del media;torch.cuda.empty_cache()
    except Exception as exc:
        import traceback
        print('BASELINE_ERROR',r['idx'],traceback.format_exc(),flush=True)
        with (OUT/f'research40_errors_s{R40_SERVER}.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase='baseline',idx=r['idx'],error=repr(exc)))+'\n')
        torch.cuda.empty_cache()
RESULT=r40_snapshot(records,'baseline_done' if len(records)==len(R40_SAMPLES) else 'baseline_incomplete')
print('BASELINE_SUMMARY',json.dumps(RESULT),flush=True)

"""M2 mechanism audit and reversible evidence focus, frozen development40.

Diagnostic conditions preserve old components explicitly. General candidates
remove answer-word clock, speech-window selection, and fixed modality order.
No native causal edge is opened by any of the general candidates.
"""
import gc, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch

LR_SERVER = globals().get('LR_SERVER', 1)
LR_LIMIT = globals().get('LR_LIMIT')
LR_DIR = OUT / 'latent_reopen40'
LR_DIR.mkdir(exist_ok=True)
LR_MANIFEST = json.loads((OUT/'uploads/research40_manifest.json').read_text())
LR_SAMPLES = LR_MANIFEST[f'server{LR_SERVER}']
LR_SHA = LR_MANIFEST['manifest_sha256']
assert LR_MANIFEST['total'] == 40
assert all(x['duration'] <= 45 and x['decoded_audio_duration'] <= 45
           for x in LR_MANIFEST['selected'])

# Load definitions, never run an old completed condition again.
ARGON_SERVER = LR_SERVER
_source = (OUT/'uploads/argon_latent40.py').read_text()
exec(compile(_source.split('RESULT=argon_run(',1)[0], 'argon_definitions_only', 'exec'), globals())

LR_METHODS = [
    'D1_M2_no_clock',
    'D2_M2_reopen24',
    'D3_M2_global_query',
    'G1_local_encode_global_read',
    'G2_local_encode_reopen24',
    'G3_latent_source_reflection',
]

def lr_forward(payload, mask=None, mask_until=36, reflection=None, capture=False):
    """Native RoPE; query-only same-layer latent feedback; all hooks removed."""
    inp = payload['decoder']
    device = MODEL_.get_input_embeddings().weight.device
    kw = {k:v.to(device) for k,v in inp.items()}
    hooks=[]; masks={}; captured={}
    last=payload['last_media']
    def pre(li):
        def hook(module, args, kwargs):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            key=str(hidden.device)
            if key not in masks: masks[key]=mask.to(hidden.device)
            kwargs['attention_mask']=masks[key]
            return args,kwargs
        return hook
    def post(li):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            if reflection is not None and li in reflection:
                h=h.clone(); q=h[0,last+1:].float()
                # Intervene at the same layer where the contrasting state was measured.
                delta=reflection[li].to(h.device)
                scale=(.20*q.norm(dim=-1,keepdim=True)/delta.norm(dim=-1,keepdim=True).clamp_min(1e-5))
                h[0,last+1:]=(q+scale*delta).to(h.dtype)
            if capture and li in (18,22): captured[li]=h[0,last+1:].detach().float().cpu()
            return (h,)+out[1:] if isinstance(out,tuple) else h
        return hook
    try:
        for li,layer in enumerate(MODEL_.model.layers):
            if mask is not None and li < mask_until:
                hooks.append(layer.register_forward_pre_hook(pre(li),with_kwargs=True))
            if (capture and li in (18,22)) or (reflection is not None and li in reflection):
                hooks.append(layer.register_forward_hook(post(li)))
        with torch.inference_mode():
            out=MODEL_.model(**kw,use_cache=False,return_dict=True)
            logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    finally:
        for h in hooks: h.remove()
        masks.clear()
    assert torch.isfinite(logits).all()
    return logits,captured

def lr_local_mask(payload):
    """Symmetric 3s source-local encoding; question keeps every AV key."""
    v=payload['is_v']; a=payload['is_a']; sec=payload['sec'].float()
    L=payload['tokens']; dtype=MODEL_.get_input_embeddings().weight.dtype
    mask=torch.zeros((L,L),dtype=dtype)
    upper=torch.ones((L,L),dtype=torch.bool).triu_(1)
    mask.masked_fill_(upper,torch.finfo(dtype).min)
    del upper
    ix=torch.where(v|a)[0]; start=int(ix.min());end=int(ix.max())+1
    # Operate only on existing causal media edges. Native text edges are unchanged.
    vv=v[start:end]; aa=a[start:end]; ss=sec[start:end]
    cross=(vv[:,None]&aa[None,:]) | (aa[:,None]&vv[None,:])
    far=(ss[:,None]-ss[None,:]).abs()>3.
    sub=mask[start:end,start:end]
    eligible=cross & far & (sub==0)
    sub[eligible]-=3.5
    return mask[None,None]

def lr_rotation(payload, audio, duration, k, old):
    inp=payload['decoder'];pos=inp['position_ids'];v=payload['is_v'];a=payload['is_a']
    sec=payload['sec'];last=payload['last_media'];L=payload['tokens']
    device=MODEL_.get_input_embeddings().weight.device
    dtype=MODEL_.get_input_embeddings().weight.dtype
    native,cap=forward_latent_pass(inp,pos,last_media=last,record_layers=(16,22))
    error=float((native[LETTERS[:k]]-payload['native_logits']).abs().max())
    assert error<=.12, error
    win=detect_active_window(audio,duration)
    clock,thought,sal=build_latent_aligned_value_clock(cap[22],sec,v,a,last,win,duration)
    fg=compute_dynamic_visual_foreground(cap[16],v,sec,duration,chunk_salience=sal)
    del cap,thought
    prior=np.array(old['prior_logprobs'])
    scores={}
    def record(name,logits,alpha=0):
        scores[name]=(torch.log_softmax(logits[LETTERS[:k]],dim=-1).numpy()-alpha*prior).tolist()
    hard=make_additive_mask(L,v,a,sec,last,win,dtype,device,spg=payload['spg'],
        desmear_win_s=3.,desmear_pen=3.5,gate_mode='hard',out_v_pen=2.8,fg_bias=fg)
    if 'scores' not in old:
        m2,_=forward_latent_pass(inp,pos,mask=hard,clock_vec=clock,last_media=last)
        reference_scores=dict(R3_M0_base=torch.log_softmax(native[LETTERS[:k]],dim=-1).numpy().tolist(),
            R3_M2_true_gate_audio=(torch.log_softmax(m2[LETTERS[:k]],dim=-1).numpy()-.45*prior).tolist())
    else:
        reference_scores={m:old['scores'][m] for m in ('R3_M0_base','R3_M2_true_gate_audio')}
    lg,_=forward_latent_pass(inp,pos,mask=hard,last_media=last)
    record('D1_M2_no_clock',lg,.45)
    # Exact old clock hooks, but native full context is restored after block23.
    clock_hooks=[]
    def clock_hook(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;h=h.clone()
        cv=clock.to(h.device);active=cv.norm(dim=-1)>0;hf=h[0].float()
        scale=hf[active].norm(dim=-1,keepdim=True).mean().clamp_min(1e-5)
        h[0]=(hf+.16*scale*cv).to(h.dtype)
        return (h,)+out[1:] if isinstance(out,tuple) else h
    try:
        for li in (14,16,18,20,22):
            clock_hooks.append(MODEL_.model.layers[li].register_forward_hook(clock_hook))
        lg,_=lr_forward(payload,hard,mask_until=24)
        record('D2_M2_reopen24',lg,.45)
    finally:
        for h in clock_hooks:h.remove()
    # Restore query access globally; retain the M2 media representation otherwise.
    hard[0,0,last+1:,:]=0
    qn=L-last-1
    hard[0,0,last+1:,last+1:]=torch.triu(torch.full((qn,qn),torch.finfo(dtype).min,device=device,dtype=dtype),diagonal=1)
    try:
        for li in (14,16,18,20,22):
            clock_hooks.append(MODEL_.model.layers[li].register_forward_hook(clock_hook))
        lg,_=lr_forward(payload,hard)
        record('D3_M2_global_query',lg,.45)
    finally:
        for h in clock_hooks:h.remove()
    del hard,clock,fg
    local=lr_local_mask(payload)
    lg,_=lr_forward(payload,local)
    record('G1_local_encode_global_read',lg)
    lg,local_states=lr_forward(payload,local,mask_until=24,capture=True)
    record('G2_local_encode_reopen24',lg)
    # Contrastive latent reflection between source-local and native representations.
    # No speech selector, vocabulary projection, answer-word vectors, or PMI.
    _,native_states=lr_forward(payload,capture=True)
    delta={li:local_states[li]-native_states[li] for li in local_states}
    lg,_=lr_forward(payload,reflection=delta)
    record('G3_latent_source_reflection',lg)
    del local,local_states,native_states,delta
    return dict(scores=scores,reference_scores=reference_scores,replay_error=error,tokens=L,win=list(win),
        native_slot_logits=native[LETTERS[:k]].tolist())

def lr_run():
    output=LR_DIR/f'results_s{LR_SERVER}.jsonl'
    done=argon_load_rows(output)
    samples=LR_SAMPLES[:LR_LIMIT] if LR_LIMIT else LR_SAMPLES
    for sample in samples:
        idx=sample['idx']
        if idx in done:continue
        started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
        audio=decode_audio(fetch(sample['video_path']))
        assert abs(len(audio)/16000-sample['decoded_audio_duration'])<.2
        scores={m:np.zeros(k) for m in LR_METHODS};audits=[]
        references={m:np.zeros(k) for m in ('R3_M0_base','R3_M2_true_gate_audio')}
        media=None
        for rotation,perm in enumerate(rotations(k)):
            cache=ARGON_CACHE/f'{idx}_rot{rotation}.pt'
            result_file=LR_DIR/f'{idx}_rot{rotation}.json'
            if result_file.exists():result=json.loads(result_file.read_text())
            else:
                if not cache.exists():
                    if media is None:
                        src=fetch(sample['video_path'])
                        media=prepare_video(src,max_pixels=int(160*min(1.,32/sample['duration']))*28*28)
                    prompt=argon_prompt(stem,texts,perm)
                    inputs,_=qwen_inputs(PROC,prompt,media,'av')
                    inputs=move_inputs(inputs,MODEL_)
                    payload=argon_encode(inputs,k)
                    torch.save(payload,cache)
                    del inputs
                else:payload=torch.load(cache,map_location='cpu',weights_only=False)
                assert payload['manifest_sha256']==LR_SHA
                old_path=ARGON_DIR/f'{idx}_rot{rotation}.json'
                old=json.loads(old_path.read_text()) if old_path.exists() else dict(
                    prior_logprobs=argon_text_prior(argon_prompt(stem,texts,perm),k).tolist())
                result=lr_rotation(payload,audio,sample['duration'],k,old)
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=LR_SHA)
                tmp=result_file.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(result_file)
                del payload;gc.collect();torch.cuda.empty_cache()
            for method,s in result['scores'].items():
                for slot,opt in enumerate(perm):scores[method][opt]+=s[slot]
            ref=result.get('reference_scores')
            if ref is None:
                old=json.loads((ARGON_DIR/f'{idx}_rot{rotation}.json').read_text())
                ref={m:old['scores'][m] for m in references}
            for method,s in ref.items():
                for slot,opt in enumerate(perm):references[method][opt]+=s[slot]
            audits.append({k:v for k,v in result.items() if k!='scores'})
        # Only evaluation reads task and gold.
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',
            manifest_sha256=LR_SHA,scores={m:s.tolist() for m,s in scores.items()},
            predictions={m:chr(65+int(s.argmax())) for m,s in scores.items()},
            reference_scores={m:s.tolist() for m,s in references.items()},
            audit=audits,seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done[idx]=rec
        del media
        print('REOPEN_SAMPLE',len(done),'/',len(samples),json.dumps(rec),flush=True)
    return dict(completed=len(done),expected=len(samples),server=LR_SERVER,manifest_sha256=LR_SHA)

RESULT=lr_run()
print('REOPEN_COMPLETE',json.dumps(RESULT),flush=True)

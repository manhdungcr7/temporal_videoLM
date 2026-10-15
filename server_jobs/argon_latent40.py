"""Argon LatentChrono-CoT on the frozen development40, resumable per rotation.

Load proposed math verbatim, fix transport/indexing/memory issues explicitly.
Native AV encoding, all cyclic rotations, no replacement or label-dependent routing.
"""
import gc, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ARGON_DIR=OUT/'argon_latent40';ARGON_DIR.mkdir(exist_ok=True)
ARGON_CACHE=ARGON_DIR/'decoder_cache';ARGON_CACHE.mkdir(exist_ok=True)
ARGON_RESULT=ARGON_DIR/f'results_s{ARGON_SERVER}.jsonl'
ARGON_MANIFEST=json.loads((OUT/'uploads/research40_manifest.json').read_text())
ARGON_SAMPLES=ARGON_MANIFEST[f'server{ARGON_SERVER}']
ARGON_SHA=ARGON_MANIFEST['manifest_sha256']
ARGON_ORIGINAL=(OUT/'uploads/proposal_original.py').read_text()
ARGON_SOURCE_SHA=hashlib.sha256(ARGON_ORIGINAL.encode()).hexdigest()
exec(compile(ARGON_ORIGINAL.split('def eval_sample(r):',1)[0],'argon_proposed_definitions','exec'),globals())
METHODS += ['R3_M7_clock_mask_pmi_control','R3_M8_recurrent_only','R3_M9_thought_only']
LETTERS=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC']

# Explicit repeated KV selects T4 efficient SDPA without changing attention math.
exec(compile((OUT/'uploads/avbind_sdpa_compat.py').read_text(),'sdpa_compat','exec'),globals())

def argon_prompt(stem,texts,perm):
    return render(stem,texts,perm)+'\nAnswer with only the letter of the best answer.'

def argon_text_prior(prompt,k):
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[{'type':'text','text':prompt}]}]
    text=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inputs=PROC(text=text,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    hook=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():logits=MODEL_(**inputs,use_cache=False).logits[0,-1].float().cpu()
    finally:hook.remove()
    return torch.log_softmax(logits[LETTERS[:k]],dim=-1).numpy()

def argon_encode(inp,k):
    captured={}
    def capture(_module,args,kwargs):
        for key in ('inputs_embeds','position_ids','attention_mask','cache_position'):
            if kwargs.get(key) is not None:captured[key]=kwargs[key].detach().cpu()
    hooks=[MODEL_.model.register_forward_pre_hook(capture,with_kwargs=True),
           MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))]
    try:
        with torch.inference_mode():lg=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
    finally:
        for h in hooks:h.remove()
    pos,sec,is_v,is_a,last=media_layout(MODEL_,PROC,inp)
    spg=inp.get('video_second_per_grid')
    return dict(decoder=captured,native_logits=lg[LETTERS[:k]],sec=sec.cpu(),is_v=is_v.cpu(),is_a=is_a.cpu(),
        last_media=int(last),spg=float(spg.reshape(-1)[0]) if spg is not None else None,
        tokens=int(inp['input_ids'].shape[1]),manifest_sha256=ARGON_SHA,prompt='letter_only_v1')

# Preserve the proposed masks, including its explicit synchronous V->A causal
# relaxation. Build triangle with a bool matrix, not O(L^2) int64 index pairs.
_argon_mask_source=ARGON_ORIGINAL.split('def make_additive_mask(',1)[1].split('\ndef forward_latent_pass(',1)[0]
_argon_mask_source='def make_additive_mask('+_argon_mask_source
_argon_mask_source=_argon_mask_source.replace(
    'triu_idx = torch.triu_indices(L, L, offset=1, device=device)\n    m2d[triu_idx[0], triu_idx[1]] = neg\n    del triu_idx',
    'upper = torch.ones((L, L), dtype=torch.bool, device=device).triu_(1)\n    m2d.masked_fill_(upper, neg)\n    del upper')
exec(compile(_argon_mask_source,'argon_mask_bounded_indices','exec'),globals())

def forward_latent_pass(inp,pos,mask=None,handoff_layer=None,is_a=None,last_media=0,
    clock_vec=None,clock_layers=(14,16,18,20,22),clock_scale=.16,
    cont_thought_vec=None,cont_thought_layers=(0,1,2),cont_thought_scale=.22,
    recurrent_delta=None,recurrent_layers=None,record_layers=()):
    """Equivalent decoder-only execution; install masks per device without mutation."""
    if recurrent_layers is None:recurrent_layers={10:.28,12:.35,14:.35,16:.28}
    device=MODEL_.get_input_embeddings().weight.device
    kw={key:val.to(device) for key,val in inp.items()}
    kw['position_ids']=pos.to(device)
    captured={};hooks=[];device_masks={};mask_late=None
    if handoff_layer is not None:
        assert mask is not None and is_a is not None
        mask_late=mask.clone()
        mask_late[0,0,last_media+1:,is_a.to(mask.device)]=torch.finfo(mask.dtype).min
    def mask_hook(li):
        def pre(_module,args,kwargs):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            late=mask_late is not None and li>=handoff_layer
            key=(str(hidden.device),late)
            if key not in device_masks:device_masks[key]=(mask_late if late else mask).to(hidden.device)
            kwargs['attention_mask']=device_masks[key]
            return args,kwargs
        return pre
    def output_hook(li):
        def hook(_module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            changed=False
            if cont_thought_vec is not None and li in cont_thought_layers:
                h=h.clone();changed=True;q=h[0,last_media+1:].float()
                assert q.shape[0]>0
                ct=cont_thought_vec.to(h.device,torch.float32).unsqueeze(0)
                h[0,last_media+1:]=(q+cont_thought_scale*q.norm(dim=-1,keepdim=True).mean().clamp(min=1e-5)*ct).to(h.dtype)
            if clock_vec is not None and li in clock_layers:
                if not changed:h=h.clone();changed=True
                cv=clock_vec.to(h.device,torch.float32);hf=h[0].float();active=cv.norm(dim=-1)>0
                if active.any():
                    scale=hf[active].norm(dim=-1,keepdim=True).mean().clamp(min=1e-5)
                    h[0]=(hf+clock_scale*scale*cv).to(h.dtype)
            if recurrent_delta is not None and li in recurrent_layers:
                if not changed:h=h.clone();changed=True
                q=h[0,last_media+1:].float();dq=recurrent_delta.to(h.device,torch.float32)
                assert q.shape==dq.shape,(q.shape,dq.shape,last_media)
                h[0,last_media+1:]=(q+float(recurrent_layers[li])*q.norm(dim=-1,keepdim=True).clamp(min=1e-5)
                    /dq.norm(dim=-1,keepdim=True).clamp(min=1e-5)*dq).to(h.dtype)
            if li in record_layers:captured[li]=h[0].detach().float().cpu()
            return (h,)+out[1:] if isinstance(out,tuple) else h
        return hook
    try:
        for li,layer in enumerate(MODEL_.model.layers):
            if mask is not None:hooks.append(layer.register_forward_pre_hook(mask_hook(li),with_kwargs=True))
            if li in record_layers or (clock_vec is not None and li in clock_layers) or (
                cont_thought_vec is not None and li in cont_thought_layers) or (
                recurrent_delta is not None and li in recurrent_layers):
                hooks.append(layer.register_forward_hook(output_hook(li)))
        with torch.inference_mode():
            out=MODEL_.model(**kw,use_cache=False,return_dict=True)
            logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    finally:
        for hook in hooks:hook.remove()
        device_masks.clear()
    assert torch.isfinite(logits).all(),'nonfinite final logits'
    return logits,captured

def argon_rotation(payload,raw_audio,dur,k,lp_prior):
    device=MODEL_.get_input_embeddings().weight.device
    dtype=MODEL_.get_input_embeddings().weight.dtype
    inp=payload['decoder'];pos=inp['position_ids'];sec=payload['sec'];v=payload['is_v'];a=payload['is_a']
    last=payload['last_media'];L=payload['tokens'];win=detect_active_window(raw_audio,dur)
    common=dict(last_media=last)
    lg_base,cap=forward_latent_pass(inp,pos,record_layers=(16,22),**common)
    replay=float((lg_base[LETTERS[:k]]-payload['native_logits']).abs().max())
    assert replay<=.12,f'decoder/native replay drift {replay}'
    clock,thought,sal=build_latent_aligned_value_clock(cap[22],sec,v,a,last,win,dur)
    fg=compute_dynamic_visual_foreground(cap[16],v,sec,dur,chunk_salience=sal)
    base_q=cap[22][last+1:];del cap
    def mask(mode,foreground=True,desmear=True):
        return make_additive_mask(L,v,a,sec,last,win,dtype,device,spg=payload['spg'],
            desmear_win_s=3. if desmear else 0.,desmear_pen=3.5 if desmear else 0.,
            gate_mode=mode,out_v_pen=2.8,fg_bias=fg if foreground else None)
    m=mask('hard');hard,cap=forward_latent_pass(inp,pos,mask=m,clock_vec=clock,record_layers=(22,),**common)
    sync_q=cap[22][last+1:];del cap,m
    m=mask('anti',False,False);_,cap=forward_latent_pass(inp,pos,mask=m,record_layers=(22,),**common)
    delta=.5*(base_q+sync_q)-cap[22][last+1:];del cap,m,sync_q,base_q
    m=mask('none')
    rec,_=forward_latent_pass(inp,pos,mask=m,clock_vec=clock,recurrent_delta=delta,**common)
    ct,_=forward_latent_pass(inp,pos,mask=m,clock_vec=clock,cont_thought_vec=thought,**common)
    control,_=forward_latent_pass(inp,pos,mask=m,clock_vec=clock,**common);del m
    m=mask('soft')
    handoff,_=forward_latent_pass(inp,pos,mask=m,handoff_layer=14,is_a=a,clock_vec=clock,
        cont_thought_vec=thought,cont_thought_scale=.20,recurrent_delta=delta,**common);del m
    rec_only,_=forward_latent_pass(inp,pos,recurrent_delta=delta,**common)
    ct_only,_=forward_latent_pass(inp,pos,cont_thought_vec=thought,**common)
    def lp(logits):return torch.log_softmax(logits[LETTERS[:k]],dim=-1).numpy()
    scores=dict(zip(METHODS,[lp(lg_base),lp(lg_base)-.75*lp_prior,lp(hard)-.45*lp_prior,
        lp(rec)-.75*lp_prior,lp(ct)-.75*lp_prior,lp(handoff)-.70*lp_prior,None,
        lp(control)-.75*lp_prior,lp(rec_only),lp(ct_only)]))
    w=float(np.clip((float(np.ptp(lp_prior))-.30)/.55,.15,.85))
    scores['R3_M6_latent_chrono_unified']=w*(.45*scores[METHODS[5]]+.30*scores[METHODS[3]]+.25*scores[METHODS[4]])+(
        1-w)*(.60*scores[METHODS[2]]+.40*scores[METHODS[5]])
    assert all(np.isfinite(s).all() for s in scores.values())
    return dict(scores={m:s.tolist() for m,s in scores.items()},win=list(win),replay_error=replay,
        tokens=L,last_media=last,question_rows=L-last-1,n_visual=int(v.sum()),n_audio=int(a.sum()),
        mask_MB=round(L*L*2/2**20,2),salience=sal.tolist(),thought_norm=float(thought.norm()),
        delta_norm=float(delta.norm()),prior_logprobs=lp_prior.tolist(),blend_weight=w)

def argon_load_rows(path):
    rows={}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row=json.loads(line);rows[row['idx']]=row
    return rows

def argon_run(limit=None):
    samples=ARGON_SAMPLES[:limit] if limit else ARGON_SAMPLES
    done=argon_load_rows(ARGON_RESULT)
    for r in samples:
        if r['idx'] in done:continue
        started=time.time();idx=r['idx'];stem,texts=parse_question(r['question']);k=len(texts)
        src=fetch(r['video_path']);audio=decode_audio(src);dur=r['duration']
        assert abs(len(audio)/16000-r['decoded_audio_duration'])<.2
        totals={m:np.zeros(k) for m in METHODS};audits=[];media=None
        try:
            for rotation,perm in enumerate(rotations(k)):
                result_file=ARGON_DIR/f'{idx}_rot{rotation}.json'
                if result_file.exists():result=json.loads(result_file.read_text())
                else:
                    print('ARGON_ROT_START',ARGON_SERVER,idx,rotation,flush=True)
                    cache=ARGON_CACHE/f'{idx}_rot{rotation}.pt'
                    prompt=argon_prompt(stem,texts,perm)
                    if cache.exists():payload=torch.load(cache,map_location='cpu',weights_only=False)
                    else:
                        if media is None:media=prepare_video(src,max_pixels=int(160*min(1.,32/dur))*28*28)
                        inputs,_=qwen_inputs(PROC,prompt,media,'av');inputs=move_inputs(inputs,MODEL_)
                        payload=argon_encode(inputs,k);del inputs
                        torch.save(payload,cache)
                    assert payload['manifest_sha256']==ARGON_SHA
                    prior=argon_text_prior(prompt,k)
                    result=argon_rotation(payload,audio,dur,k,prior)
                    result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=ARGON_SHA)
                    tmp=result_file.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(result_file)
                    del payload;gc.collect();torch.cuda.empty_cache()
                    print('ARGON_ROT_DONE',ARGON_SERVER,idx,rotation,'tokens',result['tokens'],'replay',result['replay_error'],flush=True)
                assert result['manifest_sha256']==ARGON_SHA
                for m,score in result['scores'].items():
                    for slot,opt in enumerate(perm):totals[m][opt]+=score[slot]
                audits.append({a:b for a,b in result.items() if a!='scores'})
            # Gold is read only after inference, for reporting.
            rec=dict(idx=idx,task=r['task'],gold=r['gold'],status='ok',duration=dur,
                manifest_sha256=ARGON_SHA,proposal_sha256=ARGON_SOURCE_SHA,
                scores={m:s.tolist() for m,s in totals.items()},audit=audits,
                predictions={m:chr(65+int(np.argmax(s))) for m,s in totals.items()},seconds=round(time.time()-started,2))
            with ARGON_RESULT.open('a') as f:f.write(json.dumps(rec)+'\n')
            done[idx]=rec
            print('ARGON_SAMPLE',ARGON_SERVER,len(done),'/',len(ARGON_SAMPLES),idx,r['task'],r['gold'],rec['predictions'],rec['seconds'],flush=True)
        except Exception:
            with (ARGON_DIR/f'errors_s{ARGON_SERVER}.jsonl').open('a') as f:
                f.write(json.dumps(dict(idx=idx,error=traceback.format_exc()))+'\n')
            raise
        finally:
            del media;gc.collect();torch.cuda.empty_cache()
    return dict(server=ARGON_SERVER,completed=len(done),expected=len(ARGON_SAMPLES),
        missing=[r['idx'] for r in ARGON_SAMPLES if r['idx'] not in done],methods=METHODS,manifest_sha256=ARGON_SHA)

RESULT=argon_run(globals().get('ARGON_LIMIT'))
print('ARGON_COMPLETE',json.dumps(RESULT),flush=True)

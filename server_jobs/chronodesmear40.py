"""Round5 proposal mathematics, frozen40 harness and explicitly corrected boundaries."""
import ast,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CD_SERVER=globals().get('CD_SERVER',1)
CD_LIMIT=globals().get('CD_LIMIT')
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(),'cache_definitions','exec'),globals())
exec(compile((OUT/'uploads/chronodesmear_math.py').read_text(),'chrono_math','exec'),globals())
CD_DIR=OUT/'chronodesmear40';CD_DIR.mkdir(exist_ok=True)
CD_ORIGINAL=(OUT/'uploads/chronodesmear_proposal_original.py').read_text()
CD_PROPOSAL_SHA=hashlib.sha256(CD_ORIGINAL.encode()).hexdigest()
CD_RUNTIME_SHA=hashlib.sha256((OUT/'uploads/chronodesmear40.py').read_bytes()
    +(OUT/'uploads/chronodesmear_math.py').read_bytes()).hexdigest()
_wanted={'hidden_to_manifold_thought','detect_active_window','center_option_slice',
         'compute_av_resonance_and_shift_diagnostics','make_shuffled_delta'}
_parts=[n for n in ast.parse(CD_ORIGINAL).body if isinstance(n,ast.FunctionDef) and n.name in _wanted]
assert len(_parts)==len(_wanted)
exec(compile(ast.Module(body=_parts,type_ignores=[]),'original_round5_helpers','exec'),globals())
CD_METHODS=['R5_C0_baseline','R5_C1_causal_desmear_1p','R5_C2_stratified_dyn_cd',
    'R5_C3_balanced_modal_cd','R5_C4_stem_plus_media','R5_C5_chronodesmear_full',
    'R5_C6_multiseed_shuffled']
CD_REC={10:.24,12:.30,14:.30,16:.24}
CD_DS_LAYERS=(8,10,12,14,16)

def cd_find_span(payload,stem):
    """Exact lexical embedding matches, replacing the proposal's token-count guess."""
    start=payload['last_media']+1
    tail=payload['decoder']['inputs_embeds'][0,start:].cpu()
    device=MODEL_.get_input_embeddings().weight.device
    def find(texts,minimum=0):
        matches=[]
        for text in texts:
            ids=PROC.tokenizer.encode(text,add_special_tokens=False)
            with torch.inference_mode():
                embeds=MODEL_.get_input_embeddings()(torch.tensor(ids,device=device)).cpu()
            for offset in range(minimum,len(tail)-len(ids)+1):
                if torch.equal(tail[offset:offset+len(ids)],embeds): matches.append((offset,offset+len(ids)))
        if not matches: raise ValueError(('Could not exactly locate text boundary',texts))
        return min(matches)
    stem_start,stem_end=find((stem,' '+stem,'\n'+stem))
    option_start,_=find(('(A)',' (A)','\n(A)'),minimum=stem_end)
    assert 0<=stem_start<stem_end<=option_start<len(tail)
    return dict(span_start=start,option_start=start+option_start,
        stem_len=option_start,lexical_stem_start=start+stem_start,lexical_stem_end=start+stem_end)

def cd_forward(payload,mask=None,thought=None,delta=None,span_end=None,
               recurrent_layers=None,desmear=False,capture=False,plans=None,
               strengths=(.26,.18),record_layers=(22,)):
    device=MODEL_.get_input_embeddings().weight.device
    kwargs={k:v.to(device) for k,v in payload['decoder'].items()}
    start=payload['last_media']+1;end=payload['tokens'] if span_end is None else span_end
    rec=CD_REC if recurrent_layers is None else recurrent_layers
    handles=[];masks={};captured={}
    def pre(module,args,kw):
        h=kw.get('hidden_states',args[0] if args else None);key=str(h.device)
        if key not in masks: masks[key]=mask.to(h.device)
        kw['attention_mask']=masks[key]
        return args,kw
    def post(layer):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            changed=False
            if thought is not None and layer in (0,1,2):
                h=h.clone();changed=True;q=h[0,start:end].float()
                h[0,start:end]=(q+.18*q.norm(dim=-1,keepdim=True).mean().clamp_min(1e-5)
                    *thought.to(h.device)[None]).to(h.dtype)
            if desmear and layer in CD_DS_LAYERS:
                if not changed: h=h.clone();changed=True
                for plan,strength in zip(plans,strengths):
                    if not len(plan['indices']): continue
                    indices=plan['indices'].to(h.device)
                    h[0,indices]=desmear_values(h[0,indices].float(),plan,strength).to(h.dtype)
            if delta is not None and layer in rec:
                if not changed: h=h.clone();changed=True
                q=h[0,start:end].float();d=delta.to(h.device)
                assert q.shape==d.shape,(q.shape,d.shape)
                h[0,start:end]=(q+rec[layer]*q.norm(dim=-1,keepdim=True).clamp_min(1e-5)
                    /d.norm(dim=-1,keepdim=True).clamp_min(1e-5)*d).to(h.dtype)
            if capture and layer in record_layers: captured[layer]=h[0].detach().float().cpu()
            return (h,)+out[1:] if isinstance(out,tuple) else h
        return hook
    try:
        for layer,block in enumerate(MODEL_.model.layers):
            if mask is not None: handles.append(block.register_forward_pre_hook(pre,with_kwargs=True))
            if (thought is not None and layer in (0,1,2)) or (desmear and layer in CD_DS_LAYERS) or (
                delta is not None and layer in rec) or (capture and layer in record_layers):
                handles.append(block.register_forward_hook(post(layer)))
        with torch.inference_mode():
            output=MODEL_.model(**kwargs,use_cache=False,return_dict=True)
            logits=MODEL_.lm_head(output.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    finally:
        for handle in handles: handle.remove()
        masks.clear()
    assert torch.isfinite(logits).all()
    return logits,captured

def cd_rotation(payload,audio,duration,k,stem):
    started=time.time();last=payload['last_media'];v=payload['is_v'];a=payload['is_a'];sec=payload['sec']
    dtype=MODEL_.get_input_embeddings().weight.dtype
    plans=[interval_plan(torch.where(source)[0],sec) for source in (v,a)]
    span=cd_find_span(payload,stem)
    native,cap=cd_forward(payload,capture=True)
    replay=float((native[CD_LABELS[:k]]-payload['native_logits']).abs().max())
    assert replay<=.12,('Native replay',replay)
    base=cap[22];base_q=base[last+1:]
    static,innovation,v_indices,bin_counts=static_visual_mask(base,v,sec,duration)
    thought,resonance=compute_av_resonance_and_shift_diagnostics(base,v,a,static,sec,duration)
    del cap
    ds,cap=cd_forward(payload,desmear=True,plans=plans,capture=True)
    ds_q=cap[22][last+1:];del cap
    window=detect_active_window(audio,duration)
    branch={}
    for mode in ('static_plus_aud','v_all_only','hard_sync','anti_sync'):
        mask=counterfactual_mask(payload,mode,dtype,static,window)
        _,cap=cd_forward(payload,mask=mask,capture=True)
        branch[mode]=cap[22][last+1:].clone()
        del mask,cap;torch.cuda.empty_cache()
    # Exact supplied code, including ds and V-only terms absent from the prose formula.
    delta_raw=(.5*(ds_q+base_q)-branch['static_plus_aud'])+.35*(
        branch['v_all_only']-branch['static_plus_aud'])
    centered=center_option_slice(delta_raw,span['stem_len'])
    balanced=.5*centered+.5*center_option_slice(.5*(branch['v_all_only']+
        branch['hard_sync'])-branch['anti_sync'],span['stem_len'])
    logits={'R5_C0_baseline':native,'R5_C1_causal_desmear_1p':ds}
    logits['R5_C2_stratified_dyn_cd'],_=cd_forward(payload,delta=centered)
    logits['R5_C3_balanced_modal_cd'],_=cd_forward(payload,delta=balanced,desmear=True,plans=plans)
    logits['R5_C4_stem_plus_media'],_=cd_forward(payload,delta=delta_raw[:span['stem_len']],
        span_end=span['option_start'],recurrent_layers={10:.28,12:.35,14:.35,16:.28},
        desmear=True,plans=plans)
    logits['R5_C5_chronodesmear_full'],_=cd_forward(payload,thought=thought,delta=centered,
        desmear=True,plans=plans)
    scores={m:torch.log_softmax(lg[CD_LABELS[:k]],dim=-1).numpy().tolist() for m,lg in logits.items()}
    seed_scores={};norm_errors={}
    for seed in (42,1337,2027):
        shuffled=make_shuffled_delta(centered,seed)
        norm_errors[str(seed)]=float((shuffled.norm(dim=-1)-centered.norm(dim=-1)).abs().max())
        assert torch.allclose(shuffled.norm(dim=-1),centered.norm(dim=-1),rtol=1e-5,atol=1e-5)
        lg,_=cd_forward(payload,delta=shuffled)
        seed_scores[str(seed)]=torch.log_softmax(lg[CD_LABELS[:k]],dim=-1).numpy().tolist()
    scores['R5_C6_multiseed_shuffled']=np.mean(list(seed_scores.values()),axis=0).tolist()
    del base,ds_q,base_q,branch,centered,balanced,delta_raw,logits
    return dict(scores=scores,shuffled_seed_scores=seed_scores,replay_error=replay,tokens=payload['tokens'],
        span=span,static_bins=bin_counts,static_count=int(static.sum()),visual_count=int(v.sum()),
        av_resonance_EML=resonance,resonance_is_event_grounding=False,
        audio_window_for_C3_only=list(window),shuffle_norm_errors=norm_errors,
        echo_valid_counts=[int(p['has_previous'].sum()) for p in plans],seconds=time.time()-started)

def cd_run():
    samples=CD_MANIFEST[f'server{CD_SERVER}']
    if CD_LIMIT: samples=samples[:CD_LIMIT]
    output=CD_DIR/f'results_s{CD_SERVER}.jsonl'
    done={json.loads(line)['idx'] for line in output.read_text().splitlines() if line.strip()} if output.exists() else set()
    for sample in samples:
        idx=sample['idx']
        if idx in done: continue
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
        audio=decode_audio(fetch(sample['video_path']))
        assert abs(len(audio)/16000-sample['decoded_audio_duration'])<.2
        totals={m:np.zeros(k) for m in CD_METHODS};seeds={str(s):np.zeros(k) for s in (42,1337,2027)};audits=[]
        for rotation,perm in enumerate(rotations(k)):
            path=CD_DIR/f'{idx}_rot{rotation}.json'
            if path.exists(): result=json.loads(path.read_text())
            else:
                print('CHRONO_ROT_START',CD_SERVER,idx,rotation,flush=True)
                payload=torch.load(CD_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
                assert payload['manifest_sha256']==CD_SHA
                result=cd_rotation(payload,audio,sample['duration'],k,stem)
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=CD_SHA,
                    proposal_sha256=CD_PROPOSAL_SHA,runtime_sha256=CD_RUNTIME_SHA)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                del payload;gc.collect();torch.cuda.empty_cache()
            assert result['manifest_sha256']==CD_SHA and result['runtime_sha256']==CD_RUNTIME_SHA
            for method,values in result['scores'].items():
                for slot,option in enumerate(perm): totals[method][option]+=values[slot]
            for seed,values in result['shuffled_seed_scores'].items():
                for slot,option in enumerate(perm): seeds[seed][option]+=values[slot]
            audits.append({key:value for key,value in result.items() if key not in ('scores','shuffled_seed_scores')})
        record=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=CD_SHA,
            runtime_sha256=CD_RUNTIME_SHA,proposal_sha256=CD_PROPOSAL_SHA,
            scores={m:values.tolist() for m,values in totals.items()},
            predictions={m:chr(65+int(values.argmax())) for m,values in totals.items()},
            shuffled_seed_scores={s:values.tolist() for s,values in seeds.items()},audit=audits,
            seconds=round(time.time()-started,2))
        with output.open('a') as handle: handle.write(json.dumps(record)+'\n')
        done.add(idx)
        print('CHRONO_SAMPLE',len(done),'/',len(samples),json.dumps({k:v for k,v in record.items() if k!='audit'}),flush=True)
    return dict(server=CD_SERVER,completed=len(done),expected=len(samples),manifest_sha256=CD_SHA)

try: RESULT=cd_run()
except Exception:
    with (CD_DIR/f'errors_s{CD_SERVER}.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(error=traceback.format_exc()))+'\n')
    raise
print('CHRONO_COMPLETE',json.dumps(RESULT),flush=True)

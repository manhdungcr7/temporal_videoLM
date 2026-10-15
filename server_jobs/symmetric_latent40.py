"""Same-layer, norm-bounded AV interaction repair; no clock or modality order.

This is an independent hypothesis inspired by the CCLD audit, not an established
causal decomposition. The old research40 interaction-logit controls are distinct:
this experiment acts on same-layer query representations with bounded residuals.
"""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import DynamicCache

SL_SERVER=globals().get('SL_SERVER',1)
SL_DIR=OUT/'symmetric_latent40_v2';SL_DIR.mkdir(exist_ok=True)
SL_MANIFEST=json.loads((OUT/'uploads/research40_manifest.json').read_text())
SL_SAMPLES=SL_MANIFEST[f'server{SL_SERVER}'];SL_SHA=SL_MANIFEST['manifest_sha256']
SL_RUNTIME_SHA=hashlib.sha256((OUT/'uploads/symmetric_latent40.py').read_bytes()).hexdigest()
SL_CACHE=OUT/'argon_latent40/decoder_cache'
SL_METHODS=['S0_native','S1_visual_same_layer','S2_joint_same_layer',
            'S3_orthogonal_joint','S4_two_layer_orthogonal_joint',
            'S5_question_stem_joint','S6_question_stem_argon_delta']
SL_LABELS=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC']
SL_CACHE_VALIDATED=False

def sl_project(interaction,visual,audio):
    """Remove span of two unimodal differences without inflating small residuals."""
    basis=torch.stack((visual,audio),dim=-1).double()
    u,s,_=torch.linalg.svd(basis,full_matrices=False)
    keep=s>torch.maximum(s[:,:1]*1e-6,torch.full_like(s,1e-8))
    u=u*keep[:,None,:]
    result=interaction.double()-torch.einsum('qdr,qr->qd',u,torch.einsum('qdr,qd->qr',u,interaction.double()))
    return result.float()

def sl_mask(payload,absent):
    """Block all outbound paths from the omitted source, preserving causal edges."""
    L=payload['tokens'];dtype=MODEL_.get_input_embeddings().weight.dtype
    mask=torch.zeros((L,L),dtype=dtype)
    upper=torch.ones((L,L),dtype=torch.bool).triu_(1)
    mask.masked_fill_(upper,torch.finfo(dtype).min)
    # Omitting both streams blocks both media keys from every non-media row.
    # Omitting one also blocks its keys from the other stream, preventing leaks.
    blocked=torch.zeros(L,dtype=torch.bool)
    for source in absent:blocked|=payload['is_v'] if source=='visual' else payload['is_a']
    mask[(~blocked)[:,None]&blocked[None,:]]=torch.finfo(dtype).min
    return mask[None,None]

def sl_question_rows(payload,stem):
    """Exact text-embedding match; exclude choices and answer-format tokens."""
    tail=payload['decoder']['inputs_embeds'][0,payload['last_media']+1:].cpu()
    for text in (stem,' '+stem,'\n'+stem):
        ids=PROC.tokenizer.encode(text,add_special_tokens=False)
        device=MODEL_.get_input_embeddings().weight.device
        with torch.inference_mode():
            embeds=MODEL_.get_input_embeddings()(torch.tensor(ids,device=device)).cpu()
        for start in range(len(tail)-len(ids)+1):
            if torch.equal(tail[start:start+len(ids)],embeds):
                selected=torch.zeros(len(tail),dtype=torch.bool)
                selected[start:start+len(ids)]=True
                return selected
    raise ValueError('Question stem did not match frozen lexical embeddings exactly')

def sl_forward(payload,mask=None,deltas=None,gain=.15,capture=False,build_bank=False,prefix_bank=None,rescale=False):
    device=MODEL_.get_input_embeddings().weight.device
    kw={k:v.to(device) for k,v in payload['decoder'].items()}
    last=payload['last_media'];prefix=last+1;hooks=[];masks={};states={}
    if prefix_bank is not None:
        assert mask is None,'Counterfactual media representations cannot reuse native KV'
        kw['inputs_embeds']=kw['inputs_embeds'][:,prefix:]
        kw['position_ids']=kw['position_ids'][:,:,prefix:]
        kw['cache_position']=torch.arange(prefix,payload['tokens'],device=device)
        kw['past_key_values']=DynamicCache(ddp_cache_data=prefix_bank,config=MODEL_.model.config)
        last=-1
    def pre(module,args,kwargs):
        h=kwargs.get('hidden_states',args[0] if args else None);key=str(h.device)
        if key not in masks:masks[key]=mask.to(h.device)
        kwargs['attention_mask']=masks[key]
        return args,kwargs
    def post(li):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            if deltas is not None and li in deltas:
                h=h.clone();q=h[0,last+1:].float();d=deltas[li].to(h.device)
                alpha=gain[li] if isinstance(gain,dict) else gain
                scale=alpha*q.norm(dim=-1,keepdim=True)/d.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                if not rescale:scale=scale.clamp(max=1.)
                h[0,last+1:]=(q+scale*d).to(h.dtype)
            if capture and li in (18,22):states[li]=h[0,last+1:].detach().float().cpu()
            return (h,)+out[1:] if isinstance(out,tuple) else h
        return hook
    try:
        for li,layer in enumerate(MODEL_.model.layers):
            if mask is not None:hooks.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
            if (capture and li in (18,22)) or (deltas is not None and li in deltas):
                hooks.append(layer.register_forward_hook(post(li)))
        with torch.inference_mode():
            out=MODEL_.model(**kw,use_cache=build_bank or prefix_bank is not None,return_dict=True)
            logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
            if build_bank:
                states['_prefix_bank']=tuple((k[:,:,:prefix].detach(),v[:,:,:prefix].detach())
                    for k,v in out.past_key_values.to_legacy_cache())
    finally:
        for h in hooks:h.remove()
        masks.clear()
    assert torch.isfinite(logits).all()
    return logits,states

def sl_rotation(payload,k,stem):
    global SL_CACHE_VALIDATED
    native,joint=sl_forward(payload,capture=True,build_bank=True)
    bank=joint.pop('_prefix_bank')
    replay=float((native[SL_LABELS[:k]]-payload['native_logits']).abs().max())
    assert replay<=.12,replay
    cached_native,_=sl_forward(payload,prefix_bank=bank)
    cache_error=float((native[SL_LABELS[:k]]-cached_native[SL_LABELS[:k]]).abs().max())
    use_prefix=cache_error<=.03
    branches={}
    for name,absent in [('visual',('audio',)),('audio',('visual',)),('neither',('visual','audio'))]:
        mask=sl_mask(payload,absent)
        _,branches[name]=sl_forward(payload,mask=mask,capture=True)
        del mask;torch.cuda.empty_cache()
    visual_delta={li:joint[li]-branches['audio'][li] for li in joint}
    interaction={li:joint[li]-branches['visual'][li]-branches['audio'][li]+branches['neither'][li] for li in joint}
    projected={li:sl_project(interaction[li],branches['visual'][li]-branches['neither'][li],
        branches['audio'][li]-branches['neither'][li]) for li in joint}
    question_rows=sl_question_rows(payload,stem)
    stem_joint=projected[22]*question_rows[:,None]
    stem_visual=visual_delta[22]*question_rows[:,None]
    logits={'S0_native':native}
    conditions=[('S1_visual_same_layer',{22:visual_delta[22]},.15,False),
        ('S2_joint_same_layer',{22:interaction[22]},.15,False),
        ('S3_orthogonal_joint',{22:projected[22]},.15,False),
        ('S4_two_layer_orthogonal_joint',projected,.10,False),
        ('S5_question_stem_joint',{22:stem_joint},.15,False),
        ('S6_question_stem_argon_delta',{li:stem_visual for li in (10,12,14,16,18)},
            {10:.25,12:.30,14:.30,16:.25,18:.20},True)]
    intervention_replay={}
    fallback=[]
    for method,delta,gain,rescale in conditions:
        logits[method],_=sl_forward(payload,deltas=delta,gain=gain,prefix_bank=bank if use_prefix else None,rescale=rescale)
        if not SL_CACHE_VALIDATED and use_prefix:
            reference,_=sl_forward(payload,deltas=delta,gain=gain,rescale=rescale)
            error=float((reference[SL_LABELS[:k]]-logits[method][SL_LABELS[:k]]).abs().max())
            if error>.03 or reference[SL_LABELS[:k]].argmax()!=logits[method][SL_LABELS[:k]].argmax():
                logits[method]=reference;fallback.append(method)
            intervention_replay[method]=error
    SL_CACHE_VALIDATED=True
    return dict(scores={m:torch.log_softmax(lg[SL_LABELS[:k]],dim=-1).numpy().tolist() for m,lg in logits.items()},
        replay_error=replay,prefix_replay_error=cache_error,intervention_cache_replay=intervention_replay,
        tokens=payload['tokens'],stem_rows=int(question_rows.sum()),tail_rows=len(question_rows),
        prefix_used=use_prefix,full_intervention_fallback=fallback,
        interaction_norm={li:float(x.norm()) for li,x in interaction.items()},
        projection_retention={li:float(projected[li].norm()/interaction[li].norm().clamp_min(1e-8)) for li in joint})

def sl_run():
    output=SL_DIR/f'results_s{SL_SERVER}.jsonl'
    done={json.loads(x)['idx'] for x in output.read_text().splitlines()} if output.exists() else set()
    for sample in SL_SAMPLES:
        idx=sample['idx']
        if idx in done:continue
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
        totals={m:np.zeros(k) for m in SL_METHODS};audits=[]
        for rotation,perm in enumerate(rotations(k)):
            path=SL_DIR/f'{idx}_rot{rotation}.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                print('SYMMETRIC_ROT_START',SL_SERVER,idx,rotation,flush=True)
                payload=torch.load(SL_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
                assert payload['manifest_sha256']==SL_SHA
                result=sl_rotation(payload,k,stem)
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=SL_SHA,runtime_sha256=SL_RUNTIME_SHA)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                del payload;gc.collect();torch.cuda.empty_cache()
            assert result['manifest_sha256']==SL_SHA and result['runtime_sha256']==SL_RUNTIME_SHA
            for m,scores in result['scores'].items():
                for slot,option in enumerate(perm):totals[m][option]+=scores[slot]
            audits.append({k:v for k,v in result.items() if k!='scores'})
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',
            manifest_sha256=SL_SHA,runtime_sha256=SL_RUNTIME_SHA,
            scores={m:s.tolist() for m,s in totals.items()},
            predictions={m:chr(65+int(s.argmax())) for m,s in totals.items()},
            audit=audits,seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done.add(idx);print('SYMMETRIC_SAMPLE',len(done),'/',len(SL_SAMPLES),json.dumps(rec),flush=True)
    return dict(server=SL_SERVER,completed=len(done),expected=len(SL_SAMPLES))

RESULT=sl_run()
print('SYMMETRIC_COMPLETE',json.dumps(RESULT),flush=True)

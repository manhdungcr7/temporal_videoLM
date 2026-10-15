"""Faithful CCLD mathematics on frozen40; bounded decoder replay and safe masks.

The author's sampling/deletion harness is deliberately not executed. Definitions
are extracted from the archived original, and all scientific deviations are
declared in results/argon_ccld40/protocol.md.
"""
import ast,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CC_SERVER=globals().get('CC_SERVER',1)
CC_LIMIT=globals().get('CC_LIMIT')
CC_DIR=OUT/'argon_ccld40';CC_DIR.mkdir(exist_ok=True)
CC_MANIFEST=json.loads((OUT/'uploads/research40_manifest.json').read_text())
CC_SHA=CC_MANIFEST['manifest_sha256']
CC_SAMPLES=CC_MANIFEST[f'server{CC_SERVER}']
CC_ORIGINAL=(OUT/'uploads/ccld_proposal_original.py').read_text()
CC_SOURCE_SHA=hashlib.sha256(CC_ORIGINAL.encode()).hexdigest()
ARGON_SERVER=CC_SERVER
_base=(OUT/'uploads/argon_latent40.py').read_text().split('RESULT=argon_run(',1)[0]
exec(compile(_base,'argon_round3_definitions_only','exec'),globals())
_names={'hidden_to_manifold_thought_batch','detect_active_window',
        'extract_continuous_thoughts','make_clean_mask','make_shuffled_control'}
_tree=ast.parse(CC_ORIGINAL)
_parts=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in _names]
assert len(_parts)==len(_names)
_definitions=ast.unparse(ast.Module(body=_parts,type_ignores=[]))
_definitions=_definitions.replace(
    'triu_idx = torch.triu_indices(L, L, offset=1, device=device)\n    m2d[triu_idx[0], triu_idx[1]] = neg\n    del triu_idx',
    'upper = torch.ones((L, L), dtype=torch.bool, device=device).triu_(1)\n    m2d.masked_fill_(upper, neg)\n    del upper')
assert 'triu_indices' not in _definitions
exec(compile(_definitions,'ccld_original_math','exec'),globals())
CC_METHODS=['R4_C0_baseline','R4_C1_m9_thought_ctrl','R4_C2_m8_noclk_nopmi',
    'R4_C3_causal_vis_rec','R4_C4_shuffled_rec_ctrl','R4_C5_m9_plus_causal_rec',
    'R4_C6_ccld_unmasked','R4_C7_ccld_late_handoff']
CC_REC={10:.25,12:.30,14:.30,16:.25,18:.20}
CC_REC_FULL={10:.25,12:.32,14:.32,16:.25,18:.20}

def cc_forward(payload,mask=None,thought=None,thought_scale=.22,delta=None,
               recurrent_layers=None,late_audio=False,capture=False):
    """Same residual math as proposal; no shared mask mutation across devices."""
    device=MODEL_.get_input_embeddings().weight.device
    dtype=MODEL_.get_input_embeddings().weight.dtype
    kw={k:v.to(device) for k,v in payload['decoder'].items()}
    last=payload['last_media'];L=payload['tokens']
    hooks=[];captured={};masks={};late=None
    if late_audio:
        late=torch.zeros((1,1,L,L),dtype=dtype)
        late.masked_fill_(torch.ones((L,L),dtype=torch.bool).triu_(1)[None,None],torch.finfo(dtype).min)
        late[0,0,last+1:,payload['is_a']]-=2.5
    def pre(li):
        def hook(module,args,kwargs):
            h=kwargs.get('hidden_states',args[0] if args else None)
            selected=late if late_audio and li>=16 else mask
            key=(str(h.device),bool(late_audio and li>=16))
            if key not in masks:masks[key]=selected.to(h.device)
            kwargs['attention_mask']=masks[key]
            return args,kwargs
        return hook
    def post(li):
        def hook(module,args,out):
            h=out[0] if isinstance(out,tuple) else out
            if thought is not None and li in (0,1,2):
                h=h.clone();q=h[0,last+1:].float();ct=thought.to(h.device).unsqueeze(0)
                h[0,last+1:]=(q+thought_scale*q.norm(dim=-1,keepdim=True).mean().clamp_min(1e-5)*ct).to(h.dtype)
            if delta is not None and li in recurrent_layers:
                h=h.clone();q=h[0,last+1:].float();d=delta.to(h.device)
                assert q.shape==d.shape
                h[0,last+1:]=(q+recurrent_layers[li]*q.norm(dim=-1,keepdim=True).clamp_min(1e-5)
                    /d.norm(dim=-1,keepdim=True).clamp_min(1e-5)*d).to(h.dtype)
            if capture and li==22:captured[22]=h[0].detach().float().cpu()
            return (h,)+out[1:] if isinstance(out,tuple) else h
        return hook
    try:
        for li,layer in enumerate(MODEL_.model.layers):
            if mask is not None or (late_audio and li>=16):
                hooks.append(layer.register_forward_pre_hook(pre(li),with_kwargs=True))
            if (thought is not None and li in (0,1,2)) or (
                delta is not None and li in recurrent_layers) or (capture and li==22):
                hooks.append(layer.register_forward_hook(post(li)))
        with torch.inference_mode():
            out=MODEL_.model(**kw,use_cache=False,return_dict=True)
            logits=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    finally:
        for hook in hooks:hook.remove()
        masks.clear()
    assert torch.isfinite(logits).all()
    return logits,captured

def cc_rotation(payload,audio,duration,k):
    native,cap=cc_forward(payload,capture=True)
    replay=float((native[LETTERS[:k]]-payload['native_logits']).abs().max())
    assert replay<=.12,('native replay',replay)
    v=payload['is_v'];a=payload['is_a'];sec=payload['sec'];last=payload['last_media'];L=payload['tokens']
    device=MODEL_.get_input_embeddings().weight.device;dtype=MODEL_.get_input_embeddings().weight.dtype
    win=detect_active_window(audio,duration)
    base=cap[22];base_q=base[last+1:]
    ct_m9,ct_global=extract_continuous_thoughts(base,sec,v,a,last,win)
    del cap
    branches={}
    for mode in ('audio_only','hard_sync','anti_sync'):
        mask=make_clean_mask(L,v,a,sec,last,win,dtype,device,spg=payload['spg'],gate_mode=mode)
        _,cap=cc_forward(payload,mask=mask,capture=True)
        branches[mode]=cap[22][last+1:].clone()
        del cap,mask;torch.cuda.empty_cache()
    m8=.5*(base_q+branches['hard_sync'])-branches['anti_sync']
    visual=base_q-branches['audio_only']
    shuffled=make_shuffled_control(visual)
    shuffle_norm_error=float((visual.norm(dim=-1)-shuffled.norm(dim=-1)).abs().max())
    assert torch.allclose(visual.norm(dim=-1),shuffled.norm(dim=-1),rtol=1e-5,atol=1e-5)
    ccld=.55*visual+.45*(branches['hard_sync']-branches['audio_only'])
    results={'R4_C0_baseline':native}
    results['R4_C1_m9_thought_ctrl'],_=cc_forward(payload,thought=ct_m9)
    results['R4_C2_m8_noclk_nopmi'],_=cc_forward(payload,delta=m8,
        recurrent_layers={10:.28,12:.35,14:.35,16:.28})
    results['R4_C3_causal_vis_rec'],_=cc_forward(payload,delta=visual,recurrent_layers=CC_REC)
    results['R4_C4_shuffled_rec_ctrl'],_=cc_forward(payload,delta=shuffled,recurrent_layers=CC_REC)
    results['R4_C5_m9_plus_causal_rec'],_=cc_forward(payload,thought=ct_global,thought_scale=.20,
        delta=visual,recurrent_layers=CC_REC)
    results['R4_C6_ccld_unmasked'],_=cc_forward(payload,thought=ct_global,thought_scale=.20,
        delta=ccld,recurrent_layers=CC_REC_FULL)
    results['R4_C7_ccld_late_handoff'],_=cc_forward(payload,thought=ct_global,thought_scale=.20,
        delta=ccld,recurrent_layers=CC_REC_FULL,late_audio=True)
    scores={m:torch.log_softmax(lg[LETTERS[:k]],dim=-1).numpy().tolist() for m,lg in results.items()}
    return dict(scores=scores,win=list(win),replay_error=replay,tokens=L,
        delta_visual_norm=float(visual.norm()),delta_m8_norm=float(m8.norm()),
        delta_ccld_norm=float(ccld.norm()),shuffle_norm_error=shuffle_norm_error,
        ct_m9_global_cosine=float(F.cosine_similarity(ct_m9,ct_global,dim=0)),
        question_rows=L-last-1,clock=False,pmi=False)

def cc_run():
    samples=CC_SAMPLES[:CC_LIMIT] if CC_LIMIT else CC_SAMPLES
    output=CC_DIR/f'results_s{CC_SERVER}.jsonl';done=argon_load_rows(output)
    for sample in samples:
        idx=sample['idx']
        if idx in done:continue
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        started=time.time();_,texts=parse_question(sample['question']);k=len(texts)
        audio=decode_audio(fetch(sample['video_path']))
        assert abs(len(audio)/16000-sample['decoded_audio_duration'])<.2
        totals={m:np.zeros(k) for m in CC_METHODS};audits=[]
        for rotation,perm in enumerate(rotations(k)):
            path=CC_DIR/f'{idx}_rot{rotation}.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                print('CCLD_ROT_START',CC_SERVER,idx,rotation,flush=True)
                payload=torch.load(ARGON_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
                assert payload['manifest_sha256']==CC_SHA
                result=cc_rotation(payload,audio,sample['duration'],k)
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=CC_SHA,proposal_sha256=CC_SOURCE_SHA)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                del payload;gc.collect();torch.cuda.empty_cache()
            assert result['manifest_sha256']==CC_SHA and result['proposal_sha256']==CC_SOURCE_SHA
            for m,score in result['scores'].items():
                for slot,option in enumerate(perm):totals[m][option]+=score[slot]
            audits.append({k:v for k,v in result.items() if k!='scores'})
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',
            manifest_sha256=CC_SHA,proposal_sha256=CC_SOURCE_SHA,
            scores={m:s.tolist() for m,s in totals.items()},
            predictions={m:chr(65+int(s.argmax())) for m,s in totals.items()},
            audit=audits,seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done[idx]=rec
        print('CCLD_SAMPLE',len(done),'/',len(samples),json.dumps(rec),flush=True)
    return dict(server=CC_SERVER,completed=len(done),expected=len(samples),manifest_sha256=CC_SHA)

try:RESULT=cc_run()
except Exception:
    with (CC_DIR/f'errors_s{CC_SERVER}.jsonl').open('a') as f:
        f.write(json.dumps(dict(error=traceback.format_exc()))+'\n')
    raise
print('CCLD_COMPLETE',json.dumps(RESULT),flush=True)

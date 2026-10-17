"""Matched micro8 check of the proposal's layer-specific VO interpretation.

Primary F2 uses the input media states and VO weights of each target block.
Original salience selection, clock, bivector, gate, scale and block-output hook
remain fixed. F1 isolates weights; F3 separately checks the attention-residual
hook interpretation. No fit, option rotations, gold-dependent routing or sweep.
"""
import ast
import copy
import gc
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F

LF_DIR=OUT/'pcebr_v5_layerfix_micro8';LF_DIR.mkdir(exist_ok=True)
if (LF_DIR/'error.json').exists():
    prior_error_dir=LF_DIR/'device_error_attempt';prior_error_dir.mkdir(exist_ok=True)
    for oldname in ('error.json','protocol.json','runtime_source.json','hook_audit.json'):
        oldpath=LF_DIR/oldname
        if oldpath.exists():oldpath.rename(prior_error_dir/oldname)
(LF_DIR/'states').mkdir(exist_ok=True)
LF_SOURCE=OUT/'uploads/stage0_pcebr_v5_original.py'
LF_TREE=ast.parse(LF_SOURCE.read_text(encoding='utf-8'))
LF_TREE.body=[n for n in LF_TREE.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RESULT' for t in n.targets))]
LF_NS=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=LF_SERVER,__name__='v5_layerfix_original')
exec(compile(LF_TREE,str(LF_SOURCE),'exec'),LF_NS)
LF_MID=(14,16,18,20,22,24);LF_LATE=(28,29,30,31,32,33);LF_ALL=LF_MID+LF_LATE
LF_MANIFEST=json.loads((OUT/'uploads/research40_manifest.json').read_text(encoding='utf-8'))
LF_SAMPLES=LF_MANIFEST[f'server{LF_SERVER}'][:4]
LF_LABELS=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD']
LF_DEV=MODEL_.get_input_embeddings().weight.device
LF_OBSERVATIONS={}

def lf_write(name,value):
    (LF_DIR/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

# Observe the exact top-patch indices selected by the original function.
sel=next(copy.deepcopy(n) for n in LF_TREE.body if isinstance(n,ast.FunctionDef) and n.name=='extract_query_conditioned_salient_slots')
class SelectionObserver(ast.NodeTransformer):
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('chosen_v','chosen_a'):
            key=node.targets[0].id
            obs=ast.parse(f"LF_SELECTIONS['{key}'].append({key}.detach().cpu().clone())").body[0]
            return [node,obs]
        return node
sel=SelectionObserver().visit(sel)
sel_tree=ast.fix_missing_locations(ast.Module(body=[sel],type_ignores=[]))
LF_SEL_NS=dict(LF_NS,LF_SELECTIONS={})
exec(compile(sel_tree,'<observe_original_patch_selection>','exec'),LF_SEL_NS)

def lf_logits(kw):
    with torch.inference_mode():
        out=MODEL_.model(**kw,use_cache=False,return_dict=True)
        return MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()

def lf_capture(kw):
    pre={};post={};handles=[]
    for li in LF_ALL:
        def make_pre(i):
            def hk(mod,args,kwargs):
                h=args[0] if args else kwargs['hidden_states']
                pre[i]=h[0].detach().float().cpu()
            return hk
        handles.append(MODEL_.model.layers[li].register_forward_pre_hook(make_pre(li),with_kwargs=True))
    for li in (18,27):
        def make_post(i):
            def hk(mod,args,out):
                h=out[0] if isinstance(out,tuple) else out
                post[i]=h[0].detach().float().cpu()
            return hk
        handles.append(MODEL_.model.layers[li].register_forward_hook(make_post(li)))
    try:logits=lf_logits(kw)
    finally:
        for h in handles:h.remove()
    return pre,post,logits

def lf_pool_input(h,clock,selections):
    banks=[]
    for key,index in [('chosen_v',clock['v_indices']),('chosen_a',clock['a_indices'])]:
        x=h[index]
        dynamic=F.normalize(x-.85*x.mean(0,keepdim=True),dim=-1)
        banks.append(torch.stack([F.normalize(dynamic[ix].mean(0),dim=-1) for ix in selections[key]]))
    return banks

def lf_build(V,A,lv,la,iv):
    fwd,gate,phi,af=LF_NS['build_orthogonal_bivector_routed_state'](V,A,lv,la,iv,reverse_timeline=False)
    rev,rg,rphi,ar=LF_NS['build_orthogonal_bivector_routed_state'](V,A,lv,la,iv,reverse_timeline=True)
    parity=dict(even_error=float((af['c_ev']-ar['c_ev']).abs().max()),odd_error=float((af['tau_unit']+ar['tau_unit']).abs().max()),cosine=float(fwd@rev),gate=gate)
    assert parity['even_error']<1e-5 and parity['odd_error']<1e-5
    assert gate<1e-6 or parity['cosine']<.5
    return dict(fwd=fwd,rev=rev,content=af['c_ev'],gate=gate,phi=phi,audit=parity,V_vo=V,A_vo=A)

def lf_tangent(hq,direction,scale):
    unit=F.normalize(hq,dim=-1);dv=direction.to(hq.device,dtype=torch.float32).unsqueeze(0)
    tangent=F.normalize(dv-(dv*unit).sum(-1,keepdim=True)*unit,dim=-1)
    norm=hq.norm(dim=-1,keepdim=True).clamp(min=1e-5)
    return F.normalize(hq+scale*norm*tangent,dim=-1)*norm

def lf_route(kw,banks,rows,gate,channel='fwd',point='block_output',scale_override=None):
    scale=.045*gate if scale_override is None else scale_override
    handles=[];pre={};calls={li:0 for li in banks};norm_errors=[]
    for li,b in banks.items():
        direction=b[channel]
        if point=='attention_residual':
            def make_pre(i):
                def hk(mod,args,kwargs):
                    h=args[0] if args else kwargs['hidden_states']
                    ix=torch.tensor(rows,device=h.device,dtype=torch.long)
                    pre[i]=h[0,ix].detach().clone()
                return hk
            handles.append(MODEL_.model.layers[li].register_forward_pre_hook(make_pre(li),with_kwargs=True))
        def make_hook(i,vec):
            def hk(mod,args,out):
                calls[i]+=1
                if scale<=1e-7:return out
                h=out[0] if isinstance(out,tuple) else out
                ix=torch.tensor(rows,device=h.device,dtype=torch.long);new=h.clone()
                if point=='block_output':
                    hq=h[0,ix].float();updated=lf_tangent(hq,vec,scale)
                    new[0,ix]=updated.to(h.dtype)
                    actual=new[0,ix].float()
                else:
                    # Accelerate may move a block's input after its PyTorch pre-hook.
                    residual=pre[i].to(device=h.device,dtype=h.dtype)
                    hq=(residual+h[0,ix]).float()
                    updated=lf_tangent(hq,vec,scale)
                    new[0,ix]=(updated-residual.float()).to(h.dtype)
                    actual=(residual+new[0,ix]).float()
                err=((actual.norm(dim=-1)-hq.norm(dim=-1)).abs()/hq.norm(dim=-1).clamp(min=1e-5)).max()
                norm_errors.append(float(err))
                return (new,)+out[1:] if isinstance(out,tuple) else new
            return hk
        target=MODEL_.model.layers[li] if point=='block_output' else MODEL_.model.layers[li].self_attn
        handles.append(target.register_forward_hook(make_hook(li,direction)))
    try:logits=lf_logits(kw)
    finally:
        for h in handles:h.remove()
    assert set(calls.values())=={1}
    LF_OBSERVATIONS['last_hook_audit']=dict(point=point,calls=calls,max_relative_norm_roundoff=max(norm_errors,default=0.))
    return logits

def lf_result(logits,labels,native):
    selected=logits[labels];p=F.softmax(selected,dim=-1)
    return dict(logits=selected.tolist(),probabilities=p.tolist(),prediction='ABCD'[int(p.argmax())],tv_native=float((p-native).abs().sum()/2),hook_audit=copy.deepcopy(LF_OBSERVATIONS.get('last_hook_audit')))

lf_write('protocol.json',dict(server=LF_SERVER,ids=[s['idx'] for s in LF_SAMPLES],rotation=0,source_sha256=hashlib.sha256(LF_SOURCE.read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256((OUT/'uploads/pcebr_v5_layerfix_micro8.py').read_bytes()).hexdigest(),manifest_sha256=LF_MANIFEST['manifest_sha256'],
    primary='F2_layer_inputbanks_block',mid_layers=LF_MID,late_layers=LF_LATE,scale=.045,training=False,
    selection='Original H18/H27 output salience, original top-patch indices and clocks; fixed across variants',
    F1='Original H18 post-block banks, target-specific VO weights, original block-output tangent hook',
    F2='Media inputs of each target block, original residual pooling on same selected patches, target-specific VO, original block-output tangent hook',
    F3='Same vectors as F2; tangent applied to attention residual before MLP, via attention-output delta',
    limitation='Proposal inconsistently mentions VO of each layer and VO18; no learned M1/M2-to-QA connection is specified or added.',
    freeform='Native and primary F2, 32 tokens, no options; fixed original prefix hooks; vectors frozen from prepass'))
lf_write('runtime_source.json',dict(block=inspect.getsource(type(MODEL_.model.layers[18]).forward),attention=inspect.getsource(type(MODEL_.model.layers[18].self_attn).forward)))
LF_ROWS=[];LF_START=time.time()
try:
    for sample in LF_SAMPLES:
        started=time.time();idx=sample['idx']
        payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=True)
        assert payload['manifest_sha256']==LF_MANIFEST['manifest_sha256']
        kw={k:v.to(LF_DEV) for k,v in payload['decoder'].items()}
        clock=LF_NS['audit_and_interpolate_grid_clocks'](payload,sample)
        stem=sample['question'].split('(A)')[0].split('A.')[0].strip()
        stem_rows,option_rows,assistant_rows,rows=LF_NS['find_option_free_and_option_spans'](payload,stem,PROC.tokenizer)
        assert rows and not set(rows)&set(option_rows)
        pre,post,native_full=lf_capture(kw);k=len(payload['native_logits']);labels=LF_LABELS[:k]
        replay=float((native_full[labels]-payload['native_logits'].float()).abs().max());assert replay==0.
        LF_SEL_NS['LF_SELECTIONS']={'chosen_v':[],'chosen_a':[]}
        V,A,lv,la,iv=LF_SEL_NS['extract_query_conditioned_salient_slots'](post[18],post[27],stem_rows,clock,top_patch_frac=.15)
        selections=LF_SEL_NS['LF_SELECTIONS']
        ref=LF_NS['extract_query_conditioned_salient_slots'](post[18],post[27],stem_rows,clock,top_patch_frac=.15)
        assert all(torch.equal(x,y) for x,y in zip((V,A,lv,la,iv),ref))
        project=LF_NS['project_to_layer_vo_subspace']
        old=lf_build(project(MODEL_.model.layers[18],V),project(MODEL_.model.layers[18],A),lv,la,iv)
        original={li:old for li in LF_MID};F1={};F2={};raw_banks={}
        for li in LF_ALL:
            layer=MODEL_.model.layers[li]
            if li in LF_MID:F1[li]=lf_build(project(layer,V),project(layer,A),lv,la,iv)
            Vi,Ai=lf_pool_input(pre[li],clock,selections)
            raw_banks[li]=dict(V=Vi,A=Ai)
            F2[li]=lf_build(project(layer,Vi),project(layer,Ai),lv,la,iv)
        gate=old['gate'];assert all(abs(b['gate']-gate)<1e-12 for b in F2.values())
        native_p=F.softmax(native_full[labels],dim=-1)
        LF_OBSERVATIONS['last_hook_audit']=None
        conditions={'native':lf_result(native_full,labels,native_p)}
        mid={li:F2[li] for li in LF_MID};late={li:F2[li] for li in LF_LATE}
        random_banks={}
        g=torch.Generator().manual_seed(idx+101)
        for li in LF_MID:random_banks[li]={'fwd':F.normalize(torch.randn(old['fwd'].shape,generator=g),dim=-1)}
        specs=[('v5_original',original,'fwd','block_output',None),
               ('F1_layer_vo18banks_block',F1,'fwd','block_output',None),
               ('F2_layer_inputbanks_block',mid,'fwd','block_output',None),
               ('F2_reversed',mid,'rev','block_output',None),
               ('F2_content_only',mid,'content','block_output',None),
               ('F2_random',random_banks,'fwd','block_output',None),
               ('F2_late',late,'fwd','block_output',None),
               ('F2_late_reversed',late,'rev','block_output',None),
               ('F3_layer_inputbanks_attention_residual',mid,'fwd','attention_residual',None),
               ('zero_scale_replay',mid,'fwd','block_output',0.)]
        for name,banks,channel,point,scale in specs:
            lg=lf_route(kw,banks,rows,gate,channel,point,scale)
            conditions[name]=lf_result(lg,labels,native_p)
        assert conditions['zero_scale_replay']['logits']==conditions['native']['logits']
        prior=json.loads((OUT/'pcebr_stage0_v5/m3_audit.json').read_text(encoding='utf-8'))
        oldaudit=next(a for a in prior if a['result']['idx']==idx)
        original_error=max(abs(a-b) for a,b in zip(conditions['v5_original']['probabilities'],oldaudit['probabilities']['p_mid_fwd']))
        assert original_error<1e-7
        input_audit={li:dict(shape=list(h.shape),media_video_mean=h[clock['v_indices']].mean(0),media_audio_mean=h[clock['a_indices']].mean(0)) for li,h in pre.items()}
        torch.save(dict(pre_input_audit=input_audit,original_banks=dict(V=V,A=A,lv=lv,la=la,intervals=iv),selections=selections,
                        F1=F1,F2=F2,raw_banks=raw_banks,original=old,random_banks=random_banks,hook_rows=rows,option_rows=option_rows),LF_DIR/'states'/f'{idx}.pt')
        row=dict(idx=idx,task=sample['task'],gold=sample['gold'],conditions=conditions,native_replay_error=replay,
                 original_v5_probability_replay_error=original_error,gate=gate,clock=clock['clock_audit'],
                 layer_parity={str(li):b['audit'] for li,b in F2.items()},seconds=round(time.time()-started,2))
        for value in conditions.values():value['correct']=value['prediction']==row['gold']
        LF_ROWS.append(row);lf_write('results.json',LF_ROWS)
        print('LAYERFIX_MC',idx,sample['task'],sample['gold'],json.dumps({n:c['prediction'] for n,c in conditions.items()}),flush=True)
        # Generation uses exactly the original option-removal and position scheme.
        last_m=int(payload['last_media']);keep=sorted(list(range(last_m+1))+stem_rows+assistant_rows)
        keep_t=torch.tensor(keep,device=LF_DEV,dtype=torch.long)
        ff_emb=kw['inputs_embeds'][:,keep_t,:].clone();ff_pos=kw['position_ids'][:,:,keep_t].clone()
        tail_start=int(ff_pos[0,0,last_m])+1;n_tail=len(keep)-(last_m+1)
        ff_pos[:,:,last_m+1:]=(tail_start+torch.arange(n_tail,device=LF_DEV)).view(1,1,-1).to(ff_pos.dtype)
        ff_rows=list(range(last_m+1,last_m+1+n_tail));eos={PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')}
        ff={}
        for name in ('native','F2_layer_inputbanks_block'):
            c_emb=ff_emb.clone();c_pos=ff_pos.clone();gen=[]
            for _ in range(32):
                length=c_emb.shape[1]
                gen_kw=dict(inputs_embeds=c_emb,position_ids=c_pos,attention_mask=torch.ones((1,length),device=LF_DEV,dtype=torch.long),cache_position=torch.arange(length,device=LF_DEV))
                lg=lf_logits(gen_kw) if name=='native' else lf_route(gen_kw,mid,ff_rows,gate)
                token=int(lg.argmax())
                if token in eos and gen:break
                gen.append(token)
                with torch.inference_mode():emb=MODEL_.get_input_embeddings()(torch.tensor([[token]],device=LF_DEV)).to(c_emb.dtype)
                c_emb=torch.cat([c_emb,emb],dim=1);c_pos=torch.cat([c_pos,c_pos[:,:,-1:]+1],dim=-1)
            ff[name]=dict(text=PROC.tokenizer.decode(gen,skip_special_tokens=True).strip(),token_ids=gen)
        row['freeform']=ff;row['seconds']=round(time.time()-started,2);lf_write('results.json',LF_ROWS)
        print('LAYERFIX_FREEFORM',idx,json.dumps(ff,ensure_ascii=False),flush=True)
        del pre,post,kw,payload,F1,F2,raw_banks,mid,late,original,ff_emb,ff_pos,c_emb,c_pos
        gc.collect();torch.cuda.empty_cache()
    RESULT=dict(server=LF_SERVER,completed=len(LF_ROWS),seconds=round(time.time()-LF_START,2))
    lf_write('completion.json',RESULT);print('LAYERFIX_DONE',json.dumps(RESULT),flush=True)
except Exception:
    lf_write('error.json',dict(completed=len(LF_ROWS),traceback=traceback.format_exc()));raise
finally:
    modules=list(MODEL_.modules())
    lf_write('hook_audit.json',dict(forward_hooks=sum(len(m._forward_hooks) for m in modules),pre_hooks=sum(len(m._forward_pre_hooks) for m in modules),cuda_ok=bool(torch.isfinite(torch.ones(1,device='cuda:0')).all())))

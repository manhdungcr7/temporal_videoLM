"""Exploratory matched routing ablations and algebra audits; original run stays intact."""
import inspect
import json
import math
import torch
import torch.nn.functional as F

checks={'server':PC_SERVER,'routing_ablations':[],'freeform_regex_counterexamples':[]}
# Compare dual-SVD against dense regularized covariance on an independent small matrix.
rng=torch.Generator().manual_seed(808)
X=torch.randn(20,32,generator=rng)
Xn=F.normalize(X,dim=-1);xc=Xn-Xn.mean(0,keepdim=True)
cov=xc.T@xc/len(X);lam=max(.22*float(cov.trace())/X.shape[1],1e-5)
eig,U=torch.linalg.eigh(.78*cov+lam*torch.eye(X.shape[1]))
W=(U*eig.rsqrt().unsqueeze(0))@U.T
zca=V5_NS['fit_dual_svd_zca_whitener'](X,.22)
actual=V5_NS['apply_dual_svd_zca'](X,zca)
expected=F.normalize(xc@W,dim=-1)
checks['zca_dense_normalized_output_max_error']=float((actual-expected).abs().max())

# Original uniform test is sim==0. Equal nonzero similarities are a separate case.
iv=torch.tensor([[0.,2.],[2.,4.],[4.,6.],[6.,8.],[8.,10.]])
checks['constant_retrieval_features']=[]
for constant in (0.,.2,-.2):
    raw=torch.full((5,),constant)
    phi,null,std,*_=V5_NS['compute_shrinkage_calibrated_relational_features'](raw*14,raw*14,iv,iv,raw_sim_1=raw,raw_sim_2=raw)
    checks['constant_retrieval_features'].append(dict(cosine=constant,phi=phi.tolist(),null=null.tolist(),std=std.tolist()))
for text,task,gold in [("No, it happens at the same time.",'TA','Yes'),("Yes, it is not happening now.",'TA','No'),("The video shows flour being put away; milk and egg are not being added.",'CMTB','add milk and one egg whisk together'),("After the speech, the scene changes. The narrated event was already over.",'TF','Future')]:
    checks['freeform_regex_counterexamples'].append(dict(text=text,task=task,gold=gold,audit=V5_NS['audit_freeform_generation_quality'](text,30.,task,gold)))

layer=MODEL_.model.layers[18]
checks['vo_projection_layer_index']=18
checks['vo_layer_weight_shapes']={k:list(getattr(layer.self_attn,k).weight.shape) for k in ('v_proj','o_proj')}
checks['runtime_decoder_block_source']=inspect.getsource(type(layer).forward)
checks['mod_zca_media_source']=inspect.getsource(V5_NS['ResidualStreamAligner']._pre_whiten_media)
checks['main_training_source_hashes']=V5_INFO['source_sha256']

tok=PROC.tokenizer; labels=[tok.convert_tokens_to_ids(c) for c in 'ABCD']
device=MODEL_.get_input_embeddings().weight.device
for audit in V5_M3:
    row=audit['result'];idx=row['idx']
    raw=torch.load(V5_DIR/'artifacts'/f'm3_{idx}.pt',map_location='cpu',weights_only=True)
    payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=True)
    kw={k:v.to(device) for k,v in payload['decoder'].items()}
    n=len(payload['native_logits']);target_rows=audit['hook_rows']
    vector_raw,gate,_,_=V5_NS['build_orthogonal_bivector_routed_state'](raw['V18_raw'],raw['A18_raw'],raw['lg_v'],raw['lg_a'],raw['iv_t'])
    directions={'low_strength_raw_tangent':vector_raw,'vo_content_only_tangent':raw['aud_fwd']['c_ev'],'zero_scale_parity':raw['vec_fwd']}
    sample=dict(idx=idx,task=row['task'],gold=row['gold_letter'],main_native=row['mc_pred_native'],main_full=row['mc_pred_routed_fwd'],conditions={})
    for name,direction in directions.items():
        scale=0. if name=='zero_scale_parity' else .045*gate
        handles=[]
        for li in (14,16,18,20,22,24):
            def make_hook(vec=direction,s=scale):
                def hook(mod,args,out):
                    if s<=1e-7:return out
                    h=out[0] if isinstance(out,tuple) else out
                    new=h.clone();ix=torch.tensor(target_rows,device=h.device)
                    hq=h[0,ix].float();hu=F.normalize(hq,dim=-1);dv=vec.to(h.device).unsqueeze(0)
                    dt=F.normalize(dv-(dv*hu).sum(-1,keepdim=True)*hu,dim=-1)
                    norms=hq.norm(dim=-1,keepdim=True).clamp(min=1e-5)
                    new[0,ix]=(F.normalize(hq+s*norms*dt,dim=-1)*norms).to(h.dtype)
                    return (new,)+out[1:] if isinstance(out,tuple) else new
                return hook
            handles.append(MODEL_.model.layers[li].register_forward_hook(make_hook()))
        try:
            with torch.inference_mode():
                o=MODEL_.model(**kw,use_cache=False,return_dict=True)
                logits=MODEL_.lm_head(o.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()[labels[:n]]
        finally:
            for handle in handles:handle.remove()
        probs=F.softmax(logits,dim=-1);pred='ABCD'[int(probs.argmax())]
        sample['conditions'][name]=dict(prediction=pred,correct=pred==row['gold_letter'],probabilities=probs.tolist(),logits=logits.tolist(),native_logit_max_diff=float((logits-payload['native_logits']).abs().max()))
    checks['routing_ablations'].append(sample)
    v5_write('controls.json',checks)
    print('PCV5_MATCHED_ABLATION',json.dumps(sample),flush=True)
checks['intervention_hook_count']=sum(len(l._forward_hooks) for l in MODEL_.model.layers)
checks['cuda_ok']=bool(torch.isfinite(torch.ones(1,device='cuda:0')).all())
v5_write('controls.json',checks)
RESULT=checks
print('PCV5_CONTROLS_DONE',json.dumps(checks),flush=True)

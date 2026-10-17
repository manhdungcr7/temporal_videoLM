"""Original control accuracies plus preread, fixed-seed CLC specificity audits."""
import ast
import copy
import json
import torch
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as v7_qmod

v7_m3_ast=next(n for n in ast.parse(V7_ORIGINAL).body if isinstance(n,ast.FunctionDef) and n.name=='run_stage0_m3_bivector_vo_tangent_routing')
v7_route_ast=next(n for n in ast.walk(v7_m3_ast) if isinstance(n,ast.FunctionDef) and n.name=='run_caor_routed')
v7_control_rows=[]
v7_micro=json.loads((OUT/'uploads/research40_manifest.json').read_text())[f'server{PC_SERVER}'][:4]
for row in v7_micro:
    idx=row['idx'];raw=torch.load(V7_DIR/'artifacts'/f'm3_{idx}.pt',map_location='cpu',weights_only=False)
    payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
    device=MODEL_.get_input_embeddings().weight.device
    kw={k:x.to(device) for k,x in payload['decoder'].items()}
    impl=MODEL_.model.layers[0].self_attn.config._attn_implementation
    tail=torch.arange(int(payload['last_media'])+1,payload['tokens'],dtype=torch.long)
    scope=dict(V7_NS,model_llm=MODEL_,kw=kw,all_tail_rows_t=tail,
        v_indices=raw['clock_info']['v_indices'],a_indices=raw['clock_info']['a_indices'],last_m=int(payload['last_media']),
        orig_impl=impl,orig_kernel=v7_qmod.ALL_ATTENTION_FUNCTIONS[impl],qwen_omni_modeling=v7_qmod)
    exec(compile(ast.Module(body=[v7_route_ast],type_ignores=[]),'original_v7_route_control_audit','exec'),scope)
    route=scope['run_caor_routed'];layers=(14,16,18,20,22,24)
    labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD'];k=len(payload['native_logits'])
    fullprof=raw['prof_full_fwd'];scale=0.038*fullprof['gamma_gate']
    caor=torch.softmax(route(layers,fullprof)[labels[:k]],dim=-1)
    full=torch.softmax(route(layers,fullprof,clc_vec_1d=raw['clc_dir_fwd'],clc_scale=scale)[labels[:k]],dim=-1)
    replay=float((full-raw['p_mid_fwd']).abs().max())
    caor_replay=float((caor-raw['p_mid_caor_only']).abs().max())
    assert replay<1e-6 and caor_replay<1e-6
    worst_k=int((raw['lg_v']+raw['lg_a']).argmin())
    irrel=V7_NS['build_caor_token_bias_profiles'](payload['tokens'],raw['clock_info'],raw['slot_meta'],raw['aud_fwd'],
        m2_probe_bundle=V7_PROBE,exp_phi=raw['phi_fwd'],mode='irrel',irrel_slot_k=worst_k)
    conditions={}
    def record(name,prof,**opts):
        logits=route(layers,prof,**opts)[labels[:k]];prob=torch.softmax(logits,dim=-1);pred='ABCD'[int(prob.argmax())]
        conditions[name]=dict(logits=logits.tolist(),probabilities=prob.tolist(),prediction=pred,correct=pred==row['gold'])
    for name,prof in [('original_irrelevant_slot_attn_only',irrel)]+[(f'original_random_profile_{seed}_attn_only',p) for seed,p in zip((101,202,303),raw['profs_rand'])]:
        record(name,prof)
    # Additional controls chosen before observing M3: fixed seeds, same full attention routing and CLC strength.
    for seed in (101,202,303):
        gen=torch.Generator().manual_seed(idx+seed)
        direction=torch.nn.functional.normalize(torch.randn(raw['clc_dir_fwd'].shape,generator=gen),dim=-1)
        record(f'expl_full_caor_random_clc_{seed}',fullprof,clc_vec_1d=direction,clc_scale=scale)
    # Isolate the sink correction difference in the event contrast; no mainrun change.
    foil=V7_NS['build_caor_token_bias_profiles'](payload['tokens'],raw['clock_info'],raw['slot_meta'],raw['aud_fwd'],
        m2_probe_bundle=V7_PROBE,exp_phi=raw['phi_fwd'],mode='foil_suppressed')
    foil['kappa_sink']=fullprof['kappa_sink'];box={}
    route(layers,foil,capture_l22_box=box)
    ev=torch.nn.functional.normalize(raw['h22_supp']-box['h22_last'],dim=-1)
    direction=torch.nn.functional.normalize(0.45*ev+0.35*raw['dh_ord']+0.20*raw['vec_fwd'],dim=-1)
    record('expl_full_caor_sink_matched_event_clc',fullprof,clc_vec_1d=direction,clc_scale=scale)
    # The original CLC direction is captured from a prompt containing MC options, even when later reused in freeform.
    # Recompute all three prepasses with options actually absent, keeping profiles, coefficients and strength fixed.
    stem=row['question'].split('(A)')[0].split('A.')[0].strip()
    spans=V7_NS['find_option_free_and_option_spans'](payload,stem,PROC.tokenizer)
    keep=sorted(list(range(int(payload['last_media'])+1))+spans[0]+spans[2])
    keep_t=torch.tensor(keep,device=device)
    ff_emb=kw['inputs_embeds'][:,keep_t,:].clone();ff_pos=kw['position_ids'][:,:,keep_t].clone()
    first=int(ff_pos[0,0,int(payload['last_media'])])+1
    n_tail=len(keep)-(int(payload['last_media'])+1)
    ff_pos[:,:,int(payload['last_media'])+1:]=(first+torch.arange(n_tail,device=device)).view(1,1,-1)
    ff_kw=dict(inputs_embeds=ff_emb,position_ids=ff_pos,attention_mask=torch.ones((1,len(keep)),dtype=torch.long,device=device),cache_position=torch.arange(len(keep),device=device))
    ff_rows=torch.arange(int(payload['last_media'])+1,len(keep),dtype=torch.long)
    revprof=V7_NS['build_caor_token_bias_profiles'](payload['tokens'],raw['clock_info'],raw['slot_meta'],raw['aud_fwd'],
        m2_probe_bundle=V7_PROBE,exp_phi=raw['phi_fwd'],mode='full',reverse_order=True)
    foilprof=V7_NS['build_caor_token_bias_profiles'](payload['tokens'],raw['clock_info'],raw['slot_meta'],raw['aud_fwd'],
        m2_probe_bundle=V7_PROBE,exp_phi=raw['phi_fwd'],mode='foil_suppressed')
    ff_boxes={}
    for name,prof in [('support',fullprof),('foil',foilprof),('reverse',revprof)]:
        b={};route(layers,prof,custom_kw=ff_kw,target_rows=ff_rows,capture_l22_box=b);ff_boxes[name]=b['h22_last']
    diff_ev=ff_boxes['support']-ff_boxes['foil'];diff_ord=ff_boxes['support']-ff_boxes['reverse']
    ff_ev=torch.nn.functional.normalize(diff_ev,dim=-1) if float(diff_ev.norm())>1e-6 else raw['vec_fwd']
    ff_ord=torch.nn.functional.normalize(diff_ord,dim=-1) if float(diff_ord.norm())>1e-6 else raw['aud_fwd']['tau_unit']
    ff_dir=torch.nn.functional.normalize(.45*ff_ev+.35*ff_ord+.20*raw['vec_fwd'],dim=-1)
    record('expl_optionfree_prepass_clc',fullprof,clc_vec_1d=ff_dir,clc_scale=scale)
    v7_control_rows.append(dict(idx=idx,task=row['task'],gold=row['gold'],full_replay_probability_error=replay,
        caor_replay_probability_error=caor_replay,event_contrast_cosine_original_vs_sinkmatched=float(torch.dot(ev,raw['dh_ev'])),
        clc_cosine_mc_vs_optionfree_prepass=float(torch.dot(ff_dir,raw['clc_dir_fwd'])),
        optionfree_prepass_tail_text=PROC.tokenizer.decode(payload['input_ids'][0,keep][int(payload['last_media'])+1:],skip_special_tokens=False),conditions=conditions))
    v7_write('control_accuracy.json',v7_control_rows)
    print('PCV7_CONTROL_ACCURACY',json.dumps(v7_control_rows[-1]),flush=True)
assert len(v7_control_rows)==4
RESULT=dict(server=PC_SERVER,rows=len(v7_control_rows),hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
print('PCV7_CONTROLS_DONE',json.dumps(RESULT),flush=True)

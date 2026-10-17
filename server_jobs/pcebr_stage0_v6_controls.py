"""Record accuracy for the original three random and irrelevant-profile controls."""
import ast
import json
import torch
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as v6_qmod

v6_m3_ast=next(n for n in ast.parse(V6_ORIGINAL).body if isinstance(n,ast.FunctionDef) and n.name=='run_stage0_m3_bivector_vo_tangent_routing')
v6_route_ast=next(n for n in ast.walk(v6_m3_ast) if isinstance(n,ast.FunctionDef) and n.name=='run_caor_routed')
v6_control_rows=[]
v6_micro=json.loads((OUT/'uploads/research40_manifest.json').read_text())[f'server{PC_SERVER}'][:4]
for row in v6_micro:
    idx=row['idx'];raw=torch.load(V6_DIR/'artifacts'/f'm3_{idx}.pt',map_location='cpu',weights_only=False)
    payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
    device=MODEL_.get_input_embeddings().weight.device
    kw={k:x.to(device) for k,x in payload['decoder'].items()}
    stem=row['question'].split('(A)')[0].split('A.')[0].strip()
    spans=V6_NS['find_option_free_and_option_spans'](payload,stem,PROC.tokenizer)
    impl=MODEL_.model.layers[0].self_attn.config._attn_implementation
    scope=dict(V6_NS,model_llm=MODEL_,kw=kw,hook_rows_t=torch.tensor(spans[3],dtype=torch.long),
               v_indices=raw['clock_info']['v_indices'],a_indices=raw['clock_info']['a_indices'],last_m=int(payload['last_media']),
               orig_impl=impl,orig_kernel=v6_qmod.ALL_ATTENTION_FUNCTIONS[impl],qwen_omni_modeling=v6_qmod)
    exec(compile(ast.Module(body=[v6_route_ast],type_ignores=[]),'original_caor_route_accuracy_audit','exec'),scope)
    route=scope['run_caor_routed'];layers=(14,16,18,20,22,24)
    labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD'];k=len(payload['native_logits'])
    full=route(layers,raw['prof_full_fwd'])[labels[:k]]
    replay=float((torch.softmax(full,dim=-1)-raw['p_mid_fwd']).abs().max())
    assert replay<1e-6
    worst_k=int((raw['lg_v']+raw['lg_a']).argmin())
    irrel=V6_NS['build_caor_token_bias_profiles'](payload['tokens'],raw['clock_info'],raw['slot_meta'],raw['aud_fwd'],mode='irrel',irrel_slot_k=worst_k)
    conditions={}
    for name,prof in [('irrelevant_slot',irrel)]+[(f'random_seed_{seed}',prof) for seed,prof in zip((101,202,303),raw['profs_rand'])]:
        logits=route(layers,prof)[labels[:k]];prob=torch.softmax(logits,dim=-1);pred='ABCD'[int(prob.argmax())]
        conditions[name]=dict(logits=logits.tolist(),probabilities=prob.tolist(),prediction=pred,correct=pred==row['gold'])
    v6_control_rows.append(dict(idx=idx,task=row['task'],gold=row['gold'],full_replay_probability_error=replay,conditions=conditions))
    v6_write('control_accuracy.json',v6_control_rows)
    print('PCV6_CONTROL_ACCURACY',json.dumps(v6_control_rows[-1]),flush=True)
assert len(v6_control_rows)==4
RESULT=dict(server=PC_SERVER,rows=len(v6_control_rows),hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
print('PCV6_CONTROLS_DONE',json.dumps(RESULT),flush=True)

"""Replay saved final interventions, isolate CAOR / CLC / continuous thought without new rollout."""
import ast,gc,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling
D=OUT/'pcebr_stage0_v8/corrected'
source=(D/'effective_source.py').read_text(encoding='utf-8')
t=ast.parse(source)
main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='run_stage0_m3_bivector_vo_tangent_routing')
route=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='run_caor_routed')
saved=json.loads((D/'samples.json').read_text())
rows=[]
for row in saved:
    idx=row['idx'];model_llm=MODEL_
    payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
    raw=torch.load(D/'artifacts'/f'm3_{idx}.pt',map_location='cpu',weights_only=False)
    device=model_llm.get_input_embeddings().weight.device
    kw={k:v.to(device) for k,v in payload['decoder'].items()}
    last_m=int(payload['last_media']);prefix_len=last_m
    v_indices=raw['clock_info']['v_indices'];a_indices=raw['clock_info']['a_indices']
    all_tail_rows_t=torch.arange(last_m+1,payload['tokens'],dtype=torch.long)
    mid_layers=(14,16,18,20,22,24)
    orig_impl=model_llm.model.layers[0].self_attn.config._attn_implementation
    orig_kernel=qwen_omni_modeling.ALL_ATTENTION_FUNCTIONS[orig_impl]
    with torch.inference_mode():
        native=model_llm.model(**kw,use_cache=True,return_dict=True)
        legacy=native.past_key_values.to_legacy_cache() if hasattr(native.past_key_values,'to_legacy_cache') else native.past_key_values
        prefix_bank=tuple((k[:,:,:prefix_len].detach(),v[:,:,:prefix_len].detach()) for k,v in legacy)
        del native,legacy
    scope=dict(globals())
    exec(compile(ast.Module(body=[route],type_ignores=[]),'saved_final_route','exec'),scope)
    run=scope['run_caor_routed']
    prof=raw['prof_full_fwd'];gamma=prof['gamma_gate']
    conditions={
        'CAOR_only':(0.78,0.,0.),
        'CAOR_CLC_no_CT':(0.78,.042*gamma,0.),
        'CAOR_CT_no_CLC':(0.78,0.,.16*gamma),
        'CT_only':(0.,0.,.16*gamma),
        'CLC_only':(0.,.042*gamma,0.),
        'CAOR_CLC_CT_direct_replay':(.78,.042*gamma,.16*gamma)}
    labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD'][:len(payload['native_logits'])]
    out={}
    for tag,(eta,cs,ts) in conditions.items():
        logits=run(mid_layers,prof,eta_scale=eta,clc_vec_1d=raw['clc_dir_fwd'],clc_scale=cs,ct_vec_1d=raw['slot_meta']['z_thought'],ct_scale=ts,
            use_prefix_cache=row['prefix_cache_used_for_routed_passes'])
        p=F.softmax(logits[labels],dim=-1)
        out[tag]=dict(pred='ABCD'[int(p.argmax())],probabilities=p.tolist())
    replay_err=float((torch.tensor(out['CAOR_CLC_CT_direct_replay']['probabilities'])-torch.tensor(row['probabilities']['p_mid_direct'])).abs().max())
    result=dict(idx=idx,task=row['task'],gold=row['gold_letter'],conditions=out,direct_replay_max_probability_error=replay_err)
    rows.append(result)
    (D/'component_ablation.json').write_text(json.dumps(rows,indent=2))
    print('V8_COMPONENTS',json.dumps(result),flush=True)
    del scope,raw,payload,prefix_bank
    gc.collect();torch.cuda.empty_cache()
RESULT=dict(n_samples=len(rows),scores={k:sum(x['conditions'][k]['pred']==x['gold'] for x in rows) for k in conditions})
print('V8_COMPONENT_COMPLETE',json.dumps(RESULT),flush=True)

"""Resume M2/M3 with frozen feature teachers; never retrain completed M1."""
import json
import time
import torch

_v3_resume_start=time.time()
_v3_freeze_audit=[]
for prefix,bundle in _v3_bundles.items():
    saved=torch.load(_v3_folder/'artifacts'/f'{prefix}_bundle.pt',map_location='cpu',weights_only=True)
    for key in ('m_ctx','m_loc'):
        model=bundle[key]
        before={k:v.detach().cpu() for k,v in model.state_dict().items()}
        assert all(torch.equal(before[k],saved[key+'_state'][k]) for k in before),('Weights changed before freeze',prefix,key)
        model.requires_grad_(False)
        for parameter in model.parameters():parameter.grad=None
        after={k:v.detach().cpu() for k,v in model.state_dict().items()}
        assert all(torch.equal(before[k],after[k]) for k in before)
        _v3_freeze_audit.append(dict(prefix=prefix,model=key,weights_unchanged=True,trainable_parameters=0))
_v3_write('freeze_repair_audit.json',_v3_freeze_audit)
_v3_info['failed_attempt_episode_files']=[r['file'] for r in _v3_info['episodes']]
_v3_info['repair']='Freeze pretrained M1 feature teachers before downstream probe training; no parameter/value/hyperparameter changes.'
_v3_bundle=_v3_bundles.get('avbind_pooled_pairs') or _v3_bundles['avbind_native_pairs']
_v3_m2_result=_v3_ns['run_stage0_m2_retrieved_interval_counterfactuals'](_v3_bundle)
_v3_m3_result=_v3_ns['run_stage0_m3_mid_vs_late_llm_routing'](_v3_bundle)
RESULT=dict(protocol='PC-EBR v3, original math with frozen M1 feature teachers for probe autograd',
    Algebraic_Unit_Checks=_v3_ns['run_algebraic_unit_checks']('cuda:0'),
    M1_native_pairs=_v3_state['avbind_native_pairs'],M1_pooled_pairs=_v3_state['avbind_pooled_pairs'],
    M2_interval_algebra_and_counterfactuals=_v3_m2_result,M3_mid_vs_late_routing_micro_probe=_v3_m3_result,
    resumed_elapsed_sec=time.time()-_v3_resume_start)
_v3_write('main_results.json',RESULT);_v3_write('instrumentation.json',_v3_info)
print('PCV3_RESUMED_COMPLETE',json.dumps(RESULT),flush=True)

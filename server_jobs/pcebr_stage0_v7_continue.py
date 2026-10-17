import shutil
failed=V7_DIR/'original_m0_failure'
failed.mkdir(exist_ok=True)
for name in ('error.json','completion.json','progress.json','instrumentation.json'):
    if (V7_DIR/name).exists(): shutil.copy2(V7_DIR/name,failed/name)
V7_STATE['M0']=dict(status='failed_original_assertion',failed_lines=[x['line'] for x in v7_m0_checks if not x['passed']],diagnostic=v7_m0_output)
v7_start=time.time()
try:
    for V7_PREFIX in ('avbind_native_pairs','avbind_pooled_pairs'):
        result,bundle=V7_NS['run_stage0_m1_orthogonal_factorial'](V7_PREFIX)
        V7_STATE['M1_native' if 'native' in V7_PREFIX else 'M1_pooled']=result
        V7_BUNDLES[V7_PREFIX]=bundle
        torch.save(v7_cpu({k:x for k,x in bundle.items() if not k.startswith('m_')}),V7_DIR/'artifacts'/f'{V7_PREFIX}_bundle.pt')
        torch.save(v7_cpu({k:x.state_dict() for k,x in bundle.items() if k.startswith('m_')}),V7_DIR/'artifacts'/f'{V7_PREFIX}_model_states.pt')
        v7_write('progress.json',V7_STATE)
        print('PCV7_M1_COMPLETE',V7_PREFIX,json.dumps(result),flush=True)
    V7_STATE['M2'],V7_PROBE,V7_BRIDGE=V7_NS['run_stage0_m2_connected'](V7_BUNDLES['avbind_pooled_pairs'])
    torch.save(v7_cpu(dict(probe={k:(x.state_dict() if k=='clf' else x) for k,x in V7_PROBE.items()},bridge=V7_BRIDGE.state_dict())),V7_DIR/'artifacts/connected_m2_models.pt')
    v7_write('progress.json',V7_STATE)
    print('PCV7_M2_COMPLETE',json.dumps(V7_STATE['M2']),flush=True)
    V7_STATE['M3']=V7_NS['run_stage0_m3_bivector_vo_tangent_routing'](m1_aligner=V7_BUNDLES['avbind_pooled_pairs']['m_c6'],m2_probe_bundle=V7_PROBE,hop2_bridge=V7_BRIDGE)
    V7_STATE.update(protocol='Stage-0 v7 final original functions with observation-only artifacts and separate validation',elapsed_sec=round(time.time()-v7_start,2))
    v7_write('main_results.json',V7_STATE)
    print('PCV7_DONE',json.dumps(V7_STATE),flush=True)
except Exception:
    v7_write('error.json',dict(traceback=traceback.format_exc(),elapsed_sec=time.time()-v7_start))
    raise
finally:
    v7_write('progress.json',V7_STATE)
    v7_write('instrumentation.json',V7_INFO)
    health=[]
    for dev in range(torch.cuda.device_count()):
        health.append(float((torch.ones(1,device=f'cuda:{dev}')+1).cpu()))
    v7_write('completion.json',dict(parameter_versions_unchanged=v7_param_versions=={n:p._version for n,p in MODEL_.named_parameters()},
        hooks_before=v7_hooks_before,hooks_after=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()),cuda_health=health))

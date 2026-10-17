"""Resume M2/M3 after an observation-only JSON serialization error."""
import shutil
import traceback

def v6_json_default(value):
    if torch.is_tensor(value):return dict(tensor_shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def v6_write(name,value):
    (V6_DIR/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=v6_json_default),encoding='utf-8')

errdir=V6_DIR/'audit_wrapper_error';errdir.mkdir(exist_ok=True)
shutil.copy2(OUT/'uploads/pcebr_stage0_v6_run.py',errdir/'runner_original.py')
if (V6_DIR/'error.json').exists():shutil.move(str(V6_DIR/'error.json'),str(errdir/'error.json'))
# Preserve the interrupted M2 attempt; the completed M1 fits/scores are reused.
for p in (V6_DIR/'artifacts').glob('*.pt'):
    if p.name.startswith(('hop2_bridge_','m2_')):shutil.move(str(p),str(errdir/p.name))
(errdir/'instrumentation.json').write_text(json.dumps(V6_INFO,indent=2,default=v6_json_default),encoding='utf-8')
for key in ('episodes','bridges','probes'):V6_INFO[key]=[]
v6_write('instrumentation.json',V6_INFO)
try:
    V6_STATE['M2']=V6_NS['run_stage0_m2_connected'](V6_BUNDLES['avbind_pooled_pairs'])
    v6_write('progress.json',V6_STATE)
    print('PCV6_M2_COMPLETE',json.dumps(V6_STATE['M2']),flush=True)
    V6_STATE['M3']=V6_NS['run_stage0_m3_bivector_vo_tangent_routing']()
    V6_STATE.update(protocol='Stage-0 v6 final original functions with observation-only artifacts and separate validation',elapsed_sec=round(time.time()-v6_start,2))
    v6_write('main_results.json',V6_STATE)
    print('PCV6_DONE',json.dumps(V6_STATE),flush=True)
except Exception:
    v6_write('error.json',dict(traceback=traceback.format_exc(),elapsed_sec=time.time()-v6_start))
    raise
finally:
    v6_write('progress.json',V6_STATE);v6_write('instrumentation.json',V6_INFO)
    health=[float((torch.ones(1,device=f'cuda:{dev}')+1).cpu()) for dev in range(torch.cuda.device_count())]
    v6_write('completion.json',dict(parameter_versions_unchanged=v6_param_versions=={n:p._version for n,p in MODEL_.named_parameters()},
        hooks_before=v6_hooks_before,hooks_after=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()),cuda_health=health))

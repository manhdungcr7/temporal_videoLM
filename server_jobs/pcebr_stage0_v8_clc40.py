"""Frozen40 confirmation of the micro8 CLC-only ablation; fixed .042 scale, no rollout."""
import ast,gc,hashlib,json,time,torch
import numpy as np
import torch.nn.functional as F
D=OUT/'pcebr_stage0_v8/clc40';D.mkdir(exist_ok=True)
previous=json.loads((D/'samples.json').read_text()) if (D/'samples.json').exists() else []
if (D/'error.json').exists():
    (D/'initial_wrapper_error.json').write_text((D/'error.json').read_text())
source=(OUT/'uploads/stage0_pcebr_v8_corrected_standalone.py').read_text(encoding='utf-8')
assert hashlib.sha256(source.encode()).hexdigest()=='25043c391ac1503280c2e4b70c4e38d27d3f8831741e4a38f8f96ffc4b758843'
tree=ast.parse(source)
tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='RESULT' for v in n.targets))]
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_stage0_m3_bivector_vo_tangent_routing')
# Replace only the fixed four-per-shard limit, not selection, duration or order.
for st in fn.body:
    if isinstance(st,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='samples' for t in st.targets):
        st.value=ast.parse("[r for r in manifest[f'server{server_id}'] if r['idx'] not in V8_COMPLETED]",mode='eval').body
    if isinstance(st,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows_out' for t in st.targets):
        st.value=ast.parse('list(V8_PREVIOUS)',mode='eval').body
loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and any(isinstance(v,ast.Name) and v.id=='s_i' for v in n.target.elts))
def assigned(st,name):
    return isinstance(st,ast.Assign) and any(any(isinstance(x,ast.Name) and x.id==name for x in ast.walk(t)) for t in st.targets)
legacy_begin=next(i for i,st in enumerate(loop.body) if assigned(st,'V18_v5_raw'))
legacy_end=next(i for i,st in enumerate(loop.body) if assigned(st,'worst_k'))
loop.body[legacy_begin:legacy_end]=ast.parse('gate_legacy_v5=0.0').body
cut=next(i for i,st in enumerate(loop.body) if assigned(st,'ct_eff_scale'))
loop.body=loop.body[:cut+1]
loop.body += ast.parse('''
p_nat40 = torch.softmax(lg_nat[labels[:k_opts]],dim=-1)
p_clc40 = torch.softmax(run_caor_routed(mid_layers,prof_full_fwd,eta_scale=0.0,clc_vec_1d=clc_dir_fwd,clc_scale=clc_eff_scale,ct_scale=0.0,use_prefix_cache=use_fast_cache)[labels[:k_opts]],dim=-1)
p_full40 = torch.softmax(run_caor_routed(mid_layers,prof_full_fwd,clc_vec_1d=clc_dir_fwd,clc_scale=clc_eff_scale,ct_vec_1d=z_thought_vec,ct_scale=ct_eff_scale,use_prefix_cache=use_fast_cache)[labels[:k_opts]],dim=-1)
p_caor40 = torch.softmax(lg_caor_full[labels[:k_opts]],dim=-1)
row40 = dict(idx=idx,task=sample['task'],gold=sample['gold'],duration=sample['duration'],gamma=gate_fwd,regime=slot_meta['detected_pair_regime'],endpoint_audit=slot_meta['endpoint_role_split_audit'],
    native='ABCD'[int(p_nat40.argmax())],CLC_only='ABCD'[int(p_clc40.argmax())],full_direct='ABCD'[int(p_full40.argmax())],CAOR_only='ABCD'[int(p_caor40.argmax())],
    probabilities=dict(native=p_nat40.tolist(),CLC_only=p_clc40.tolist(),full_direct=p_full40.tolist(),CAOR_only=p_caor40.tolist()),
    cache_parity_err=cached_routed_parity_err,cache_used=use_fast_cache,native_replay_err=native_replay_err,kernel_noop_err=kernel_noop_err,elapsed_sec=time.time()-t_sample_start)
rows_out.append(row40)
V8_WRITE40(rows_out)
print('V8_CLC40_SAMPLE',json.dumps(row40),flush=True)
gc.collect();torch.cuda.empty_cache()
''').body
loop_i=fn.body.index(loop)
fn.body=fn.body[:loop_i+1]+ast.parse("return dict(n_samples=len(rows_out),samples=rows_out)").body
ast.fix_missing_locations(tree)
def write40(rows):
    (D/'samples.json').write_text(json.dumps(rows,indent=2))
ns=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=PC_SERVER,V8_WRITE40=write40,gc=gc,V8_PREVIOUS=previous,V8_COMPLETED={r['idx'] for r in previous})
exec(compile(tree,'v8_clc40_frozen_protocol','exec'),ns)
versions={n:p._version for n,p in MODEL_.named_parameters()}
t0=time.time()
try:
    result=ns['run_stage0_m3_bivector_vo_tangent_routing']()
    assert result['n_samples']==20,result['n_samples']
    result.update(seconds=time.time()-t0,backbone_unchanged=all(p._version==versions[n] for n,p in MODEL_.named_parameters()),
        protocol='Fixed CLC-only scale .042, CAOR-only and full-direct controls; frozen40 rotation0; no generation/bridge; development evaluation, not a fresh test')
    (D/'completion.json').write_text(json.dumps(result,indent=2))
    RESULT=dict(n=20,scores={k:sum(r[k]==r['gold'] for r in result['samples']) for k in ('native','CLC_only','CAOR_only','full_direct')})
    print('V8_CLC40_DONE',json.dumps(RESULT),flush=True)
except Exception:
    import traceback
    (D/'error.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2));raise

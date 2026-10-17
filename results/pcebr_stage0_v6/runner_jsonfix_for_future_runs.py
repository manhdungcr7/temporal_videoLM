"""Run the supplied final source with observation-only artifact capture."""
import ast
import collections
import hashlib
import json
import time
import traceback
import numpy as np
import torch

V6_DIR = OUT / 'pcebr_stage0_v6'
V6_DIR.mkdir(exist_ok=True)
(V6_DIR / 'artifacts').mkdir(exist_ok=True)
V6_SOURCE = OUT / 'uploads/stage0_pcebr_v6_final_original.py'
V6_ORIGINAL = V6_SOURCE.read_text(encoding='utf-8')
assert hashlib.sha256(V6_ORIGINAL.encode()).hexdigest() == '6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515'
V6_STATE, V6_BUNDLES, V6_M3 = {}, {}, []
V6_INFO = dict(source_lf_sha256=hashlib.sha256(V6_ORIGINAL.encode()).hexdigest(), trained=[], retrieval=[], episodes=[], probes=[], bridges=[])
V6_PREFIX = ''

def v6_cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: v6_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [v6_cpu(v) for v in value]
    return value

def v6_json_default(value):
    if torch.is_tensor(value): return dict(tensor_shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)

def v6_write(name, value):
    (V6_DIR / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,default=v6_json_default), encoding='utf-8')

def v6_tree(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RESULT' for t in n.targets))]
    return tree

V6_V5_TREE = v6_tree((OUT / 'uploads/stage0_pcebr_v5_original.py').read_text(encoding='utf-8'))
V6_V5_NS = dict(OUT=OUT, MODEL_=MODEL_, PROC=PROC, LAB_SERVER=PC_SERVER)
exec(compile(V6_V5_TREE, 'archived_v5_definitions', 'exec'), V6_V5_NS)
V6_OLD_M3_AST = next(n for n in V6_V5_TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'run_stage0_m3_bivector_vo_tangent_routing')
V6_OLD_ROUTE_AST = next(n for n in ast.walk(V6_OLD_M3_AST) if isinstance(n, ast.FunctionDef) and n.name == 'run_routed')

def v6_capture_probe(v):
    name = f'm2_probe_{len(V6_INFO["probes"]):02d}.pt'
    keys = ('X_tr', 'y_tr', 'X_he', 'y_he', 'mu', 'sd', 'valid_col', 'logits_he', 'preds', 'preds_3way')
    raw = {k: v6_cpu(v[k]) for k in keys if k in v}
    raw.update(state_dict=v6_cpu(v['clf'].state_dict()), heldout_metadata=[{k: e.get(k) for k in ('label', 'dir_tag', 'pair_key')} for e in v['he_eps']])
    torch.save(raw, V6_DIR / 'artifacts' / name)
    row = dict(file=name, n_train=len(v['tr_eps']), n_eval=len(v['he_eps']), metrics=v['out'], paired_hits=v['paired_hits'], paired_total=v['paired_tot'])
    V6_INFO['probes'].append(row)
    v6_write('instrumentation.json', V6_INFO)
    print('PCV6_PROBE', json.dumps(row), flush=True)

def v6_capture_m3(v):
    name = f'm3_{v["idx"]}.pt'
    keys = ('V18_raw', 'A18_raw', 'V18_vo', 'A18_vo', 'lg_v', 'lg_a', 'iv_t', 'vec_fwd', 'vec_rev', 'phi_fwd', 'phi_rev',
            'V18_v5_raw', 'A18_v5_raw', 'lg_v_v5', 'lg_a_v5', 'iv_t_v5', 'vec_legacy_v5')
    raw = {k: v6_cpu(v[k]) for k in keys}
    raw.update({k: v6_cpu(x) for k, x in v.items() if k.startswith('p_') and torch.is_tensor(x)})
    raw.update({k: v6_cpu(v[k]) for k in ('slot_meta', 'aud_fwd', 'aud_rev', 'clock_info', 'deltas_fwd', 'deltas_rev', 'profs_rand', 'prof_full_fwd', 'prof_zero_noop')})
    old = V6_V5_NS
    old_clock = old['audit_and_interpolate_grid_clocks'](v['payload'], v['sample'])
    old_rows = old['find_option_free_and_option_spans'](v['payload'], v['stem_text'], PROC.tokenizer)
    old_banks = old['extract_query_conditioned_salient_slots'](v['h_post_block_cap'][18], v['h_post_block_cap'][27], old_rows[0], old_clock, top_patch_frac=0.15)
    old_V = old['project_to_layer_vo_subspace'](MODEL_.model.layers[18], old_banks[0])
    old_A = old['project_to_layer_vo_subspace'](MODEL_.model.layers[18], old_banks[1])
    old_vec, old_gate, _, _ = old['build_orthogonal_bivector_routed_state'](old_V, old_A, *old_banks[2:], reverse_timeline=False)
    old_scope = dict(old, model_llm=MODEL_, kw=v['kw'], eff_scale=0.045*old_gate, hook_rows_t=torch.tensor(old_rows[3], dtype=torch.long))
    exec(compile(ast.Module(body=[V6_OLD_ROUTE_AST], type_ignores=[]), 'archived_v5_route', 'exec'), old_scope)
    old_logits = old_scope['run_routed'](v['mid_layers'], old_vec)
    old_prob = torch.softmax(old_logits[v['labels'][:v['k_opts']]], dim=-1)
    raw.update(legacy_archived_source_logits=old_logits, legacy_archived_source_prob=old_prob, legacy_archived_source_vec=old_vec)
    torch.save(raw, V6_DIR / 'artifacts' / name)
    parity = dict(probability_max_abs_error=float((old_prob-v['p_legacy_v5']).abs().max()),
                  vector_max_abs_error=float((old_vec-v['vec_legacy_v5']).abs().max()),
                  gate_abs_error=abs(old_gate-v['gate_legacy_v5']), hook_rows_equal=old_rows[3]==v['hook_rows'])
    tok = PROC.tokenizer
    tail_ids = v['payload']['input_ids'][0, v['last_m']+1:].tolist()
    tail = tok.decode(tail_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    row = dict(result=v['rows_out'][-1], raw_file=name, legacy_same_runtime_archived_source_parity=parity,
               legacy_parity_kind='same-input same-runtime original v5 source replay; historical bytewise parity not presumed',
               tail_roundtrip=tok(tail, add_special_tokens=False)['input_ids']==tail_ids,
               hook_option_disjoint=not bool(set(v['hook_rows']) & set(v['option_rows'])),
               freeform_input_text=tok.decode(v['payload']['input_ids'][0,v['keep_indices']], skip_special_tokens=False),
               probabilities={k:x.tolist() for k,x in raw.items() if k.startswith('p_') and torch.is_tensor(x)})
    V6_M3.append(row)
    v6_write('m3_audit.json', V6_M3)
    print('PCV6_M3_SAMPLE', json.dumps(row['result']), flush=True)

V6_TREE = v6_tree(V6_ORIGINAL)
for node in ast.walk(V6_TREE):
    if isinstance(node, ast.FunctionDef) and node.name == 'eval_linear_probe':
        node.body.insert(-1, ast.Expr(value=ast.Call(func=ast.Name(id='V6_CAPTURE_PROBE', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])))
    if isinstance(node, ast.For):
        for i, statement in enumerate(list(node.body)):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                c = statement.value
                if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id == 'rows_out' and c.func.attr == 'append':
                    node.body.insert(i+1, ast.Expr(value=ast.Call(func=ast.Name(id='V6_CAPTURE_M3', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])))
ast.fix_missing_locations(V6_TREE)
V6_NS = dict(OUT=OUT, MODEL_=MODEL_, PROC=PROC, LAB_SERVER=PC_SERVER, V6_CAPTURE_PROBE=v6_capture_probe, V6_CAPTURE_M3=v6_capture_m3)
exec(compile(V6_TREE, str(V6_SOURCE), 'exec'), V6_NS)

v6_orig_train = V6_NS['train_single_stream_aligner']
def v6_train(*args, **kwargs):
    model = v6_orig_train(*args, **kwargs)
    name = f'{V6_PREFIX}_model_{len(V6_INFO["trained"]):02d}.pt'
    torch.save(dict(state_dict=v6_cpu(model.state_dict()), kwargs=kwargs, whiten_mode=model.whiten_mode,
                    zca_txt=v6_cpu(model.zca_txt), zca_med=v6_cpu(model.zca_med)), V6_DIR/'artifacts'/name)
    V6_INFO['trained'].append(dict(file=name, kwargs=kwargs, parameters=sum(p.numel() for p in model.parameters())))
    v6_write('instrumentation.json', V6_INFO)
    return model
V6_NS['train_single_stream_aligner'] = v6_train

v6_orig_bridge = V6_NS['train_two_hop_bridge_on_train_split']
def v6_bridge(*args, **kwargs):
    model = v6_orig_bridge(*args, **kwargs)
    name = f'hop2_bridge_{len(V6_INFO["bridges"]):02d}.pt'
    torch.save(dict(state_dict=v6_cpu(model.state_dict()), kwargs=kwargs), V6_DIR/'artifacts'/name)
    V6_INFO['bridges'].append(dict(file=name, parameters=sum(p.numel() for p in model.parameters()), kwargs=kwargs))
    return model
V6_NS['train_two_hop_bridge_on_train_split'] = v6_bridge

v6_orig_eval = V6_NS['evaluate_sim_matrix']
def v6_eval(sim, examples):
    metrics = v6_orig_eval(sim, examples)
    name = f'{V6_PREFIX}_retrieval_{len(V6_INFO["retrieval"]):03d}.pt'
    torch.save(dict(sim=v6_cpu(sim), metrics=metrics, examples=examples), V6_DIR/'artifacts'/name)
    V6_INFO['retrieval'].append(dict(file=name, prefix=V6_PREFIX, metrics=metrics))
    return metrics
V6_NS['evaluate_sim_matrix'] = v6_eval

v6_orig_build = V6_NS['build_connected_m2_episodes']
def v6_build(*args, **kwargs):
    episodes, audit = v6_orig_build(*args, **kwargs)
    name = f'm2_episodes_{len(V6_INFO["episodes"]):02d}.pt'
    torch.save(dict(episodes=v6_cpu(episodes), kwargs=kwargs, audit=audit), V6_DIR/'artifacts'/name)
    V6_INFO['episodes'].append(dict(file=name, n=len(episodes), labels=dict(collections.Counter(e['label'] for e in episodes)), kwargs=kwargs, audit=audit))
    v6_write('instrumentation.json', V6_INFO)
    return episodes, audit
V6_NS['build_connected_m2_episodes'] = v6_build

v6_param_versions = {n:p._version for n,p in MODEL_.named_parameters()}
v6_hooks_before = sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules())
v6_write('input_audit.json', dict(server=PC_SERVER, runtime_out=str(OUT), torch=torch.__version__, model=MODEL_.config._name_or_path,
    source_lf_sha256=V6_INFO['source_lf_sha256'], caches={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'uploads').glob('avbind_*pairs_s*.pt')},
    manifest_sha256=json.loads((OUT/'uploads/research40_manifest.json').read_text())['manifest_sha256']))
v6_start = time.time()
try:
    V6_STATE['M0'] = V6_NS['run_algebraic_unit_checks']('cuda:0')
    # Independent validation only; no change to the supplied feature computation.
    iv1=torch.tensor([[0.,4.],[4.,8.]])
    iv2=torch.tensor([[1.,3.],[3.,8.]])
    phi=V6_NS['compute_wasserstein_synchrony_relational_features'](torch.tensor([.5,.5]),torch.tensor([1.,0.]),iv1,iv2,torch.tensor(1.))
    grid=np.linspace(0.,8.,240001)
    d=grid/8.-np.clip((grid-1.)/2.,0.,1.)
    dense=np.array([np.trapezoid(np.maximum(d,0.),grid)/8.,np.trapezoid(np.maximum(-d,0.),grid)/8.])
    analytical=np.array([1./96.,25./96.])
    extra=dict(computed=phi[3:5].tolist(), analytic_expected=analytical.tolist(), fine_grid=dense.tolist(),
               max_error_analytic=float(np.max(np.abs(phi[3:5].numpy()-analytical))),
               max_error_fine_grid=float(np.max(np.abs(phi[3:5].numpy()-dense))), points=len(grid))
    assert extra['max_error_analytic']<1e-6 and extra['max_error_fine_grid']<1e-6
    v6_write('extra_unit_audit.json', extra)
    v6_write('progress.json', V6_STATE)
    print('PCV6_M0_COMPLETE',json.dumps(V6_STATE['M0']),flush=True)
    for V6_PREFIX in ('avbind_native_pairs','avbind_pooled_pairs'):
        result,bundle=V6_NS['run_stage0_m1_orthogonal_factorial'](V6_PREFIX)
        V6_STATE['M1_native' if 'native' in V6_PREFIX else 'M1_pooled']=result
        V6_BUNDLES[V6_PREFIX]=bundle
        torch.save(v6_cpu({k:x for k,x in bundle.items() if not k.startswith('m_')}),V6_DIR/'artifacts'/f'{V6_PREFIX}_bundle.pt')
        v6_write('progress.json',V6_STATE)
        print('PCV6_M1_COMPLETE',V6_PREFIX,json.dumps(result),flush=True)
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
    v6_write('progress.json',V6_STATE)
    v6_write('instrumentation.json',V6_INFO)
    health=[]
    for dev in range(torch.cuda.device_count()):
        health.append(float((torch.ones(1,device=f'cuda:{dev}')+1).cpu()))
    v6_write('completion.json',dict(parameter_versions_unchanged=v6_param_versions=={n:p._version for n,p in MODEL_.named_parameters()},
        hooks_before=v6_hooks_before,hooks_after=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()),cuda_health=health))

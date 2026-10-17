"""Run the supplied final source with observation-only artifact capture."""
import ast
import collections
import hashlib
import json
import time
import traceback
import numpy as np
import torch

V7_DIR = OUT / 'pcebr_stage0_v7'
V7_DIR.mkdir(exist_ok=True)
(V7_DIR / 'artifacts').mkdir(exist_ok=True)
V7_SOURCE = OUT / 'uploads/stage0_pcebr_v7_final_original.py'
V7_ORIGINAL = V7_SOURCE.read_text(encoding='utf-8')
assert hashlib.sha256(V7_ORIGINAL.encode()).hexdigest() == 'd27eb87a7014e941bd7a7a920dbe3ce074338cf616757608cc476712d7c8ee80'
V7_STATE, V7_BUNDLES, V7_M3 = {}, {}, []
V7_INFO = dict(source_lf_sha256=hashlib.sha256(V7_ORIGINAL.encode()).hexdigest(), trained=[], retrieval=[], episodes=[], probes=[], bridges=[])
V7_PREFIX = ''

def v7_cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: v7_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [v7_cpu(v) for v in value]
    return value

def v7_json_default(value):
    if torch.is_tensor(value): return dict(tensor_shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)

def v7_write(name, value):
    (V7_DIR / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,default=v7_json_default), encoding='utf-8')

def v7_tree(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RESULT' for t in n.targets))]
    return tree

V7_V5_TREE = v7_tree((OUT / 'uploads/stage0_pcebr_v5_original.py').read_text(encoding='utf-8'))
V7_V5_NS = dict(OUT=OUT, MODEL_=MODEL_, PROC=PROC, LAB_SERVER=PC_SERVER)
exec(compile(V7_V5_TREE, 'archived_v5_definitions', 'exec'), V7_V5_NS)
V7_OLD_M3_AST = next(n for n in V7_V5_TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'run_stage0_m3_bivector_vo_tangent_routing')
V7_OLD_ROUTE_AST = next(n for n in ast.walk(V7_OLD_M3_AST) if isinstance(n, ast.FunctionDef) and n.name == 'run_routed')

def v7_capture_probe(v):
    if 'clf' not in v:
        return
    name = f'm2_probe_{len(V7_INFO["probes"]):02d}.pt'
    keys = ('X_tr', 'y_tr', 'X_he', 'y_he', 'mu', 'sd', 'valid_col', 'logits_he', 'preds', 'preds_3way')
    raw = {k: v7_cpu(v[k]) for k in keys if k in v}
    raw.update(state_dict=v7_cpu(v['clf'].state_dict()), heldout_metadata=[{k: e.get(k) for k in ('label', 'dir_tag', 'pair_key')} for e in v['he_eps']])
    torch.save(raw, V7_DIR / 'artifacts' / name)
    row = dict(file=name, n_train=len(v['tr_eps']), n_eval=len(v['he_eps']), metrics=v['out'], paired_hits=v['paired_hits'], paired_total=v['paired_tot'])
    V7_INFO['probes'].append(row)
    v7_write('instrumentation.json', V7_INFO)
    print('PCV7_PROBE', json.dumps(row), flush=True)

def v7_capture_m3(v):
    name = f'm3_{v["idx"]}.pt'
    keys = ('dustbin_theta', 'clc_dir_fwd', 'clc_dir_rev', 'h22_supp', 'h22_foil', 'h22_rev', 'dh_ev', 'dh_ord', 'V18_raw', 'A18_raw', 'V18_vo', 'A18_vo', 'lg_v', 'lg_a', 'iv_t', 'vec_fwd', 'vec_rev', 'phi_fwd', 'phi_rev',
            'V18_v5_raw', 'A18_v5_raw', 'lg_v_v5', 'lg_a_v5', 'iv_t_v5', 'vec_legacy_v5')
    raw = {k: v7_cpu(v[k]) for k in keys if k in v}
    raw.update({k: v7_cpu(x) for k, x in v.items() if k.startswith('p_') and torch.is_tensor(x)})
    raw.update({k: v7_cpu(v[k]) for k in ('slot_meta', 'aud_fwd', 'aud_rev', 'clock_info', 'deltas_fwd', 'deltas_rev', 'profs_rand', 'prof_full_fwd', 'prof_zero_noop')})
    old = V7_V5_NS
    old_clock = old['audit_and_interpolate_grid_clocks'](v['payload'], v['sample'])
    old_rows = old['find_option_free_and_option_spans'](v['payload'], v['stem_text'], PROC.tokenizer)
    old_banks = old['extract_query_conditioned_salient_slots'](v['h_post_block_cap'][18], v['h_post_block_cap'][27], old_rows[0], old_clock, top_patch_frac=0.15)
    old_V = old['project_to_layer_vo_subspace'](MODEL_.model.layers[18], old_banks[0])
    old_A = old['project_to_layer_vo_subspace'](MODEL_.model.layers[18], old_banks[1])
    old_vec, old_gate, _, _ = old['build_orthogonal_bivector_routed_state'](old_V, old_A, *old_banks[2:], reverse_timeline=False)
    old_scope = dict(old, model_llm=MODEL_, kw=v['kw'], eff_scale=0.045*old_gate, hook_rows_t=torch.tensor(old_rows[3], dtype=torch.long))
    exec(compile(ast.Module(body=[V7_OLD_ROUTE_AST], type_ignores=[]), 'archived_v5_route', 'exec'), old_scope)
    old_logits = old_scope['run_routed'](v['mid_layers'], old_vec)
    old_prob = torch.softmax(old_logits[v['labels'][:v['k_opts']]], dim=-1)
    raw.update(legacy_archived_source_logits=old_logits, legacy_archived_source_prob=old_prob, legacy_archived_source_vec=old_vec)
    torch.save(raw, V7_DIR / 'artifacts' / name)
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
    V7_M3.append(row)
    v7_write('m3_audit.json', V7_M3)
    print('PCV7_M3_SAMPLE', json.dumps(row['result']), flush=True)

V7_TREE = v7_tree(V7_ORIGINAL)
class V7Observe(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name == 'eval_linear_probe':
            class ProbeReturns(ast.NodeTransformer):
                def visit_Return(self,ret):
                    call=ast.parse('V7_CAPTURE_PROBE(locals())').body[0]
                    return [call,ret]
            node=ProbeReturns().visit(node)
        return self.generic_visit(node)
    def visit_For(self,node):
        self.generic_visit(node)
        body=[]
        for statement in node.body:
            body.append(statement)
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
                c=statement.value
                if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='rows_out' and c.func.attr=='append':
                    body.append(ast.parse('V7_CAPTURE_M3(locals())').body[0])
        node.body=body
        return node
V7_TREE=V7Observe().visit(V7_TREE)
ast.fix_missing_locations(V7_TREE)
V7_NS = dict(OUT=OUT, MODEL_=MODEL_, PROC=PROC, LAB_SERVER=PC_SERVER, V7_CAPTURE_PROBE=v7_capture_probe, V7_CAPTURE_M3=v7_capture_m3)
exec(compile(V7_TREE, str(V7_SOURCE), 'exec'), V7_NS)

v7_orig_train = V7_NS['train_single_stream_aligner']
def v7_train(*args, **kwargs):
    model = v7_orig_train(*args, **kwargs)
    name = f'{V7_PREFIX}_model_{len(V7_INFO["trained"]):02d}.pt'
    torch.save(dict(state_dict=v7_cpu(model.state_dict()), kwargs=kwargs, whiten_mode=model.whiten_mode,
                    zca_txt=v7_cpu(model.zca_txt), zca_med=v7_cpu(model.zca_med)), V7_DIR/'artifacts'/name)
    V7_INFO['trained'].append(dict(file=name, kwargs=kwargs, parameters=sum(p.numel() for p in model.parameters())))
    v7_write('instrumentation.json', V7_INFO)
    return model
V7_NS['train_single_stream_aligner'] = v7_train

v7_orig_bridge = V7_NS['train_two_hop_bridge_on_train_split']
def v7_bridge(*args, **kwargs):
    model = v7_orig_bridge(*args, **kwargs)
    name = f'hop2_bridge_{len(V7_INFO["bridges"]):02d}.pt'
    torch.save(dict(state_dict=v7_cpu(model.state_dict()), kwargs=kwargs), V7_DIR/'artifacts'/name)
    V7_INFO['bridges'].append(dict(file=name, parameters=sum(p.numel() for p in model.parameters()), kwargs=kwargs))
    return model
V7_NS['train_two_hop_bridge_on_train_split'] = v7_bridge

v7_orig_eval = V7_NS['evaluate_sim_matrix']
def v7_eval(sim, examples):
    metrics = v7_orig_eval(sim, examples)
    name = f'{V7_PREFIX}_retrieval_{len(V7_INFO["retrieval"]):03d}.pt'
    torch.save(dict(sim=v7_cpu(sim), metrics=metrics, examples=examples), V7_DIR/'artifacts'/name)
    V7_INFO['retrieval'].append(dict(file=name, prefix=V7_PREFIX, metrics=metrics))
    return metrics
V7_NS['evaluate_sim_matrix'] = v7_eval

v7_orig_build = V7_NS['build_connected_m2_episodes']
def v7_build(*args, **kwargs):
    episodes, audit = v7_orig_build(*args, **kwargs)
    name = f'm2_episodes_{len(V7_INFO["episodes"]):02d}.pt'
    torch.save(dict(episodes=v7_cpu(episodes), kwargs=kwargs, audit=audit), V7_DIR/'artifacts'/name)
    V7_INFO['episodes'].append(dict(file=name, n=len(episodes), labels=dict(collections.Counter(e['label'] for e in episodes)), kwargs=kwargs, audit=audit))
    v7_write('instrumentation.json', V7_INFO)
    return episodes, audit
V7_NS['build_connected_m2_episodes'] = v7_build

v7_param_versions = {n:p._version for n,p in MODEL_.named_parameters()}
v7_hooks_before = sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules())
v7_write('input_audit.json', dict(server=PC_SERVER, runtime_out=str(OUT), torch=torch.__version__, model=MODEL_.config._name_or_path,
    source_lf_sha256=V7_INFO['source_lf_sha256'], caches={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'uploads').glob('avbind_*pairs_s*.pt')},
    manifest_sha256=json.loads((OUT/'uploads/research40_manifest.json').read_text())['manifest_sha256']))
v7_start = time.time()
try:
    V7_STATE['M0'] = V7_NS['run_algebraic_unit_checks']('cuda:0')
    v7_write('progress.json', V7_STATE)
    print('PCV7_M0_COMPLETE',json.dumps(V7_STATE['M0']),flush=True)
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

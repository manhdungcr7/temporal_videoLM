"""Execute unchanged supplied v5 mathematics, saving auditable intermediate results."""
import ast
import collections
import hashlib
import json
import time
import torch

V5_DIR = OUT / 'pcebr_stage0_v5'
V5_DIR.mkdir(exist_ok=True)
(V5_DIR / 'artifacts').mkdir(exist_ok=True)
V5_SOURCE = OUT / 'uploads/stage0_pcebr_v5_original.py'
V5_TREE = ast.parse(V5_SOURCE.read_text(encoding='utf-8'))
V5_TREE.body = [n for n in V5_TREE.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RESULT' for t in n.targets))]
V5_NS = dict(OUT=OUT, MODEL_=MODEL_, PROC=PROC, LAB_SERVER=PC_SERVER, __name__='pcebr_v5_original')
V5_STATE = {}
V5_BUNDLES = {}
V5_PREFIX = ''
V5_INFO = dict(source_sha256=hashlib.sha256(V5_SOURCE.read_bytes()).hexdigest(), trained=[], retrieval=[], episodes=[], probes=[])
V5_M3 = []

def v5_cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:v5_cpu_tree(v) for k,v in value.items()}
    return value

def v5_write(name, data):
    (V5_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def v5_capture_probe(v):
    name = f'm2_probe_{len(V5_INFO["probes"]):02d}.pt'
    keys = ('X_tr', 'y_tr', 'X_he', 'y_he', 'mu', 'sd', 'valid_col', 'logits_he', 'preds', 'preds_3way')
    raw = {k: v[k].detach().cpu() for k in keys if k in v}
    raw.update(state_dict={k: x.detach().cpu() for k, x in v['clf'].state_dict().items()},
               heldout_metadata=[{k: e[k] for k in ('label', 'dir_tag', 'pair_key')} for e in v['he_eps']])
    torch.save(raw, V5_DIR / 'artifacts' / name)
    row = dict(file=name, n_train=len(v['tr_eps']), n_eval=len(v['he_eps']), metrics=v['out'], paired_hits=v['paired_hits'], paired_total=v['paired_tot'])
    V5_INFO['probes'].append(row)
    v5_write('instrumentation.json', V5_INFO)
    print('PCV5_PROBE', json.dumps(row), flush=True)

def v5_capture_m3(v):
    name = f'm3_{v["idx"]}.pt'
    tensor_keys = ('V18_raw', 'A18_raw', 'V18_vo', 'A18_vo', 'lg_v', 'lg_a', 'iv_t', 'vec_fwd', 'vec_rev', 'phi_fwd', 'phi_rev', 'p_nat', 'p_late_fwd', 'p_late_rev', 'p_mid_fwd', 'p_mid_rev', 'p_mid_irrel')
    raw = {k: v[k].detach().cpu() for k in tensor_keys}
    raw['clock_info'] = v['clock_info']
    raw['random_vectors'] = v['rand_vecs']
    raw['aud_fwd']=v5_cpu_tree(v['aud_fwd'])
    raw['aud_rev']=v5_cpu_tree(v['aud_rev'])
    raw['q18_true']=v5_cpu_tree(v['h_cap'][18][v['stem_rows']])
    raw['q18_wrong']=v5_cpu_tree(v['q18_wrong'])
    torch.save(raw, V5_DIR / 'artifacts' / name)
    payload = v['payload']
    tok = PROC.tokenizer
    tail_ids = payload['input_ids'][0, v['last_m'] + 1:].tolist()
    tail = tok.decode(tail_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    row = dict(result=v['rows_out'][-1], raw_file=name,
               tail_roundtrip=tok(tail, add_special_tokens=False)['input_ids'] == tail_ids,
               hook_rows=v['hook_rows'], option_rows=v['option_rows'],
               freeform_input_text=tok.decode(payload['input_ids'][0, v['keep_indices']], skip_special_tokens=False),
               vec_reverse_max_abs=float((v['vec_fwd'] - v['vec_rev']).abs().max()),
               probabilities={k: v[k].tolist() for k in ('p_nat', 'p_late_fwd', 'p_late_rev', 'p_mid_fwd', 'p_mid_rev', 'p_mid_irrel')})
    V5_M3.append(row)
    v5_write('m3_audit.json', V5_M3)
    print('PCV5_M3_SAMPLE', json.dumps(row['result']), flush=True)

for node in ast.walk(V5_TREE):
    if isinstance(node, ast.FunctionDef) and node.name == 'eval_linear_probe':
        node.body.insert(-1, ast.Expr(value=ast.Call(func=ast.Name(id='V5_CAPTURE_PROBE', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])))
    if isinstance(node, ast.For):
        for i, statement in enumerate(list(node.body)):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                c = statement.value
                if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id == 'rows_out' and c.func.attr == 'append':
                    node.body.insert(i + 1, ast.Expr(value=ast.Call(func=ast.Name(id='V5_CAPTURE_M3', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])))
ast.fix_missing_locations(V5_TREE)
V5_NS.update(V5_CAPTURE_PROBE=v5_capture_probe, V5_CAPTURE_M3=v5_capture_m3)
exec(compile(V5_TREE, str(V5_SOURCE), 'exec'), V5_NS)

v5_orig_train = V5_NS['train_single_stream_aligner']
def v5_train(*args, **kwargs):
    model = v5_orig_train(*args, **kwargs)
    name = f'{V5_PREFIX}_model_{len(V5_INFO["trained"]):02d}.pt'
    torch.save(dict(state_dict={k: x.detach().cpu() for k, x in model.state_dict().items()}, kwargs=kwargs, whiten_mode=model.whiten_mode, zca_txt=v5_cpu_tree(model.zca_txt), zca_med=v5_cpu_tree(model.zca_med)), V5_DIR / 'artifacts' / name)
    V5_INFO['trained'].append(dict(file=name, kwargs=kwargs))
    v5_write('instrumentation.json', V5_INFO)
    print('PCV5_TRAIN', name, flush=True)
    return model
V5_NS['train_single_stream_aligner'] = v5_train

v5_orig_bridge=V5_NS['train_two_hop_bridge_on_train_split']
def v5_bridge(*args,**kwargs):
    model=v5_orig_bridge(*args,**kwargs)
    name=f'hop2_bridge_{len(V5_INFO.setdefault("bridges",[])):02d}.pt'
    torch.save(dict(state_dict=v5_cpu_tree(model.state_dict()),kwargs=kwargs),V5_DIR/'artifacts'/name)
    V5_INFO['bridges'].append(dict(file=name,parameters=sum(p.numel() for p in model.parameters())))
    v5_write('instrumentation.json',V5_INFO)
    print('PCV5_BRIDGE',name,flush=True)
    return model
V5_NS['train_two_hop_bridge_on_train_split']=v5_bridge

v5_orig_eval = V5_NS['evaluate_sim_matrix']
def v5_eval(sim, examples):
    result = v5_orig_eval(sim, examples)
    name = f'{V5_PREFIX}_retrieval_{len(V5_INFO["retrieval"]):03d}.pt'
    torch.save(dict(sim=sim.detach().cpu(), metrics=result, examples=[{k: x for k, x in e.items() if k not in ('raw', 'text')} for e in examples]), V5_DIR / 'artifacts' / name)
    V5_INFO['retrieval'].append(dict(file=name, prefix=V5_PREFIX, metrics=result))
    print('PCV5_RETRIEVAL', name, json.dumps(result), flush=True)
    return result
V5_NS['evaluate_sim_matrix'] = v5_eval

v5_orig_build = V5_NS['build_connected_m2_episodes']
def v5_build(*args, **kwargs):
    episodes, audit = v5_orig_build(*args, **kwargs)
    name = f'm2_episodes_{len(V5_INFO["episodes"]):02d}.pt'
    raw = {k: torch.stack([e[k].detach().cpu() for e in episodes]) for k in ('phi', 'null_feat', 'std_attn')} if episodes else {}
    raw.update(metadata=[{k: e[k] for k in ('label', 'dir_tag', 'pair_key')} for e in episodes], kwargs=kwargs, foil_audit=audit)
    torch.save(raw, V5_DIR / 'artifacts' / name)
    row = dict(file=name, n=len(episodes), labels=dict(collections.Counter(e['label'] for e in episodes)), kwargs=kwargs, foil_audit=audit)
    V5_INFO['episodes'].append(row)
    v5_write('instrumentation.json', V5_INFO)
    print('PCV5_EPISODES', json.dumps(row), flush=True)
    return episodes, audit
V5_NS['build_connected_m2_episodes'] = v5_build

v5_input = dict(server=PC_SERVER, runtime_out=str(OUT), torch=torch.__version__, model=MODEL_.config._name_or_path,
                source_sha256=V5_INFO['source_sha256'], caches={})
for prefix in ('avbind_native_pairs', 'avbind_pooled_pairs'):
    for shard in (1, 2):
        path = OUT / 'uploads' / f'{prefix}_s{shard}.pt'
        v5_input['caches'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
v5_write('input_audit.json', v5_input)
v5_start = time.time()
try:
    V5_STATE['Algebraic_Bivector_and_ZeroFloor_Unit_Checks'] = V5_NS['run_algebraic_unit_checks']('cuda:0')
    for V5_PREFIX in ('avbind_native_pairs', 'avbind_pooled_pairs'):
        result, bundle = V5_NS['run_stage0_m1_orthogonal_factorial'](V5_PREFIX)
        V5_STATE['M1_native_orthogonal_factorial' if V5_PREFIX == 'avbind_native_pairs' else 'M1_pooled_orthogonal_factorial'] = result
        V5_BUNDLES[V5_PREFIX] = bundle
        torch.save(dict(sim_train=bundle['sim_tr_c1'].cpu(), sim_heldout=bundle['sim_he_c1'].cpu(), train_ex=bundle['train_ex'], held_ex=bundle['held_ex']), V5_DIR / 'artifacts' / f'{V5_PREFIX}_bundle.pt')
        v5_write('progress.json', V5_STATE)
        print('PCV5_M1_COMPLETE', V5_PREFIX, json.dumps(result), flush=True)
    V5_STATE['M2_incremental_ablation_and_learned_hop2'] = V5_NS['run_stage0_m2_connected'](V5_BUNDLES['avbind_pooled_pairs'])
    v5_write('progress.json', V5_STATE)
    print('PCV5_M2_COMPLETE', json.dumps(V5_STATE['M2_incremental_ablation_and_learned_hop2']), flush=True)
    V5_STATE['M3_bivector_vo_tangent_routing'] = V5_NS['run_stage0_m3_bivector_vo_tangent_routing']()
    V5_STATE.update(protocol='Stage-0 v5 original functions with observation-only checkpoints', elapsed_sec=round(time.time() - v5_start, 2))
    v5_write('main_results.json', V5_STATE)
    print('PCV5_DONE', json.dumps(V5_STATE), flush=True)
finally:
    v5_write('progress.json', V5_STATE)
    v5_write('instrumentation.json', V5_INFO)

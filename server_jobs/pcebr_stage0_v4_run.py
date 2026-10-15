"""Execute unchanged supplied v4 mathematics, saving auditable intermediate results."""
import ast
import collections
import hashlib
import json
import time
import torch

V4_DIR = OUT / 'pcebr_stage0_v4'
V4_DIR.mkdir(exist_ok=True)
(V4_DIR / 'artifacts').mkdir(exist_ok=True)
V4_SOURCE = OUT / 'uploads/stage0_pcebr_v4_original.py'
V4_TREE = ast.parse(V4_SOURCE.read_text(encoding='utf-8'))
V4_TREE.body = [n for n in V4_TREE.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RESULT' for t in n.targets))]
V4_NS = dict(OUT=OUT, MODEL_=MODEL_, PROC=PROC, LAB_SERVER=PC_SERVER, __name__='pcebr_v4_original')
V4_STATE = {}
V4_BUNDLES = {}
V4_PREFIX = ''
V4_INFO = dict(source_sha256=hashlib.sha256(V4_SOURCE.read_bytes()).hexdigest(), trained=[], retrieval=[], episodes=[], probes=[])
V4_M3 = []

def v4_write(name, data):
    (V4_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def v4_capture_probe(v):
    name = f'm2_probe_{len(V4_INFO["probes"]):02d}.pt'
    keys = ('X_tr', 'y_tr', 'X_he', 'y_he', 'mu', 'sd', 'valid_col', 'logits_he', 'preds', 'preds_3way')
    raw = {k: v[k].detach().cpu() for k in keys if k in v}
    raw.update(state_dict={k: x.detach().cpu() for k, x in v['clf'].state_dict().items()},
               heldout_metadata=[{k: e[k] for k in ('label', 'dir_tag', 'pair_key')} for e in v['he_eps']])
    torch.save(raw, V4_DIR / 'artifacts' / name)
    row = dict(file=name, n_train=len(v['tr_eps']), n_eval=len(v['he_eps']), metrics=v['out'], paired_hits=v['paired_hits'], paired_total=v['paired_tot'])
    V4_INFO['probes'].append(row)
    v4_write('instrumentation.json', V4_INFO)
    print('PCV4_PROBE', json.dumps(row), flush=True)

def v4_capture_m3(v):
    name = f'm3_{v["idx"]}.pt'
    tensor_keys = ('V18_bank', 'A18_bank', 'lg_v', 'lg_a', 'iv_t', 'vec_fwd', 'vec_rev', 'phi_fwd', 'phi_rev', 'p_nat', 'p_late_fwd', 'p_late_rev', 'p_mid_fwd', 'p_mid_rev', 'p_mid_irrel')
    raw = {k: v[k].detach().cpu() for k in tensor_keys}
    raw['clock_info'] = v['clock_info']
    raw['random_vectors'] = v['rand_vecs']
    torch.save(raw, V4_DIR / 'artifacts' / name)
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
    V4_M3.append(row)
    v4_write('m3_audit.json', V4_M3)
    print('PCV4_M3_SAMPLE', json.dumps(row['result']), flush=True)

for node in ast.walk(V4_TREE):
    if isinstance(node, ast.FunctionDef) and node.name == 'eval_linear_probe':
        node.body.insert(-1, ast.Expr(value=ast.Call(func=ast.Name(id='V4_CAPTURE_PROBE', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])))
    if isinstance(node, ast.For):
        for i, statement in enumerate(list(node.body)):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                c = statement.value
                if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id == 'rows_out' and c.func.attr == 'append':
                    node.body.insert(i + 1, ast.Expr(value=ast.Call(func=ast.Name(id='V4_CAPTURE_M3', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])))
ast.fix_missing_locations(V4_TREE)
V4_NS.update(V4_CAPTURE_PROBE=v4_capture_probe, V4_CAPTURE_M3=v4_capture_m3)
exec(compile(V4_TREE, str(V4_SOURCE), 'exec'), V4_NS)

v4_orig_train = V4_NS['train_single_stream_aligner']
def v4_train(*args, **kwargs):
    model = v4_orig_train(*args, **kwargs)
    name = f'{V4_PREFIX}_model_{len(V4_INFO["trained"]):02d}.pt'
    torch.save(dict(state_dict={k: x.detach().cpu() for k, x in model.state_dict().items()}, kwargs=kwargs, use_whitening=model.use_whitening), V4_DIR / 'artifacts' / name)
    V4_INFO['trained'].append(dict(file=name, kwargs=kwargs))
    v4_write('instrumentation.json', V4_INFO)
    print('PCV4_TRAIN', name, flush=True)
    return model
V4_NS['train_single_stream_aligner'] = v4_train

v4_orig_eval = V4_NS['evaluate_sim_matrix']
def v4_eval(sim, examples):
    result = v4_orig_eval(sim, examples)
    name = f'{V4_PREFIX}_retrieval_{len(V4_INFO["retrieval"]):03d}.pt'
    torch.save(dict(sim=sim.detach().cpu(), metrics=result, examples=[{k: x for k, x in e.items() if k not in ('raw', 'text')} for e in examples]), V4_DIR / 'artifacts' / name)
    V4_INFO['retrieval'].append(dict(file=name, prefix=V4_PREFIX, metrics=result))
    print('PCV4_RETRIEVAL', name, json.dumps(result), flush=True)
    return result
V4_NS['evaluate_sim_matrix'] = v4_eval

v4_orig_build = V4_NS['build_connected_m2_episodes']
def v4_build(*args, **kwargs):
    episodes, audit = v4_orig_build(*args, **kwargs)
    name = f'm2_episodes_{len(V4_INFO["episodes"]):02d}.pt'
    raw = {k: torch.stack([e[k].detach().cpu() for e in episodes]) for k in ('phi', 'null_feat', 'std_attn')} if episodes else {}
    raw.update(metadata=[{k: e[k] for k in ('label', 'dir_tag', 'pair_key')} for e in episodes], kwargs=kwargs, foil_audit=audit)
    torch.save(raw, V4_DIR / 'artifacts' / name)
    row = dict(file=name, n=len(episodes), labels=dict(collections.Counter(e['label'] for e in episodes)), kwargs=kwargs, foil_audit=audit)
    V4_INFO['episodes'].append(row)
    v4_write('instrumentation.json', V4_INFO)
    print('PCV4_EPISODES', json.dumps(row), flush=True)
    return episodes, audit
V4_NS['build_connected_m2_episodes'] = v4_build

v4_input = dict(server=PC_SERVER, runtime_out=str(OUT), torch=torch.__version__, model=MODEL_.config._name_or_path,
                source_sha256=V4_INFO['source_sha256'], caches={})
for prefix in ('avbind_native_pairs', 'avbind_pooled_pairs'):
    for shard in (1, 2):
        path = OUT / 'uploads' / f'{prefix}_s{shard}.pt'
        v4_input['caches'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
v4_write('input_audit.json', v4_input)
v4_start = time.time()
try:
    V4_STATE['Algebraic_and_TinyNoise_Unit_Checks'] = V4_NS['run_algebraic_unit_checks']('cuda:0')
    for V4_PREFIX in ('avbind_native_pairs', 'avbind_pooled_pairs'):
        result, bundle = V4_NS['run_stage0_m1_clean_comparison'](V4_PREFIX)
        V4_STATE['M1_native_clean_comparison' if V4_PREFIX == 'avbind_native_pairs' else 'M1_pooled_clean_comparison'] = result
        V4_BUNDLES[V4_PREFIX] = bundle
        torch.save(dict(sim_train=bundle['sim_tr_c1'].cpu(), sim_heldout=bundle['sim_he_c1'].cpu(), train_ex=bundle['train_ex'], held_ex=bundle['held_ex']), V4_DIR / 'artifacts' / f'{V4_PREFIX}_bundle.pt')
        v4_write('progress.json', V4_STATE)
        print('PCV4_M1_COMPLETE', V4_PREFIX, json.dumps(result), flush=True)
    V4_STATE['M2_shrinkage_calibrated_connected_to_C1'] = V4_NS['run_stage0_m2_connected'](V4_BUNDLES['avbind_pooled_pairs'])
    v4_write('progress.json', V4_STATE)
    print('PCV4_M2_COMPLETE', json.dumps(V4_STATE['M2_shrinkage_calibrated_connected_to_C1']), flush=True)
    V4_STATE['M3_clock_calibrated_patch_salience_routing'] = V4_NS['run_stage0_m3_clock_and_patch_salience_routing']()
    V4_STATE.update(protocol='Stage-0 v4 original functions with observation-only checkpoints', elapsed_sec=round(time.time() - v4_start, 2))
    v4_write('main_results.json', V4_STATE)
    print('PCV4_DONE', json.dumps(V4_STATE), flush=True)
finally:
    v4_write('progress.json', V4_STATE)
    v4_write('instrumentation.json', V4_INFO)

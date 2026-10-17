"""Fixed40: layer-local CLC vs shared native contrasts, signed/random controls.

AST adapter preserves corrected extraction/profiles and routes; no QA-dependent selection.
"""
import ast, gc, hashlib, json, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F

D = OUT / 'pcebr_stage0_v8_layerlocal'
D.mkdir(exist_ok=True)
(D / 'artifacts').mkdir(exist_ok=True)
source = (OUT / 'uploads/stage0_pcebr_v8_corrected_standalone.py').read_text(encoding='utf-8')
assert hashlib.sha256(source.encode()).hexdigest() == '25043c391ac1503280c2e4b70c4e38d27d3f8831741e4a38f8f96ffc4b758843'
tree = ast.parse(source)
tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RESULT' for t in n.targets))]
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_stage0_m3_bivector_vo_tangent_routing')
previous = json.loads((D / 'samples.json').read_text()) if (D / 'samples.json').exists() else []
old40 = {r['idx']: r for r in json.loads((OUT / 'pcebr_stage0_v8/clc40/samples.json').read_text())}

def assigned(st, name):
    return isinstance(st, ast.Assign) and any(any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(t)) for t in st.targets)

for st in main.body:
    if assigned(st, 'samples'):
        st.value = ast.parse("[r for r in manifest[f'server{server_id}'] if r['idx'] not in LL_COMPLETED]", mode='eval').body
    if assigned(st, 'rows_out'):
        st.value = ast.parse('list(LL_PREVIOUS)', mode='eval').body
loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and any(isinstance(t, ast.Name) and t.id == 's_i' for t in n.target.elts))
legacy_begin = next(i for i, st in enumerate(loop.body) if assigned(st, 'V18_v5_raw'))
legacy_end = next(i for i, st in enumerate(loop.body) if assigned(st, 'worst_k'))
loop.body[legacy_begin:legacy_end] = ast.parse('gate_legacy_v5 = 0.0').body
cut = next(i for i, st in enumerate(loop.body) if assigned(st, 'ct_eff_scale'))
loop.body = loop.body[:cut + 1]
route = next(n for n in ast.walk(loop) if isinstance(n, ast.FunctionDef) and n.name == 'run_caor_routed')

# Capture all five layer states during each existing option-free counterfactual pass.
cap_if = next(n for n in ast.walk(route) if isinstance(n, ast.If) and ast.unparse(n.test) == 'capture_l22_box is not None')
cap_if.body = ast.parse('''
for capture_layer in target_layers[:5]:
    def make_local_cap(layer_index):
        def local_cap(mod, args, out):
            h_o = out[0] if isinstance(out, tuple) else out
            capture_l22_box[layer_index] = h_o[0,-1].detach().float().cpu().clone()
            if layer_index == 22:
                capture_l22_box['h22_last'] = capture_l22_box[layer_index]
            return out
        return local_cap
    handles_clc.append(model_llm.model.layers[capture_layer].register_forward_hook(make_local_cap(capture_layer)))
''').body
# Keep vector path bitwise identical; add a layer-indexed direction path.
replaced = 0
for call in ast.walk(route):
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'make_clc_hk':
        call.args[0] = ast.parse('clc_vec_1d[l_i] if isinstance(clc_vec_1d, dict) else clc_vec_1d', mode='eval').body
        replaced += 1
assert replaced == 1
loop.body += ast.parse('''
ll_layers = mid_layers[:5]
ev, od, local_native, local_ev = {}, {}, {}, {}
contrast_audit = {}
for ll in ll_layers:
    assert all(ll in box for box in (cap_supp_ff,cap_foil_ff,cap_rev_ff)), ('capture_missing',idx,ll)
    diff_e = cap_supp_ff[ll]-cap_foil_ff[ll]
    diff_o = cap_supp_ff[ll]-cap_rev_ff[ll]
    ne,no = float(diff_e.norm()),float(diff_o.norm())
    assert min(ne,no)>1e-6, ('degenerate_contrast',idx,ll,ne,no)
    ev[ll],od[ll] = F.normalize(diff_e,dim=-1),F.normalize(diff_o,dim=-1)
    local_native[ll] = F.normalize(.45*ev[ll]+.35*od[ll],dim=-1)
    local_ev[ll] = ev[ll]
shared_native = F.normalize(.45*ev[22]+.35*od[22],dim=-1)
for ll in ll_layers:
    contrast_audit[ll] = dict(ev_norm=float((cap_supp_ff[ll]-cap_foil_ff[ll]).norm()),ord_norm=float((cap_supp_ff[ll]-cap_rev_ff[ll]).norm()),
        ev_vs_ord_cosine=float(torch.dot(ev[ll],od[ll])),local_vs_shared_cosine=float(torch.dot(local_native[ll],shared_native)))
directions = dict(shared22_mixed_replay=clc_dir_fwd,shared22_native=shared_native,layerlocal_native=local_native,layerlocal_ev=local_ev,
    layerlocal_negated={ll:-local_native[ll] for ll in ll_layers})
for seed in (101,202,303):
    rng = torch.Generator().manual_seed(idx+seed)
    directions[f'random_{seed}'] = {ll:F.normalize(torch.randn(ev[ll].shape,generator=rng),dim=-1) for ll in ll_layers}
conditions = dict(native=dict(pred='ABCD'[int(lg_nat[labels[:k_opts]].argmax())],probabilities=F.softmax(lg_nat[labels[:k_opts]],dim=-1).tolist()),
    CAOR_only=dict(pred='ABCD'[int(lg_caor_full[labels[:k_opts]].argmax())],probabilities=F.softmax(lg_caor_full[labels[:k_opts]],dim=-1).tolist()))
for name,direction in directions.items():
    d_list = direction.values() if isinstance(direction,dict) else [direction]
    assert all(abs(float(d.norm())-1.0)<1e-5 for d in d_list)
    logits = run_caor_routed(mid_layers,prof_full_fwd,eta_scale=0.0,clc_vec_1d=direction,clc_scale=clc_eff_scale,ct_scale=0.0,use_prefix_cache=use_fast_cache)
    probs = F.softmax(logits[labels[:k_opts]],dim=-1)
    conditions[name] = dict(pred='ABCD'[int(probs.argmax())],probabilities=probs.tolist())
prior = LL_OLD40[idx]
replay_errors = {}
for tag,old_tag in (('native','native'),('shared22_mixed_replay','CLC_only'),('CAOR_only','CAOR_only')):
    err = float((torch.tensor(conditions[tag]['probabilities'])-torch.tensor(prior['probabilities'][old_tag])).abs().max())
    replay_errors[tag] = err
    assert err<=1e-6, ('prior40_replay',idx,tag,err)
artifact = dict(idx=idx,directions=directions,ev=ev,od=od,counterfactual_states=dict(supp=cap_supp_ff,foil=cap_foil_ff,rev=cap_rev_ff),
    gamma=gate_fwd,profiles=dict(supp=prof_full_fwd,foil=prof_foil_suppressed,rev=prof_ord_sign_rev),input_cache_sha256=hashlib.sha256(pt_file.read_bytes()).hexdigest())
art_path = LL_D/'artifacts'/f'{idx}.pt'
torch.save(artifact,art_path)
record = dict(idx=idx,task=sample['task'],gold=sample['gold'],gamma=gate_fwd,conditions=conditions,contrast_audit=contrast_audit,
    prior40_replay_errors=replay_errors,cache_parity_err=cached_routed_parity_err,cache_used=use_fast_cache,native_replay_err=native_replay_err,kernel_noop_err=kernel_noop_err,
    artifact_sha256=hashlib.sha256(art_path.read_bytes()).hexdigest(),input_cache_sha256=artifact['input_cache_sha256'],elapsed_sec=time.time()-t_sample_start)
rows_out.append(record)
LL_WRITE(rows_out)
print('LAYERLOCAL_SAMPLE',json.dumps(dict(idx=idx,gold=sample['gold'],preds={k:v['pred'] for k,v in conditions.items()},elapsed_sec=record['elapsed_sec'])),flush=True)
gc.collect();torch.cuda.empty_cache()
''').body
li = main.body.index(loop)
main.body = main.body[:li+1] + ast.parse('return dict(n_samples=len(rows_out),samples=rows_out)').body
ast.fix_missing_locations(tree)
effective = ast.unparse(tree)+'\n'
(D/'effective_source.py').write_text(effective)

def save(rows):
    tmp = D/'samples.tmp'
    tmp.write_text(json.dumps(rows,indent=2))
    tmp.replace(D/'samples.json')

ns = dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=PC_SERVER,LL_COMPLETED={r['idx'] for r in previous},LL_PREVIOUS=previous,
    LL_OLD40=old40,LL_WRITE=save,LL_D=D,gc=gc,hashlib=hashlib)
exec(compile(tree,'layerlocal_fixed40','exec'),ns)
versions = {n:p._version for n,p in MODEL_.named_parameters()}
t0 = time.time()
try:
    result = ns['run_stage0_m3_bivector_vo_tangent_routing']()
    assert result['n_samples']==20
    result.update(seconds=time.time()-t0,server=PC_SERVER,backbone_versions_unchanged=all(p._version==versions[n] for n,p in MODEL_.named_parameters()),
        effective_source_sha256=hashlib.sha256(effective.encode()).hexdigest())
    assert result['backbone_versions_unchanged']
    (D/'completion.json').write_text(json.dumps(result,indent=2))
    RESULT = dict(n=20,scores={k:sum(r['conditions'][k]['pred']==r['gold'] for r in result['samples']) for k in result['samples'][0]['conditions']})
    print('LAYERLOCAL_DONE',json.dumps(RESULT),flush=True)
except Exception:
    (D/'error.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2))
    raise

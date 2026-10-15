"""Read-only algebra and provenance controls for v4; no retuning or QA expansion."""
import ast
import json
import torch
import torch.nn.functional as F

v4_checks = {'server': PC_SERVER, 'permutation_examples': [], 'uniform_gate': {}, 'real_sample_checks': [], 'sim_mat_ast': {}}
for K in (5, 6, 9):
    for seed in (101, 202, 303):
        rng = torch.Generator().manual_seed(seed)
        V = F.normalize(torch.randn(K, 2048, generator=rng), dim=-1)
        A = F.normalize(torch.randn(K, 2048, generator=rng), dim=-1)
        lv = torch.randn(K, generator=rng) * 3
        la = torch.randn(K, generator=rng) * 3
        iv = torch.stack((torch.arange(K) * 2, torch.arange(1, K + 1) * 2), -1).float()
        f, gf, pf = V4_NS['build_symmetric_routed_vector'](V, A, lv, la, iv)
        r, gr, pr = V4_NS['build_symmetric_routed_vector'](V, A, lv, la, iv, reverse_timeline=True)
        v4_checks['permutation_examples'].append(dict(K=K, seed=seed, cosine=float(f @ r), max_abs_difference=float((f-r).abs().max()), gate_fwd=gf, gate_rev=gr, signed_phi_fwd=float(pf[5]), signed_phi_rev=float(pr[5])))
        if K == 5 and seed == 101:
            _, gu, pu = V4_NS['build_symmetric_routed_vector'](V, A, torch.zeros(K), torch.zeros(K), iv)
            v4_checks['uniform_gate'] = dict(gate=gu, effective_scale=.22*gu, phi_max_abs=float(pu.abs().max()))

for path in sorted((V4_DIR/'artifacts').glob('m3_*.pt')):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    idx = int(path.stem.split('_')[1])
    ci = obj['clock_info']; sec=ci['sec_calibrated']; vi=ci['v_indices']; ai=ci['a_indices']
    iv=obj['iv_t']
    count_v=[int(((sec[vi]>=x[0])&(sec[vi]<x[1]+1e-4)).sum()) for x in iv]
    count_a=[int(((sec[ai]>=x[0])&(sec[ai]<x[1]+1e-4)).sum()) for x in iv]
    v4_checks['real_sample_checks'].append(dict(idx=idx, vector_cosine=float(obj['vec_fwd'] @ obj['vec_rev']), vector_max_abs_diff=float((obj['vec_fwd']-obj['vec_rev']).abs().max()), calibrated_tokens_per_slot_v=count_v, calibrated_tokens_per_slot_a=count_a))

source_tree=ast.parse(V4_SOURCE.read_text(encoding='utf-8'))
for node in ast.walk(source_tree):
    if isinstance(node,ast.FunctionDef) and node.name=='build_connected_m2_episodes':
        v4_checks['sim_mat_ast']={'load_count':sum(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='sim_mat' for n in ast.walk(node)), 'lines':[n.lineno for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='sim_mat']}
v4_checks['intervention_hook_count']=sum(len(layer._forward_hooks) for layer in MODEL_.model.layers)
v4_checks['cuda_ok']=bool(torch.isfinite(torch.ones(1,device='cuda:0')).all())
v4_write('controls.json',v4_checks)
RESULT=v4_checks
print('PCV4_CONTROLS',json.dumps(RESULT),flush=True)

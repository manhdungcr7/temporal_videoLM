"""Exploratory fixed-seed controls; kept separate from supplied Stage-0.

Compare no centering, media-only centering, original dual centering, and
the original dual-centering shuffle. Do not tune thresholds or read research40.
"""
import ast
import collections
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F

_audit_started = time.time()
_audit_prefix = 'avbind_native_pairs' if PC_SERVER == 1 else 'avbind_pooled_pairs'
_audit_source = OUT / 'uploads/stage0_pcebr_mechanism_check_devicefix.py'
_audit_tree = ast.parse(_audit_source.read_text(encoding='utf-8'))
_audit_tree.body = [node for node in _audit_tree.body
                    if not (isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp))]
_audit_ns = {'__name__': 'pcebr_audit_helpers'}
exec(compile(_audit_tree, str(_audit_source), 'exec'), _audit_ns)
_audit_ns['OUT']=OUT
deconfound = _audit_ns['deconfound_by_recording']
Projector = _audit_ns['DeconfoundedEventProjector']
_audit_parts = [torch.load(OUT/'uploads'/f'{_audit_prefix}_s{s}.pt',map_location='cpu',weights_only=True) for s in (1,2)]
_audit_examples = [x for p in _audit_parts for x in p['examples']]
_audit_train = [x for x in _audit_examples if x['split']=='train']
_audit_held = [x for x in _audit_examples if x['split']=='heldout']
_audit_device = 'cuda:0'

def tensors(items):
    return (torch.stack([x['raw'] for x in items]).float().to(_audit_device),
            torch.stack([x['text'] for x in items]).float().to(_audit_device),
            torch.tensor([0 if x['source']=='video' else 1 for x in items],device=_audit_device),
            [x['recording_id'] for x in items])

_tx,_tt,_ts,_tr=tensors(_audit_train)
_hx,_ht,_hs,_hr=tensors(_audit_held)
_n=len(_audit_train)
_norms=[x['content'].strip().lower() for x in _audit_train]
_pos=torch.tensor([[_norms[i]==_norms[j] for j in range(_n)] for i in range(_n)],device=_audit_device)
_same=torch.tensor([[_tr[i]==_tr[j] and _ts[i]==_ts[j] for j in range(_n)] for i in range(_n)],device=_audit_device)
_boost=torch.where(_same&~_pos,0.8,0.0)
_audit_folder=OUT/'pcebr_stage0'
_audit_folder.mkdir(exist_ok=True)
_audit_sims={}
_audit_original_eval=_audit_ns['evaluate_retrieval_and_within_rec_localization']

def evaluate(zm,zt,name):
    sim=zt@zm.T
    _audit_sims[name]=sim.cpu()
    base=_audit_original_eval(zm,zt,_audit_held)
    hn=[x['content'].strip().lower() for x in _audit_held]
    gp=sim.argmax(-1).cpu().tolist()
    rows=[]
    for i,x in enumerate(_audit_held):
        js=[j for j,y in enumerate(_audit_held) if y['recording_id']==x['recording_id'] and y['source']==x['source']]
        lp=js[int(sim[i,js].argmax())]
        maxscore=float(sim[i,js].max())
        tied=[j for j in js if abs(float(sim[i,j])-maxscore)<1e-7]
        rows.append(dict(index=i,recording=x['recording_id'],source=x['source'],start=x['start'],end=x['end'],
            candidates=len(js),global_pred=gp[i],global_correct=hn[i]==hn[gp[i]],
            local_pred=lp,local_correct=hn[i]==hn[lp] and abs(_audit_held[lp]['start']-x['start'])<1e-3,
            local_ties=len(tied),query=hn[i]))
    base['per_query']=rows
    return base

_audit_results={}
# Use the original training AST so even operation/gradient accumulation order is
# preserved. V1 used an algebraically equivalent rewrite and changed seed-0
# predictions; V1 is archived separately and is not the main sensitivity table.
_audit_fn=copy.deepcopy(next(n for n in _audit_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_stage0_m1_factorial'))
for _node in ast.walk(_audit_fn):
    if isinstance(_node,ast.Call) and isinstance(_node.func,ast.Attribute) and _node.func.attr in ('manual_seed','Random'):
        if _node.args and isinstance(_node.args[0],ast.Constant) and _node.args[0].value==20261015:
            _node.args[0]=ast.Name(id='PC_AUDIT_SEED',ctx=ast.Load())
    if isinstance(_node,ast.IfExp) and isinstance(_node.body,ast.Name) and _node.body.id in ('tr_txt_dec','he_txt_dec'):
        _node.test=ast.BoolOp(op=ast.And(),values=[_node.test,ast.Compare(left=ast.Name(id='mode_name',ctx=ast.Load()),
            ops=[ast.NotEq()],comparators=[ast.Constant(value='6_media_only_center_r24')])])
    if isinstance(_node,ast.List) and _node.elts and isinstance(_node.elts[0],ast.Tuple):
        _first=_node.elts[0].elts[0]
        if isinstance(_first,ast.Constant) and _first.value=='3_pcebr_uncentered_r24':
            _node.elts.insert(2,ast.Tuple(elts=[ast.Constant('6_media_only_center_r24'),ast.Constant(True),ast.Constant(False)],ctx=ast.Load()))
_audit_module=ast.fix_missing_locations(ast.Module(body=[_audit_fn],type_ignores=[]))
exec(compile(_audit_module,'pcebr_exact_ast_controls','exec'),_audit_ns)
_audit_order=['raw','raw_center','uncentered','dual_center_original','media_only_center','dual_center_shuffle']
_audit_calls=0
def _record_evaluation(zm,zt,examples):
    global _audit_calls
    assert len(examples)==len(_audit_held)
    variant=_audit_order[_audit_calls];_audit_calls+=1
    result=evaluate(zm,zt,f'{variant}_seed{_seed}')
    if variant not in ('raw','raw_center'):
        key=f'{variant}_seed{_seed}'
        result.update(seed=_seed,variant=variant,params=294985)
        _audit_results[key]=result
        print('PCEBR_AUDIT',key,{k:v for k,v in result.items() if k!='per_query'},flush=True)
    return result
_audit_ns['evaluate_retrieval_and_within_rec_localization']=_record_evaluation
for _seed in (20261015,20261016,20261017):
    _audit_calls=0
    _audit_ns['PC_AUDIT_SEED']=_seed
    _audit_ns['run_stage0_m1_factorial'](_audit_prefix)
    assert _audit_calls==6

_groups=collections.defaultdict(list)
for i,x in enumerate(_audit_held):_groups[(x['recording_id'],x['source'])].append(i)
_chance=sum(1/len(_groups[(x['recording_id'],x['source'])]) for x in _audit_held)/len(_audit_held)
_duplicate_groups=[]
for (r,s),ix in _groups.items():
    bytext=collections.defaultdict(list)
    for i in ix:bytext[_audit_held[i]['content'].strip().lower()].append(i)
    for txt,ids in bytext.items():
        if len(ids)>1:_duplicate_groups.append(dict(recording=r,source=s,indices=ids,content=txt))

_phi=_audit_ns['compute_directed_interval_relations']
_early=torch.tensor([[2.,4.]])
_late=torch.tensor([[10.,12.]])
_pa=_phi(_early,_late);_pb=_phi(_late,_early)
_weights=torch.tensor([.9,.1]);_centers=torch.tensor([3.,11.])
_algebra=dict(translation_error=float((_pa-_phi(_early+12.5,_late+12.5)).abs().max()),
    common_scale_full_phi_error=float((_pa-_phi(2*_early,2*_late)).abs().max()),
    full_phi_not_antisymmetric_max=float((_pa+_pb).abs().max()),
    content_weighted_time_before=float((_weights*_centers).sum()),
    content_weighted_time_after=float((_weights*_centers.flip(0)).sum()),
    coverage_anchor_example=float(_phi(torch.tensor([[0.,10.]]),torch.tensor([[2.,4.]]))[0,0,1]))
_probe_model=Projector()
_audit_output=dict(prefix=_audit_prefix,audit_version=2,source_sha256=hashlib.sha256((OUT/'uploads/pcebr_stage0_audit.py').read_bytes()).hexdigest(),
    seeds=[20261015,20261016,20261017],train_pairs=len(_audit_train),heldout_pairs=len(_audit_held),
    groups=[dict(recording=r,source=s,n=len(ix)) for (r,s),ix in _groups.items()],
    uniform_random_local_chance=_chance,duplicate_content_groups=_duplicate_groups,
    params=sum(p.numel() for p in _probe_model.parameters()),algebra_limits=_algebra,
    pair_metadata=[p['metadata'] for p in _audit_parts],results=_audit_results,
    elapsed_sec=time.time()-_audit_started)
torch.save(_audit_sims,_audit_folder/f'audit_similarities_s{PC_SERVER}.pt')
(_audit_folder/f'audit_results_s{PC_SERVER}.json').write_text(json.dumps(_audit_output,indent=2),encoding='utf-8')
RESULT=dict(prefix=_audit_prefix,rows=len(_audit_results),elapsed_sec=_audit_output['elapsed_sec'],algebra_limits=_algebra)
print('PCEBR_AUDIT_DONE',json.dumps(RESULT),flush=True)

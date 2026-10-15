"""Execute supplied v2 functions with read-only checkpoint instrumentation."""
import ast
import collections
import hashlib
import json
import time
import torch
from pathlib import Path

_v2_folder=OUT/'pcebr_stage0_v2';_v2_folder.mkdir(exist_ok=True)
(_v2_folder/'artifacts').mkdir(exist_ok=True)
_v2_source=OUT/'uploads/stage0_pcebr_mechanism_check_v2_original.py'
_v2_tree=ast.parse(_v2_source.read_text(encoding='utf-8'))
_v2_tree.body=[node for node in _v2_tree.body if not (isinstance(node,ast.If) and isinstance(node.test,ast.BoolOp))]
_v2_ns=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=PC_SERVER,__name__='pcebr_v2_original_helpers')
_v2_state={};_v2_instrumentation=dict(source_sha256=hashlib.sha256(_v2_source.read_bytes()).hexdigest(),conditions=[],probes=[],episodes=[],evaluations=[])

def _v2_write(name,data):
    (_v2_folder/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def _v2_capture_episodes(examples,sim_mat,episodes):
    tag='train' if examples[0]['split']=='train' else 'heldout'
    data={k:torch.stack([e[k].detach().cpu() for e in episodes]) for k in ('phi','phi_shuf','null_feat','std_attn')}
    data.update(labels=[e['label'] for e in episodes],directions=[e['dir_tag'] for e in episodes],
        examples=[{k:v for k,v in x.items() if k not in ('raw','text')} for x in examples],sim=sim_mat.detach().cpu())
    torch.save(data,_v2_folder/'artifacts'/f'm2_episodes_{tag}.pt')
    summary=dict(split=tag,episodes=len(episodes),labels=dict(collections.Counter(data['labels'])),directions=dict(collections.Counter(data['directions'])))
    _v2_instrumentation['episodes'].append(summary)
    print('PCV2_EPISODES',json.dumps(summary),flush=True)

def _v2_capture_probe(values):
    index=len(_v2_instrumentation['probes'])+1
    payload=dict(X_train=values['X_tr'].detach().cpu(),X_heldout=values['X_he'].detach().cpu(),
        y_train=values['y_tr'].detach().cpu(),y_heldout=values['y_he'].detach().cpu(),
        predictions=values['preds'].detach().cpu(),state_dict={k:v.detach().cpu() for k,v in values['clf'].state_dict().items()})
    torch.save(payload,_v2_folder/'artifacts'/f'm2_probe_B{index}.pt')
    _v2_instrumentation['probes'].append(dict(index=index,n_train=len(payload['y_train']),n_heldout=len(payload['y_heldout']),features=payload['X_train'].shape[1],overall=values['overall']))
    print('PCV2_PROBE',json.dumps(_v2_instrumentation['probes'][-1]),flush=True)

# Insert observation-only calls immediately before the two nested returns.
for _v2_node in ast.walk(_v2_tree):
    if isinstance(_v2_node,ast.FunctionDef) and _v2_node.name=='build_noncircular_retrieved_episodes':
        _v2_node.body.insert(-1,ast.Expr(value=ast.Call(func=ast.Name(id='PCV2_CAPTURE_EPISODES',ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load()) for n in ('examples','sim_mat','episodes')],keywords=[])))
    if isinstance(_v2_node,ast.FunctionDef) and _v2_node.name=='eval_linear_probe':
        _v2_node.body.insert(-1,ast.Expr(value=ast.Call(func=ast.Name(id='PCV2_CAPTURE_PROBE',ctx=ast.Load()),args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
ast.fix_missing_locations(_v2_tree)
_v2_ns.update(PCV2_CAPTURE_EPISODES=_v2_capture_episodes,PCV2_CAPTURE_PROBE=_v2_capture_probe)
exec(compile(_v2_tree,str(_v2_source),'exec'),_v2_ns)
_v2_current_prefix='';_v2_current_condition='baseline';_v2_baseline_index=0
_v2_eval_original=_v2_ns['evaluate_sim_matrix']

def _v2_evaluate(sim,examples):
    global _v2_baseline_index
    result=_v2_eval_original(sim,examples)
    key=_v2_current_condition
    if key=='baseline':
        key=f'baseline{_v2_baseline_index}';_v2_baseline_index+=1
    file=f'{_v2_current_prefix}_{key}_retrieval.pt'
    torch.save(dict(sim=sim.detach().cpu(),metrics=result,examples=[{k:v for k,v in e.items() if k not in ('raw','text')} for e in examples]),_v2_folder/'artifacts'/file)
    _v2_instrumentation['evaluations'].append(dict(prefix=_v2_current_prefix,condition=key,metrics=result,file=file))
    return result
_v2_ns['evaluate_sim_matrix']=_v2_evaluate
_v2_train_original=_v2_ns['train_and_eval_m1_condition']

def _v2_train(*args,**kwargs):
    global _v2_current_condition
    _v2_current_condition=f"{kwargs['mode']}_hard{int(kwargs['use_hard_neg'])}_shuffle{int(kwargs['shuffle_within_rec'])}_seed{kwargs['seed']}"
    result,model=_v2_train_original(*args,**kwargs)
    file=f'{_v2_current_prefix}_{_v2_current_condition}_model.pt'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mode=model.mode,seed=kwargs['seed'],metrics=result),_v2_folder/'artifacts'/file)
    _v2_instrumentation['conditions'].append(dict(prefix=_v2_current_prefix,condition=_v2_current_condition,metrics=result,file=file))
    _v2_write('instrumentation.json',_v2_instrumentation)
    print('PCV2_M1_CONDITION',_v2_current_prefix,_v2_current_condition,json.dumps(result),flush=True)
    return result,model
_v2_ns['train_and_eval_m1_condition']=_v2_train

_v2_m1_original=_v2_ns['run_stage0_m1_crossed_factorial']
def _v2_m1(prefix):
    global _v2_current_prefix,_v2_current_condition,_v2_baseline_index
    _v2_current_prefix=prefix;_v2_current_condition='baseline';_v2_baseline_index=0
    result,ctx=_v2_m1_original(prefix)
    _v2_state[prefix]=result
    _v2_write('progress.json',_v2_state)
    return result,ctx
_v2_ns['run_stage0_m1_crossed_factorial']=_v2_m1

_v2_m2_original=_v2_ns['run_stage0_m2_retrieved_interval_counterfactuals']
def _v2_m2(ctx):
    result=_v2_m2_original(ctx)
    _v2_state['M2']=result;_v2_write('progress.json',_v2_state);_v2_write('instrumentation.json',_v2_instrumentation)
    return result
_v2_ns['run_stage0_m2_retrieved_interval_counterfactuals']=_v2_m2

_v2_input_audit=dict(server=PC_SERVER,torch=torch.__version__,model=MODEL_.config._name_or_path,caches={})
for _v2_prefix in ('avbind_native_pairs','avbind_pooled_pairs'):
    for _v2_s in (1,2):
        _v2_path=OUT/'uploads'/f'{_v2_prefix}_s{_v2_s}.pt'
        _v2_input_audit['caches'][_v2_path.name]=hashlib.sha256(_v2_path.read_bytes()).hexdigest()
_v2_write('input_audit.json',_v2_input_audit)
try:
    RESULT=_v2_ns['run_all_stage0_v2']()
    _v2_write('main_results.json',RESULT)
finally:
    _v2_write('instrumentation.json',_v2_instrumentation)

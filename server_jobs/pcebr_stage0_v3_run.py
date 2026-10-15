"""Run the supplied v3 functions with observation-only artifacts/checkpoints."""
import ast
import collections
import hashlib
import json
import time
import torch

_v3_folder=OUT/'pcebr_stage0_v3';_v3_folder.mkdir(exist_ok=True)
(_v3_folder/'artifacts').mkdir(exist_ok=True)
_v3_source=OUT/'uploads/stage0_pcebr_mechanism_check_v3_original.py'
_v3_tree=ast.parse(_v3_source.read_text(encoding='utf-8'))
_v3_tree.body=[n for n in _v3_tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RESULT' for t in n.targets))]
_v3_ns=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=PC_SERVER,__name__='pcebr_v3_original')
_v3_state={};_v3_bundles={};_v3_prefix='';_v3_counter=0
_v3_info=dict(source_sha256=hashlib.sha256(_v3_source.read_bytes()).hexdigest(),trained=[],retrieval=[],probes=[],episodes=[])
_v3_m3audit=[]

def _v3_write(name,data):(_v3_folder/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def _v3_capture_probe(v):
    name=f'm2_probe_{len(_v3_info["probes"]):02d}.pt'
    payload=dict(X_train=v['X_tr'].detach().cpu(),y_train=v['y_tr'].detach().cpu(),X_heldout=v['X_he'].detach().cpu(),
        y_heldout=v['y_he'].detach().cpu(),predictions=v['preds'].detach().cpu(),predictions_3way=v['preds_3way'].detach().cpu(),
        state_dict={k:x.detach().cpu() for k,x in v['clf'].state_dict().items()},
        heldout_metadata=[{k:e[k] for k in ('label','dir_tag')} for e in v['he_eps']])
    torch.save(payload,_v3_folder/'artifacts'/name)
    row=dict(file=name,n_train=len(v['y_tr']),n_eval=len(v['y_he']),accuracy4=v['overall_4way'],accuracy3=v['overall_3way_no_foil'])
    _v3_info['probes'].append(row);_v3_write('instrumentation.json',_v3_info)
    print('PCV3_PROBE',json.dumps(row),flush=True)

def _v3_capture_m3(v):
    tok=PROC.tokenizer;payload=v['payload'];lm=v['last_m']
    tail_ids=payload['input_ids'].cpu()[0].tolist()[lm+1:]
    tail=tok.decode(tail_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    reencoded=tok(tail,add_special_tokens=False)['input_ids']
    row=dict(idx=v['idx'],native_replay_error=v['native_replay_err'],
        exact_tail_token_roundtrip=reencoded==tail_ids,hook_rows=v['hook_rows'],option_rows=v['option_rows'],
        actual_stem_text=tok.decode(payload['input_ids'][0,v['stem_rows']],skip_special_tokens=False),
        actual_option_text=tok.decode(payload['input_ids'][0,v['option_rows']],skip_special_tokens=False),
        actual_assistant_text=tok.decode(payload['input_ids'][0,v['assistant_rows']],skip_special_tokens=False),
        freeform_input_text=tok.decode(payload['input_ids'][0,v['keep_indices']],skip_special_tokens=False),
        freeform_generated_text=v['freeform_text'],generated_tokens=v['gen_tokens'],
        probabilities={k:v[k].tolist() for k in ('p_nat','p_late_fwd','p_late_rev','p_mid_fwd','p_mid_rev','p_mid_irrel')},
        visual_slots=len(v['V18']),audio_slots=len(v['A18']),
        cosine=v['fwd_rev_cos'],phi_fwd=v['phi_fwd'].tolist(),phi_rev=v['phi_rev'].tolist())
    _v3_m3audit.append(row);_v3_write('m3_audit.json',_v3_m3audit)
    print('PCV3_M3_SAMPLE',json.dumps(v['rows_out'][-1]),flush=True)

for _v3_node in ast.walk(_v3_tree):
    if isinstance(_v3_node,ast.FunctionDef) and _v3_node.name=='eval_linear_probe':
        _v3_node.body.insert(-1,ast.Expr(value=ast.Call(func=ast.Name(id='PCV3_CAPTURE_PROBE',ctx=ast.Load()),
            args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
    if isinstance(_v3_node,ast.For):
        for i,statement in enumerate(list(_v3_node.body)):
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
                call=statement.value
                if isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='rows_out' and call.func.attr=='append':
                    _v3_node.body.insert(i+1,ast.Expr(value=ast.Call(func=ast.Name(id='PCV3_CAPTURE_M3',ctx=ast.Load()),
                        args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
ast.fix_missing_locations(_v3_tree)
_v3_ns.update(PCV3_CAPTURE_PROBE=_v3_capture_probe,PCV3_CAPTURE_M3=_v3_capture_m3)
exec(compile(_v3_tree,str(_v3_source),'exec'),_v3_ns)
_v3_orig_train=_v3_ns['train_single_stream_aligner']
def _v3_train(*args,**kwargs):
    global _v3_counter
    m=_v3_orig_train(*args,**kwargs)
    name=f'{_v3_prefix}_model_{_v3_counter:02d}.pt';_v3_counter+=1
    row=dict(file=name,kwargs=kwargs)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},proj_type=m.proj_type,kwargs=kwargs),_v3_folder/'artifacts'/name)
    _v3_info['trained'].append(row);_v3_write('instrumentation.json',_v3_info)
    print('PCV3_TRAIN',name,json.dumps(kwargs),flush=True)
    return m
_v3_ns['train_single_stream_aligner']=_v3_train

_v3_orig_eval=_v3_ns['evaluate_sim_matrix']
def _v3_eval(sim,examples):
    metrics=_v3_orig_eval(sim,examples)
    name=f'{_v3_prefix}_retrieval_{len(_v3_info["retrieval"]):03d}.pt'
    torch.save(dict(sim=sim.detach().cpu(),metrics=metrics,examples=[{k:v for k,v in x.items() if k not in ('raw','text')} for x in examples]),_v3_folder/'artifacts'/name)
    _v3_info['retrieval'].append(dict(file=name,prefix=_v3_prefix,metrics=metrics))
    print('PCV3_RETRIEVAL',name,json.dumps(metrics),flush=True)
    return metrics
_v3_ns['evaluate_sim_matrix']=_v3_eval

_v3_orig_m1=_v3_ns['run_stage0_m1_crossed_factorial']
def _v3_m1(prefix):
    global _v3_prefix
    _v3_prefix=prefix
    result,bundle=_v3_orig_m1(prefix)
    _v3_bundles[prefix]=bundle;_v3_state[prefix]=result
    torch.save(dict(m_ctx_state={k:v.detach().cpu() for k,v in bundle['m_ctx'].state_dict().items()},
        m_loc_state={k:v.detach().cpu() for k,v in bundle['m_loc'].state_dict().items()},
        sim_train=bundle['sim_tr'].detach().cpu(),sim_heldout=bundle['sim_he'].detach().cpu()),_v3_folder/'artifacts'/f'{prefix}_bundle.pt')
    _v3_write('progress.json',_v3_state);_v3_write('instrumentation.json',_v3_info)
    print('PCV3_M1_COMPLETE',prefix,json.dumps(result),flush=True)
    return result,bundle
_v3_ns['run_stage0_m1_crossed_factorial']=_v3_m1

_v3_orig_build=_v3_ns['build_zero_shortcut_episodes']
def _v3_build(*args,**kwargs):
    episodes=_v3_orig_build(*args,**kwargs)
    tag=f'm2_episodes_{len(_v3_info["episodes"]):02d}.pt'
    data={k:torch.stack([e[k].detach().cpu() for e in episodes]) for k in ('phi','null_feat','std_attn')} if episodes else {}
    data.update(labels=[e['label'] for e in episodes],directions=[e['dir_tag'] for e in episodes],kwargs=kwargs)
    torch.save(data,_v3_folder/'artifacts'/tag)
    row=dict(file=tag,episodes=len(episodes),kwargs=kwargs,labels=dict(collections.Counter(data['labels'])))
    _v3_info['episodes'].append(row);_v3_write('instrumentation.json',_v3_info)
    print('PCV3_EPISODES',json.dumps(row),flush=True)
    return episodes
_v3_ns['build_zero_shortcut_episodes']=_v3_build

_v3_orig_m2=_v3_ns['run_stage0_m2_retrieved_interval_counterfactuals']
def _v3_m2(bundle):
    result=_v3_orig_m2(bundle);_v3_state['M2']=result;_v3_write('progress.json',_v3_state)
    print('PCV3_M2_COMPLETE',json.dumps(result),flush=True)
    return result
_v3_ns['run_stage0_m2_retrieved_interval_counterfactuals']=_v3_m2
_v3_audit=dict(server=PC_SERVER,torch=torch.__version__,model=MODEL_.config._name_or_path,
    runtime_out=str(OUT),layer_devices={str(i):str(next(MODEL_.model.layers[i].parameters()).device) for i in (0,18,27,35)},caches={})
for prefix in ('avbind_native_pairs','avbind_pooled_pairs'):
    for shard in (1,2):
        path=OUT/'uploads'/f'{prefix}_s{shard}.pt'
        _v3_audit['caches'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
_v3_write('input_audit.json',_v3_audit)
try:
    RESULT=_v3_ns['run_all_stage0_v3']()
    _v3_write('main_results.json',RESULT)
finally:_v3_write('instrumentation.json',_v3_info)

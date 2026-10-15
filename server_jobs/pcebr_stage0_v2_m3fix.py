"""Resume original M3 with mask-device repair, preserving completed M1/M2."""
import ast
import hashlib
import json
import time
import torch

_m3f_start=time.time();_m3f_folder=OUT/'pcebr_stage0_v2'
_m3f_source=OUT/'uploads/stage0_pcebr_mechanism_check_v2_devicefix.py'
_m3f_tree=ast.parse(_m3f_source.read_text(encoding='utf-8'))
_m3f_tree.body=[node for node in _m3f_tree.body if not (isinstance(node,ast.If) and isinstance(node.test,ast.BoolOp))]
_m3f_ns=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=PC_SERVER,__name__='pcv2_m3_devicefix')
_m3f_audit=[]

def _m3f_capture(values):
    payload=values['payload'];k=values['k_opts'];labels=values['labels'][:k]
    parity=float((values['lg_nat'][labels]-payload['native_logits'].float()).abs().max())
    row=dict(idx=values['idx'],k=k,native_replay_max_error=parity,
        choice_tokens=len(values['kw']['inputs_embeds'][0])-values['last_media']-1,
        early_late_direction_cosine=float(torch.dot(values['vec_fwd'],values['vec_rev'])),
        native_label_logits=values['lg_nat'][labels].tolist(),
        probabilities={name:values[name].tolist() for name in ('p_nat','p_late_fwd','p_late_rev','p_mid_fwd','p_mid_rev','p_mid_shuf')})
    assert parity<=.06,('M3 native replay discrepancy',row)
    _m3f_audit.append(row)
    (_m3f_folder/'m3_audit.json').write_text(json.dumps(_m3f_audit,indent=2),encoding='utf-8')
    print('PCV2_M3_SAMPLE',json.dumps(row),flush=True)

for _m3f_node in ast.walk(_m3f_tree):
    if isinstance(_m3f_node,ast.For):
        for _m3f_i,_m3f_statement in enumerate(list(_m3f_node.body)):
            if isinstance(_m3f_statement,ast.Expr) and isinstance(_m3f_statement.value,ast.Call):
                _m3f_call=_m3f_statement.value
                if isinstance(_m3f_call.func,ast.Attribute) and isinstance(_m3f_call.func.value,ast.Name) and _m3f_call.func.value.id=='rows_out' and _m3f_call.func.attr=='append':
                    _m3f_node.body.insert(_m3f_i+1,ast.Expr(value=ast.Call(func=ast.Name(id='PCV2_CAPTURE_M3',ctx=ast.Load()),args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
ast.fix_missing_locations(_m3f_tree)
_m3f_ns['PCV2_CAPTURE_M3']=_m3f_capture
exec(compile(_m3f_tree,str(_m3f_source),'exec'),_m3f_ns)
_m3f_result=_m3f_ns['run_stage0_m3_mid_vs_late_llm_routing']()
_m3f_result['devicefix_elapsed_sec']=time.time()-_m3f_start
(_m3f_folder/'m3_devicefix_results.json').write_text(json.dumps(_m3f_result,indent=2),encoding='utf-8')
_m3f_progress=json.loads((_m3f_folder/'progress.json').read_text(encoding='utf-8'))
RESULT=dict(M1_native_pairs_crossed_factorial=_m3f_progress['avbind_native_pairs'],
    M1_pooled_pairs_crossed_factorial=_m3f_progress['avbind_pooled_pairs'],
    M2_affine_and_retrieved_counterfactuals=_m3f_progress['M2'],M3_mid_vs_late_llm_routing=_m3f_result,
    original_source_sha256=hashlib.sha256((OUT/'uploads/stage0_pcebr_mechanism_check_v2_original.py').read_bytes()).hexdigest(),
    devicefix_source_sha256=hashlib.sha256(_m3f_source.read_bytes()).hexdigest())
(_m3f_folder/'main_results.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('PCV2_COMPLETE',json.dumps(_m3f_result),flush=True)

"""Exploratory controls for v2 claims; never substitute for main results."""
import ast
import json
import math
import torch
from pathlib import Path

_c2_folder=OUT/'pcebr_stage0_v2'
_c2_src=(OUT/'uploads/stage0_pcebr_mechanism_check_v2_original.py').read_text(encoding='utf-8')
_c2_tree=ast.parse(_c2_src)
_c2_tree.body=[node for node in _c2_tree.body if not (isinstance(node,ast.If) and isinstance(node.test,ast.BoolOp))]
_c2_ns={'__name__':'pcv2_control_helpers','OUT':OUT}
exec(compile(_c2_tree,'pcv2_controls_helpers','exec'),_c2_ns)

# The implementation's einsum preserves d in the output: the alleged low-rank
# map is actually x_d * sum_r down[d,r] up[r,d], a diagonal operation.
_c2_x=torch.tensor([[1.,0.,0.,0.]])
_c2_down=torch.ones(1,4,2);_c2_up=torch.ones(1,2,4)
_c2_actual=torch.einsum('nd,ndr,nrd->nd',_c2_x,_c2_down,_c2_up)
_c2_true_lr=torch.einsum('nd,ndr,nre->ne',_c2_x,_c2_down,_c2_up)
_c2_diagonal=_c2_x*torch.einsum('ndr,nrd->nd',_c2_down,_c2_up)
_c2_ctx=torch.tensor([.9,.8,.7,.6])
_c2_adv=torch.tensor([0.,0.,.4,-.4])
_c2_out=dict(algebra=dict(einsum_actual=_c2_actual.tolist(),proper_low_rank=_c2_true_lr.tolist(),
    diagonal_identity_error=float((_c2_actual-_c2_diagonal).abs().max()),
    group_adv_sums=[float(_c2_adv[:2].sum()),float(_c2_adv[2:].sum())],
    global_winner_before=int(_c2_ctx.argmax()),global_winner_after=int((_c2_ctx+_c2_adv).argmax())))

_c2_parts=[torch.load(OUT/'uploads'/f'avbind_native_pairs_s{s}.pt',map_location='cpu',weights_only=True) for s in (1,2)]
_c2_examples=[x for p in _c2_parts for x in p['examples']]
_c2_train=[x for x in _c2_examples if x['split']=='train'];_c2_held=[x for x in _c2_examples if x['split']=='heldout']
_c2_device='cuda:0'

if PC_SERVER==1:
    # Reuse the exact supplied synthetic episode generator and linear probes.
    # Every similarity is zero, so both retrieval distributions are uniform and
    # contain no event-specific semantic signal whatsoever.
    _c2_builder=next(n for n in ast.walk(_c2_tree) if isinstance(n,ast.FunctionDef) and n.name=='build_noncircular_retrieved_episodes')
    _c2_probe=next(n for n in ast.walk(_c2_tree) if isinstance(n,ast.FunctionDef) and n.name=='eval_linear_probe')
    _c2_probe_records=[]
    def _c2_capture_uniform(values):
        _c2_probe_records.append(dict(X_heldout=values['X_he'].detach().cpu(),y_heldout=values['y_he'].detach().cpu(),
            predictions=values['preds'].detach().cpu(),state_dict={k:v.detach().cpu() for k,v in values['clf'].state_dict().items()},
            correct=int((values['preds']==values['y_he']).sum()),n=len(values['y_he'])))
    _c2_probe.body.insert(-1,ast.Expr(value=ast.Call(func=ast.Name(id='PCV2_CAPTURE_UNIFORM',ctx=ast.Load()),args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
    _c2_ns['PCV2_CAPTURE_UNIFORM']=_c2_capture_uniform
    _c2_ns['device']=_c2_device
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_c2_builder,_c2_probe],type_ignores=[])),'pcv2_uniform_retrieval_control','exec'),_c2_ns)
    _c2_ns['tr_eps']=_c2_ns['build_noncircular_retrieved_episodes'](_c2_train,torch.zeros(len(_c2_train),len(_c2_train),device=_c2_device))
    _c2_ns['he_eps']=_c2_ns['build_noncircular_retrieved_episodes'](_c2_held,torch.zeros(len(_c2_held),len(_c2_held),device=_c2_device))
    _c2_out['uniform_retrieval_synthetic_probes']={
        'B1_std':_c2_ns['eval_linear_probe'](lambda e:e['std_attn']),
        'B3_phi':_c2_ns['eval_linear_probe'](lambda e:e['phi']),
        'B5_phi_null':_c2_ns['eval_linear_probe'](lambda e:torch.cat([e['phi'],e['null_feat']],dim=-1)),
    }
    torch.save(_c2_probe_records,_c2_folder/'uniform_control_probes.pt')
    _c2_out['uniform_raw_counts']=[dict(correct=x['correct'],n=x['n']) for x in _c2_probe_records]
else:
    _c2_model_file=_c2_folder/'artifacts/avbind_native_pairs_two_stream_hard1_shuffle0_seed20261015_model.pt'
    _c2_state=torch.load(_c2_model_file,map_location='cpu',weights_only=True)
    _c2_model=_c2_ns['IdentityAnchoredTwoStreamAligner'](mode='two_stream').to(_c2_device)
    _c2_model.load_state_dict(_c2_state['state_dict']);_c2_model.eval()
    _c2_raw=torch.stack([x['raw'] for x in _c2_held]).to(_c2_device)
    _c2_text=torch.stack([x['text'] for x in _c2_held]).to(_c2_device)
    _c2_sources=torch.tensor([0 if x['source']=='video' else 1 for x in _c2_held],device=_c2_device)
    _c2_recs=[x['recording_id'] for x in _c2_held]
    _c2_streams=_c2_ns['build_media_streams'](_c2_raw,_c2_recs,_c2_sources,[x['start'] for x in _c2_held])
    with torch.no_grad():
        _c2_full=_c2_model.score_matrix(_c2_text,*_c2_streams,_c2_sources,_c2_recs)
        _c2_single=torch.cat([_c2_model.score_matrix(t.unsqueeze(0),*_c2_streams,_c2_sources,_c2_recs) for t in _c2_text])
    _c2_out['query_batch_invariance_max_error']=float((_c2_full-_c2_single).abs().max())
_c2_path=_c2_folder/f'controls_s{PC_SERVER}.json'
_c2_path.write_text(json.dumps(_c2_out,indent=2),encoding='utf-8')
RESULT=_c2_out
print('PCV2_CONTROLS',json.dumps(RESULT),flush=True)

"""Inference-only replay of archived R14/R18 alignment checkpoints."""
import ast
import hashlib
import json
import torch
import torch.nn.functional as F

_base_source=OUT/'uploads/avbind_alignment.py'
_base_ns={'__name__':'pcebr_legacy_alignment'}
exec(compile(_base_source.read_text(encoding='utf-8'),str(_base_source),'exec'),_base_ns)
_base_suffix='r14' if PC_SERVER==1 else 'r18'
_base_prefix='avbind_native_pairs' if PC_SERVER==1 else 'avbind_pooled_pairs'
_base_checkpoint=OUT/'uploads'/f'avbind_native_alignment_{_base_suffix}.pt'
_base_saved=torch.load(_base_checkpoint,map_location='cpu',weights_only=True)
_base_model=_base_ns['NativeSemanticAlignment']().to('cuda:0')
_base_model.load_state_dict(_base_saved['state_dict'])
_base_model.eval()
_base_parts=[torch.load(OUT/'uploads'/f'{_base_prefix}_s{s}.pt',map_location='cpu',weights_only=True) for s in (1,2)]
_base_held=[x for part in _base_parts for x in part['examples'] if x['split']=='heldout']
_base_raw=torch.stack([x['raw'] for x in _base_held]).to('cuda:0')
_base_text=torch.stack([x['text'] for x in _base_held]).to('cuda:0')
_base_sources=torch.tensor([0 if x['source']=='video' else 1 for x in _base_held],device='cuda:0')
_helper_path=OUT/'uploads/stage0_pcebr_mechanism_check_devicefix.py'
_helper_tree=ast.parse(_helper_path.read_text(encoding='utf-8'))
_helper_tree.body=[node for node in _helper_tree.body if not (isinstance(node,ast.If) and isinstance(node.test,ast.BoolOp))]
_helper_ns={'__name__':'pcebr_legacy_eval'}
exec(compile(_helper_tree,str(_helper_path),'exec'),_helper_ns)
with torch.no_grad():
    _base_aligned=F.normalize(_base_model(_base_raw,_base_sources).float(),dim=-1)
    _base_query=F.normalize(_base_text.float(),dim=-1)
    _base_metrics=_helper_ns['evaluate_retrieval_and_within_rec_localization'](_base_aligned,_base_query,_base_held)
RESULT=dict(checkpoint=_base_checkpoint.name,sha256=hashlib.sha256(_base_checkpoint.read_bytes()).hexdigest(),
    params=sum(p.numel() for p in _base_model.parameters()),metrics=_base_metrics,metadata=_base_saved['metadata'])
(_base_path:=OUT/'pcebr_stage0'/f'baseline_s{PC_SERVER}.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('PCEBR_LEGACY_REPLAY',json.dumps(RESULT),flush=True)

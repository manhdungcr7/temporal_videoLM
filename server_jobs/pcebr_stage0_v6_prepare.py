"""Prepare exactly rotation0 of the four Stage-0 v6 micro rows per server."""
import gc
import hashlib
import inspect
import json
import torch
from pathlib import Path

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
_pv6_samples=CD_MANIFEST[f'server{PC_SERVER}'][:4]
_pv6_records=[]
for _pv6_row in _pv6_samples:
    assert _pv6_row['duration']<=45 and _pv6_row['decoded_audio_duration']<=45
    _pv6_stem,_pv6_texts=parse_question(_pv6_row['question'])
    _pv6_perm=next(iter(rotations(len(_pv6_texts))))
    _pv6_path=CD_CACHE/f"{_pv6_row['idx']}_rot0.pt"
    if not _pv6_path.exists() or 'input_ids' not in torch.load(_pv6_path,map_location='cpu',weights_only=True):
        _pv6_media=prepare_video(fetch(_pv6_row['video_path']),max_pixels=int(160*min(1.,32/_pv6_row['duration']))*28*28)
        _pv6_inputs,_=qwen_inputs(PROC,cd_prompt(_pv6_stem,_pv6_texts,_pv6_perm),_pv6_media,'av')
        _pv6_inputs=move_inputs(_pv6_inputs,MODEL_)
        _pv6_payload=cd_encode(_pv6_inputs,len(_pv6_texts))
        _pv6_payload['input_ids'] = _pv6_inputs['input_ids'].detach().cpu()
        torch.save(_pv6_payload,_pv6_path)
        del _pv6_inputs,_pv6_media,_pv6_payload
        gc.collect();torch.cuda.empty_cache()
    _pv6_payload=torch.load(_pv6_path,map_location='cpu',weights_only=True)
    _pv6_record=dict(idx=_pv6_row['idx'],tokens=_pv6_payload['tokens'],native_logits=_pv6_payload['native_logits'].tolist(),
        mask_shape=list(_pv6_payload['is_v'].shape),sec_shape=list(_pv6_payload['sec'].shape),
        decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in _pv6_payload['decoder'].items()},
        bytes=_pv6_path.stat().st_size)
    _pv6_records.append(_pv6_record)
    print('PCV6_MICRO_CACHE',json.dumps(_pv6_record),flush=True)
_pv6_folder=OUT/'pcebr_stage0_v6';_pv6_folder.mkdir(exist_ok=True)
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,out=str(OUT),rows=_pv6_records,manifest_sha256=CD_SHA,
    media_layout_source=inspect.getsource(media_layout),sdpa_compat=AVB_SDPA_COMPAT)
(_pv6_folder/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

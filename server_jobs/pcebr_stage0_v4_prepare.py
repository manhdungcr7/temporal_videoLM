"""Prepare exactly rotation0 of the four Stage-0 v4 micro rows per server."""
import gc
import hashlib
import inspect
import json
import torch
from pathlib import Path

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
_pv4_samples=CD_MANIFEST[f'server{PC_SERVER}'][:4]
_pv4_records=[]
for _pv4_row in _pv4_samples:
    assert _pv4_row['duration']<=45 and _pv4_row['decoded_audio_duration']<=45
    _pv4_stem,_pv4_texts=parse_question(_pv4_row['question'])
    _pv4_perm=next(iter(rotations(len(_pv4_texts))))
    _pv4_path=CD_CACHE/f"{_pv4_row['idx']}_rot0.pt"
    if not _pv4_path.exists() or 'input_ids' not in torch.load(_pv4_path,map_location='cpu',weights_only=True):
        _pv4_media=prepare_video(fetch(_pv4_row['video_path']),max_pixels=int(160*min(1.,32/_pv4_row['duration']))*28*28)
        _pv4_inputs,_=qwen_inputs(PROC,cd_prompt(_pv4_stem,_pv4_texts,_pv4_perm),_pv4_media,'av')
        _pv4_inputs=move_inputs(_pv4_inputs,MODEL_)
        _pv4_payload=cd_encode(_pv4_inputs,len(_pv4_texts))
        _pv4_payload['input_ids'] = _pv4_inputs['input_ids'].detach().cpu()
        torch.save(_pv4_payload,_pv4_path)
        del _pv4_inputs,_pv4_media,_pv4_payload
        gc.collect();torch.cuda.empty_cache()
    _pv4_payload=torch.load(_pv4_path,map_location='cpu',weights_only=True)
    _pv4_record=dict(idx=_pv4_row['idx'],tokens=_pv4_payload['tokens'],native_logits=_pv4_payload['native_logits'].tolist(),
        mask_shape=list(_pv4_payload['is_v'].shape),sec_shape=list(_pv4_payload['sec'].shape),
        decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in _pv4_payload['decoder'].items()},
        bytes=_pv4_path.stat().st_size)
    _pv4_records.append(_pv4_record)
    print('PCV4_MICRO_CACHE',json.dumps(_pv4_record),flush=True)
_pv4_folder=OUT/'pcebr_stage0_v4';_pv4_folder.mkdir(exist_ok=True)
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,out=str(OUT),rows=_pv4_records,manifest_sha256=CD_SHA,
    media_layout_source=inspect.getsource(media_layout),sdpa_compat=AVB_SDPA_COMPAT)
(_pv4_folder/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

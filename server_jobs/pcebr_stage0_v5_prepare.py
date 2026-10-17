"""Prepare exactly rotation0 of the four Stage-0 v5 micro rows per server."""
import gc
import hashlib
import inspect
import json
import torch
from pathlib import Path

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
_pv5_samples=CD_MANIFEST[f'server{PC_SERVER}'][:4]
_pv5_records=[]
for _pv5_row in _pv5_samples:
    assert _pv5_row['duration']<=45 and _pv5_row['decoded_audio_duration']<=45
    _pv5_stem,_pv5_texts=parse_question(_pv5_row['question'])
    _pv5_perm=next(iter(rotations(len(_pv5_texts))))
    _pv5_path=CD_CACHE/f"{_pv5_row['idx']}_rot0.pt"
    if not _pv5_path.exists() or 'input_ids' not in torch.load(_pv5_path,map_location='cpu',weights_only=True):
        _pv5_media=prepare_video(fetch(_pv5_row['video_path']),max_pixels=int(160*min(1.,32/_pv5_row['duration']))*28*28)
        _pv5_inputs,_=qwen_inputs(PROC,cd_prompt(_pv5_stem,_pv5_texts,_pv5_perm),_pv5_media,'av')
        _pv5_inputs=move_inputs(_pv5_inputs,MODEL_)
        _pv5_payload=cd_encode(_pv5_inputs,len(_pv5_texts))
        _pv5_payload['input_ids'] = _pv5_inputs['input_ids'].detach().cpu()
        torch.save(_pv5_payload,_pv5_path)
        del _pv5_inputs,_pv5_media,_pv5_payload
        gc.collect();torch.cuda.empty_cache()
    _pv5_payload=torch.load(_pv5_path,map_location='cpu',weights_only=True)
    _pv5_record=dict(idx=_pv5_row['idx'],tokens=_pv5_payload['tokens'],native_logits=_pv5_payload['native_logits'].tolist(),
        mask_shape=list(_pv5_payload['is_v'].shape),sec_shape=list(_pv5_payload['sec'].shape),
        decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in _pv5_payload['decoder'].items()},
        bytes=_pv5_path.stat().st_size)
    _pv5_records.append(_pv5_record)
    print('PCV5_MICRO_CACHE',json.dumps(_pv5_record),flush=True)
_pv5_folder=OUT/'pcebr_stage0_v5';_pv5_folder.mkdir(exist_ok=True)
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,out=str(OUT),rows=_pv5_records,manifest_sha256=CD_SHA,
    media_layout_source=inspect.getsource(media_layout),sdpa_compat=AVB_SDPA_COMPAT)
(_pv5_folder/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

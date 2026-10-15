"""Prepare exactly rotation0 of the four Stage-0 v2 micro rows per server."""
import gc
import hashlib
import inspect
import json
import torch
from pathlib import Path

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
_pv2_samples=CD_MANIFEST[f'server{PC_SERVER}'][:4]
_pv2_records=[]
for _pv2_row in _pv2_samples:
    assert _pv2_row['duration']<=45 and _pv2_row['decoded_audio_duration']<=45
    _pv2_stem,_pv2_texts=parse_question(_pv2_row['question'])
    _pv2_perm=next(iter(rotations(len(_pv2_texts))))
    _pv2_path=CD_CACHE/f"{_pv2_row['idx']}_rot0.pt"
    if not _pv2_path.exists():
        _pv2_media=prepare_video(fetch(_pv2_row['video_path']),max_pixels=int(160*min(1.,32/_pv2_row['duration']))*28*28)
        _pv2_inputs,_=qwen_inputs(PROC,cd_prompt(_pv2_stem,_pv2_texts,_pv2_perm),_pv2_media,'av')
        _pv2_inputs=move_inputs(_pv2_inputs,MODEL_)
        _pv2_payload=cd_encode(_pv2_inputs,len(_pv2_texts))
        torch.save(_pv2_payload,_pv2_path)
        del _pv2_inputs,_pv2_media,_pv2_payload
        gc.collect();torch.cuda.empty_cache()
    _pv2_payload=torch.load(_pv2_path,map_location='cpu',weights_only=True)
    _pv2_record=dict(idx=_pv2_row['idx'],tokens=_pv2_payload['tokens'],native_logits=_pv2_payload['native_logits'].tolist(),
        mask_shape=list(_pv2_payload['is_v'].shape),sec_shape=list(_pv2_payload['sec'].shape),
        decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in _pv2_payload['decoder'].items()},
        bytes=_pv2_path.stat().st_size)
    _pv2_records.append(_pv2_record)
    print('PCV2_MICRO_CACHE',json.dumps(_pv2_record),flush=True)
_pv2_folder=OUT/'pcebr_stage0_v2';_pv2_folder.mkdir(exist_ok=True)
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,out=str(OUT),rows=_pv2_records,manifest_sha256=CD_SHA,
    media_layout_source=inspect.getsource(media_layout),sdpa_compat=AVB_SDPA_COMPAT)
(_pv2_folder/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

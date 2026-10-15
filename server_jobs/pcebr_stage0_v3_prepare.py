"""Prepare exactly rotation0 of the four Stage-0 v3 micro rows per server."""
import gc
import hashlib
import inspect
import json
import torch
from pathlib import Path

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
_pv3_samples=CD_MANIFEST[f'server{PC_SERVER}'][:4]
_pv3_records=[]
for _pv3_row in _pv3_samples:
    assert _pv3_row['duration']<=45 and _pv3_row['decoded_audio_duration']<=45
    _pv3_stem,_pv3_texts=parse_question(_pv3_row['question'])
    _pv3_perm=next(iter(rotations(len(_pv3_texts))))
    _pv3_path=CD_CACHE/f"{_pv3_row['idx']}_rot0.pt"
    if not _pv3_path.exists() or 'input_ids' not in torch.load(_pv3_path,map_location='cpu',weights_only=True):
        _pv3_media=prepare_video(fetch(_pv3_row['video_path']),max_pixels=int(160*min(1.,32/_pv3_row['duration']))*28*28)
        _pv3_inputs,_=qwen_inputs(PROC,cd_prompt(_pv3_stem,_pv3_texts,_pv3_perm),_pv3_media,'av')
        _pv3_inputs=move_inputs(_pv3_inputs,MODEL_)
        _pv3_payload=cd_encode(_pv3_inputs,len(_pv3_texts))
        _pv3_payload['input_ids'] = _pv3_inputs['input_ids'].detach().cpu()
        torch.save(_pv3_payload,_pv3_path)
        del _pv3_inputs,_pv3_media,_pv3_payload
        gc.collect();torch.cuda.empty_cache()
    _pv3_payload=torch.load(_pv3_path,map_location='cpu',weights_only=True)
    _pv3_record=dict(idx=_pv3_row['idx'],tokens=_pv3_payload['tokens'],native_logits=_pv3_payload['native_logits'].tolist(),
        mask_shape=list(_pv3_payload['is_v'].shape),sec_shape=list(_pv3_payload['sec'].shape),
        decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in _pv3_payload['decoder'].items()},
        bytes=_pv3_path.stat().st_size)
    _pv3_records.append(_pv3_record)
    print('PCV3_MICRO_CACHE',json.dumps(_pv3_record),flush=True)
_pv3_folder=OUT/'pcebr_stage0_v3';_pv3_folder.mkdir(exist_ok=True)
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,out=str(OUT),rows=_pv3_records,manifest_sha256=CD_SHA,
    media_layout_source=inspect.getsource(media_layout),sdpa_compat=AVB_SDPA_COMPAT)
(_pv3_folder/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

"""Prepare exactly rotation0 of the four Stage-0 v7 micro rows per server."""
import gc
import hashlib
import inspect
import json
import torch
from pathlib import Path

CD_SERVER=PC_SERVER
CD_CACHE_RUN=False
exec(compile((OUT/'uploads/chronodesmear_cache40.py').read_text(encoding='utf-8'),'chronodesmear_cache40.py','exec'),globals())
_pv7_samples=CD_MANIFEST[f'server{PC_SERVER}'][:4]
_pv7_records=[]
for _pv7_row in _pv7_samples:
    assert _pv7_row['duration']<=45 and _pv7_row['decoded_audio_duration']<=45
    _pv7_stem,_pv7_texts=parse_question(_pv7_row['question'])
    _pv7_perm=next(iter(rotations(len(_pv7_texts))))
    _pv7_path=CD_CACHE/f"{_pv7_row['idx']}_rot0.pt"
    if not _pv7_path.exists() or 'input_ids' not in torch.load(_pv7_path,map_location='cpu',weights_only=True):
        _pv7_media=prepare_video(fetch(_pv7_row['video_path']),max_pixels=int(160*min(1.,32/_pv7_row['duration']))*28*28)
        _pv7_inputs,_=qwen_inputs(PROC,cd_prompt(_pv7_stem,_pv7_texts,_pv7_perm),_pv7_media,'av')
        _pv7_inputs=move_inputs(_pv7_inputs,MODEL_)
        _pv7_payload=cd_encode(_pv7_inputs,len(_pv7_texts))
        _pv7_payload['input_ids'] = _pv7_inputs['input_ids'].detach().cpu()
        torch.save(_pv7_payload,_pv7_path)
        del _pv7_inputs,_pv7_media,_pv7_payload
        gc.collect();torch.cuda.empty_cache()
    _pv7_payload=torch.load(_pv7_path,map_location='cpu',weights_only=True)
    _pv7_record=dict(idx=_pv7_row['idx'],tokens=_pv7_payload['tokens'],native_logits=_pv7_payload['native_logits'].tolist(),
        mask_shape=list(_pv7_payload['is_v'].shape),sec_shape=list(_pv7_payload['sec'].shape),
        decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in _pv7_payload['decoder'].items()},
        bytes=_pv7_path.stat().st_size)
    _pv7_records.append(_pv7_record)
    print('PCV7_MICRO_CACHE',json.dumps(_pv7_record),flush=True)
_pv7_folder=OUT/'pcebr_stage0_v7';_pv7_folder.mkdir(exist_ok=True)
RESULT=dict(server=PC_SERVER,model=MODEL_.config._name_or_path,out=str(OUT),rows=_pv7_records,manifest_sha256=CD_SHA,
    media_layout_source=inspect.getsource(media_layout),sdpa_compat=AVB_SDPA_COMPAT)
(_pv7_folder/'prepare_audit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

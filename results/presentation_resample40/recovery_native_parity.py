_sample=SL_SAMPLES[0]
assert _sample['idx']==1898
_stem,_texts=parse_question(_sample['question']);_k=len(_texts)
_media=prepare_video(fetch(_sample['video_path']),max_pixels=int(160*min(1.,32/_sample['duration']))*28*28)
_parity=[]
for _rotation,_perm in enumerate(rotations(_k)):
    _inputs,_=qwen_inputs(PROC,argon_prompt(_stem,_texts,_perm),_media,'av')
    _inputs=move_inputs(_inputs,MODEL_)
    _payload=argon_encode(_inputs,_k)
    _lp=torch.log_softmax(_payload['native_logits'],dim=-1).numpy()
    _ref=json.loads((OUT/'presentation_resample_recovery_refs'/f'1898_rot{_rotation}.json').read_text())['scores']['R4_C0_baseline']
    _error=float(np.max(abs(_lp-np.asarray(_ref))))
    assert _error<=.12,('Cross-worker original-native mismatch',_error)
    _parity.append(dict(rotation=_rotation,max_error=_error))
    del _inputs,_payload;gc.collect();torch.cuda.empty_cache()
_RESULT=dict(idx=1898,model='Qwen2.5-Omni-3B FP16',native_reference_parity=_parity,max_error=max(x['max_error'] for x in _parity))
(OUT/'presentation_resample40/recovery_native_parity.json').write_text(json.dumps(_RESULT,indent=2))
print('RECOVERY_NATIVE_PARITY',json.dumps(_RESULT),flush=True)

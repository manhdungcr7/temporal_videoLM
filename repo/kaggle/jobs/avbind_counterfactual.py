"""Counterfactual sensor keys: preserve content, accentuate changing evidence.

Vision control freezes the first decoded frame; audio control is silence.
Native answer media and factual value states remain unmodified. No question,
options, timestamps or benchmark label is exposed to either key encoding.
"""
def sensor_control(media,source):
    result=dict(media)
    if source=='video':
        frames=media['videos'][0]
        static=frames[:1].expand_as(frames).clone() if torch.is_tensor(frames) else np.repeat(frames[:1],len(frames),axis=0)
        result['videos']=[static]
    elif source=='audio':result['audio']=[np.zeros_like(media['audio'][0])]
    else:raise ValueError('Unknown modality')
    return result

def differential_sensor_bank(src,duration,cache_path=None):
    cache=Path(cache_path) if cache_path else None
    metadata=dict(media_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),revision=BIND_METADATA['revision'],layer=BIND_LAYER,
                  representation='layer28 raw + (raw-control); fixed coefficient 1',interval_sec=2,question_used=False)
    if cache and cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        if saved['metadata']!=metadata:raise ValueError('Counterfactual key cache mismatch')
        return saved['nodes'],saved['features']
    nodes=[];features=[]
    for index in range(int(math.ceil(duration/2))):
        start,end=2*index,min(2*index+2,duration)
        if end-start<.2:continue
        crop=cut(src,start,end,f'avbind_latent_{index}');media=prepare_video(crop)
        for source,mode in (('video','vision'),('audio','audio')):
            raw=latent_sensor_feature(media,mode)
            control=latent_sensor_feature(sensor_control(media,source),mode)
            features.append(F.layer_norm(2*raw-control,(raw.shape[-1],)))
            nodes.append(dict(id=f'{source[0]}{index}',source=source,interval=[start,end]))
        print('DIFFERENTIAL_SLOT',json.dumps(dict(start=start,end=end,coefficient=1,original_answer_media_unchanged=True)),flush=True)
    features=torch.stack(features)
    if cache:torch.save(dict(nodes=nodes,features=features,metadata=metadata),cache)
    return nodes,features

def avbind_counterfactual_answer(video_path,question,answer_options=None,slot_cache=None,value_cache=None,gain=1.0):
    original=globals()['latent_sensor_bank']
    globals()['latent_sensor_bank']=differential_sensor_bank
    try:
        result=avbind_kv_answer(video_path,question,answer_options,slot_cache,value_cache,gain)
        result['counterfactual_keys']=True;result['coefficient']=1
        return result
    finally:globals()['latent_sensor_bank']=original

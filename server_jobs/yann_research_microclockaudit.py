import json,torch
YR_DIR=OUT/'yann_temporal_research'
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text());rows=[]
for row in manifest[f'server{YR_SERVER}'][:4]:
    c=torch.load(YR_DIR/'microfeatures'/f"{row['idx']}.pt",map_location='cpu',weights_only=True)
    p=torch.load(OUT/'argon_latent40/decoder_cache'/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=True)
    rows.append(dict(idx=row['idx'],duration=row['duration'],bank_video_times=c['times'][c['mods']==1].tolist(),
        bank_audio_times=c['times'][c['mods']==0].tolist(),native_video_times=torch.unique(p['sec'][p['is_v']],sorted=True).tolist(),
        cache_keys=list(p),spg=p.get('spg'),bank_spans=c['spans'][c['mods']==1].tolist(),
        decoder_v_times=torch.unique((p['decoder']['position_ids'][0,0,p['is_v']].float()-p['decoder']['position_ids'][0,0,p['is_a']].min())/25,sorted=True).tolist()))
RESULT=dict(rows=rows)
(YR_DIR/'microclockaudit.json').write_text(json.dumps(RESULT,indent=2));print(json.dumps(RESULT),flush=True)

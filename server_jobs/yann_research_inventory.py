import inspect,json,torch
from pathlib import Path
YR_DIR=OUT/'yann_temporal_research';YR_DIR.mkdir(exist_ok=True)
yr_manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
yr_row=yr_manifest[f'server{YR_SERVER}'][0]
yr_p=torch.load(OUT/'argon_latent40/decoder_cache'/f"{yr_row['idx']}_rot0.pt",map_location='cpu',weights_only=True)
RESULT=dict(server=YR_SERVER,out=str(OUT),globals_present={k:k in globals() for k in ('MODEL_','PROC','CD_CACHE','CD_MANIFEST','V7_NS','DATA','ROWS','df','annotations','LAB_CONFIG')},
    source_model_forward=inspect.getsource(MODEL_.model.forward),source_layer_forward=inspect.getsource(MODEL_.model.layers[0].forward),
    cache={k:dict(shape=list(v.shape),dtype=str(v.dtype)) if torch.is_tensor(v) else str(type(v)) for k,v in yr_p.items()},
    decoder={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in yr_p['decoder'].items()},
    helpers={k:inspect.getsource(globals()[k]) for k in ('prepare_video','qwen_inputs','fetch','decode_audio','media_layout') if k in globals()},
    files=[str(p.relative_to(OUT)) for p in OUT.glob('*')],
    config={k:v for k,v in MODEL_.config.to_dict().items() if k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','rope_scaling')},
    sample=yr_row)
(YR_DIR/'inventory.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in RESULT.items() if k not in ('source_model_forward','source_layer_forward','helpers')},ensure_ascii=False),flush=True)

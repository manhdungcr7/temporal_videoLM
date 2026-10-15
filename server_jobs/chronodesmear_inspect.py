import inspect, json, shutil
from pathlib import Path
RESULT = dict(
    model=MODEL_.config._name_or_path,
    layers=len(MODEL_.model.layers),
    out=str(OUT),
    out_files=[p.name for p in OUT.iterdir()],
    uploads=[p.name for p in (OUT/'uploads').iterdir()] if (OUT/'uploads').exists() else [],
    decoder_cache_files=len(list((OUT/'argon_latent40/decoder_cache').glob('*.pt'))),
    available=[n for n in ('fetch','prepare_video','qwen_inputs','move_inputs','media_layout',
        'parse_question','rotations','render','decode_audio','SYSTEM') if n in globals()],
    disk_free_gb=round(shutil.disk_usage(OUT).free/2**30,2),
    model_forward=str(inspect.signature(MODEL_.model.forward)),
)
print(json.dumps(RESULT),flush=True)

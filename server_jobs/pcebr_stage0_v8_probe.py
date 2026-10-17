import json, torch
from pathlib import Path
RESULT = dict(out=str(OUT), model=str(MODEL_.config._name_or_path),
    helpers={k: k in globals() for k in ['PROC','prepare_video','fetch','qwen_inputs','move_inputs','media_layout','render','parse_question','rotations']},
    files=[str(p.relative_to(OUT)) for p in Path(OUT).glob('*')],
    cache_count=len(list((Path(OUT)/'argon_latent40/decoder_cache').glob('*.pt'))),
    dtype=str(next(MODEL_.parameters()).dtype), torch=torch.__version__)
print(json.dumps(RESULT),flush=True)

import inspect, json, shutil
import torch
RESULT = dict(model=MODEL_.config._name_or_path, layers=len(MODEL_.model.layers),
    dtype=str(MODEL_.get_input_embeddings().weight.dtype),
    decoder_signature=str(inspect.signature(MODEL_.model.forward)),
    layer_forward=inspect.getsource(type(MODEL_.model.layers[0]).forward),
    globals_available=[x for x in ('prepare_video','media_layout','speech_window','render','MODEL_','PROC') if x in globals()],
    cache_files=len(list((OUT/'research40_decoder_cache').glob('*_rot*.pt'))),
    out_files=[x.name for x in OUT.iterdir()], ffmpeg=shutil.which('ffmpeg'))
print(json.dumps(RESULT),flush=True)

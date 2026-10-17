import inspect,json,torch
from transformers.cache_utils import DynamicCache
YR_DIR=OUT/'yann_temporal_research'
RESULT=dict(model=inspect.getsource(type(MODEL_.model).forward),block=inspect.getsource(type(MODEL_.model.layers[-1]).forward),
    attention=inspect.getsource(type(MODEL_.model.layers[-1].self_attn).forward),cache=inspect.getsource(DynamicCache),
    config=MODEL_.config.to_dict(),device_layers=[str(next(l.parameters()).device) for l in MODEL_.model.layers],
    disk=__import__('shutil').disk_usage('/kaggle/working')._asdict())
(YR_DIR/'decoderinspect.json').write_text(json.dumps(RESULT,indent=2,default=str));print(json.dumps(dict(devices=RESULT['device_layers'],disk=RESULT['disk'])),flush=True)

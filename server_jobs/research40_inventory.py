import sys, json, inspect
import torch
main_module=sys.modules['__main__']
jobs=getattr(main_module,'JOBS',{})
details={}
for name in ('prepare_video','qwen_inputs','media_layout','parse_question','prompt_of','rotations','fetch','decode_audio','speech_window'):
    fn=globals().get(name)
    if callable(fn):
        try: details[name]=inspect.getsource(fn)
        except (OSError,TypeError): details[name]=dict(signature=str(inspect.signature(fn)),filename=fn.__code__.co_filename)
    else: details[name]=None
RESULT=dict(model=MODEL_.config.to_dict(),layers=len(MODEL_.model.layers),
            root=str(ROOT),out=str(OUT),helpers=details,
            jobs=[dict(id=i,name=j.get('name'),status=j.get('status'),tail=j.get('out','')[-1200:]) for i,j in jobs.items()],
            torch_version=torch.__version__,namespace_keys=[k for k in globals() if not k.startswith('_')],
            attention_source=inspect.getsource(type(MODEL_.model.layers[0].self_attn)),
            decoder_source=inspect.getsource(type(MODEL_.model.layers[0])))
print('INVENTORY',json.dumps({k:RESULT[k] for k in ('layers','root','out','jobs','torch_version')}),flush=True)
(OUT/'research40_inventory.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

import inspect,json,shutil,torch
RESULT=dict(out=str(OUT),torch=torch.__version__,ffmpeg=shutil.which('ffmpeg'),
    helpers={n:inspect.getsource(globals()[n]) for n in ('prepare_video','qwen_inputs','move_inputs')},
    live_model=str(MODEL_.config._name_or_path),hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()),
    perception_files=[str(p) for p in OUT.rglob('video_*.mp4')][:4])
print(json.dumps(RESULT),flush=True)

import inspect,json,numpy as np,torch
import qwen_omni_utils
from qwen_omni_utils import audio_process
row=json.loads((OUT/'uploads/perception48_manifest.json').read_text())['rows'][YR_SERVER-1]
path=OUT/'yann_temporal_research/perception48'/f"{row['video_id']}.mp4"
m=prepare_video(path,max_pixels=96*28*28)
RESULT=dict(server=YR_SERVER,video_id=row['video_id'],annotation_duration=row['duration'],annotation_audio_duration=row['audio_duration'],
    audio_type=str(type(m['audio'][0])),audio_shape=list(np.asarray(m['audio'][0]).shape),fps=m['fps'],video_shape=list(m['videos'][0].shape),
    audio_utils_source=inspect.getsource(audio_process.process_audio_info),processor_audio_rate=PROC.feature_extractor.sampling_rate)
(OUT/'yann_temporal_research/mediaaudit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print(json.dumps(RESULT),flush=True)

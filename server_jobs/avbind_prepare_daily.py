"""Small cross-domain diagnostic from official example raw media and QA only.

Never load the published captions, explanations, or alignment annotations
into inference. This is not a representative Daily-Omni benchmark result.
"""
import requests,hashlib
from huggingface_hub import hf_hub_download,HfApi
from concurrent.futures import ThreadPoolExecutor
daily=OUT/'avbind_daily_examples';daily.mkdir(exist_ok=True)
dataset_revision=HfApi().dataset_info('liarliar/Daily-Omni').sha
qa_path=Path(hf_hub_download('liarliar/Daily-Omni','qa.json',repo_type='dataset',revision=dataset_revision))
qa=json.loads(qa_path.read_text())
video_ids=('G_VTkkb34gw','bswbQtOPk6E','d6b4OmUFt7I')
commit=requests.get('https://api.github.com/repos/Lliar-liar/Daily-Omni/commits/main',timeout=30).json()['sha']
paths={}
for video_id in video_ids:
    files=[]
    for suffix in ('video.mp4','audio.wav'):
        path=daily/f'{video_id}_{suffix}'
        if not path.exists():
            url=f'https://raw.githubusercontent.com/Lliar-liar/Daily-Omni/{commit}/example_videos/{video_id}/{video_id}_{suffix}'
            response=requests.get(url,timeout=60);response.raise_for_status();path.write_bytes(response.content)
        files.append(path)
    dst=daily/f'{video_id}_av.mp4'
    if not dst.exists():
        ffmpeg('-i',str(files[0]),'-i',str(files[1]),'-map','0:v:0','-map','1:a:0',
               '-c:v','copy','-c:a','aac','-shortest',str(dst))
    with av.open(str(dst)) as c:duration=float(c.duration/av.time_base)
    if not 29<duration<31:raise RuntimeError(f'Unexpected example duration: {video_id} {duration}')
    paths[video_id]=str(dst)
    print('DAILY_MEDIA_READY',video_id,duration,flush=True)
available=[(i,row) for i,row in enumerate(qa) if row['video_id'] in paths and row.get('video_duration')=='30s']
by={}
for i,row in available:by.setdefault(row['Type'],[]).append(dict(idx=i,video_id=row['video_id'],path=paths[row['video_id']],question=row['Question']))
selected=[values[0] for key,values in sorted(by.items())]
manifest=dict(dataset_revision=dataset_revision,github_commit=commit,
              qa_sha256=hashlib.sha256(qa_path.read_bytes()).hexdigest(),
              selected=selected,available_by_type={k:len(v) for k,v in by.items()},
              selection='first official 30s QA per available type on the three public example videos; not success-selected')
(daily/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(daily/'qa.json').write_bytes(qa_path.read_bytes())
RESULT=manifest
print('DAILY_EXAMPLES_READY',json.dumps(dict(n=len(selected),available_by_type=manifest['available_by_type'])),flush=True)

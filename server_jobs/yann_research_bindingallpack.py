"""Audit native tubelet support, preserve raw support, then package R4 cases."""
import hashlib,json,tarfile,torch
YR_DIR=OUT/'yann_temporal_research'
manifest=json.loads((OUT/'uploads/perception48_manifest.json').read_text());lookup={r['video_id']:r for r in manifest['rows']}
files=[];audits=[]
for p in sorted(list((YR_DIR/'binding48_v2').glob('*.pt'))+list((YR_DIR/'binding_evalextra').glob('*.pt'))):
    if p.stem.endswith('_other'):continue
    c=torch.load(p,map_location='cpu',weights_only=True);row=lookup[c['video_id']]
    fps=c['original_frames']/row['duration'];v=c['mods']==1
    if 'original_halfsecond_spans' not in c:c['original_halfsecond_spans']=c['spans'].clone()
    # One visual temporal patch spans two native sampled frames, not half a second.
    c['spans'][v,0]=c['times'][v].clamp_min(0)
    c['spans'][v,1]=(c['times'][v]+2/fps).clamp_max(c['duration'])
    c.update(native_video_fps=fps,bank_support_version='native visual tubelet2/fps; audio half-second bins')
    torch.save(c,p);files.append(p)
    audits.append(dict(video_id=c['video_id'],variant=c['variant'],fps=fps,video_support_seconds=2/fps,
        prompt_len=c['input_prompt_len'],tokens=c['tokens'],duration=c['duration']))
index=[dict(name=str(p.relative_to(YR_DIR)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
p=YR_DIR/f'binding_all_features_s{YR_SERVER}.tar.gz'
with tarfile.open(p,'w:gz',compresslevel=1) as z:
    for f in files:z.add(f,arcname=str(f.relative_to(YR_DIR)),recursive=False)
RESULT=dict(server=YR_SERVER,files=index,archive=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),support_audits=audits)
(YR_DIR/'bindingallpack.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('YR_BINDINGPACK',json.dumps(dict(files=len(files),bytes=RESULT['bytes'],sha256=RESULT['sha256'])),flush=True)

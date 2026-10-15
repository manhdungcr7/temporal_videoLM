"""Read-only media timing audit; no baseline/QA scoring, no media modification."""
import json,importlib,inspect,time,gc
import numpy as np
import torch
import av
import decord
from PIL import Image,ImageDraw

R40_AUDIT_DIR=OUT/'research40_sampling_audit';R40_AUDIT_DIR.mkdir(exist_ok=True)
rows=[]
for row in R40_SAMPLES:
    path=fetch(row['video_path']);vr=decord.VideoReader(str(path))
    n_original=len(vr);original_fps=float(vr.get_avg_fps())
    media=prepare_video(path,max_pixels=base_by_id[row['idx']]['pixels'])
    frames=media['videos'][0];n=len(frames)
    indices=torch.linspace(0,n_original-1,n).round().long().tolist()
    # Supported Decord reader uses this uniform index rule. Save actual source
    # frame PTS as well as nominal sampler fps; they are not identical clocks.
    timestamps=np.asarray(vr.get_frame_timestamp(indices),dtype=float)
    chunks=[]
    for start in np.arange(0,row['decoded_audio_duration'],2.):
        start=float(start);end=min(start+2.,row['decoded_audio_duration'])
        if end-start<.05:continue
        lo=max(0,int(np.floor(start*media['fps'])));hi=min(n,int(np.ceil(end*media['fps'])))
        if hi<=lo:lo=max(0,n-1);hi=n
        actual=timestamps[lo:hi,0]
        chunks.append(dict(start=start,end=end,nominal_indices=list(range(lo,hi)),
            source_frame_pts=actual.tolist(),frames_before_start=int((actual<start-1e-6).sum()),
            frames_at_or_after_end=int((actual>=end-1e-6).sum()),
            early_excursion=max(0.,start-float(actual.min())),
            late_excursion=max(0.,float(actual.max())-end)))
    with av.open(str(path)) as container:
        first=next(container.decode(audio=0))
        first_audio_pts=None if first.pts is None else float(first.pts*first.time_base)
    rec=dict(idx=row['idx'],frames=n,original_frames=n_original,original_fps=original_fps,
        sampler_fps=media['fps'],source_indices=indices,source_pts=timestamps.tolist(),
        first_decoded_audio_pts=first_audio_pts,chunks=chunks)
    rows.append(rec)
    if row['idx'] in (1449,1528,2054):
        picked=np.linspace(0,n-1,min(16,n)).round().astype(int)
        width,height=320,210;sheet=Image.new('RGB',(width*4,height*4),'white');draw=ImageDraw.Draw(sheet)
        for slot,index in enumerate(picked):
            pixels=frames[index].permute(1,2,0).cpu().numpy()
            thumb=Image.fromarray(np.clip(pixels,0,255).astype('uint8'));thumb.thumbnail((320,185))
            x=(slot%4)*width;y=(slot//4)*height
            sheet.paste(thumb,(x,y+22));draw.text((x+4,y+4),f"sample {row['idx']} frame {index} @ {timestamps[index,0]:.3f}s",fill='black')
        sheet.save(R40_AUDIT_DIR/f"{row['idx']}_native_frames.jpg")
    print('SAMPLING_AUDIT',row['idx'],round(max(c['early_excursion'] for c in chunks),4),flush=True)
    del media,frames,vr;gc.collect()
summary=dict(server=R40_SERVER,manifest_sha256=R40_MANIFEST['manifest_sha256'],count=len(rows),rows=rows,
    max_early_excursion=max(c['early_excursion'] for r in rows for c in r['chunks']),
    affected_chunks=sum(c['frames_before_start']>0 or c['frames_at_or_after_end']>0 for r in rows for c in r['chunks']))
(OUT/f'research40_sampling_audit_s{R40_SERVER}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
RESULT={k:v for k,v in summary.items() if k!='rows'};print('SAMPLING_AUDIT_SUMMARY',json.dumps(RESULT),flush=True)

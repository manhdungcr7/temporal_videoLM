"""Cross-check three previously flagged recordings without occupying GPU workers."""
import concurrent.futures,json,re,subprocess,urllib.request
from pathlib import Path
import av
import imageio_ffmpeg
import numpy as np
import decord

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/presentation_clock_audit';FOLDER.mkdir(exist_ok=True)
MANIFEST=json.loads((ROOT/'results/research40_manifest.json').read_text())
REV='211b4f14ab66ae9557b69e302d1f090e0e610839'

def audit(row):
    idx=row['idx'];path=FOLDER/f'{idx}.mp4'
    if not path.exists():
        url='https://huggingface.co/datasets/zcs15/SVHalluc/resolve/'+REV+'/'+row['video_path']
        with urllib.request.urlopen(url,timeout=60) as response:path.write_bytes(response.read())
    pts=[];discard=[];first_rgb=None
    with av.open(str(path)) as container:
        origin=float((container.start_time or 0)/av.time_base)
        for frame in container.decode(video=0):
            if first_rgb is None:first_rgb=frame.to_ndarray(format='rgb24')
            pts.append(float(frame.pts*frame.time_base))
            discard.append(bool(getattr(frame,'is_corrupt',False)))
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-copyts','-i',str(path),
         '-map','0:v:0','-vf','showinfo','-an','-f','null','-']
    ff=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    ff_pts=[float(x) for x in re.findall(r'\bpts_time:([+\-\d.eE]+)',ff.stderr)]
    assert ff.returncode==0 and len(ff_pts)>1,(ff.returncode,ff.stderr[-300:])
    common=min(len(pts),len(ff_pts))
    error=float(np.max(np.abs(np.array(pts[:common])-np.array(ff_pts[:common]))))
    vr=decord.VideoReader(str(path))
    decord_pts=np.asarray(vr.get_frame_timestamp(list(range(len(vr))))).astype(float)[:,0]
    expected_offset=int((decord_pts<origin).sum())
    probe_indices=sorted(set([0,1,2]+list(range(max(0,expected_offset-2),min(len(vr),expected_offset+3)))))
    rgb=vr.get_batch(probe_indices).asnumpy()
    pixel_errors=np.abs(rgb.astype(float)-first_rgb[None].astype(float)).mean(axis=(1,2,3))
    matched_index=probe_indices[int(pixel_errors.argmin())]
    result=dict(idx=idx,manifest_sha256=MANIFEST['manifest_sha256'],duration=row['duration'],
        container_origin=origin,pyav_frames=len(pts),ffmpeg_frames=len(ff_pts),
        pyav_pts=pts,ffmpeg_pts=ff_pts,cross_decoder_index_pts_error=error,
        negative_pts_frames=sum(t<origin for t in pts),first_pts=pts[0],last_pts=pts[-1],
        max_frame_interval=float(max(np.diff(pts))),min_frame_interval=float(min(np.diff(pts))),
        pyav_nonmonotone_steps=int((np.diff(pts)<0).sum()),
        decord_frames=len(vr),decord_first_pts=float(decord_pts[0]),
        decord_negative_frames=expected_offset,decord_matching_first_display_frame=matched_index,
        decord_first_display_pixel_mae=float(pixel_errors.min()),
        decord_probe_pixel_mae={str(i):float(e) for i,e in zip(probe_indices,pixel_errors)},
        note='Agreement on timestamps is not by itself a proof of the intended display clock or QA correctness.')
    (FOLDER/f'{idx}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('pyav_pts','ffmpeg_pts')}),flush=True)
    return result

if __name__=='__main__':
    selected=[r for r in MANIFEST['selected'] if r['idx'] in (1449,1528,2054)]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:results=list(pool.map(audit,selected))
    (FOLDER/'summary.json').write_text(json.dumps(results,indent=2))

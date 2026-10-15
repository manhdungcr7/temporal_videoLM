"""Match sampled decoded pixels to displayed frames instead of trusting Decord PTS."""
import json
from pathlib import Path
import av,decord
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/presentation_clock_audit'
catalog={r['idx']:r for s in (1,2) for r in
         json.loads((ROOT/f'results/research40_sampling_audit_s{s}.json').read_text())['rows']}

def run(idx):
    path=FOLDER/f'{idx}.mp4';reference=[];pts=[];stride=None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb=frame.to_ndarray(format='rgb24')
            if stride is None:stride=(max(1,rgb.shape[0]//40),max(1,rgb.shape[1]//60))
            reference.append(rgb[::stride[0],::stride[1]].reshape(-1))
            pts.append(float(frame.pts*frame.time_base))
    reference=np.stack(reference).astype('float32');pts=np.array(pts)
    vr=decord.VideoReader(str(path));indices=catalog[idx]['source_indices']
    sampled=vr.get_batch(indices).asnumpy()
    sampled=sampled[:,::stride[0],::stride[1]].reshape(len(indices),-1).astype('float32')
    fps=catalog[idx]['sampler_fps'];matches=[]
    for slot,pixels in enumerate(sampled):
        mse=((reference-pixels[None])**2).mean(axis=1)
        best=int(mse.argmin());near=np.flatnonzero(mse<=float(mse[best])+.5)
        clock_range=[float(pts[near].min()),float(pts[near].max())]
        matches.append(dict(slot=slot,native_index=indices[slot],nominal_time=slot/fps,
            best_display_index=best,best_display_pts=float(pts[best]),pixel_rmse=float(np.sqrt(mse[best])),
            plausible_time_range=clock_range,ambiguous=clock_range[1]-clock_range[0]>.5,
            best_time_difference=slot/fps-float(pts[best])))
    reliable=[r for r in matches if not r['ambiguous'] and r['pixel_rmse']<8]
    result=dict(idx=idx,native_samples=len(matches),display_frames=len(reference),
        matches=matches,reliable_matches=len(reliable),
        max_abs_time_error_reliable=max([abs(x['best_time_difference']) for x in reliable],default=None),
        median_abs_time_error_reliable=float(np.median([abs(x['best_time_difference']) for x in reliable])) if reliable else None,
        caveat='Pixel nearest-neighbor is ambiguous for static/repeated frames; this is an audit, not event-time annotation.')
    (FOLDER/f'{idx}_pixel_match.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='matches'}),flush=True)
    return result

if __name__=='__main__':
    results=[run(idx) for idx in (1449,1528,2054)]
    (FOLDER/'pixel_match_summary.json').write_text(json.dumps(results,indent=2))

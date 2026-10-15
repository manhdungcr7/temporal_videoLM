"""Presentation-clock rendering control, with untouched original baseline.

Use decoded presentation timestamps, not frame index/average-fps assumptions.
Common clip clock preserves VFR, edit preroll, and audio gaps without any QA
router or ASR dependency. This is an input-correctness control, not yet a novel
learned joint-AV method or a guarantee of benchmark accuracy.
"""
import json,time,gc,math,hashlib,importlib
from pathlib import Path
import numpy as np
import torch
import av
import decord
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

R40_PTS_DIR=OUT/'research40_pts_decoder_cache';R40_PTS_DIR.mkdir(exist_ok=True)
R40_PTS_FILE=OUT/f'research40_pts_clock_s{R40_SERVER}.jsonl'
R40_PTS_VERSION='presentation_grid_v1'

def r40_pts_media(row,pixels):
    path=fetch(row['video_path']);duration=float(row['duration'])
    vr=decord.VideoReader(str(path))
    decord_pts=np.asarray(vr.get_frame_timestamp(list(range(len(vr)))),dtype=float)[:,0]
    pts=[];dimensions=None
    with av.open(str(path)) as container:
        common_origin=float((container.start_time or 0)/av.time_base)
        for frame in container.decode(video=0):
            assert frame.pts is not None,'No presentation timestamp; refuse invented clock'
            pts.append(float(frame.pts*frame.time_base)-common_origin)
            dimensions=(frame.height,frame.width)
    pts=np.array(pts,dtype=float)
    assert len(pts)>1 and bool(np.all(np.diff(pts)>=-1e-8)),'Non-monotone decoded presentation clock'
    # Validate decoder clocks without assuming the same preroll/frame count.
    # Each PyAV PTS must have a corresponding Decord PTS within one source frame.
    sorted_decord=np.sort(decord_pts-common_origin)
    insertion=np.searchsorted(sorted_decord,pts)
    lower=sorted_decord[np.clip(insertion-1,0,len(sorted_decord)-1)]
    upper=sorted_decord[np.clip(insertion,0,len(sorted_decord)-1)]
    cross_error=float(np.minimum(abs(pts-lower),abs(pts-upper)).max())
    assert cross_error<=.05,('Decord/PyAV timestamp disagreement',cross_error)
    last=float(min(duration,pts[-1]));assert last>0
    n=max(4,2*int(math.ceil(last)))
    targets=np.linspace(0.,last,n);fps=(n-1)/last
    selected=np.searchsorted(pts,targets,side='right')-1
    # Before any decoded picture is active, use unknown/black, not a future frame.
    indices=set(int(i) for i in selected if i>=0);pictures={}
    with av.open(str(path)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in indices:pictures[i]=torch.from_numpy(frame.to_ndarray(format='rgb24').copy()).permute(2,0,1)
    shape=next(iter(pictures.values())).shape
    black=torch.zeros(shape,dtype=torch.uint8)
    frames=torch.stack([pictures[int(i)] if i>=0 else black for i in selected])
    # Exactly the same native utility's spatial-budget rule as the fixed baseline.
    vision=None
    for module in ('qwen_omni_utils.v2_5.vision_process','qwen_omni_utils.vision_process'):
        try:vision=importlib.import_module(module);break
        except ModuleNotFoundError:continue
    assert vision is not None
    height,width=vision.smart_resize(dimensions[0],dimensions[1],factor=28,
                                    min_pixels=vision.VIDEO_MIN_PIXELS,max_pixels=pixels)
    frames=resize(frames,[height,width],interpolation=InterpolationMode.BICUBIC,antialias=True).float()
    # Render the audio clock explicitly; preserving gaps is different from
    # concatenating decoded segments. No speech- or question-specific masking.
    length=int(round(duration*16000));wave=np.zeros(length,dtype='float32');written=np.zeros(length,dtype=bool)
    pieces=[];missing_pts=0
    with av.open(str(path)) as container:
        resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
        cumulative=0
        for frame in container.decode(audio=0):
            for piece in resampler.resample(frame):
                values=piece.to_ndarray().reshape(-1).astype('float32',copy=False)
                if piece.pts is None:offset=cumulative;missing_pts+=1
                else:offset=int(round((float(piece.pts*piece.time_base)-common_origin)*16000))
                pieces.append((offset,values));cumulative=offset+len(values)
        for piece in resampler.resample(None):
            values=piece.to_ndarray().reshape(-1).astype('float32',copy=False)
            offset=cumulative if piece.pts is None else int(round((float(piece.pts*piece.time_base)-common_origin)*16000))
            pieces.append((offset,values));cumulative=offset+len(values)
    assert missing_pts==0,'Audio segments lack PTS; do not guess gap positions'
    for offset,values in pieces:
        lo=max(0,offset);hi=min(length,offset+len(values))
        if hi>lo:wave[lo:hi]=values[lo-offset:hi-offset];written[lo:hi]=True
    original=decode_audio(path);common=min(len(original),len(wave))
    waveform_rms=float(np.sqrt(np.mean((original[:common]-wave[:common])**2)))
    metadata=dict(version=R40_PTS_VERSION,common_origin=common_origin,
        pyav_frames=len(pts),decord_frames=len(decord_pts),pyav_first_pts=float(pts[0]),
        decord_first_pts=float(decord_pts[0]),cross_decoder_max_pts_error=cross_error,
        target_times=targets.tolist(),selected_frame_indices=selected.tolist(),
        selected_source_pts=[float(pts[i]) if i>=0 else None for i in selected],
        max_frame_hold_delay=float(max(t-pts[i] for t,i in zip(targets,selected) if i>=0)),
        native_negative_frames=int((decord_pts<common_origin).sum()),frames=n,fps=fps,
        original_audio_samples=len(original),canonical_audio_samples=len(wave),audio_rms_difference=waveform_rms,
        audio_uncovered_samples=int((~written).sum()),spatial_pixels=pixels,dimensions=[height,width])
    (R40_PTS_DIR/f"{row['idx']}_clock.json").write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    media=dict(path=str(path),audio=[wave],videos=[frames],fps=fps,max_pixels_per_frame=pixels)
    del vr,pictures,pieces;gc.collect()
    return media,metadata

base_by_id={r['idx']:r for r in r40_json_rows(R40_BASE_FILE)}
records=r40_json_rows(R40_PTS_FILE);done={r['idx'] for r in records}
for row in R40_SAMPLES:
    if row['idx'] in done:continue
    started=time.time();question,texts=parse_question(row['question']);k=len(texts)
    media,clock=r40_pts_media(row,base_by_id[row['idx']]['pixels'])
    totals={name:np.zeros(k) for name in ('pts_native_av','pts_native_av_boost','pts_source_kv')}
    first=None;native_bank=None;source_bank=None;replay_error=0.
    for rotation,perm in enumerate(rotations(k)):
        cache_path=R40_PTS_DIR/f"{row['idx']}_rot{rotation}.pt"
        if cache_path.exists():
            payload=torch.load(cache_path,map_location='cpu',weights_only=False)
            assert payload['pts_version']==R40_PTS_VERSION
        else:
            inputs,_=qwen_inputs(PROC,r40_prompt(question,texts,perm),media,'av');inputs=move_inputs(inputs,MODEL_)
            if rotation==0:payload=r40_native_and_cache(inputs,cache_path,k)
            else:
                prefix=first['last_media']+1
                assert torch.equal(inputs['input_ids'].cpu()[:,:prefix],first['input_ids'][:,:prefix])
                pos,sec,is_v,is_a,last=media_layout(MODEL_,PROC,inputs)
                device=MODEL_.get_input_embeddings().weight.device
                with torch.inference_mode():tail=MODEL_.get_input_embeddings()(inputs['input_ids'][:,prefix:].to(device)).cpu()
                embeds=torch.cat((first['decoder']['inputs_embeds'][:,:prefix],tail),dim=1)
                payload=dict(decoder=dict(inputs_embeds=embeds,position_ids=pos.cpu(),
                    attention_mask=inputs['attention_mask'].cpu(),cache_position=torch.arange(embeds.shape[1])),
                    is_v=is_v.cpu(),is_a=is_a.cpu(),sec=sec.cpu(),last_media=last,input_ids=inputs['input_ids'].cpu(),
                    fingerprint=R40_FINGERPRINT)
            payload.update(pts_version=R40_PTS_VERSION,num_options=k,clock=clock,
                           sample_idx=f"pts_v1_{row['idx']}")
            torch.save(payload,cache_path);del inputs
        if rotation==0:
            first=payload
            native_bank,_=r40_prefix(payload)
            source_bank=r40_source_bank(payload)
        native=r40_bank_scores(payload,native_bank)
        if 'logits' in payload:
            error=float(np.max(abs(native-payload['logits'].numpy())))
            assert error<=.06,('Corrected native replay',error)
            replay_error=max(replay_error,error)
        with r40_av_boost(payload,question):boost=r40_bank_scores(payload,native_bank)
        independent=r40_bank_scores(payload,source_bank)
        for name,lg in zip(totals,(native,boost,independent)):
            lp=lg-np.logaddexp.reduce(lg)
            for slot,i in enumerate(perm):totals[name][i]+=lp[slot]
    rec=dict(idx=row['idx'],task=row['task'],gold=row['gold'],status='ok',
        manifest_sha256=R40_MANIFEST['manifest_sha256'],scores={name:value.tolist() for name,value in totals.items()},
        clock=clock,replay_error=replay_error,seconds=round(time.time()-started,2),
        no_asr_or_question_router=True,original_baseline_preserved=True)
    with R40_PTS_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(row['idx'])
    merged=[dict(r,scores=dict(base_by_id[r['idx']]['scores'],**r['scores'])) for r in records]
    r40_snapshot(merged,'pts_clock');print('PTS_CLOCK_SAMPLE',json.dumps(rec),flush=True)
    del media,first,payload,native_bank,source_bank;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'pts_clock_done');print('PTS_CLOCK_SUMMARY',json.dumps(RESULT),flush=True)

"""Frame-index versus presentation-time sampling; audio/native budget unchanged.

Input-correctness control, not a claimed novel hallucination method. Small
nonmonotone decoder PTS are stable-sorted, never replaced by Decord metadata.
"""
import gc,hashlib,json,time
import av
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

PC_SERVER=globals().get('PC_SERVER',1)
PC_LIMIT=globals().get('PC_LIMIT')
SL_SERVER=PC_SERVER
exec(compile((OUT/'uploads/symmetric_latent40.py').read_text().split('RESULT=sl_run()',1)[0],
    'native_decoder_helpers','exec'),globals())
ARGON_SERVER=PC_SERVER
exec(compile((OUT/'uploads/argon_latent40.py').read_text().split('RESULT=argon_run(',1)[0],
    'native_encoder_helpers','exec'),globals())
PC_DIR=OUT/'presentation_resample40';PC_DIR.mkdir(exist_ok=True)
PC_SHA=hashlib.sha256((OUT/'uploads/presentation_resample40.py').read_bytes()).hexdigest()
PC_METHODS=['P0_original_native_reference','P1_pyav_frame_index','P2_pyav_presentation_time']


def pc_select(pts,targets):
    """Latest displayed image at/before each target; -1 means not yet available."""
    pts=np.asarray(pts,dtype=float);targets=np.asarray(targets,dtype=float)
    assert len(pts)>1 and np.isfinite(pts).all() and np.isfinite(targets).all()
    order=np.argsort(pts,kind='stable')
    ranks=np.searchsorted(pts[order],targets,side='right')-1
    return np.where(ranks>=0,order[ranks.clip(min=0)],-1)


def pc_media(sample):
    pixels=int(160*min(1.,32/sample['duration']))*28*28
    original=prepare_video(fetch(sample['video_path']),max_pixels=pixels)
    frames=original['videos'][0];n,h,w=int(frames.shape[0]),int(frames.shape[-2]),int(frames.shape[-1])
    assert frames.ndim==4 and frames.shape[1]==3
    fps=original['fps'];targets=np.arange(n,dtype=float)/fps
    pts=[];sizes=[]
    with av.open(original['path']) as container:
        origin=float((container.start_time or 0)/av.time_base)
        for frame in container.decode(video=0):
            assert frame.pts is not None,'Missing PTS; refuse invented timestamps'
            pts.append(float(frame.pts*frame.time_base)-origin);sizes.append((frame.height,frame.width))
    pts=np.asarray(pts);assert len(set(sizes))==1,'Variable frame geometry needs explicit treatment'
    uniform=torch.linspace(0,len(pts)-1,n).round().long().numpy()
    presented=pc_select(pts,targets)
    needed=set(int(i) for i in np.concatenate([uniform,presented]) if i>=0);pictures={}
    with av.open(original['path']) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in needed:pictures[i]=torch.from_numpy(frame.to_ndarray(format='rgb24').copy()).permute(2,0,1)
    black=torch.zeros_like(next(iter(pictures.values())))
    variants={}
    for method,indices in zip(PC_METHODS[1:],(uniform,presented)):
        stack=torch.stack([pictures[int(i)] if i>=0 else black for i in indices])
        stack=resize(stack,[h,w],interpolation=InterpolationMode.BICUBIC,antialias=True).float()
        assert tuple(stack.shape)==tuple(frames.shape)
        variants[method]=dict(original,videos=[stack])
    metadata=dict(common_origin=origin,pyav_frames=len(pts),native_frames=n,native_fps=fps,
        geometry=[h,w],pixels=pixels,source_geometry=list(sizes[0]),
        target_times=targets.tolist(),source_pts=pts.tolist(),nonmonotone_steps=int((np.diff(pts)<0).sum()),
        uniform_indices=uniform.tolist(),presentation_indices=presented.tolist(),
        selected_pts=[float(pts[i]) if i>=0 else None for i in presented],
        targets_after_last_pts=int((targets>pts.max()).sum()),black_slots=int((presented<0).sum()),
        same_selected_indices=int((uniform==presented).sum()),audio_samples=len(original['audio'][0]),
        original_audio_unchanged=True)
    (PC_DIR/f"{sample['idx']}_clock.json").write_text(json.dumps(metadata,indent=2))
    return variants,metadata


def pc_payload(media,prompt,k,first=None,first_ids=None):
    inputs,_=qwen_inputs(PROC,prompt,media,'av');inputs=move_inputs(inputs,MODEL_)
    if first is None:
        result=argon_encode(inputs,k);ids=inputs['input_ids'].cpu()
    else:
        prefix=first['last_media']+1;ids=inputs['input_ids'].cpu()
        assert torch.equal(ids[:,:prefix],first_ids[:,:prefix])
        pos,sec,v,a,last=media_layout(MODEL_,PROC,inputs)
        with torch.inference_mode():
            tail=MODEL_.get_input_embeddings()(inputs['input_ids'][:,prefix:]).cpu()
        embeds=torch.cat((first['decoder']['inputs_embeds'][:,:prefix],tail),dim=1)
        result=dict(decoder=dict(inputs_embeds=embeds,position_ids=pos.cpu(),
            attention_mask=inputs['attention_mask'].cpu(),cache_position=torch.arange(embeds.shape[1])),
            sec=sec.cpu(),is_v=v.cpu(),is_a=a.cpu(),last_media=int(last),spg=first['spg'],
            tokens=int(ids.shape[1]),manifest_sha256=SL_SHA)
    del inputs
    return result,ids


def pc_run():
    output=PC_DIR/f'results_s{PC_SERVER}.jsonl'
    done={json.loads(x)['idx'] for x in output.read_text().splitlines()} if output.exists() else set()
    originals={r['idx']:r for r in [json.loads(x) for x in
        (OUT/'argon_ccld40'/f'results_s{PC_SERVER}.jsonl').read_text().splitlines()]}
    samples=SL_SAMPLES[:PC_LIMIT] if PC_LIMIT else SL_SAMPLES
    for sample in samples:
        idx=sample['idx']
        if idx in done:continue
        started=time.time();assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        stem,texts=parse_question(sample['question']);k=len(texts)
        media,clock=pc_media(sample)
        results={rotation:dict(idx=idx,rotation=rotation,perm=list(perm),scores={},manifest_sha256=SL_SHA,
            runtime_sha256=PC_SHA,replay_error=0.) for rotation,perm in enumerate(rotations(k))}
        for rotation in results:
            old=json.loads((OUT/'argon_ccld40'/f'{idx}_rot{rotation}.json').read_text())
            results[rotation]['scores'][PC_METHODS[0]]=old['scores']['R4_C0_baseline']
        for method in PC_METHODS[1:]:
            first=None;first_ids=None
            for rotation,perm in enumerate(rotations(k)):
                payload,ids=pc_payload(media[method],argon_prompt(stem,texts,perm),k,first,first_ids)
                native,_=sl_forward(payload)
                if rotation==0:
                    error=float((native[SL_LABELS[:k]]-payload['native_logits']).abs().max())
                    assert error<=.12,('Encoder replay mismatch',method,error)
                    first=payload;first_ids=ids
                    results[rotation]['replay_error']=max(results[rotation]['replay_error'],error)
                results[rotation]['scores'][method]=torch.log_softmax(native[SL_LABELS[:k]],dim=-1).numpy().tolist()
                del native;gc.collect();torch.cuda.empty_cache()
            del first,first_ids,payload
        totals={m:np.zeros(k) for m in PC_METHODS};audits=[]
        for rotation,result in results.items():
            (PC_DIR/f'{idx}_rot{rotation}.json').write_text(json.dumps(result))
            for method,score in result['scores'].items():
                for slot,opt in enumerate(result['perm']):totals[method][opt]+=score[slot]
            audits.append({a:b for a,b in result.items() if a!='scores'})
        assert totals[PC_METHODS[0]].tolist()==originals[idx]['scores']['R4_C0_baseline']
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=SL_SHA,
            runtime_sha256=PC_SHA,scores={m:s.tolist() for m,s in totals.items()},
            predictions={m:chr(65+int(s.argmax())) for m,s in totals.items()},audit=audits,
            clock=clock,seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done.add(idx);print('PRESENTATION_SAMPLE',len(done),'/',len(samples),json.dumps(rec),flush=True)
        del media;gc.collect();torch.cuda.empty_cache()
    return dict(server=PC_SERVER,completed=len(done),expected=len(samples))


RESULT=pc_run()
print('PRESENTATION_COMPLETE',json.dumps(RESULT),flush=True)

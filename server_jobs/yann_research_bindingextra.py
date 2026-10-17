"""R4: actual waveform/frame O/A/V/AV translations with human-derived event-order labels."""
import ast,copy,gc,json,math,random,time,torch,shutil
import numpy as np
YR_DIR=OUT/'yann_temporal_research';YR_BIND=YR_DIR/'binding_evalextra';YR_BIND.mkdir(exist_ok=True)
tree=ast.parse((OUT/'uploads/yann_research_supervision_resume.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='yr_prepcase'],type_ignores=[]),'yr_prepare_same_hook','exec'),globals())
manifest=json.loads((OUT/'uploads/perception48_manifest.json').read_text());rows_out=[];skipped=[]
def endpoints(row,source):
    es=row['sounds'] if source=='audio' else row['actions'];group={}
    for e in es:
        if source=='audio' and e['label'].strip()=='Muted':continue
        group.setdefault(e['label'],[]).append([e['timestamps'][0]/1e6,e['timestamps'][1]/1e6])
    return [dict(source=source,label=lab,interval=min(intervals),description=f'the first occurrence of the {"sound" if source=="audio" else "action"} "{lab}"') for lab,intervals in sorted(group.items())]
for i,row in enumerate(manifest['rows']):
    if i%2!=YR_SERVER-1:continue
    if row['split']=='train':continue
    started=time.time();vid=row['video_id'];base=prepare_video(YR_DIR/'perception48'/f'{vid}.mp4',max_pixels=int(96*min(1.,32/row['duration']))*28*28)
    frames=base['videos'][0];audio=np.asarray(base['audio'][0]);fps=base['fps']
    assert torch.is_tensor(frames) and frames.ndim==4,(type(frames),getattr(frames,'shape',None))
    assert audio.ndim==1 and abs(len(audio)/16000-row['audio_duration'])<.25
    actual_duration=max(row['duration'],len(audio)/16000,len(frames)/fps)
    # Actual native cadence is often1.91–1.99fps, not exactly2. Keep temporal tubelet parity.
    nv=2*math.floor(min(4.,44.8-actual_duration)*fps/2)
    delta=nv/fps
    if delta<.5:skipped.append(dict(video_id=vid,reason='insufficient non-clipping room below45s'));continue
    na=int(round(delta*16000));assert nv%2==0 and abs(na/16000-delta)<1/16000
    groups={s:endpoints(row,s) for s in ('audio','video')}
    intended=('audio','video') if i%2==0 else ('video','audio')
    pairs=[];actual=intended
    for direction in (intended,):
        for a in groups[direction[0]]:
            for b in groups[direction[1]]:
                if (a['source'],a['label'])==(b['source'],b['label']):continue
                gaps=[]
                for da,dv in ((0,0),(delta,0),(0,delta),(delta,delta)):
                    gaps.append((a['interval'][0]+(da if a['source']=='audio' else dv))-(b['interval'][0]+(da if b['source']=='audio' else dv)))
                if min(abs(g) for g in gaps)<.50:continue
                if not any(g*gaps[0]<0 for g in gaps):continue
                pairs.append((a,b))
        if pairs:actual=direction;break
    if not pairs:skipped.append(dict(video_id=vid,reason='no unambiguous distinct-label endpoint pair'));continue
    rng=random.Random(20261017+int(vid.split('_')[-1]));a,b=pairs[rng.randrange(len(pairs))]
    # Both source orders are represented; no task label, version label, or times enter the question.
    question=f'Which starts first, {a["description"]} or {b["description"]}? Answer with the event description.'
    outputs=[]
    for name,da,dv in (('O',0.,0.),('A',delta,0.),('V',0.,delta),('AV',delta,delta)):
        shifted_audio=np.concatenate((np.zeros(na if da else 0,dtype=audio.dtype),audio,np.zeros(0 if da else na,dtype=audio.dtype)))
        zero=torch.zeros((nv,*frames.shape[1:]),dtype=frames.dtype,device=frames.device)
        shifted_frames=torch.cat((zero,frames),dim=0) if dv else torch.cat((frames,zero),dim=0)
        assert shifted_audio.shape[0]==len(audio)+na and shifted_frames.shape[0]==len(frames)+nv
        assert np.array_equal(shifted_audio[na if da else 0:(na if da else 0)+len(audio)],audio)
        assert torch.equal(shifted_frames[nv if dv else 0:(nv if dv else 0)+len(frames)],frames)
        yr_media=dict(base,audio=[shifted_audio],videos=[shifted_frames])
        rrow=dict(row,duration=row['duration']+delta,audio_duration=len(shifted_audio)/16000)
        assert max(rrow['duration'],rrow['audio_duration'])<=45
        ea=copy.deepcopy(a);eb=copy.deepcopy(b)
        for e in (ea,eb):e['interval']=[t+(da if e['source']=='audio' else dv) for t in e['interval']]
        answer=ea['description'] if ea['interval'][0]<eb['interval'][0] else eb['description']
        other=eb['description'] if answer==ea['description'] else ea['description']
        path=YR_BIND/f'{vid}_{name}.pt'
        if not path.exists():
            yr_prepcase(path,rrow,question,answer,'binding_qa',None,None)
            cap=torch.load(path,map_location='cpu',weights_only=True)
            # Candidate completion is a diagnostic, not a task-specific inference rule.
            otherpath=YR_BIND/f'{vid}_{name}_other.pt'
            yr_prepcase(otherpath,rrow,question,other,'binding_foil_completion',None,None)
            foil=torch.load(otherpath,map_location='cpu',weights_only=True)
            cap.update(answer_h_other=foil['answer_h'],target_other=foil['target'],other_answer=other,
                endpoints=[ea,eb],variant=name,delta=delta,actual_direction=actual,intended_direction=intended,
                original_audio_samples=len(audio),shifted_audio_samples=len(shifted_audio),original_frames=len(frames),shifted_frames=len(shifted_frames),
                transform='actual waveform zero-padding and frame black-padding before native encoders; all4 equal length, no clipping')
            torch.save(cap,path)
        cap=torch.load(path,map_location='cpu',weights_only=True)
        outputs.append(dict(variant=name,answer=cap['answer'],endpoints=cap['endpoints'],tokens=cap['tokens'],file=str(path.relative_to(YR_DIR))))
    result=dict(video_id=vid,split=row['split'],question=question,direction=actual,intended_direction=intended,delta=delta,
        label_changes={v['variant']:v['answer']!=outputs[0]['answer'] for v in outputs},variants=outputs)
    assert outputs[0]['answer']==outputs[3]['answer'],'Joint shift changed relative onset ordering'
    assert len({v['tokens'] for v in outputs})<=2,'Unexpected large token-budget mismatch'
    rows_out.append(result);(YR_DIR/'bindingextra_partial.json').write_text(json.dumps(dict(server=YR_SERVER,rows=rows_out,skipped=skipped),indent=2),encoding='utf-8')
    print('YR_BINDINGDATA',vid,row['split'],actual,result['label_changes'],round(time.time()-started,1),flush=True)
    del base,frames,audio,yr_media,shifted_audio,shifted_frames,cap;gc.collect();torch.cuda.empty_cache()
RESULT=dict(server=YR_SERVER,rows=rows_out,skipped=skipped,hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
assert RESULT['hooks']==0
(YR_DIR/'bindingextra.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('YR_BINDINGDATA_DONE',len(rows_out),len(skipped),flush=True)

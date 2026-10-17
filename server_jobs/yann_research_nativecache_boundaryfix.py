"""R11 public paired/native QA features for native upper-decoder training."""
import collections,copy,gc,importlib.util,json,math,random,time,torch
import numpy as np
YR_DIR=OUT/'yann_temporal_research';dest=YR_DIR/'native_tail48';dest.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('native_tail',OUT/'uploads/yann_native_tail.py');nt=importlib.util.module_from_spec(sp);sp.loader.exec_module(nt)
manifest=json.loads((OUT/'uploads/perception48_manifest.json').read_text());groups={}
for folder in ('binding48_v2','binding_evalextra'):
    for p in sorted((YR_DIR/folder).glob('*_O.pt')):
        c=torch.load(p,map_location='cpu',weights_only=True);gs=groups.setdefault(c['video_id'],[])
        if not any(x['question']==c['question'] for x in gs):gs.append(c)
def valid_events(row,source):
    out={}
    for e in row['sounds' if source=='audio' else 'actions']:
        label=e['label'].strip()
        if source=='audio' and label in ('Muted','Other','Other:background sound','Object:sound'):continue
        interval=[t/1e6 for t in e['timestamps']]
        if label not in out or interval[0]<out[label]['interval'][0]:
            out[label]=dict(label=label,source=source,interval=interval,description=f'the first occurrence of the {"sound" if source=="audio" else "action"} "{label}"')
    return list(out.values())
def pattern(endpoints,delta):
    a,b=endpoints;truth=[]
    for da,dv in ((0,0),(delta,0),(0,delta),(delta,delta)):
        truth.append(a['interval'][0]+(da if a['source']=='audio' else dv)<b['interval'][0]+(da if b['source']=='audio' else dv))
    return [v!=truth[0] for v in truth]
rows=[]
for i,row in enumerate(manifest['rows']):
    if i%2!=YR_SERVER-1:continue
    vid=row['video_id'];start=time.time();questions=copy.deepcopy(groups.get(vid,[]))
    if questions and row['split']=='train':
        delta=questions[0]['delta'];oldpat=pattern(questions[0]['endpoints'],delta);candidates=[]
        ev=valid_events(row,'audio')+valid_events(row,'video')
        for a in ev:
            for b in ev:
                if (a['source'],a['label'])==(b['source'],b['label']):continue
                if pattern([a,b],delta)==oldpat:continue
                gaps=[a['interval'][0]+(da if a['source']=='audio' else dv)-b['interval'][0]-(da if b['source']=='audio' else dv) for da,dv in ((0,0),(delta,0),(0,delta),(delta,delta))]
                if min(abs(x) for x in gaps)<.5:continue
                candidates.append((a,b))
        if candidates:
            rng=random.Random(20261020+int(vid.split('_')[-1]));a,b=rng.choice(candidates)
            questions.append(dict(question=f'Which starts first, {a["description"]} or {b["description"]}? Answer with the event description.',endpoints=[a,b],delta=delta,added_train_question=True))
    human=torch.load(YR_DIR/'perception48'/f'{vid}_q0.pt',map_location='cpu',weights_only=True)
    media=prepare_video(YR_DIR/'perception48'/f'{vid}.mp4',max_pixels=int(96*min(1.,32/row['duration']))*28*28)
    frames=media['videos'][0];audio=np.asarray(media['audio'][0]);nv=int(round(questions[0]['delta']*media['fps'])) if questions else 0
    if questions:assert nv%2==0
    na=int(round(nv/media['fps']*16000)) if nv else 0
    for variant in (['N']+['O','A','V','AV'] if questions else ['N']):
        prefixfile=dest/f'{vid}_{variant}_prefix.pt';casesfile=dest/f'{vid}_{variant}_cases.pt'
        if prefixfile.exists() and casesfile.exists():continue
        if variant=='N':current=media;qs=[dict(question=human['question'],answer=human['answer'],kind='human_qa')]
        else:
            da=variant in ('A','AV');dv=variant in ('V','AV')
            z=torch.zeros((nv,*frames.shape[1:]),dtype=frames.dtype,device=frames.device)
            current=dict(media,audio=[np.concatenate((np.zeros(na if da else 0,dtype=audio.dtype),audio,np.zeros(0 if da else na,dtype=audio.dtype)))],
                videos=[torch.cat((z,frames),0) if dv else torch.cat((frames,z),0)])
            qs=[]
            for q in questions:
                ends=copy.deepcopy(q['endpoints'])
                for e in ends:
                    delay=(na/16000 if da else 0.) if e['source']=='audio' else (nv/media['fps'] if dv else 0.)
                    e['interval']=[t+delay for t in e['interval']]
                a,b=ends;answer=a['description'] if a['interval'][0]<b['interval'][0] else b['description'];other=b['description'] if answer==a['description'] else a['description']
                qs.append(dict(question=q['question'],answer=answer,other_answer=other,kind='binding_qa',endpoints=ends,
                    flip_pattern=pattern(q['endpoints'],nv/media['fps']),added_train_question=q.get('added_train_question',False)))
        oldq=qs[0]['question'];inp,_=qwen_inputs(PROC,oldq+'\nAnswer the question directly.',current,'av');inp=move_inputs(inp,MODEL_)
        base=nt.capture_native(MODEL_,PROC,inp)
        saved=[]
        for q in qs:
            c=nt.make_tail(MODEL_,PROC,base,oldq,q['question'],q['answer'],q.get('other_answer'))
            c.update(video_id=vid,split=row['split'],variant=variant,kind=q['kind'],flip_pattern=q.get('flip_pattern'),added_train_question=q.get('added_train_question',False),endpoints=q.get('endpoints'))
            saved.append(c)
        keys={j:tuple(t.detach().cpu().clone() for t in kv) for j,kv in base['keys'].items() if j>=28}
        torch.save(dict(keys=keys,prefix=base['prefix'],start_position=base['start_position']),prefixfile);torch.save(saved,casesfile)
        del keys,base,inp,current;gc.collect();torch.cuda.empty_cache()
    rows.append(dict(video_id=vid,split=row['split'],paired_questions=len(questions),mixed_patterns=len({tuple(pattern(q['endpoints'],q['delta'])) for q in questions})>1 if questions else False,seconds=round(time.time()-start,2)))
    (YR_DIR/'nativecache_partial.json').write_text(json.dumps(dict(server=YR_SERVER,rows=rows),indent=2));print('YR_NATIVE_CACHE',json.dumps(rows[-1]),flush=True)
    del media,frames,audio;gc.collect();torch.cuda.empty_cache()
RESULT=dict(server=YR_SERVER,rows=rows,files=len(list(dest.glob('*.pt'))),bytes=sum(p.stat().st_size for p in dest.glob('*.pt')))
(YR_DIR/'nativecache.json').write_text(json.dumps(RESULT,indent=2));print('YR_NATIVE_CACHE_DONE',json.dumps(RESULT),flush=True)

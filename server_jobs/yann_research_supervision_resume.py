"""Prepare independent human-supervised memory/readout probes on public train videos."""
import gc,hashlib,io,json,random,time,urllib.request,zipfile,torch
from pathlib import Path
YR_DIR=OUT/'yann_temporal_research';YR_DIR.mkdir(exist_ok=True)
YR_DATA=YR_DIR/'perception48';YR_DATA.mkdir(exist_ok=True)
yr_manifest=json.loads((OUT/'uploads/perception48_manifest.json').read_text())
class YRRemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:self.length=int(r.headers['Content-Length'])
        self.pos=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,x,w=0):self.pos=x if w==0 else self.pos+x if w==1 else self.length+x;return self.pos
    def read(self,n=-1):
        n=self.length-self.pos if n<0 else min(n,self.length-self.pos)
        if not n:return b''
        if n>16_000_000:raise ValueError('single range too large')
        with urllib.request.urlopen(urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'}),timeout=120) as r:
            if r.status!=206:raise ValueError('range ignored')
            b=r.read(n+1)
        assert len(b)==n
        self.pos+=n;self.transferred+=n;return b
rf=YRRemoteZip('https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip')
yr_out=[];yr_rng=random.Random(20261016)
def yr_prepcase(path,row,question,answer,kind,intervals=None,source=None):
    prompt=question+'\nAnswer the question directly.'
    inp,_=qwen_inputs(PROC,prompt,yr_media,'av');prompt_len=int(inp['input_ids'].shape[1])
    target=PROC.tokenizer(answer,add_special_tokens=False)['input_ids']
    # Human answers/interval annotations, never native predictions, supervise the probe.
    assert target and len(target)<=768,(row['video_id'],len(target))
    inp['input_ids']=torch.cat([inp['input_ids'],torch.tensor([target],dtype=torch.long)],dim=1)
    inp['attention_mask']=torch.ones_like(inp['input_ids'])
    inp=move_inputs(inp,MODEL_)
    _,sec,iv,ia,lm=media_layout(MODEL_,PROC,inp)
    cap={}
    def mid(module,args,out):
        h=out[0] if isinstance(out,tuple) else out
        cap['q']=h[0,prompt_len-1].detach().cpu()
        ix=torch.where(iv|ia)[0].cpu();times=sec[ix.to(sec.device)].cpu()
        bins=torch.floor(times*2+1e-4).long()
        features=[];ts=[];mods=[];spans=[]
        for mod,mask in ((0,ia),(1,iv)):
            local=torch.where(mask)[0].cpu();localbins=torch.floor(sec[local.to(sec.device)].cpu()*2+1e-4).long()
            for k in torch.unique(localbins,sorted=True).tolist():
                inds=local[localbins==k];features.append(h[0,inds.to(h.device)].float().mean(0).half().cpu())
                ts.append(float(sec[inds.to(sec.device)].mean()));mods.append(mod);spans.append([max(0.,k*.5),min(row['duration'],(k+1)*.5)])
        cap['bank']=torch.stack(features);cap['times']=torch.tensor(ts);cap['mods']=torch.tensor(mods);cap['spans']=torch.tensor(spans)
    def final(module,args):
        h=args[0];cap['answer_h']=h[0,prompt_len-1:prompt_len+len(target)-1].detach().cpu()
        return (h[:,-1:],)
    hooks=[MODEL_.model.layers[18].register_forward_hook(mid),MODEL_.lm_head.register_forward_pre_hook(final)]
    try:
        with torch.inference_mode():MODEL_(**inp,use_audio_in_video=True,use_cache=False)
    finally:
        for hook in hooks:hook.remove()
    assert cap['answer_h'].shape[0]==len(target)
    cap.update(target=torch.tensor(target),video_id=row['video_id'],split=row['split'],question=question,answer=answer,
        kind=kind,source=source,intervals=intervals,duration=row['duration'],tokens=int(inp['input_ids'].shape[1]),
        input_prompt_len=prompt_len,bank_clock='native TMRoPE-derived seconds; 0.5s aggregation',hook_index=18)
    torch.save(cap,path)
    del inp;gc.collect();torch.cuda.empty_cache()
    return {k:cap[k] for k in ('video_id','split','question','answer','kind','tokens')}
with zipfile.ZipFile(rf) as z:
    for i,row in enumerate(yr_manifest['rows']):
        if i%2!=YR_SERVER-1:continue
        started=time.time();vid=row['video_id'];mp4=YR_DATA/f'{vid}.mp4'
        if not mp4.exists():mp4.write_bytes(z.read(row['zip']['name']))
        assert mp4.stat().st_size==row['zip']['bytes']
        # Metadata was filtered before selection; decoded durations are checked without replacing failures.
        import av
        with av.open(str(mp4)) as c:
            streams={s.type:s for s in c.streams}
            decoded_durations={k:float(v.duration*v.time_base) for k,v in streams.items() if v.duration is not None}
        assert 'audio' in streams and 'video' in streams
        assert all(v<=45.15 for v in decoded_durations.values()),decoded_durations
        yr_media=prepare_video(mp4,max_pixels=int(96*min(1.,32/row['duration']))*28*28)
        qs=row['questions'];q=qs[yr_rng.randrange(len(qs))]
        cases=[(q['question'],q['options'][q['answer_id']],'human_qa',None,None)]
        for source,items in (('video',row['actions']),('audio',row['sounds'])):
            labels=sorted(set(e['label'] for e in items));label=labels[yr_rng.randrange(len(labels))]
            intervals=sorted([round(e['timestamps'][0]/1e6,2),round(e['timestamps'][1]/1e6,2)] for e in items if e['label']==label)
            # Multi-instance target; no false unique-start labels.
            noun='action' if source=='video' else 'sound';verb='visible' if source=='video' else 'audible'
            question=f'At what times is the {noun} "{label}" {verb}? Give its start and end times in seconds, including each occurrence.'
            answer='; '.join(f'{a:.2f} to {b:.2f} seconds' for a,b in intervals)
            cases.append((question,answer,'human_grounding',intervals,source))
        records=[]
        for ci,(question,answer,kind,ints,source) in enumerate(cases):
            p=YR_DATA/f'{vid}_q{ci}.pt'
            if p.exists():
                r=torch.load(p,map_location='cpu',weights_only=True);record={k:r[k] for k in ('video_id','split','question','answer','kind','tokens')}
            else:record=yr_prepcase(p,row,question,answer,kind,ints,source)
            record['file']=str(p.relative_to(YR_DIR));records.append(record)
        yr_out.extend(records)
        partial=dict(server=YR_SERVER,rows=yr_out,videos=len(yr_out)//3,range_bytes_transferred=rf.transferred)
        (YR_DIR/'supervision_partial.json').write_text(json.dumps(partial,indent=2),encoding='utf-8')
        print('YR_SUPERVISION',vid,row['split'],len(yr_out),round(time.time()-started,1),flush=True)
        del yr_media;gc.collect();torch.cuda.empty_cache()
RESULT=dict(server=YR_SERVER,rows=yr_out,videos=len(yr_out)//3,range_bytes_transferred=rf.transferred,
    hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
assert RESULT['hooks']==0
(YR_DIR/'supervision.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('YR_SUPERVISION_DONE',json.dumps(dict(server=YR_SERVER,videos=RESULT['videos'],cases=len(yr_out))),flush=True)

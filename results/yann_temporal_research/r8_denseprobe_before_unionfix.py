"""R8 feature diagnostic: human dense labels, within-recording localization, no QA model gain."""
import collections,gc,json,math,random,torch
from torch import nn
from torch.nn import functional as F
YR_DIR=OUT/'yann_temporal_research';manifest=json.loads((OUT/'uploads/perception48_manifest.json').read_text());lookup={r['video_id']:r for r in manifest['rows']}
cases=[torch.load(p,map_location='cpu',weights_only=True) for p in sorted((YR_DIR/'binding48_v2').glob('*.pt')) if not p.stem.endswith('_other')]
def events(row):
    return [(source,e['label'],[t/1e6 for t in e['timestamps']]) for source,key in ((0,'sounds'),(1,'actions')) for e in row[key]
        if not(source==0 and e['label'] in ('Muted','Other','Other:background sound','Object:sound'))]
labels=sorted(set((s,l) for c in cases if c['split']=='train' for s,l,t in events(lookup[c['video_id']])))
labindex={x:i for i,x in enumerate(labels)};dev=torch.device('cuda:0');rows=[]
for c in cases:
    spans=c['spans'].float();width=(spans[:,1]-spans[:,0]).clamp_min(1e-4)
    y=torch.zeros(len(spans),len(labels));mask=(c['mods'][:,None]==torch.tensor([s for s,l in labels])[None,:]).float()
    for source,label,interval in events(lookup[c['video_id']]):
        if (source,label) not in labindex:continue
        delay=c['delta'] if c['variant'] in (('A','AV') if source==0 else ('V','AV')) else 0.
        lo,hi=[t+delay for t in interval];ov=(torch.minimum(spans[:,1],torch.tensor(hi))-torch.maximum(spans[:,0],torch.tensor(lo))).clamp_min(0)/width
        # Union over repeated annotated instances, not sum of overlapping intervals.
        j=labindex[(source,label)];y[:,j]=torch.maximum(y[:,j],ov.clamp_max(1)*(c['mods']==source))
    t=spans.mean(-1)/c['duration'];prior=torch.stack((t,t*t,torch.sin(math.pi*t),torch.cos(math.pi*t)),dim=-1)
    rows.append(dict(video_id=c['video_id'],split=c['split'],variant=c['variant'],x=F.layer_norm(c['bank'].float(),(2048,)),prior=prior,y=y,mask=mask))
train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];diag=[r for r in rows if r['split']=='diagnostic']
torch.manual_seed(20261018);random.seed(20261018)
class ContentHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Linear(2048,len(labels)) if YR_SERVER==1 else nn.Sequential(nn.Linear(2048,128),nn.GELU(),nn.Linear(128,len(labels)))
    def forward(self,x):return self.net(x)
def evaluate(head,rs,key):
    head.eval();out=[]
    with torch.no_grad():
        for r in rs:
            logits=head(r[key].to(dev)).float().cpu()
            for j,(source,label) in enumerate(labels):
                valid=r['mask'][:,j]>0;positive=(r['y'][:,j]>=.5)&valid;negative=(r['y'][:,j]==0)&valid
                if not bool(positive.any()) or not bool(negative.any()):continue
                pos=logits[positive,j];neg=logits[negative,j];diff=pos[:,None]-neg[None,:]
                auc=float(((diff>0).float()+.5*(diff==0).float()).mean())
                candidates=torch.where(valid)[0];top=int(candidates[logits[valid,j].argmax()])
                out.append(dict(video_id=r['video_id'],variant=r['variant'],source=source,label=label,auc=auc,top1=bool(positive[top]),positive=int(positive.sum()),negative=int(negative.sum())))
    grouped={}
    for r in out:grouped.setdefault(r['video_id'],[]).append(r)
    return dict(recordings=len(grouped),label_variant_tasks=len(out),macro_recording_auc=sum(sum(x['auc'] for x in rs)/len(rs) for rs in grouped.values())/max(1,len(grouped)),
        macro_recording_top1=sum(sum(x['top1'] for x in rs)/len(rs) for rs in grouped.values())/max(1,len(grouped)),rows=out)
X={key:torch.cat([r[key] for r in train]).to(dev) for key in ('x','prior')};Y=torch.cat([r['y'] for r in train]).to(dev);M=torch.cat([r['mask'] for r in train]).to(dev)
rate=(Y*M).sum(0)/M.sum(0).clamp_min(1);posweight=((1-rate)/rate.clamp_min(.01)).clamp(1,50)
outputs={}
for key in ('x','prior'):
    torch.manual_seed(20261018);head=(ContentHead() if key=='x' else nn.Linear(4,len(labels))).to(dev)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.1);history=[];best=-1
    for step in range(601):
        if step in (0,50,100,200,400,600):
            v=evaluate(head,val,key);metric=v['macro_recording_auc']
            history.append(dict(step=step,validation={k:x for k,x in v.items() if k!='rows'}))
            if metric>best:best=metric;beststep=step;torch.save(head.state_dict(),YR_DIR/f'denseprobe_{key}_selected.pt')
        if step==600:break
        head.train();optimizer.zero_grad(set_to_none=True);ix=torch.randint(len(Y),(512,),device=dev)
        loss=F.binary_cross_entropy_with_logits(head(X[key][ix]),Y[ix],pos_weight=posweight,reduction='none')
        loss=(loss*M[ix]).sum()/M[ix].sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
    head.load_state_dict(torch.load(YR_DIR/f'denseprobe_{key}_selected.pt',map_location=dev,weights_only=True))
    outputs[key]=dict(selected_step=beststep,history=history,train=evaluate(head,train,key),validation=evaluate(head,val,key),diagnostic=evaluate(head,diag,key))
RESULT=dict(server=YR_SERVER,condition='linear native feature probe' if YR_SERVER==1 else '128-hidden native feature probe',labels=labels,outputs=outputs,
    note='Fixed-ontology privileged-label diagnostic only. Within-recording AUC, not QA accuracy. Time-only head controls for occurrence priors. Weighted BCE scores are not calibrated absence probabilities.')
(YR_DIR/'denseprobe.json').write_text(json.dumps(RESULT,indent=2));print(json.dumps({key:{split:{k:v for k,v in out[split].items() if k!='rows'} for split in ('train','validation','diagnostic')} for key,out in outputs.items()}),flush=True)
del X,Y,M,head,optimizer;gc.collect();torch.cuda.empty_cache()

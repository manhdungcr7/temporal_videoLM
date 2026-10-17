"""R9 open-vocabulary dense-field mechanism check, with unseen-label and semantic-shuffle controls."""
import collections,gc,json,math,random,torch
from torch import nn
from torch.nn import functional as F
YR_DIR=OUT/'yann_temporal_research'
source=(OUT/'uploads/yann_research_denseprobe_unionfix.py').read_text()
setup=source.split('outputs={}')[0]
assert setup!=source
exec(compile(setup,'r9_shared_dense_human_labels','exec'),globals())
saved_labels=labels;seed=20261019;rng=random.Random(seed);torch.manual_seed(seed)
held=[]
for mod in (0,1):
    inds=[i for i,(m,l) in enumerate(labels) if m==mod];rng.shuffle(inds);held+=inds[:max(1,round(.2*len(inds)))]
seen=[i for i in range(len(labels)) if i not in held]
texts=[];vectors=[];textdev=MODEL_.get_input_embeddings().weight.device
for mod,label in labels:
    clean=label.strip().replace(':',' ').replace('(',' ').replace(')',' ')
    text=f'The {"sound" if mod==0 else "action"} of {clean}.';texts.append(text)
    inp=PROC.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(textdev);cap={}
    def capture(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;cap['q']=h[0].float().mean(0).detach().cpu()
    hook=MODEL_.model.layers[18].register_forward_hook(capture)
    try:
        with torch.inference_mode():MODEL_.model(input_ids=inp,attention_mask=torch.ones_like(inp),use_cache=False)
    finally:hook.remove()
    vectors.append(cap['q'])
textvectors=torch.stack(vectors).to(dev);textvectors=F.layer_norm(textvectors,(2048,));textindex=list(range(len(labels)))
if YR_SERVER==2:
    for mod in (0,1):
        inds=[i for i in seen if labels[i][0]==mod];perm=inds.copy();rng.shuffle(perm)
        for a,b in zip(inds,perm):textindex[a]=b
class SemanticField(nn.Module):
    def __init__(self):
        super().__init__();self.media=nn.Linear(2048,128,bias=False);self.text=nn.Linear(2048,128,bias=False)
        self.scale=nn.Parameter(torch.tensor(math.log(8.)));self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x,queries):
        k=F.normalize(self.media(x),dim=-1);q=F.normalize(self.text(queries),dim=-1)
        return self.scale.exp().clamp(1,30)*(k@q.T)+self.bias
head=SemanticField().to(dev);optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.05)
def score(rs):
    head.eval();out=[]
    with torch.no_grad():
        for r in rs:
            lg=head(r['x'].to(dev),textvectors).float().cpu()
            for j,(mod,label) in enumerate(labels):
                valid=r['mask'][:,j]>0;positive=(r['y'][:,j]>=.5)&valid;negative=(r['y'][:,j]==0)&valid
                if not bool(positive.any()) or not bool(negative.any()):continue
                diff=lg[positive,j,None]-lg[negative,j][None,:]
                auc=float(((diff>0).float()+.5*(diff==0).float()).mean());ix=torch.where(valid)[0];top=int(ix[lg[valid,j].argmax()])
                out.append(dict(video_id=r['video_id'],variant=r['variant'],label=label,source=mod,unseen=j in held,auc=auc,top1=bool(positive[top])))
    metrics={}
    for subset in ('all','seen','unseen'):
        rr=out if subset=='all' else [r for r in out if r['unseen']==(subset=='unseen')];group={}
        for r in rr:group.setdefault(r['video_id'],[]).append(r)
        metrics[subset]=dict(recordings=len(group),label_variant_tasks=len(rr),macro_recording_auc=sum(sum(r['auc'] for r in rs)/len(rs) for rs in group.values())/max(1,len(group)),
            macro_recording_top1=sum(sum(r['top1'] for r in rs)/len(rs) for rs in group.values())/max(1,len(group)))
    return dict(metrics=metrics,rows=out)
config=dict(seed=seed,steps=1200,lr=.0003,weight_decay=.05,batch_cells=512,seen_labels=[labels[j] for j in seen],held_labels=[labels[j] for j in held],
    selection='max validation seen-label recording-macro AUC, includes step0; unseen labels never select',
    condition='true_text_semantics' if YR_SERVER==1 else 'seen_label_text_shuffle_control',textindex=textindex,
    inference='native media features versus arbitrary input text embedding; fixed-class inventory only for this probe evaluation')
history=[];best=-1;beststep=None;qs=textvectors[textindex][seen]
for step in range(1201):
    if step in (0,100,300,600,1200):
        v=score(val);metric=v['metrics']['seen']['macro_recording_auc'];history.append(dict(step=step,validation=v['metrics']))
        if metric>best:best=metric;beststep=step;torch.save(dict(state_dict=head.state_dict(),config=config,step=step,text_vectors=textvectors.cpu(),labels=labels),YR_DIR/'semanticfield_selected.pt')
        print('YR_SEMANTIC_VAL',json.dumps(history[-1]),flush=True)
    if step==1200:break
    head.train();optimizer.zero_grad(set_to_none=True);ix=torch.randint(len(Y),(512,),device=dev)
    logits=head(X['x'][ix],qs);loss=F.binary_cross_entropy_with_logits(logits,Y[ix][:,seen],pos_weight=posweight[seen],reduction='none')
    loss=(loss*M[ix][:,seen]).sum()/M[ix][:,seen].sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
ck=torch.load(YR_DIR/'semanticfield_selected.pt',map_location=dev,weights_only=True);head.load_state_dict(ck['state_dict'])
RESULT=dict(server=YR_SERVER,config=config,selected_step=beststep,history=history,texts=texts,train=score(train),validation=score(val),diagnostic=score(diag),
    note='Closed inventory evaluation of an arbitrary-text scorer, human category labels and tiny recording split. AUC is not QA accuracy or calibrated absence. Held category embeddings occur in evaluation only; no held-category BCE target is optimized.')
(YR_DIR/'semanticfield.json').write_text(json.dumps(RESULT,indent=2));print('YR_SEMANTIC_DONE',json.dumps({k:RESULT[k]['metrics'] for k in ('train','validation','diagnostic')}),flush=True)
del X,Y,M,head,optimizer;gc.collect();torch.cuda.empty_cache()

import json
import torch
import torch.nn.functional as F
checks=json.loads((V5_DIR/'controls.json').read_text(encoding='utf-8'))
checks['std_plus_presence_no_phi_c5']=[]
# Missing matched ablation: retain C5 retrieval/mean-time/presence and remove only Phi.
for suite_index in range(3):
    epdir=V5_DIR/'artifacts'
    tr=torch.load(epdir/f'm2_episodes_{suite_index*10+2:02d}.pt',map_location='cpu',weights_only=True)
    he=torch.load(epdir/f'm2_episodes_{suite_index*10+3:02d}.pt',map_location='cpu',weights_only=True)
    dev='cuda:0'
    Xtr=torch.cat([tr['std_attn'],tr['null_feat']],-1).to(dev)
    Xhe=torch.cat([he['std_attn'],he['null_feat']],-1).to(dev)
    ytr=torch.tensor([e['label'] for e in tr['metadata']],device=dev)
    yhe=torch.tensor([e['label'] for e in he['metadata']],device=dev)
    mu=Xtr.mean(0,keepdim=True);sd=Xtr.std(0,keepdim=True);valid=(sd>1e-5).float()
    Atr=(Xtr-mu)/sd.clamp(min=1e-4)*valid;Ahe=(Xhe-mu)/sd.clamp(min=1e-4)*valid
    torch.manual_seed(20261015)
    clf=torch.nn.Linear(Xtr.shape[-1],4).to(dev)
    torch.nn.init.zeros_(clf.weight);torch.nn.init.zeros_(clf.bias)
    opt=torch.optim.AdamW(clf.parameters(),lr=.035,weight_decay=.04)
    for _ in range(200):
        loss=F.cross_entropy(clf(Atr),ytr);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        logits=clf(Ahe);pred=logits.argmax(-1);mask=yhe<3;pred3=logits[mask,:3].argmax(-1)
    result=dict(suite_index=suite_index,correct=int((pred==yhe).sum()),total=len(yhe),accuracy4=round(float((pred==yhe).float().mean())*100,1),accuracy3=round(float((pred3==yhe[mask]).float().mean())*100,1),by_class={str(c):dict(correct=int(((yhe==c)&(pred==c)).sum()),n=int((yhe==c).sum())) for c in range(4)})
    name=f'expl_presence_probe_{suite_index:02d}.pt'
    torch.save(dict(X_train=Xtr.cpu(),X_heldout=Xhe.cpu(),labels_train=ytr.cpu(),labels_heldout=yhe.cpu(),logits=logits.cpu(),predictions=pred.cpu(),predictions3=pred3.cpu(),state_dict=v5_cpu_tree(clf.state_dict()),metrics=result),epdir/name)
    checks['std_plus_presence_no_phi_c5'].append(result)
    v5_write('controls.json',checks)
    print('PCV5_STD_PRESENCE',json.dumps(result),flush=True)

RESULT=checks['std_plus_presence_no_phi_c5']
print('PCV5_EXTRA_DONE',json.dumps(RESULT),flush=True)

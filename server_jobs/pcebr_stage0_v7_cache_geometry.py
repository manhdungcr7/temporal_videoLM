"""Descriptive pre-alignment cache audit; no training or model selection."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v7'
out={}
for prefix in ('avbind_native_pairs','avbind_pooled_pairs'):
    examples=[e for s in (1,2) for e in torch.load(R/f'results/{prefix}_s{s}.pt',map_location='cpu',weights_only=True)['examples']]
    x=F.normalize(torch.stack([e['raw'].float() for e in examples]),dim=-1)
    train=torch.tensor([e['split']=='train' for e in examples]);mu=x[train].mean(0,keepdim=True)
    data={}
    for space,features in [('normalized_raw',x),('minus_train_global_mean',x-mu)]:
        refs=[];tests=[];rows=[]
        groups=sorted(set((e['recording_id'],e['source'],e['split']) for e in examples))
        for rid,src,split in groups:
            indices=[i for i,e in enumerate(examples) if (e['recording_id'],e['source'],e['split'])==(rid,src,split)]
            xx=features[indices];mean=xx.mean(0);unit=F.normalize(mean,dim=0)
            aligned_energy=((F.normalize(xx,dim=-1)@unit).square()).mean()
            rec=dict(recording=rid,source=src,split=split,mean_unit=unit,n=len(indices),mean_direction_energy_fraction=float(aligned_energy))
            (refs if split=='train' else tests).append(rec)
        tr=torch.stack([e['mean_unit'] for e in refs]);he=torch.stack([e['mean_unit'] for e in tests])
        cos=he@tr.T
        _,sv,vh=torch.linalg.svd(tr,full_matrices=False);rank=int((sv>1e-5).sum())
        projection=(he@vh[:rank].T).square().sum(-1)
        data[space]=dict(n_train_recording_modality_groups=len(refs),n_heldout_groups=len(tests),train_background_span_rank=rank,
            heldout_vs_train_cosine_mean=float(cos.mean()),heldout_max_train_cosines=cos.max(-1).values.tolist(),
            heldout_energy_in_train_background_span=projection.tolist(),
            group_mean_direction_energy_fractions=[{k:v for k,v in e.items() if k!='mean_unit'} for e in refs+tests])
    out[prefix]=data
result=dict(note='Cache geometry BEFORE learned M1 alignment; mean decomposition is descriptive, not a causal background model. No assertion about all possible learned representations.',results=out)
(D/'cache_geometry_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({p:{s:{k:v for k,v in d.items() if k!='group_mean_direction_energy_fractions'} for s,d in data.items()} for p,data in out.items()},indent=2))

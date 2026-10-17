"""TAM-Lite: learned sensory attention with explicit addresses and frozen vocabulary readout.

Content/source determine retrieval keys; time is retained in values. This is a
small falsification prototype, not a claim of exhaustive event reasoning.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class TAMProbe(nn.Module):
    def __init__(self,hidden=2048,width=64,roles=4,use_time=True):
        super().__init__();self.hidden=hidden;self.width=width;self.roles=roles;self.use_time=use_time
        self.query=nn.Linear(hidden,width*roles,bias=False)
        self.key=nn.Linear(hidden,width,bias=False)
        self.value=nn.Linear(hidden,width,bias=False)
        self.mod_key=nn.Embedding(2,width)
        self.mod_value=nn.Embedding(2,width)
        self.time=nn.Sequential(nn.Linear(38,width),nn.GELU(),nn.Linear(width,width))
        self.memory=nn.Sequential(nn.Linear(roles*width,128),nn.GELU())
        self.context=nn.Linear(hidden,128,bias=False)
        self.out=nn.Linear(128,hidden,bias=False);nn.init.zeros_(self.out.weight)
    def time_features(self,spans,duration):
        # 6 scalar channels + 16 sin/cos channels at each endpoint =38.
        t=spans.float();dur=max(float(duration),1e-5)
        u=t/dur;freq=torch.arange(1,9,device=t.device,dtype=t.dtype)*math.pi
        phase=u.unsqueeze(-1)*freq
        base=torch.stack((u[:,0],u[:,1],u[:,1]-u[:,0],torch.log1p(t[:,0]),torch.log1p(t[:,1]),torch.full_like(u[:,0],math.log1p(dur))),dim=-1)
        return torch.cat((base,phase.sin().flatten(1),phase.cos().flatten(1)),dim=-1)
    def forward(self,bank,query,spans,mods,duration,answer_h):
        b=F.layer_norm(bank.float(),(self.hidden,));q=F.layer_norm(query.float(),(self.hidden,))
        keys=self.key(b)+self.mod_key(mods)
        queries=self.query(q).reshape(self.roles,self.width)
        scores=(queries@keys.T)/math.sqrt(self.width)
        prob=F.softmax(scores.float(),dim=-1)
        tf=self.time_features(spans,duration)
        if not self.use_time:tf=torch.zeros_like(tf)
        values=self.value(b)+self.mod_value(mods)+self.time(tf)
        mem=self.memory((prob.to(values.dtype)@values).reshape(-1))
        h=answer_h.float();ctx=self.context(F.layer_norm(h,(self.hidden,)))
        z=self.out(F.gelu(ctx+mem))
        # Zero-init gives native identity; a bounded residual prevents arbitrary norm growth.
        delta=.10*h.norm(dim=-1,keepdim=True)/math.sqrt(self.hidden)*torch.tanh(z.float())
        return h+delta,prob,dict(delta_norm=delta.norm(dim=-1).detach(),native_norm=h.norm(dim=-1).detach())

def target_density(case,device):
    spans=case['spans'].to(device).float();ints=case.get('intervals')
    if not ints:return None
    intervals=torch.tensor(ints,device=device,dtype=torch.float32)
    overlap=(torch.minimum(spans[:,None,1],intervals[None,:,1])-torch.maximum(spans[:,None,0],intervals[None,:,0])).clamp_min(0).sum(1)
    source=0 if case['source']=='audio' else 1
    overlap=overlap*(case['mods'].to(device)==source)
    if float(overlap.sum())<=0:return None
    return overlap/overlap.sum()

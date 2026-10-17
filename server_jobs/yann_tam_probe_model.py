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
    def forward(self,bank,query,spans,mods,duration,answer_h,force_attention=None):
        b=F.layer_norm(bank.float(),(self.hidden,));q=F.layer_norm(query.float(),(self.hidden,))
        keys=self.key(b)+self.mod_key(mods)
        queries=self.query(q).reshape(self.roles,self.width)
        scores=(queries@keys.T)/math.sqrt(self.width)
        prob=F.softmax(scores.float(),dim=-1)
        if force_attention is not None:prob=force_attention.to(device=prob.device,dtype=prob.dtype)
        tf=self.time_features(spans,duration)
        if not self.use_time:tf=torch.zeros_like(tf)
        values=self.value(b)+self.mod_value(mods)+self.time(tf)
        mem=self.memory((prob.to(values.dtype)@values).reshape(-1))
        h=answer_h.float();ctx=self.context(F.layer_norm(h,(self.hidden,)))
        z=self.out(F.gelu(ctx+mem))
        # Zero-init gives native identity; a bounded residual prevents arbitrary norm growth.
        delta=.10*h.norm(dim=-1,keepdim=True)/math.sqrt(self.hidden)*torch.tanh(z.float())
        return h+delta,prob,dict(delta_norm=delta.norm(dim=-1).detach(),native_norm=h.norm(dim=-1).detach())

class TAMBindingProbe(TAMProbe):
    """Correction vanishes when the retrieved temporal innovation vanishes.

    This identity says nothing about semantic correctness or calibrated absence.
    """
    def __init__(self,*args,center_time=True,contrast_path=True,**kwargs):
        super().__init__(*args,**kwargs);self.center_time=center_time;self.contrast_path=contrast_path
    def forward(self,bank,query,spans,mods,duration,answer_h,force_attention=None):
        b=F.layer_norm(bank.float(),(self.hidden,));q=F.layer_norm(query.float(),(self.hidden,))
        keys=self.key(b)+self.mod_key(mods);queries=self.query(q).reshape(self.roles,self.width)
        prob=F.softmax(((queries@keys.T)/math.sqrt(self.width)).float(),dim=-1)
        if force_attention is not None:prob=force_attention.to(device=prob.device,dtype=prob.dtype)
        content=self.value(b)+self.mod_value(mods)
        time=self.time(self.time_features(spans,duration)).float()
        with torch.autocast('cuda',enabled=False) if bank.is_cuda else torch.autocast('cpu',enabled=False):
            # Global centering retains relative source-clock offsets. Per-modality centering would not.
            st=prob.float()@time
            if self.center_time:st=st-prob.sum(-1,keepdim=True)*time.mean(0,keepdim=True)
        sc=(prob.to(content.dtype)@content).reshape(-1)
        mc=self.memory(sc);mt=self.memory(sc+st.reshape(-1).to(sc.dtype))
        h=answer_h.float();ctx=self.context(F.layer_norm(h,(self.hidden,)))
        difference=F.gelu(ctx+mt)
        if self.contrast_path:difference=difference-F.gelu(ctx+mc)
        z=self.out(difference)
        delta=.10*h.norm(dim=-1,keepdim=True)/math.sqrt(self.hidden)*torch.tanh(z.float())
        return h+delta,prob,dict(delta_norm=delta.norm(dim=-1).detach(),native_norm=h.norm(dim=-1).detach(),
            temporal_summary_norm=st.norm().detach())

def target_density(case,device):
    spans=case['spans'].to(device).float();ints=case.get('intervals')
    if not ints:return None
    intervals=torch.tensor(ints,device=device,dtype=torch.float32)
    overlap=(torch.minimum(spans[:,None,1],intervals[None,:,1])-torch.maximum(spans[:,None,0],intervals[None,:,0])).clamp_min(0).sum(1)
    source=0 if case['source']=='audio' else 1
    overlap=overlap*(case['mods'].to(device)==source)
    if float(overlap.sum())<=0:return None
    return overlap/overlap.sum()

def restore_cached_video_support(case):
    """For these caches only: each video bank row is one native two-frame tubelet.

    Sampling<=2fps and half-second grouping ensure distinct video grids were not
    merged. Live inference should use processor fps/tubelet metadata directly.
    """
    v=case['mods']==1;count=int(v.sum())
    if not count:return case
    width=float(case['duration'])/count
    if not .8<=width<=2.5:raise ValueError(('cannot infer cached visual support',width))
    case=dict(case);case['spans']=case['spans'].clone()
    case['spans'][v,0]=case['times'][v].clamp_min(0)
    case['spans'][v,1]=(case['times'][v]+width).clamp_max(float(case['duration']))
    return case

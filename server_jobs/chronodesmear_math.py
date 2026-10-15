"""Bounded-memory equivalents of Round5 temporal averages; no model globals."""
import torch
import torch.nn.functional as F

def interval_plan(indices, seconds, min_lag=1.8, max_lag=12.0):
    indices=indices.cpu().long();times=seconds.cpu().float()[indices]
    assert bool((times[1:]>=times[:-1]).all()), 'Source timestamps must be monotone'
    # searchsorted boundaries implement inclusive dt >= min_lag and <= max_lag.
    # Correct one-ULP subtraction boundary differences with explicit comparisons.
    low=torch.searchsorted(times,times-max_lag,right=False)
    high=torch.searchsorted(times,times-min_lag,right=True)
    for row in range(len(times)):
        while low[row]>0 and times[row]-times[low[row]-1]<=max_lag: low[row]-=1
        while low[row]<len(times) and times[row]-times[low[row]]>max_lag: low[row]+=1
        while high[row]>0 and times[row]-times[high[row]-1]<min_lag: high[row]-=1
        while high[row]<len(times) and times[row]-times[high[row]]>=min_lag: high[row]+=1
    return dict(indices=indices,low=low,high=high,has_previous=high>low)

def interval_mean(values, plan):
    prefix=torch.cat((torch.zeros_like(values[:1]),values.cumsum(0)),dim=0)
    low=plan['low'].to(values.device);high=plan['high'].to(values.device)
    count=(high-low).clamp_min(1).unsqueeze(-1)
    return (prefix[high]-prefix[low])/count

def desmear_values(values,plan,strength):
    if strength==0: return values
    valid=plan['has_previous'].to(values.device)
    if not bool(valid.any()): return values
    echo=interval_mean(values,plan)
    cleaned=values-strength*valid[:,None]*echo
    cleaned=cleaned*(values.norm(dim=-1,keepdim=True).clamp_min(1e-5)
        /cleaned.norm(dim=-1,keepdim=True).clamp_min(1e-5))
    # Exact identity for tokens without a past, rather than numerical norm roundoff.
    return torch.where(valid[:,None],cleaned,values)

def static_visual_mask(states,is_v,seconds,duration,n_bins=5,static_frac=.45):
    indices=torch.where(is_v.cpu().bool())[0]
    if not len(indices): raise ValueError('No visual tokens')
    vectors=F.normalize(states[indices].float(),dim=-1)
    mean=F.normalize(vectors.mean(0,keepdim=True),dim=-1)
    global_change=1-(vectors*mean).sum(-1)
    # Original local predicate is strict dt > .5 and inclusive dt <= 3.
    times=seconds.cpu().float()[indices]
    plan=interval_plan(indices,seconds,.5,3.)
    for row in range(len(times)):
        while plan['high'][row]>0 and times[row]-times[plan['high'][row]-1]<=.5:
            plan['high'][row]-=1
    plan['has_previous']=plan['high']>plan['low']
    local_mean=F.normalize(interval_mean(vectors,plan),dim=-1)
    local_change=torch.where(plan['has_previous'],1-(vectors*local_mean).sum(-1),global_change)
    innovation=.55*global_change+.45*local_change
    selected=torch.zeros(len(is_v),dtype=torch.bool)
    lower,upper=float(times.min()),float(times.max());span=max(1e-3,upper-lower)
    bin_counts=[]
    for b in range(n_bins):
        left=lower+b/n_bins*span
        right=lower+(b+1)/n_bins*span+(1e-4 if b==n_bins-1 else 0.)
        members=torch.where((times>=left)&(times<right))[0]
        count=max(1,int(round(static_frac*len(members)))) if len(members) else 0
        if count:
            chosen=members[torch.topk(innovation[members],count,largest=False).indices]
            selected[indices[chosen]]=True
        bin_counts.append(dict(bin=b,total=len(members),static=count))
    return selected,innovation,indices,bin_counts

def counterfactual_mask(payload,mode,dtype,static=None,window=None):
    length=payload['tokens'];v=payload['is_v'].bool();a=payload['is_a'].bool()
    mask=torch.zeros((length,length),dtype=dtype)
    mask.masked_fill_(torch.ones((length,length),dtype=torch.bool).triu_(1),torch.finfo(dtype).min)
    if mode=='static_plus_aud': blocked=v&~static
    elif mode=='v_all_only': blocked=a
    elif mode in ('hard_sync','anti_sync'):
        start,end=window;sec=payload['sec'].float();width=payload['spg'] or 0.
        inside=v&((sec+width)>start)&(sec<end)
        if not bool(inside.any()): inside=v.clone()
        blocked=v&~inside if mode=='hard_sync' else inside
    else: raise ValueError(mode)
    mask[(~blocked)[:,None]&blocked[None,:]]=torch.finfo(dtype).min
    if mode=='hard_sync': mask[(~a)[:,None]&a[None,:]]=torch.finfo(dtype).min
    return mask[None,None]

"""Contrastive residual CQ memory with a separate extraction/readout clock.

CQ branch states are compared against identically positioned text-only branches.
The question keeps its native media-relative position and base attention path.
Only media-dependent probe values are added in middle/late layers. No output
class offset or task rule is involved. The null memory is cached per clip.
"""
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import rotate_half

MANIFEST = json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
DATA = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
ITEMS = [(x['idx'],DATA[x['idx']]) for x in MANIFEST['selected']]
RCTX = {}
PV = ' Video content from {t0}s to {t1}s, in one short phrase:'
PA = ' Audio content from {t0}s to {t1}s, in one short phrase:'


def residual_attention(module,query,key,value,attention_mask,scaling=None,dropout=0.0,**kw):
    out,_ = sdpa_attention_forward(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kw)
    c=RCTX.get('cfg')
    li=getattr(module,'_cq_layer',-1)
    if not c or li < 0:
        return out,None
    ends=c['ends'].to(key.device)
    assert int(c['ends'].max()) < key.shape[2], 'CQ endpoint outside decoder sequence'
    if c['capture']:
        c['null'][li]=(key[:,:,ends,:].detach().cpu(),value[:,:,ends,:].detach().cpu())
        return out,None
    if li not in range(13,28) or c['gain']==0:
        return out,None
    rr=c['rows'].to(query.device)
    kn,vn=c['null'][li]
    k=key[:,:,ends,:].float()
    v=value[:,:,ends,:].float()-vn.to(value.device).float()
    if c['contrast_keys']:
        k=k-kn.to(key.device).float()
    # Endpoint K was extracted at its own chunk's clock. Only the Q/A edge
    # is rephased to the shared reader clock; the extraction is untouched.
    delta=c['read_pos']-c['end_pos'].to(key.device)
    inv=module.rotary_emb.inv_freq.to(key.device).float()
    freq=delta.float()[:,None]*inv[None,:]
    angle=torch.cat((freq,freq),-1)[None,None]
    k=k*angle.cos()+rotate_half(k)*angle.sin()
    rep=query.shape[1]//key.shape[1]
    k,v=k.repeat_interleave(rep,1),v.repeat_interleave(rep,1)
    q=query[:,:,rr,:].float()
    s=(q@k.transpose(-1,-2))*(scaling or q.shape[-1]**-0.5)
    # Normalize both modalities separately, preserving arbitrary chunk counts.
    p=s.reshape(*s.shape[:-1],-1,2).softmax(-2).reshape(*s.shape)
    mem=(0.5*p)@v
    # Bound each memory contribution by the native output magnitude. This
    # avoids amplifying near-empty probes and preserves the baseline path.
    native=out[:,rr].transpose(1,2).float()
    ratio=(native.square().mean(-1,keepdim=True).sqrt()/
           mem.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).clamp(max=1.0)
    z=native+c['gain']*ratio*mem
    out=out.clone()
    out[:,rr]=z.to(out.dtype).transpose(1,2)
    return out,None


def residual_forward(inp,pieces,null=None,gain=0.0,contrast_keys=False):
    lay=Lay(inp)
    spans=locate(inp['input_ids'].cpu(),''.join(pieces),pieces,lay.last_media)
    m,pos=lay.m.clone(),lay.pos.clone()
    pcols=torch.zeros(lay.L,dtype=torch.bool)
    for s,e in spans:
        pcols[s:e]=True
    first=int(torch.where(lay.v|lay.a)[0][0])
    safe=torch.arange(lay.L)<first
    end_pos=[]
    native_start=int(lay.pos[0,0,spans[0][0]])
    for g,(s,e) in enumerate(spans):
        own=(lay.v if g%2==0 else lay.a)&(lay.chunk==g//2)
        allowed=safe|own
        m[s:e,~allowed]=float('-inf')
        m[s:e,s:e]=torch.triu(torch.full((e-s,e-s),float('-inf')),1)
        start=int(lay.pos[:,0,own].max())+1 if own.any() else native_start
        pos[:,:,s:e]=(start+torch.arange(e-s)).view(1,1,-1).to(pos.dtype)
        end_pos.append(start+e-s-1)
        if null is None:
            m[s:e,own]=float('-inf')
    qstart=spans[-1][1]
    rows=torch.arange(qstart,lay.L)
    # Q/A cannot read literal probe scaffolds. It receives only media-dependent
    # residual features through the added attention branch.
    m[rows[:,None],torch.where(pcols)[0][None,:]]=float('-inf')
    pos[:,:,qstart:]=(native_start+torch.arange(lay.L-qstart)).view(1,1,-1).to(pos.dtype)
    capture=null is None
    c=dict(ends=torch.tensor([e-1 for s,e in spans]),end_pos=torch.tensor(end_pos),
           read_pos=native_start,rows=rows,capture=capture,null={} if capture else null,
           gain=gain,contrast_keys=contrast_keys)
    RCTX['cfg']=c
    old=ALL_ATTENTION_FUNCTIONS['sdpa']
    ALL_ATTENTION_FUNCTIONS['sdpa']=residual_attention
    hook=MODEL_.lm_head.register_forward_pre_hook(lambda mod,args:(args[0][:,-1:,:],))
    kw={k:v for k,v in inp.items() if k!='attention_mask'}
    try:
        with torch.no_grad():
            lg=MODEL_(**kw,attention_mask=lay.t(m),position_ids=pos.to(lay.dev),
                      use_audio_in_video=True,use_cache=False).logits[0,-1].float()
    finally:
        hook.remove()
        ALL_ATTENTION_FUNCTIONS['sdpa']=old
        RCTX.clear()
    return lg,c['null']


def residual_item(i,row):
    src,audio,dur,win=prep_clip(row,50)
    media=prepare_video(src)
    stem,options=parse_question(row['question'])
    inp,_=qwen_inputs(PROC,'x',media,'av')
    inp=move_inputs(inp,MODEL_)
    nc=Lay(inp).nc
    del inp
    pieces=[p.format(t0=2*t,t1=round(min(2*t+2,dur),2)) for t in range(nc) for p in (PV,PA)]
    prefix=''.join(pieces)+'\n'+stem
    inp,_=qwen_inputs(PROC,prefix,media,'av')
    inp=move_inputs(inp,MODEL_)
    _,null=residual_forward(inp,pieces)
    del inp
    rec=dict(dur=round(dur,3),options=options)
    for name,gain,keys in (('cq_native_path',0.0,False),('cq_residual03',0.3,False),
                           ('cq_residual10',1.0,False),('cq_residual_keys',0.3,True)):
        rec[name]=score(media,prefix,options,fwd=lambda inp,g=gain,k=keys:
                        residual_forward(inp,pieces,null,g,k)[0])
    return rec


CONDITIONS=('cq_native_path','cq_residual03','cq_residual10','cq_residual_keys')
R=run_loop(ITEMS,f'cq_diagnostic_round4v2_s{LAB_SERVER}.jsonl',residual_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND4_RESULT',json.dumps(RESULT),flush=True)

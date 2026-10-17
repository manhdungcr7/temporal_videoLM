"""Independent public-label native verification; all scoring has no QA task rules."""
import gc,hashlib,io,json,math,subprocess,time,traceback,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import av

D=OUT/'native_interval_verification';D.mkdir(exist_ok=True)
for name in ('originals','crops','artifacts'):(D/name).mkdir(exist_ok=True)
manifest_path=OUT/'uploads/native_interval_manifest.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='81c5c9d56d3b9dd390a0902d4aa6afcf538d3bd910f14f0e82f59cf69e10b0c8'
M=json.loads(manifest_path.read_text());pairs=[r for r in M['pairs'] if r['server']==PC_SERVER]
previous=json.loads((D/'samples.json').read_text()) if (D/'samples.json').exists() else []
records=list(previous);completed={(r['pair_id'],r['window'],r['mode']) for r in records}
versions={n:p._version for n,p in MODEL_.named_parameters()}
assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules())==0
tok=PROC.tokenizer
yes=tok.encode('Yes',add_special_tokens=False);no=tok.encode('No',add_special_tokens=False)
assert len(yes)==len(no)==1,(yes,no)
vid_id=tok.convert_tokens_to_ids(PROC.video_token);aud_id=tok.convert_tokens_to_ids(PROC.audio_token)
E=MODEL_.get_input_embeddings().weight.detach()
mu_vocab=E[::max(1,E.shape[0]//2048)].float().mean(0).cpu()

class RangeZip(io.RawIOBase):
    def __init__(self):
        self.url='https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip'
        with urllib.request.urlopen(urllib.request.Request(self.url,method='HEAD'),timeout=50) as r:self.length=int(r.headers['Content-Length'])
        self.pos=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,x,w=0):self.pos=x if w==0 else self.pos+x if w==1 else self.length+x;return self.pos
    def read(self,n=-1):
        n=self.length-self.pos if n<0 else min(n,self.length-self.pos)
        if n==0:return b''
        assert n<=16_000_000,n
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        with urllib.request.urlopen(req,timeout=60) as r:
            assert r.status==206,r.status
            b=r.read(n+1)
        assert len(b)==n,(len(b),n)
        self.pos+=n;self.transferred+=n;return b

def digest_tensor(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

def duration_audit(path):
    with av.open(str(path)) as c:
        ss={s.type:s for s in c.streams}
        d={k:float(s.duration*s.time_base) for k,s in ss.items() if s.duration is not None}
        rates={k:float(s.start_time*s.time_base) if s.start_time is not None else None for k,s in ss.items()}
    return dict(durations=d,start_times=rates)

def inference(media,mode,prompt,target=None,capture=True):
    inp,rendered=qwen_inputs(PROC,prompt,media,mode)
    plen=int(inp['input_ids'].shape[1]);prefix_ids=inp['input_ids'][0]
    masks={'vision':prefix_ids==vid_id,'audio':prefix_ids==aud_id}
    count={k:int(v.sum()) for k,v in masks.items()}
    last_media=int(torch.nonzero(masks['vision']|masks['audio'],as_tuple=False).max())
    prefix_hash=digest_tensor(prefix_ids[:last_media+1])
    if target is not None:
        target_ids=tok.encode(target,add_special_tokens=False)
        assert 0<len(target_ids)<80,target_ids
        inp['input_ids']=torch.cat((inp['input_ids'],torch.tensor([target_ids],dtype=torch.long)),dim=1)
        inp['attention_mask']=torch.ones_like(inp['input_ids'])
        lo,hi=plen-1,plen+len(target_ids)-1
    else:target_ids=None;lo,hi=plen-1,plen
    inp=move_inputs(inp,MODEL_);box={}
    def layer_cap(mod,args,out):
        h=out[0] if isinstance(out,tuple) else out
        for k,mask in masks.items():
            ids=torch.nonzero(mask,as_tuple=False).squeeze(-1).to(h.device)
            if ids.numel():box[k]=h[0,ids].detach().float().mean(0).cpu()
    def select_head(mod,args):return (args[0][:,lo:hi,:],)
    hooks=[MODEL_.lm_head.register_forward_pre_hook(select_head)]
    if capture:hooks.append(MODEL_.model.layers[22].register_forward_hook(layer_cap))
    try:
        with torch.inference_mode():
            out=MODEL_(**inp,use_audio_in_video=mode=='av',use_cache=False,return_dict=True)
            logits=out.logits[0].detach().float().cpu()
    finally:
        for h in hooks:h.remove()
    if target_ids is None:
        lg=logits[-1]
        score=dict(yes_no_log_odds=float(lg[yes[0]]-lg[no[0]]),yes_probability_given_yes_no=float(torch.softmax(lg[[yes[0],no[0]]],-1)[0]),
            yes_logit=float(lg[yes[0]]),no_logit=float(lg[no[0]]),unrestricted_argmax_id=int(lg.argmax()),unrestricted_argmax_text=tok.decode([int(lg.argmax())]))
    else:
        token_lp=F.log_softmax(logits,dim=-1)[torch.arange(len(target_ids)),torch.tensor(target_ids)]
        score=dict(description_mean_logprob=float(token_lp.mean()),description_token_logprobs=token_lp.tolist(),target_ids=target_ids)
    info=dict(prompt=prompt,rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),input_ids_sha256=digest_tensor(inp['input_ids']),
        prefix_ids_sha256=prefix_hash,prompt_tokens=plen,total_tokens=int(inp['input_ids'].shape[1]),media_token_counts=count)
    del inp,out,logits
    return score,box,info

def lens_score(h,description):
    if h is None:return None,None
    ids=tok.encode(description,add_special_tokens=False)
    nm=MODEL_.model.norm;head=MODEL_.lm_head
    with torch.inference_mode():
        x=h[None].to(device=next(nm.parameters()).device,dtype=next(nm.parameters()).dtype)
        logits=head(nm(x).to(head.weight.device))[0].float()
        vals,inds=torch.topk(logits,k=48)
        probs=F.softmax(vals/.65,-1).to(E.device)
        z=F.normalize((probs[:,None]*E[inds.to(E.device)].float()).sum(0).cpu()-mu_vocab,dim=-1)
        q=F.normalize(E[torch.tensor(ids,device=E.device)].float().mean(0).cpu()-mu_vocab,dim=-1)
        score=dict(pooled_l22_manifold_cosine=float(torch.dot(z,q)),pooled_l22_label_gap_mean=float(((logits[ids]-vals[-1])/12.).mean()),
            top48_ids=inds.cpu().tolist(),top48_texts=[tok.decode([int(i)]) for i in inds.cpu()])
    return score,dict(z=z,q=q)

t_run=time.time();range_zip=None;zip_handle=None
try:
    for pair in pairs:
        pid=pair['pair_id'];vid=pair['video_id'];original=D/'originals'/f'{vid}.mp4'
        if not original.exists():
            if range_zip is None:range_zip=RangeZip();zip_handle=zipfile.ZipFile(range_zip)
            blob=zip_handle.read(pair['zip']['name']);assert len(blob)==pair['zip']['bytes']
            original.write_bytes(blob)
        original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
        source_mode=pair['modality'];opposite_mode='vision' if source_mode=='audio' else 'audio'
        noun,verb=('sound','heard') if source_mode=='audio' else ('action','seen')
        presence=f'Can the following {noun} be {verb} in this clip: "{pair["description"]}"? Answer with only Yes or No.'
        desc_prompt='Describe the sounds heard in this clip.' if source_mode=='audio' else 'Describe the actions visible in this clip.'
        for window in ('positive','negative'):
            crop=D/'crops'/f'{pid}_{window}.mp4';a,b=pair[window]
            if not crop.exists():
                cmd=['ffmpeg','-nostdin','-v','error','-ss',f'{a:.6f}','-i',str(original),'-t',f'{b-a:.6f}',
                    '-map','0:v:0','-map','0:a:0','-c:v','libx264','-preset','fast','-crf','18','-c:a','aac','-b:a','192k','-avoid_negative_ts','make_zero','-y',str(crop)]
                subprocess.run(cmd,check=True,capture_output=True,timeout=60)
            audit=duration_audit(crop)
            assert all(abs(d-(b-a))<=.16 for d in audit['durations'].values()),audit
            assert set(audit['durations'])=={'audio','video'},audit
            media=prepare_video(crop,max_pixels=96*28*28)
            media_audit=dict(audio_samples=len(media['audio'][0]),decoded_audio_sec=len(media['audio'][0])/16000,
                video_shape=list(media['videos'][0].shape),fps=media['fps'])
            for mode in ('av',source_mode,opposite_mode):
                if (pid,window,mode) in completed:continue
                start=time.time()
                pred,means,pi=inference(media,mode,presence)
                descr,means_d,di=inference(media,mode,desc_prompt,target=pair['description'])
                assert pi['prefix_ids_sha256']==di['prefix_ids_sha256'],('prefix_input_changed',pid,mode)
                mean_errors={k:float((v-means_d[k]).abs().max()) for k,v in means.items()}
                lens,lens_vectors=lens_score(means.get(source_mode),pair['description'])
                art=D/'artifacts'/f'{pid}_{window}_{mode}.pt'
                torch.save(dict(h22_media_means=means,description_h22_media_means=means_d,lens_vectors=lens_vectors),art)
                rec=dict(pair_id=pid,video_id=vid,split=pair['split'],source_modality=source_mode,label=pair['label'],description=pair['description'],
                    window=window,interval=pair[window],mode=mode,source_present=mode in ('av',source_mode),presence=pred,description_score=descr,lens=lens,
                    prefix_mean_query_change_max_errors=mean_errors,presence_input=pi,description_input=di,crop_audit=audit,media_audit=media_audit,
                    original_sha256=original_hash,crop_sha256=hashlib.sha256(crop.read_bytes()).hexdigest(),artifact_sha256=hashlib.sha256(art.read_bytes()).hexdigest(),elapsed_sec=time.time()-start)
                records.append(rec);completed.add((pid,window,mode))
                tmp=D/'samples.tmp';tmp.write_text(json.dumps(records,indent=2));tmp.replace(D/'samples.json')
                print('INTERVAL_SAMPLE',json.dumps(dict(pair_id=pid,window=window,mode=mode,odds=pred['yes_no_log_odds'],ll=descr['description_mean_logprob'],seconds=rec['elapsed_sec'])),flush=True)
                gc.collect();torch.cuda.empty_cache()
            del media;gc.collect();torch.cuda.empty_cache()
    assert len(records)==len(pairs)*6,(len(records),len(pairs))
    assert all(p._version==versions[n] for n,p in MODEL_.named_parameters())
    assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules())==0
    result=dict(server=PC_SERVER,n_pairs=len(pairs),n_cases=len(records),records=records,backbone_versions_unchanged=True,remaining_hooks=0,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),elapsed_sec=time.time()-t_run,
        range_bytes_transferred=range_zip.transferred if range_zip else 0,torch=torch.__version__,model=MODEL_.config._name_or_path)
    (D/'completion.json').write_text(json.dumps(result,indent=2))
    RESULT=dict(n_pairs=len(pairs),n_cases=len(records),seconds=time.time()-t_run)
    print('INTERVAL_DONE',json.dumps(RESULT),flush=True)
except Exception:
    error=dict(error=traceback.format_exc(),n_cases=len(records),seconds=time.time()-t_run)
    (D/f'error_{len(list(D.glob("error_*.json")))}.json').write_text(json.dumps(error,indent=2))
    raise
finally:
    if zip_handle is not None:zip_handle.close()

"""Option-free same-hook sensory/query features; native MC is a separate readout reference."""
import gc,json,time,torch
YR_DIR=OUT/'yann_temporal_research';YR_MICRO=YR_DIR/'microfeatures';YR_MICRO.mkdir(exist_ok=True)
yr_manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text());yr_rows=[]
for row in yr_manifest[f'server{YR_SERVER}'][:4]:
    p=torch.load(OUT/'argon_latent40/decoder_cache'/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=True)
    stem,_=parse_question(row['question']);lm=p['last_media'];d=MODEL_.get_input_embeddings().weight.device
    old_ids=p['input_ids'][0];tail=PROC.tokenizer.decode(old_ids[lm+1:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
    start=tail.find(stem);end=tail.rfind('<|im_end|>');assert start>=0 and end>start+len(stem)
    newtail=tail[:start]+stem+'\nAnswer the question directly.'+tail[end:]
    nt=torch.tensor(PROC.tokenizer(newtail,add_special_tokens=False)['input_ids'],dtype=torch.long,device=d)
    assert '(A)' not in newtail and 'A.' not in newtail
    prefix=p['decoder']['inputs_embeds'][:,:lm+1].to(d)
    with torch.no_grad():emb=torch.cat([prefix,MODEL_.get_input_embeddings()(nt)[None]],dim=1)
    first=int(p['decoder']['position_ids'][0,0,lm+1]);n=emb.shape[1]
    pos=torch.cat([p['decoder']['position_ids'][:,:,:lm+1].to(d),(first+torch.arange(len(nt),device=d)).view(1,1,-1).expand(3,1,-1)],dim=-1)
    kw=dict(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones((1,n),dtype=torch.long,device=d))
    cap={};iv=p['is_v'];ia=p['is_a'];sec=p['sec'];spans=[];times=[];mods=[]
    def mid(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;features=[]
        cap['q']=h[0,-1].detach().cpu()
        for mod,mask in ((0,ia),(1,iv)):
            ix=torch.where(mask)[0];bins=torch.floor(sec[ix]*2+1e-4).long()
            for k in torch.unique(bins,sorted=True).tolist():
                inds=ix[bins==k];features.append(h[0,inds.to(h.device)].float().mean(0).half().cpu())
                times.append(float(sec[inds].mean()));mods.append(mod);spans.append([max(0.,k*.5),min(row['duration'],(k+1)*.5)])
        cap['bank']=torch.stack(features)
    hook=MODEL_.model.layers[18].register_forward_hook(mid)
    try:
        with torch.inference_mode():ff=MODEL_.model(**kw,use_cache=False,return_dict=True)
    finally:hook.remove()
    cap['answer_h']=ff.last_hidden_state[0,-1:].detach().cpu();cap.update(times=torch.tensor(times),mods=torch.tensor(mods),spans=torch.tensor(spans),duration=row['duration'],question=stem,idx=row['idx'])
    nativekw={k:v.to(d) for k,v in p['decoder'].items()}
    with torch.inference_mode():
        mc=MODEL_.model(**nativekw,use_cache=False,return_dict=True);cap['mc_h']=mc.last_hidden_state[0,-1:].detach().cpu()
        lg=MODEL_.lm_head(mc.last_hidden_state[:,-1:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
    labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC'][:len(p['native_logits'])]
    err=float((lg[labels]-p['native_logits']).abs().max());assert err<1e-5,err
    torch.save(cap,YR_MICRO/f"{row['idx']}.pt")
    payload=dict(decoder={k:v.detach().cpu() for k,v in kw.items()},last_media=lm,input_ids=torch.cat([old_ids[:lm+1],nt.cpu()])[None])
    torch.save(payload,YR_MICRO/f"{row['idx']}_optionfree_decoder.pt")
    yr_rows.append(dict(idx=row['idx'],native_mc_replay_error=err,bank_shape=list(cap['bank'].shape),optionfree_prompt=newtail,native_tokens=n))
    (YR_DIR/'microfeatures_partial.json').write_text(json.dumps(yr_rows,indent=2),encoding='utf-8')
    print('YR_MICROFEATURES',row['idx'],list(cap['bank'].shape),err,flush=True)
    del ff,mc,kw,nativekw,lg,emb,prefix,payload,cap;gc.collect();torch.cuda.empty_cache()
RESULT=dict(server=YR_SERVER,rows=yr_rows,hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
assert RESULT['hooks']==0
(YR_DIR/'microfeatures.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')

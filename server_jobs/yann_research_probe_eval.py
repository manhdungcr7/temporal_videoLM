"""R3 transfer: same selected checkpoints, original MC, and option-free generated text."""
import gc,importlib.util,json,time,torch
YR_DIR=OUT/'yann_temporal_research'
sp=importlib.util.spec_from_file_location('yann_tam_probe_model',OUT/'uploads/yann_tam_probe_model.py');tm=importlib.util.module_from_spec(sp);sp.loader.exec_module(tm)
dev=MODEL_.get_input_embeddings().weight.device;hd=MODEL_.lm_head.weight.device;yr_probes={}
for s in (1,2):
    ck=torch.load(OUT/f'uploads/r3_probe_s{s}.pt',map_location=dev,weights_only=True)
    name=ck['config']['name'];model=tm.TAMProbe(use_time=s==2).to(dev);model.load_state_dict(ck['state_dict']);model.eval();yr_probes[name]=model
yr_manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text());yr_rows=[]
def yr_adjust(c,h,probe,spans=None,bank=None):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        z,prob,di=probe((c['bank'] if bank is None else bank).to(dev),c['q'].to(dev),(c['spans'] if spans is None else spans).to(dev),c['mods'].to(dev),c['duration'],h.to(dev))
    return z,prob,di
def yr_generate(payload,c,probe):
    kw={k:v.to(dev) for k,v in payload['decoder'].items()};generated=[];past=None;first_error=None
    ep=MODEL_.generation_config.eos_token_id;eos=set(ep if isinstance(ep,list) else [ep])
    lastpos=kw['position_ids'][:,:,-1:];n=kw['inputs_embeds'].shape[1]
    with torch.inference_mode():
        for t in range(32):
            if t==0:out=MODEL_.model(**kw,use_cache=True,return_dict=True)
            else:
                token=torch.tensor([[generated[-1]]],device=dev)
                out=MODEL_.model(input_ids=token,position_ids=lastpos+t,attention_mask=torch.ones((1,n+t),dtype=torch.long,device=dev),
                    cache_position=torch.tensor([n+t-1],device=dev),past_key_values=past,use_cache=True,return_dict=True)
            past=out.past_key_values;h=out.last_hidden_state[0,-1:]
            if t==0:first_error=float((h.cpu()-c['answer_h']).abs().max());assert first_error<1e-5,first_error
            if probe is not None:h,_,_=yr_adjust(c,h,probe)
            logits=MODEL_.lm_head(h.to(device=hd,dtype=MODEL_.lm_head.weight.dtype))[0].float();token=int(logits.argmax());generated.append(token)
            if token in eos:break
    del out,past;gc.collect();torch.cuda.empty_cache()
    return dict(text=PROC.tokenizer.decode(generated,skip_special_tokens=True),token_ids=generated,prefill_hidden_parity_error=first_error)
for row in yr_manifest[f'server{YR_SERVER}'][:4]:
    c=torch.load(YR_DIR/'microfeatures'/f"{row['idx']}.pt",map_location='cpu',weights_only=True)
    payload=torch.load(YR_DIR/'microfeatures'/f"{row['idx']}_optionfree_decoder.pt",map_location='cpu',weights_only=True)
    _,options=parse_question(row['question']);labels=[PROC.tokenizer.convert_tokens_to_ids(x) for x in 'ABC'][:len(options)]
    r=dict(idx=row['idx'],task=row['task'],gold=row['gold'],mc={},generated={})
    conditions=[('native',None,None,None)]+[(name,probe,None,None) for name,probe in yr_probes.items()]
    full=yr_probes['TAM_Lite_time'];gen=torch.Generator().manual_seed(20261016+row['idx']);perm=torch.randperm(len(c['bank']),generator=gen)
    conditions += [('TAM_Lite_time_address_shuffle',full,c['spans'][perm],None),('TAM_Lite_time_zero_memory',full,None,torch.zeros_like(c['bank']))]
    with torch.no_grad():
        for name,probe,spans,bank in conditions:
            h=c['mc_h'].to(dev);prob=None
            if probe is not None:h,prob,di=yr_adjust(c,h,probe,spans,bank)
            lg=MODEL_.lm_head(h.to(device=hd,dtype=MODEL_.lm_head.weight.dtype))[0,labels].float().cpu();pred='ABC'[int(lg.argmax())]
            r['mc'][name]=dict(prediction=pred,correct=pred==row['gold'],logits=lg.tolist(),probabilities=lg.softmax(0).tolist(),
                attention=prob.cpu().tolist() if prob is not None else None)
    for name,probe in [('native',None)]+list(yr_probes.items()):r['generated'][name]=yr_generate(payload,c,probe)
    yr_rows.append(r);(YR_DIR/'probe_eval_partial.json').write_text(json.dumps(yr_rows,indent=2),encoding='utf-8')
    print('YR_PROBE_EVAL',json.dumps(dict(idx=row['idx'],mc={k:v['prediction'] for k,v in r['mc'].items()},generated={k:v['text'] for k,v in r['generated'].items()})),flush=True)
RESULT=dict(server=YR_SERVER,rows=yr_rows,counts={k:sum(r['mc'][k]['correct'] for r in yr_rows) for k in yr_rows[0]['mc']},
    hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
assert RESULT['hooks']==0
(YR_DIR/'probe_eval.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print('YR_PROBE_EVAL_DONE',json.dumps(RESULT['counts']),flush=True)

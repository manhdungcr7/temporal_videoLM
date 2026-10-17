"""Question-only states for a query representation diagnostic; no answer/options/media."""
import gc,json,torch
YR_DIR=OUT/'yann_temporal_research';folder=YR_DIR/'textquery';folder.mkdir(exist_ok=True)
cases=[]
for sub in ('binding48_v2','binding_evalextra','perception48','microfeatures'):
    for p in sorted((YR_DIR/sub).glob('*.pt')):
        if p.stem.endswith(('_other','_decoder')):continue
        c=torch.load(p,map_location='cpu',weights_only=True)
        if 'question' in c:cases.append(c)
questions=sorted(set(c['question'] for c in cases));dev=MODEL_.get_input_embeddings().weight.device;features={}
for question in questions:
    messages=[{'role':'user','content':question}]
    txt=PROC.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=PROC.tokenizer(txt,add_special_tokens=False,return_tensors='pt')['input_ids'].to(dev)
    decoded=PROC.tokenizer.decode(ids[0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    assert decoded==txt
    start=txt.index(question);end=start+len(question);offset=0;rows=[]
    for i,x in enumerate(ids[0].tolist()):
        piece=PROC.tokenizer.decode([x],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        # Unicode byte-split tokens can invalidate per-token decoded lengths;
        # these annotation-derived questions are ASCII, enforce rather than guess.
        assert piece.isascii(),(question,piece)
        if offset>=start and offset+len(piece)<=end:rows.append(i)
        offset+=len(piece)
    assert offset==len(txt) and rows
    cap={}
    def capture(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;cap['tokens']=h[0,torch.tensor(rows,device=h.device)].detach().cpu()
        cap['last']=h[0,-1].detach().cpu()
    hook=MODEL_.model.layers[18].register_forward_hook(capture)
    try:
        with torch.inference_mode():MODEL_.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:hook.remove()
    features[question]=dict(mean=cap['tokens'].float().mean(0).half(),last=cap['last'],tokens=cap['tokens'])
torch.save(features,folder/'question_states.pt')
RESULT=dict(server=YR_SERVER,questions=len(features),hook_index=18,inputs='question-only text, no options, no answer and no media')
(YR_DIR/'textquery.json').write_text(json.dumps(RESULT,indent=2));print(json.dumps(RESULT),flush=True)
gc.collect();torch.cuda.empty_cache()

"""R2: question-conditioned media states without exposing answer choices."""
import ast,gc,json,time,torch
YR_DIR=OUT/'yann_temporal_research';YR_DIR.mkdir(exist_ok=True)
_yr_tree=ast.parse((OUT/'uploads/yann_research_flow.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in _yr_tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'yr_flow_definitions','exec'),globals())
yr_manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
yr_labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC'];yr_rows=[]
yr_specs=[('native',None,(),None),('question_feedback_mid','stem',tuple(range(14,26)),'av'),
    ('question_feedback_early','stem',tuple(range(0,12)),'av'),('format_feedback_mid','format',tuple(range(14,26)),'av'),
    ('question_feedback_visual_mid','stem',tuple(range(14,26)),'v'),('question_feedback_audio_mid','stem',tuple(range(14,26)),'a')]
for sample in yr_manifest[f'server{YR_SERVER}'][:4]:
    p=torch.load(OUT/'argon_latent40/decoder_cache'/f"{sample['idx']}_rot0.pt",map_location='cpu',weights_only=True)
    stem,_=parse_question(sample['question']);lm=int(p['last_media']);ids=p['input_ids'][0,lm+1:].tolist()
    tail=PROC.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    enc=PROC.tokenizer(tail,add_special_tokens=False,return_offsets_mapping=True)
    assert enc['input_ids']==ids,'Token offsets do not roundtrip'
    qstart=tail.find(stem);assert qstart>=0;qe=qstart+len(stem)
    stems=[lm+1+i for i,(a,b) in enumerate(enc['offset_mapping']) if a<qe and b>qstart and b<=qe]
    ie=tail.rfind('<|im_end|>');assert ie>qe
    formats=[lm+1+i for i,(a,b) in enumerate(enc['offset_mapping']) if a>=ie]
    assert stems and formats and not set(stems)&set(formats)
    r=dict(idx=sample['idx'],task=sample['task'],gold=sample['gold'],stem=stem,
        exposed_stem_text=PROC.tokenizer.decode(p['input_ids'][0,stems],skip_special_tokens=False),conditions={})
    base=None
    for name,mode,layers,modality in yr_specs:
        started=time.time();mask=None;edgecount=0
        if mode:
            mask,_=yr_flow_mask(p,'native');ix=torch.where(p['is_v'] if modality=='v' else p['is_a'] if modality=='a' else p['is_v']|p['is_a'])[0]
            keys=torch.tensor(stems if mode=='stem' else formats,dtype=torch.long)
            mask[0,0,ix[:,None],keys[None,:]]=0;edgecount=len(ix)*len(keys)
            allowed=torch.zeros(p['tokens'],dtype=torch.bool);allowed[keys]=True
            assert not (mask[0,0,:lm+1,lm+1:][:,~allowed[lm+1:]]==0).any(),'Unrequested tail exposed'
        lg,_=yr_flow_forward(p,mask,layers);sl=lg[yr_labels[:len(p['native_logits'])]];prob=sl.softmax(0)
        if base is None:base=sl;assert float((sl-p['native_logits']).abs().max())<1e-5
        pred='ABC'[int(sl.argmax())]
        r['conditions'][name]=dict(prediction=pred,correct=pred==sample['gold'],logits=sl.tolist(),probabilities=prob.tolist(),
            seconds=time.time()-started,tv=float(.5*(prob-base.softmax(0)).abs().sum()),opened_edges=edgecount)
        del lg,mask;gc.collect();torch.cuda.empty_cache()
    yr_rows.append(r);(YR_DIR/'questionflow_partial.json').write_text(json.dumps(yr_rows,indent=2),encoding='utf-8');print('YR_QFLOW',json.dumps(r),flush=True)
RESULT=dict(server=YR_SERVER,rows=yr_rows,counts={k:sum(r['conditions'][k]['correct'] for r in yr_rows) for k,*_ in yr_specs},
    hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in MODEL_.modules()))
assert RESULT['hooks']==0
(YR_DIR/'questionflow.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print('YR_QFLOW_DONE',json.dumps(RESULT['counts']),flush=True)

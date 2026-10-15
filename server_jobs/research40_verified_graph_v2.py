"""Repair fragmented utterances, mention/occurrence confusion, and missing null support."""
import json,time,gc,math
import numpy as np
import torch
import av  # Restore the decoder module after older jobs used `av` for score arrays.

R40_V2_FILE=OUT/f'research40_verified_graph_v2_s{R40_SERVER}.jsonl'
R40_V2_DIR=OUT/'research40_verified_v2';R40_V2_DIR.mkdir(exist_ok=True)

def r40_semantic_speech_nodes(row):
    cache=R40_V2_DIR/f"{row['idx']}_speech.json"
    if cache.exists():return json.loads(cache.read_text())
    wave=decode_audio(fetch(row['video_path']))
    spans=get_speech_timestamps(torch.from_numpy(wave),R40_VAD,sampling_rate=16000,
        min_speech_duration_ms=150,min_silence_duration_ms=200,speech_pad_ms=80)
    groups=[]
    for span in spans:
        lo,hi=span['start'],span['end']
        if groups and (lo-groups[-1]['end'])/16000<=1.2 and (hi-groups[-1]['start'])/16000<=25:
            groups[-1]['end']=hi;groups[-1]['regions'].append((lo/16000,hi/16000))
        else:groups.append(dict(start=lo,end=hi,regions=[(lo/16000,hi/16000)]))
    nodes=[]
    for g in groups:
        lo,hi=g['start'],g['end'];a,b=lo/16000,hi/16000
        res=R40_ASR({'raw':wave[lo:hi],'sampling_rate':16000},return_timestamps='word',
            generate_kwargs={'task':'transcribe'})
        words=[]
        for w in res.get('chunks',[]):
            x,y=w.get('timestamp',(0,b-a));x=0 if x is None else x;y=b-a if y is None else y
            words.append(dict(start=max(a,min(b,a+float(x))),end=max(a,min(b,a+float(y))),text=w['text']))
        nodes.append(dict(kind='audio speech episode',start=a,end=b,text=res['text'].strip(),words=words,
                          speech_regions=g['regions'],speaker='unknown',
                          note='Episode envelope includes short pauses; fine speech regions and word times retained.'))
    cache.write_text(json.dumps(nodes,indent=2),encoding='utf-8')
    return nodes

def r40_mentioned_event(text):
    prompt=('Summarize the event referred to by these spoken words as a short search phrase. '
            'An event reference is valid even if it is planned, hypothetical, or happened at a different time. '
            'Extract the main mentioned action and its objects; do not decide whether it actually happens. '
            'Do not add details absent from the words. If the speech describes no event, copy its main subject. '
            'Return only the search phrase, without explanation.\nSpoken words: '+text)
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[{'type':'text','text':prompt}]}]
    text_prompt=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inp=PROC(text=text_prompt,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    length=inp['input_ids'].shape[1]
    handle=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():
            seq=MODEL_.generate(**inp,max_new_tokens=32,do_sample=False,use_cache=True,
                eos_token_id=[PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')])
    finally:handle.remove()
    return PROC.tokenizer.decode(seq[0,length:],skip_special_tokens=True).strip().split('\n')[0]

def r40_v2_links(row,nodes):
    cache=R40_V2_DIR/f"{row['idx']}_links.json"
    if cache.exists():return json.loads(cache.read_text())
    audio=[n for n in nodes if n['kind'].startswith('audio')];video=[n for n in nodes if n['kind']=='visual']
    media=prepare_video(fetch(row['video_path']),max_pixels=base_by_id[row['idx']]['pixels'])
    links=[]
    for i,a in enumerate(audio):
        event=r40_mentioned_event(a['text'])
        support=[]
        for j,v in enumerate(video):
            s=r40_visual_occurrence_score(media,v['start'],v['end'],event)
            support.append(dict(video_id=j,start=v['start'],end=v['end'],score=s))
        scores=np.array([x['score'] for x in support]);max_score=float(scores.max())
        # A null/no-detection alternative is always retained. Negative evidence
        # cannot become a positive detection solely by ranking within the clip.
        null_score=0.0
        all_scores=np.r_[scores,null_score]/.75;all_scores-=all_scores.max()
        p=np.exp(all_scores);p/=p.sum()
        for x,prob in zip(support,p[:-1]):x['relative_support']=float(prob)
        links.append(dict(audio_id=i,event=event,support=support,null_support=float(p[-1]),
            max_visual_log_odds=max_score,detected=max_score>0.0,
            query_or_options_visible=False,perception_source='silent video only'))
    cache.write_text(json.dumps(links,indent=2),encoding='utf-8')
    del media;gc.collect();torch.cuda.empty_cache()
    return links

def r40_v2_context(nodes,links):
    text=r40_verified_graph_text(nodes,links)
    add=['Detection uncertainty: a segment ranked highest is not necessarily evidence that an event is visible.']
    for link in links:
        add.append(f"A{link['audio_id']} event search: null/no-detection support={link['null_support']:.3f}; "
                   f"strongest visual Yes-minus-No score={link['max_visual_log_odds']:+.2f}; "
                   f"positive visible evidence detected={link['detected']}.")
        if not link['detected']:add.append('Do not infer a visual occurrence time from this negative-only search.')
    for i,a in enumerate(n for n in nodes if n['kind'].startswith('audio')):
        add.append(f"A{i} fine speech regions (the episode includes pauses): {a.get('speech_regions',[])}")
    return text+'\n'.join(add)+'\n\n'

def r40_goal_answer(state,question,texts,perm):
    prompt=('<|im_end|>\n<|im_start|>user\nOriginal user question: '+question+
            '\nAnswer exactly this question using the original recording and evidence above. '
            'Keep the subjects and reference points in the question; do not answer an inverse or different comparison. '
            'Answer choices: '+' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))+
            '\nGive only the letter of the best answer.<|im_end|>\n<|im_start|>assistant\n')
    ids=PROC.tokenizer.encode(prompt,add_special_tokens=False)
    cache=DynamicCache(ddp_cache_data=state['bank'],config=MODEL_.model.config)
    lg,_,_,_=r40_cached_text(ids,cache,state['next_pos'],state['offset'])
    return lg[R40_LABELS[:len(texts)]].cpu().numpy()

faithful=r40_json_rows(OUT/f'research40_faithful_memory_s{R40_SERVER}.jsonl')
assert len(faithful)==len(R40_SAMPLES)
faith_by_id={r['idx']:r for r in faithful}
baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
records=r40_json_rows(R40_V2_FILE);done={r['idx'] for r in records}
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time();stem,texts=parse_question(r['question']);k=len(texts)
    video=[n for n in faith_by_id[r['idx']]['observations'] if n['kind']=='visual']
    nodes=sorted(video+r40_semantic_speech_nodes(r),key=lambda n:(n['start'],n['kind']))
    links=r40_v2_links(r,nodes);supplement=r40_v2_context(nodes,links)
    payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
    raw=np.zeros(k);textonly=np.zeros(k);goal=np.zeros(k)
    state=r40_memory_analysis(payload,stem,supplement)
    for perm in rotations(k):
        choices=' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))
        lg,_,_,_=r40_memory_context(payload,stem,supplement,choices)
        x=lg[R40_LABELS[:k]].cpu().numpy();lp=x-np.logaddexp.reduce(x)
        x2=r40_text_logits(supplement+r40_prompt(stem,texts,perm))[:k];lp2=x2-np.logaddexp.reduce(x2)
        x3=r40_goal_answer(state,stem,texts,perm);lp3=x3-np.logaddexp.reduce(x3)
        for slot,i in enumerate(perm):raw[i]+=lp[slot];textonly[i]+=lp2[slot];goal[i]+=lp3[slot]
    confidence=max([float(1/(1+np.exp(-np.clip(x['max_visual_log_odds'],-30,30))))*(1-x['null_support'])
                    if x['detected'] else 0.0 for x in links],default=0.0)
    original=np.array(base_by_id[r['idx']]['scores']['base'])
    rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',manifest_sha256=R40_MANIFEST['manifest_sha256'],
        scores=dict(verified_v2_raw_av=raw.tolist(),verified_v2_text_reader=textonly.tolist(),
                    verified_v2_goal_reader=goal.tolist(),verified_v2_confidence_fusion=(confidence*goal+(1-confidence)*original).tolist()),
        observations=nodes,links=links,analysis=state['text'],confidence=confidence,seconds=round(time.time()-started,2))
    with R40_V2_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(r['idx'])
    merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
    r40_snapshot(merged,'verified_graph_v2')
    print('VERIFIED_V2_SAMPLE',json.dumps({k:rec[k] for k in ('idx','task','gold','scores','seconds','confidence','analysis')}),flush=True)
    del payload,state;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'verified_graph_v2_done')
print('VERIFIED_V2_SUMMARY',json.dumps(RESULT),flush=True)

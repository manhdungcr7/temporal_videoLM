"""Question-blind semantic grounding links on an interval/provenance graph.

Independent perception -> source-only visual occurrence probes -> general QA.
The graph supports queries rooted in either source. No task/answer template or
gold selects a speech event, temporal zone, visual window, or answer.
"""
import json,time,gc,math,re
from pathlib import Path
import numpy as np
import torch

R40_VERIFIED_FILE=OUT/f'research40_verified_graph_s{R40_SERVER}.jsonl'
R40_LINK_DIR=OUT/'research40_verified_links';R40_LINK_DIR.mkdir(exist_ok=True)
R40_YES=PROC.tokenizer.encode('Yes',add_special_tokens=False)
R40_NO=PROC.tokenizer.encode('No',add_special_tokens=False)
assert len(R40_YES)==len(R40_NO)==1

def r40_speech_event(text):
    prompt=('Identify the observable event or events mentioned by this utterance, for a recording search. '
            'Return one short, concrete event description. Use neutral tense; do not guess when it occurs. '
            'Do not add objects or actions that were not mentioned. If there is no observable event, return UNKNOWN. '
            'Do not answer any question about the recording.\nUtterance: '+text)
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[{'type':'text','text':prompt}]}]
    rendered=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inp=PROC(text=rendered,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    length=inp['input_ids'].shape[1]
    handle=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():
            seq=MODEL_.generate(**inp,max_new_tokens=36,do_sample=False,use_cache=True,
                eos_token_id=[PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')])
    finally:handle.remove()
    event=PROC.tokenizer.decode(seq[0,length:],skip_special_tokens=True).strip().split('\nHuman:')[0]
    return event

def r40_visual_occurrence_score(media,start,end,event):
    frames=media['videos'][0]
    lo=int(math.floor(start*media['fps']));hi=min(len(frames),int(math.ceil(end*media['fps'])))
    clip=frames[lo:hi]
    if len(clip)==0:clip=frames[-1:]
    if len(clip)%2:clip=torch.cat((clip,clip[-1:]),dim=0)
    prompt=('Does this silent video segment show the following event actually taking place? '
            +event+'\nJudge only visible evidence, not what might be said or intended. Answer with only Yes or No.')
    content=dict(type='video',video=media['path'],max_pixels=media['max_pixels_per_frame'])
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[content,{'type':'text','text':prompt}]}]
    rendered=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inputs=PROC(text=rendered,videos=[clip],images=None,audio=None,return_tensors='pt',padding=True,
                videos_kwargs={'fps':media['fps'],'use_audio_in_video':False})
    inputs=move_inputs(inputs,MODEL_)
    handle=MODEL_.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():lg=MODEL_(**inputs,use_audio_in_video=False,use_cache=False).logits[0,-1].float()
    finally:handle.remove()
    value=float(lg[R40_YES[0]]-lg[R40_NO[0]])
    del inputs;torch.cuda.empty_cache()
    return value

def r40_ground_links(row,nodes):
    path=R40_LINK_DIR/f"{row['idx']}_4s.json"
    if path.exists():return json.loads(path.read_text())
    audio=[n for n in nodes if n['kind'].startswith('audio')]
    video=[n for n in nodes if n['kind']=='visual']
    media=prepare_video(fetch(row['video_path']),max_pixels=base_by_id[row['idx']]['pixels'])
    links=[]
    for i,a in enumerate(audio):
        event=r40_speech_event(a['text'])
        if not event or event.upper().startswith('UNKNOWN'):
            links.append(dict(audio_id=i,event=event,observability='unknown',support=[]));continue
        support=[]
        for j,v in enumerate(video):
            score=r40_visual_occurrence_score(media,v['start'],v['end'],event)
            support.append(dict(video_id=j,start=v['start'],end=v['end'],score=score))
        scores=np.array([x['score'] for x in support]);scores-=scores.max()
        p=np.exp(scores/.75);p/=p.sum()
        for x,prob in zip(support,p):x['relative_support']=float(prob)
        links.append(dict(audio_id=i,event=event,observability='search_probe_not_ground_truth',support=support,
                          entropy=float(-(p*np.log(p+1e-12)).sum()),
                          query_or_options_visible=False,perception_source='silent video only'))
    path.write_text(json.dumps(links,indent=2),encoding='utf-8')
    del media;gc.collect();torch.cuda.empty_cache()
    return links

def r40_verified_graph_text(nodes,links):
    base=r40_interval_graph_text(nodes)
    audio=[n for n in nodes if n['kind'].startswith('audio')]
    extra=['Source-only semantic search probes. All utterance references were searched without the user question '
           'or answer choices. Relative support ranks video segments; it is not a calibrated truth probability. '
           'The links may be read in either direction and do not impose an audio-first reasoning order.']
    for link in links:
        i=link['audio_id'];extra.append(f"A{i} mentions event: {link['event']}")
        if not link['support']:continue
        strongest=sorted(link['support'],key=lambda x:x['relative_support'],reverse=True)
        for match in strongest:
            j=match['video_id']
            extra.append(f"A{i} event <-> V{j}: silent visual occurrence score={match['score']:+.2f}; relative support={match['relative_support']:.3f}.")
        best=strongest[0];a=audio[i];j=best['video_id']
        extra.append(f"Strongest silent-video search evidence for A{i}'s event is V{j} [{best['start']:.2f},{best['end']:.2f}]s; "
                     f"A{i}'s actual speech was heard [{a['start']:.2f},{a['end']:.2f}]s. "
                     'This identifies evidence to check; it does not by itself determine the answer.')
    return base+'\n'.join(extra)+'\n\n'

faithful=r40_json_rows(OUT/f'research40_faithful_memory_s{R40_SERVER}.jsonl')
assert len(faithful)==len(R40_SAMPLES)
faith_by_id={r['idx']:r for r in faithful}
baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
records=r40_json_rows(R40_VERIFIED_FILE);done={r['idx'] for r in records}
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time();stem,texts=parse_question(r['question']);k=len(texts)
    nodes=faith_by_id[r['idx']]['observations'];links=r40_ground_links(r,nodes)
    supplement=r40_verified_graph_text(nodes,links)
    payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
    raw=np.zeros(k);textonly=np.zeros(k);cot=np.zeros(k)
    state=r40_memory_analysis(payload,stem,supplement)
    for perm in rotations(k):
        choices=' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))
        lg,_,_,_=r40_memory_context(payload,stem,supplement,choices)
        v=lg[R40_LABELS[:k]].cpu().numpy();lp=v-np.logaddexp.reduce(v)
        v2=r40_text_logits(supplement+r40_prompt(stem,texts,perm))[:k];lp2=v2-np.logaddexp.reduce(v2)
        v3=r40_reason_answer(state,texts,perm);lp3=v3-np.logaddexp.reduce(v3)
        for slot,i in enumerate(perm):raw[i]+=lp[slot];textonly[i]+=lp2[slot];cot[i]+=lp3[slot]
    rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',manifest_sha256=R40_MANIFEST['manifest_sha256'],
        scores=dict(verified_graph_raw_av=raw.tolist(),verified_graph_text_reader=textonly.tolist(),
                    verified_graph_reasoning=cot.tolist()),links=links,analysis=state['text'],
        seconds=round(time.time()-started,2))
    with R40_VERIFIED_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(r['idx'])
    merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
    r40_snapshot(merged,'verified_graph')
    print('VERIFIED_GRAPH_SAMPLE',json.dumps({k:rec[k] for k in ('idx','task','gold','scores','seconds','analysis')}),flush=True)
    del payload,state;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'verified_graph_done')
print('VERIFIED_GRAPH_SUMMARY',json.dumps(RESULT),flush=True)

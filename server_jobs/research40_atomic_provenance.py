"""Atomic source/time provenance: separate a mention's word clock from occurrence.

Question/option-blind extraction and source-only perception. The general reader
may traverse either source and retain non-event utterances. No task-label router
or temporal-relation-to-answer mapping. All settings shared across the fixed40.
"""
import json,time,gc,re,hashlib,math
import numpy as np
import torch
import av
from transformers import AutoModelForCausalLM

R40_ATOMIC_FILE=OUT/f'research40_atomic_provenance_s{R40_SERVER}.jsonl'
R40_ATOMIC_DIR=OUT/'research40_atomic_provenance';R40_ATOMIC_DIR.mkdir(exist_ok=True)

# The preceding reader4 job releases its weights after completion. Its tokenizer,
# pinned revision, and functions remain available; load the same cached revision.
if 'R40_READER_MODEL' not in globals():
    gc.collect();torch.cuda.empty_cache()
    budget={d:min(5*1024**3,int(torch.cuda.mem_get_info(d)[0])-2*1024**3)
            for d in range(torch.cuda.device_count())}
    assert sum(budget.values())>9*1024**3,budget
    R40_READER_MODEL=AutoModelForCausalLM.from_pretrained(R40_READER_ID,revision=R40_READER_REV,
        torch_dtype=torch.float16,device_map='auto',max_memory=budget,attn_implementation='sdpa').eval()
    assert all(v not in ('cpu','disk') for v in R40_READER_MODEL.hf_device_map.values())

def r40_atomic_generate(messages,limit=256):
    inputs=r40_reader_inputs(messages,False);length=inputs['input_ids'].shape[1]
    handle=R40_READER_MODEL.lm_head.register_forward_pre_hook(lambda _m,args:(args[0][:,-1:,:],))
    try:
        with torch.inference_mode():seq=R40_READER_MODEL.generate(**inputs,max_new_tokens=limit,
            do_sample=False,use_cache=True)
    finally:handle.remove()
    return R40_READER_TOKENIZER.decode(seq[0,length:],skip_special_tokens=True).strip()

def r40_exact_word_span(quote,words):
    chars=[];owners=[]
    for i,w in enumerate(words):
        for c in w['text']:
            if c.isalnum():chars.append(c.lower());owners.append(i)
    needle=''.join(c.lower() for c in quote if c.isalnum())
    if not needle:return None
    start=''.join(chars).find(needle)
    if start<0:return None
    end=start+len(needle)-1;first,last=owners[start],owners[end]
    return dict(start=float(words[first]['start']),end=float(words[last]['end']),
                first_word=first,last_word=last,quote=''.join(w['text'] for w in words[first:last+1]).strip())

def r40_atomic_mentions(row,observations):
    file=R40_ATOMIC_DIR/f"{row['idx']}_mentions.json"
    if file.exists():return json.loads(file.read_text())
    episodes=[n for n in observations if n['kind'].startswith('audio')]
    heard='\n'.join(f"Episode {i}: {n['text']}" for i,n in enumerate(episodes))
    prompt=('Extract up to six distinct event/action references from these spoken words. '
        'A reference can be planned, hypothetical, or about another time. Do not infer that it happened. '
        'For each reference give its episode number, an exact contiguous quotation of the action and '
        'objects from that episode, and a short neutral event search phrase. Separate distinct actions '
        'rather than combining a whole sequence into one event. Do not invent quoted words. '
        'Return JSON only: {"mentions":[{"episode":0,"quote":"exact words","event":"neutral phrase"}]}. '
        'No question about the recording is being asked.\n'+heard)
    output=r40_atomic_generate([dict(role='system',content='Extract references faithfully without judging occurrence.'),
                                dict(role='user',content=prompt)],384)
    parsed=[]
    try:
        data=json.loads(output[output.index('{'):output.rindex('}')+1])
        for item in data.get('mentions',[])[:6]:
            i=int(item['episode']);event=str(item['event']).strip()
            if not (0<=i<len(episodes)) or not event:continue
            span=r40_exact_word_span(str(item['quote']),episodes[i].get('words',[]))
            if span is None:continue
            parsed.append(dict(span,episode=i,event=event,kind='spoken event reference',
                occurrence_asserted=False,query_or_options_visible=False))
    except (ValueError,TypeError,KeyError):pass
    # Never fabricate a word clock if extraction altered the quotation. The
    # original transcript/word bins remain accessible to the downstream reader.
    result=dict(mentions=parsed,extraction_output=output,episodes=len(episodes),
                valid=len(parsed),invalid_quotes_are_not_time_anchored=True)
    file.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def r40_atomic_search(row,mentions):
    file=R40_ATOMIC_DIR/f"{row['idx']}_search.json"
    if file.exists():links=json.loads(file.read_text())
    else:links=[]
    completed={x['mention_id'] for x in links}
    media=prepare_video(fetch(row['video_path']),max_pixels=base_by_id[row['idx']]['pixels'])
    duration=row['decoded_audio_duration']
    for i,mention in enumerate(mentions):
        if i in completed:continue
        probes=[]
        for start in np.arange(0,duration,2.):
            start=float(start);end=min(duration,start+2.)
            if end-start<.05:continue
            score=r40_visual_occurrence_score(media,start,end,mention['event'])
            # Physical relations are evidence features, not answers to QA tasks.
            if end<=mention['start']:relation='visual observation before spoken reference'
            elif start>=mention['end']:relation='visual observation after spoken reference'
            else:relation='visual observation overlaps spoken reference'
            probes.append(dict(start=start,end=end,score=score,relation=relation))
        scores=np.r_[[p['score'] for p in probes],0.]/.75;scores-=scores.max()
        p=np.exp(scores);p/=p.sum()
        for probe,weight in zip(probes,p[:-1]):probe['relative_support']=float(weight)
        links.append(dict(mention_id=i,probes=probes,null_support=float(p[-1]),
                          detected=max(p['score'] for p in probes)>0.,
                          source='silent video only',question_or_options_visible=False))
        file.write_text(json.dumps(links,indent=2),encoding='utf-8')
        print('ATOMIC_SEARCH',row['idx'],i,'/',len(mentions),flush=True)
    del media;gc.collect();torch.cuda.empty_cache()
    return links

def r40_atomic_context(observations,mentions,links):
    # Keep the original quote/visual observations; no source evidence is dropped
    # merely because it has no extractable action or no positive visual match.
    text=r40_compact_evidence(observations,[])
    lines=['Atomic source/time bindings (reference time is NOT occurrence time).',
           'Use the actual quoted phrase clock, not a long episode envelope, when referring to that phrase.',
           'A silent-video probe is uncertain perceptual evidence; a negative-only ranking is not an occurrence.']
    for i,m in enumerate(mentions):
        lines.append(f"M{i} words heard [{m['start']:.2f},{m['end']:.2f}]s: \"{m['quote']}\"; "
                     f"refers to event: {m['event']}. This alone asserts no visible occurrence.")
    for link in links:
        i=link['mention_id'];lines.append(f"M{i} silent-video search: detected={link['detected']}; "
                                        f"null support={link['null_support']:.3f}.")
        for probe in link['probes']:
            lines.append(f"M{i} event evidence at visual [{probe['start']:.2f},{probe['end']:.2f}]s: "
                         f"Yes-minus-No={probe['score']:+.2f}, support={probe['relative_support']:.3f}; "
                         f"{probe['relation']}.")
    return text+'\n'.join(lines)+'\n\n'

prefill_helper=OUT/'uploads/research40_reader4_chunked.py'
assert prefill_helper.exists(),'Upload bounded-memory reader helper before resuming.'
exec(compile(prefill_helper.read_text(encoding='utf-8'),'reader4_chunked_runtime','exec'),globals())

evidence_by_id={r['idx']:r for r in r40_json_rows(OUT/f'research40_verified_graph_v2_s{R40_SERVER}.jsonl')}
base_by_id={r['idx']:r for r in r40_json_rows(R40_BASE_FILE)}
records=r40_json_rows(R40_ATOMIC_FILE);done={r['idx'] for r in records}
for row in R40_SAMPLES:
    if row['idx'] in done:continue
    started=time.time();question,texts=parse_question(row['question']);k=len(texts)
    observations=evidence_by_id[row['idx']]['observations']
    extraction=r40_atomic_mentions(row,observations);mentions=extraction['mentions']
    links=r40_atomic_search(row,mentions);context=r40_atomic_context(observations,mentions,links)
    payload=torch.load(R40_CACHE/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=row['idx']
    raw=np.zeros(k);reader=np.zeros(k);reason=np.zeros(k)
    # Same general reader instructions, any source/order; no question-type routing.
    analysis=r40_atomic_generate([dict(role='system',content=R40_READER_SYSTEM),dict(role='user',content=
        context+'Original user question: '+question+'\nExplain the relevant evidence in at most100 words. '
        'Preserve the question referents and the actual time bounds. Do not select a final answer yet.')],160)
    for perm in rotations(k):
        choices=' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))
        lg,_,_,_=r40_memory_context(payload,question,context,choices)
        x=lg[R40_LABELS[:k]].cpu().numpy();lp=x-np.logaddexp.reduce(x)
        messages=[dict(role='system',content=R40_READER_SYSTEM),
                  dict(role='user',content=context+r40_prompt(question,texts,perm))]
        x2=r40_reader_scores(messages)[:k];lp2=x2-np.logaddexp.reduce(x2)
        x3=r40_reader_scores([dict(role='system',content=R40_READER_SYSTEM),
            dict(role='user',content=context+'Original user question: '+question),
            dict(role='assistant',content=analysis),dict(role='user',content=r40_prompt(question,texts,perm))])[:k]
        lp3=x3-np.logaddexp.reduce(x3)
        for slot,i in enumerate(perm):raw[i]+=lp[slot];reader[i]+=lp2[slot];reason[i]+=lp3[slot]
    rec=dict(idx=row['idx'],task=row['task'],gold=row['gold'],status='ok',
        manifest_sha256=R40_MANIFEST['manifest_sha256'],scores=dict(atomic_raw_av=raw.tolist(),
        atomic_reader4=reader.tolist(),atomic_reader4_reason=reason.tolist()),extraction=extraction,
        links=links,analysis=analysis,seconds=round(time.time()-started,2))
    with R40_ATOMIC_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(row['idx'])
    merged=[dict(r,scores=dict(base_by_id[r['idx']]['scores'],**r['scores'])) for r in records]
    r40_snapshot(merged,'atomic_provenance');print('ATOMIC_SAMPLE',row['idx'],json.dumps(rec['scores']),flush=True)
    del payload;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'atomic_provenance_done');print('ATOMIC_SUMMARY',json.dumps(RESULT),flush=True)
del R40_READER_MODEL;gc.collect();torch.cuda.empty_cache()

"""Refine atomic provenance: exact lexical spans, coreference, PTS crop, sparse edges.

New artifact/version; never silently replace the preceding experiment. Uses the
same source-only visual probe, with exact frame membership rather than fps slices.
Includes a bidirectionally readable source/time co-occurrence index for arbitrary
questions, not a prescribed audio-first QA path.
"""
import json,re,gc,copy,hashlib
import numpy as np
import torch
import av
import decord

source=(OUT/'uploads/research40_atomic_provenance.py').read_text(encoding='utf-8')
split_at=source.index('\nevidence_by_id=')
exec(compile(source[:split_at],'atomic_v1_definitions_only','exec'),globals())
R40_ATOMIC_FILE=OUT/f'research40_atomic_provenance_v2_s{R40_SERVER}.jsonl'
R40_ATOMIC_DIR=OUT/'research40_atomic_provenance_v2';R40_ATOMIC_DIR.mkdir(exist_ok=True)

def r40_exact_word_spans(quote,words):
    tokens=[];owners=[]
    for i,w in enumerate(words):
        for token in re.findall(r'\w+',w['text'].casefold(),flags=re.UNICODE):
            tokens.append(token);owners.append(i)
    needle=re.findall(r'\w+',quote.casefold(),flags=re.UNICODE)
    if not needle:return []
    found=[]
    for start in range(len(tokens)-len(needle)+1):
        if tokens[start:start+len(needle)]!=needle:continue
        first,last=owners[start],owners[start+len(needle)-1]
        found.append(dict(start=float(words[first]['start']),end=float(words[last]['end']),
            first_word=first,last_word=last,quote=''.join(w['text'] for w in words[first:last+1]).strip()))
    return found

def r40_atomic_mentions(row,observations):
    file=R40_ATOMIC_DIR/f"{row['idx']}_mentions.json"
    if file.exists():return json.loads(file.read_text())
    episodes=[n for n in observations if n['kind'].startswith('audio')]
    heard='\n'.join(f"Episode {i}: {n['text']}" for i,n in enumerate(episodes))
    prompt=('Extract separate predicate-argument event references from the transcript, up to six. '
        'Each item must describe ONE action, not a whole sentence or sequence. Its quote must be the '
        'minimal exact contiguous spoken phrase naming that action. The event description may resolve '
        'pronouns from the transcript, but may not guess an unmentioned entity. Distinguish the affected '
        'object from a destination, ingredient or tool; do not replace the affected object when it is '
        'moved to a new destination. Use UNKNOWN for genuinely ambiguous pronouns. Planned or negated '
        'events are still references, not evidence that they occurred. Keep their reported status.\n'
        'Example transcript: Maya will open the door, then carry a bag upstairs.\n'
        'Separate quotes: "open the door" and "carry a bag upstairs"; do not quote the whole sequence.\n'
        'Example transcript: John moves the toolbox to the shelf, then puts it on a cart.\n'
        'The second affected object is the toolbox, NOT the shelf. Quote: "puts it on a cart".\n'
        'Return JSON only: {"mentions":[{"episode":0,"quote":"exact action words",'
        '"actor":"who","affected":"what moves/changes","destination_or_tool":"where/with what",'
        '"event":"one neutral action with resolved arguments","reported_status":"as expressed"}]}.\n'+heard)
    output=r40_atomic_generate([dict(role='system',content='Extract faithfully. No question or answer about the recording is supplied.'),
                               dict(role='user',content=prompt)],512)
    with av.open(str(fetch(row['video_path']))) as container:
        first=next(container.decode(audio=0));origin=0. if first.pts is None else float(first.pts*first.time_base)
    mentions=[];rejected=0
    try:
        data=json.loads(output[output.index('{'):output.rindex('}')+1])
        for item in data.get('mentions',[])[:6]:
            i=int(item['episode']);event=str(item['event']).strip()
            if not 0<=i<len(episodes) or not event:rejected+=1;continue
            spans=r40_exact_word_spans(str(item['quote']),episodes[i].get('words',[]))
            if not spans:rejected+=1;continue
            # Preserve EVERY exact occurrence; never silently choose the first
            # repeated phrase. One semantic probe can be reused across clocks.
            for occurrence,span in enumerate(spans):
                mentions.append(dict(span,start=span['start']+origin,end=span['end']+origin,
                    episode=i,event=event,occurrence_index=occurrence,quote_occurrences=len(spans),
                    actor=item.get('actor'),affected=item.get('affected'),
                    destination_or_tool=item.get('destination_or_tool'),reported_status=item.get('reported_status'),
                    kind='spoken event reference',occurrence_asserted=False,query_or_options_visible=False))
    except (ValueError,KeyError,TypeError):rejected+=1
    result=dict(mentions=mentions,extraction_output=output,valid=len(mentions),rejected=rejected,
        first_decoded_audio_pts=origin,clock='original clip PTS, not inferred event time',
        invalid_quotes_are_not_time_anchored=True)
    file.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def r40_atomic_search(row,mentions):
    file=R40_ATOMIC_DIR/f"{row['idx']}_search.json"
    if file.exists():return json.loads(file.read_text())
    path=fetch(row['video_path']);media=prepare_video(path,max_pixels=base_by_id[row['idx']]['pixels'])
    frames=media['videos'][0];vr=decord.VideoReader(str(path))
    native_indices=torch.linspace(0,len(vr)-1,len(frames)).round().long().tolist()
    pts=np.asarray(vr.get_frame_timestamp(native_indices),dtype=float)[:,0]
    assert len(pts)==len(frames)
    links=[]
    for i,mention in enumerate(mentions):
        key=hashlib.sha256(mention['event'].encode()).hexdigest()[:20]
        probe_file=R40_ATOMIC_DIR/f"{row['idx']}_{key}_pts_probes.json"
        if probe_file.exists():stored=json.loads(probe_file.read_text())
        else:
            stored=[]
            for start in np.arange(0,row['decoded_audio_duration'],2.):
                start=float(start);end=min(row['decoded_audio_duration'],start+2.)
                members=np.flatnonzero((pts>=start)&(pts<end))
                if len(members)==0:continue  # missing visual samples are not negative event evidence
                clip=frames[torch.tensor(members,dtype=torch.long)]
                local=dict(media,videos=[clip])
                # Delegate only with a view containing the exact selected frames.
                # The old helper's local crop covers that whole view, not an
                # approximate original-video [start*fps:end*fps] range.
                score=r40_visual_occurrence_score(local,0.,len(clip)/media['fps'],mention['event'])
                stored.append(dict(start=start,end=end,score=score,actual_frame_pts=pts[members].tolist(),
                                   source_indices=[native_indices[j] for j in members]))
            probe_file.write_text(json.dumps(stored,indent=2),encoding='utf-8')
        probes=copy.deepcopy(stored)
        for p in probes:
            if p['end']<=mention['start']:p['relation']='visual observation before spoken reference'
            elif p['start']>=mention['end']:p['relation']='visual observation after spoken reference'
            else:p['relation']='visual observation overlaps spoken reference'
        scores=np.r_[[p['score'] for p in probes],0.]/.75;scores-=scores.max()
        probabilities=np.exp(scores);probabilities/=probabilities.sum()
        for p,w in zip(probes,probabilities[:-1]):p['relative_support']=float(w)
        links.append(dict(mention_id=i,probes=probes,null_support=float(probabilities[-1]),
            detected=any(p['score']>0 for p in probes),source='silent video only; exact native source PTS crop',
            question_or_options_visible=False))
        print('ATOMIC_V2_SEARCH',row['idx'],i,'/',len(mentions),flush=True)
    file.write_text(json.dumps(links,indent=2),encoding='utf-8')
    del media,frames,vr;gc.collect();torch.cuda.empty_cache()
    return links

def r40_atomic_context(observations,mentions,links):
    # Physical joins are bidirectional: visual -> words and words -> visual.
    # They assert co-occurrence of observations, NOT semantic equivalence.
    text=r40_compact_evidence(observations,[])
    lines=['Source/time index; shared physical time is not shared event meaning.',
        'Video captions are approximate observation summaries, not exact onset/duration.',
        'Quoted event mentions have word clocks; whole speech episodes need not be the correct anchor.']
    audio=[n for n in observations if n['kind'].startswith('audio')]
    video=[n for n in observations if n['kind']=='visual']
    for j,v in enumerate(video):
        words=[w for a in audio for w in a.get('words',[]) if w['end']>v['start'] and w['start']<v['end']]
        if words:
            lines.append(f"V{j} observation window [{v['start']:.2f},{v['end']:.2f}]s contains these heard words: "
                         +''.join(w['text'] for w in words).strip()+
                         ' (not a claim that the visual action and spoken subject are the same event).')
    for i,m in enumerate(mentions):
        lines.append(f"M{i} quotation [{m['start']:.2f},{m['end']:.2f}]s: \"{m['quote']}\"; refers to "
            f"{m['event']}; actor={m.get('actor')}; affected={m.get('affected')}; "
            f"destination/tool={m.get('destination_or_tool')}; reported status={m.get('reported_status')}. "
            'Mentioning the event does not prove it was visible.')
        if m['quote_occurrences']>1:lines.append('This quotation is repeated; all its observed clocks are retained, not a unique referent.')
    for link in links:
        lines.append(f"M{link['mention_id']} visual search: positive evidence detected={link['detected']}; "
                     f"null support={link['null_support']:.3f}; scores are uncertain, not truth labels.")
        strongest=sorted(link['probes'],key=lambda p:p['score'],reverse=True)[:3]
        if not link['detected']:lines.append('No positive event occurrence was found; do not infer an occurrence from ranking alone.')
        for p in strongest:
            lines.append(f"Silent observation [{p['start']:.2f},{p['end']:.2f}]s for M{link['mention_id']}: "
                f"visible-event Yes-minus-No={p['score']:+.2f}, support={p['relative_support']:.3f}; {p['relation']}. "
                'This is evidence to check, not an exact event duration.')
    return text+'\n'.join(lines)+'\n\n'

tail=source[split_at:]
for old,new in [('atomic_raw_av','atomic_v2_raw_av'),('atomic_reader4_reason','atomic_v2_reader4_reason'),
                ('atomic_reader4','atomic_v2_reader4')]:tail=tail.replace(old,new)
exec(compile(tail,'atomic_v2_evaluation_same_fixed_ids','exec'),globals())

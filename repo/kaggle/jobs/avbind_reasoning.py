"""Symmetric address-bound AV evidence readback, experimental baseline.

No dataset/task/label templates, no answer-meaning expansion, no audio-first
reference resolver. One controller chooses arbitrary locate/read operations.
This is a research prototype, not a claim of novel or universal competence.
"""
import json, math, re, time, hashlib
from pathlib import Path
import numpy as np
import torch

AVB_POLICY=(
    'You can inspect a timestamped audiovisual recording using evidence tools. '
    'Choose the next observation needed to answer the user question. You may start '
    'with any source, switch sources, revisit a source, or read both. '
    'The order is determined by the question and evidence, not by a task category. '
    'A spoken description is a mention, not proof that the described action is visible. '
    'All times refer to the original recording. Do not guess an answer here. '
    'Return one JSON object only. Available operations:\n'
    '{"op":"locate","source":"video|audio|both","query":"concrete evidence to find"}\n'
    '{"op":"read","source":"video|audio|both","ids":["V0"],"query":"what to inspect"}\n'
    '{"op":"read","source":"video|audio|both","interval":[0,2],"query":"what to inspect"}\n'
    '{"op":"stop"}\n'
    'locate checks all relevant source segments and returns supported intervals. '
    'read can use an anchor from either source to inspect another source at the same time. '
    'Use stop only when the observations already suffice. Avoid repeating an operation.'
)
AVB_READOUT=(
    'Evidence below has immutable source IDs and original-recording intervals. '
    'Use the observations to answer the question, without substituting a heard '
    'description for a visible occurrence. Time links describe intervals, not '
    'semantic identity or causation. Preparing, doing, and a finished result may '
    'be different observations. Missing evidence is not proof of the opposite.\n'
)

def avb_next_action(question,nodes,results,linked):
    """Factorized constrained tool choices; no question-type classifier."""
    context=AVB_READOUT+avb_ledger(nodes,results,relations=linked)+'\nUSER QUESTION: '+question
    # Apply the same evidence-verification requirement to every question.
    # Draft summaries alone cannot certify that the answer is grounded.
    op_names=('locate','read') if not results else ('stop','locate','read')
    descriptions={'stop':'The raw checks suffice; stop inspecting',
                  'locate':'Locate a concrete observation over the recording',
                  'read':'Read a particular interval more closely'}
    op_options=[descriptions[x] for x in op_names]
    op_scores=text_choice_scores(context+'\nWhich evidence operation is needed next?',op_options)
    op=op_names[int(np.argmax(op_scores))]
    trace=dict(op_scores=op_scores)
    if op=='stop':return dict(op='stop'),trace
    source_scores=text_choice_scores(context+f'\nNext operation: {op}. Which input is needed for this next observation?',
                                    ['Only video frames','Only the audio track','Both sources independently'])
    source=('video','audio','both')[int(np.argmax(source_scores))]
    trace['source_scores']=source_scores
    request=clean_observation(text_generate(context+f'\nNext operation: {op}; source: {source}. '
        'Describe exactly the evidence to look for or inspect. Use an actual event, object, sound, '
        'utterance, or detail from the question/observations. Give one short concrete request only, '
        'not an answer, generic placeholder, or option letter.',64)).split('\n')[0]
    action=dict(op=op,source=source,query=request)
    if op=='read':
        raw=text_generate(context+f'\nReadback request: {request}; requested source: {source}. '
            'Choose the original-recording interval to inspect. Return only a JSON object with '
            'two numeric fields start and end, in seconds. Use intervals of the relevant observed '
            'events, not the first or last segment by default.',64)
        interval=avb_json(raw);action['interval']=[interval['start'],interval['end']]
        trace['interval_raw']=raw
    return action,trace

def avb_json(text):
    text=clean_observation(text)
    decoder=json.JSONDecoder()
    for match in re.finditer(r'\{',text):
        try:
            value,_=decoder.raw_decode(text[match.start():])
            if isinstance(value,dict):return value
        except ValueError:pass
    raise ValueError('Controller did not return a JSON object')

def avb_nodes(observations,utterances):
    nodes=[]
    for i,x in enumerate(observations):
        for prefix,source,field in [('V','video','video'),('A','audio','audio')]:
            nodes.append(dict(id=f'{prefix}{i}',source=source,interval=[x['start'],x['end']],
                              content=clean_observation(x[field]),kind='chunk'))
    for i,x in enumerate(utterances):
        nodes.append(dict(id=f'S{i}',source='audio',interval=[x['start'],x['end']],
                          content=x['words'],kind='speech'))
    return nodes

def avb_link(left,right):
    a,b=left['interval'],right['interval']
    overlap=max(0.0,min(a[1],b[1])-max(a[0],b[0]))
    relation='before' if a[1]<=b[0] else 'after' if a[0]>=b[1] else 'overlaps'
    return dict(left=left['id'],right=right['id'],relation=relation,overlap_sec=round(overlap,3))

def avb_ledger(nodes,tool_results,relations=False):
    lines=['SOURCE-INDEXED OBSERVATIONS; different sources are independent evidence:']
    for n in nodes:
        lines.append(f'{n["id"]} | {n["source"]} | {n["interval"]}s | {n["kind"]}: {n["content"]}')
    if tool_results:
        lines.append('\nRAW-MEDIA READBACK RESULTS:')
        lines.extend(json.dumps(x,ensure_ascii=False) for x in tool_results)
    if relations:
        # Generic interval arithmetic, independent of question/task/answers.
        loc=[x for t in tool_results for x in t.get('matches',[]) if x.get('supported')]
        speech=[x for x in nodes if x['kind']=='speech']
        anchors=loc+speech
        lines.append('\nINTERVAL LINKS (not statements that two events are identical):')
        for i,left in enumerate(anchors):
            for right in anchors[i+1:]:
                lines.append(json.dumps(avb_link(left,right)))
    return '\n'.join(lines)+'\n'

def avb_episodes(nodes,margins,source,prefix):
    # Keep every supported episode, not just a salience peak; no class offsets.
    matches=[];active=[]
    def emit(group):
        if not group:return
        matches.append(dict(id=f'{prefix}_{len(matches)}',source=source,supported=True,
                            interval=[group[0]['interval'][0],group[-1]['interval'][1]],
                            segment_ids=[x['id'] for x in group]))
    for node,value in zip(nodes,margins):
        contiguous=not active or node['interval'][0]<=active[-1]['interval'][1]+0.01
        if value>0:
            if not contiguous:emit(active);active=[]
            active.append(node)
        else:emit(active);active=[]
    emit(active)
    return matches

def avb_locate(src,nodes,source,query,number):
    results=[]
    for modality in ('video','audio') if source=='both' else (source,):
        candidates=[x for x in nodes if x['source']==modality and
                    x['kind']==('speech' if modality=='audio' and any(n['kind']=='speech' for n in nodes) else 'chunk')]
        # Include non-speech audio chunks too: no speech-only gate.
        if modality=='audio':
            candidates += [x for x in nodes if x['source']=='audio' and x['kind']=='chunk'
                           and not any(s['kind']=='speech' and s['interval'][0]<x['interval'][1]
                                       and s['interval'][1]>x['interval'][0] for s in nodes)]
            candidates=list({x['id']:x for x in candidates}.values())
        candidates=sorted(candidates,key=lambda x:(x['interval'][0],x['interval'][1]))
        margins=[]
        for node in candidates:
            start,end=node['interval']
            chunk=cut(src,start,end,f'avb_{modality}_{start:.2f}_{end:.2f}')
            media=prepare_video(chunk)
            query_prompt=(f'Does this {modality} segment directly support the following observation: "{query}"? '
                          'Judge only the supplied modality. A mentioned action is not a directly seen action. '
                          'Do not infer missing content.')
            pair=score(media,query_prompt,['Yes','No'],mode='vision' if modality=='video' else 'audio')
            margins.append(round((pair[0]-pair[1])/2,3));del media
        matches=avb_episodes(candidates,margins,modality,f'T{number}_{modality}')
        results.append(dict(source=modality,segment_ids=[x['id'] for x in candidates],margins=margins,matches=matches))
    return dict(op='locate',query=query,source=source,
                matches=[dict(x,content=query,kind='retrieved_support') for r in results for x in r['matches']],
                profiles=results)

def avb_read(src,nodes,previous,action,number,duration):
    registry={x['id']:x for x in nodes}
    registry.update({x['id']:x for t in previous for x in t.get('matches',[])})
    if 'ids' in action:
        ids=action['ids']
        if isinstance(ids,str):ids=[ids]
        anchors=[registry[x] for x in ids if x in registry]
        if not anchors:raise ValueError('read references no known source address')
        intervals=[x['interval'] for x in anchors]
    else:intervals=[action['interval']]
    observations=[]
    for j,interval in enumerate(intervals[:4]):
        start,end=max(0,float(interval[0])),min(duration,float(interval[1]))
        if end-start<0.2:raise ValueError('read interval is empty or too short')
        chunk=cut(src,start,end,f'avb_read_{start:.2f}_{end:.2f}')
        media=prepare_video(chunk)
        source=action['source']
        prompt=(f'Inspect only this supplied {source} segment. Request: {action.get("query","Describe what is directly observable")}. '
                'Return directly observed content relevant to the request. Do not infer anything outside the segment. '
                'Do not give an option letter or answer a multiple-choice question.')
        for modality in ('video','audio') if source=='both' else (source,):
            content=clean_observation(native_generate(media,prompt,'vision' if modality=='video' else 'audio',80))
            observations.append(dict(id=f'R{number}_{j}_{modality}',source=modality,
                                     interval=[round(start,3),round(end,3)],content=content,kind='readback'))
        del media
    return dict(op='read',request=action.get('query',''),matches=observations)

def avbind_answer(video_path,question,answer_options=None,observation_cache=None,
                  utterance_cache=None,max_steps=3,variant='readback'):
    if variant not in ('ledger','readback','linked'):raise ValueError('Unknown AVBind variant')
    started=time.time();src=Path(video_path);wave=decode_audio(src);duration=len(wave)/16000
    cache=Path(observation_cache) if observation_cache else None
    observations=json.loads(cache.read_text()) if cache and cache.exists() else extraction_memory(src,duration)
    if cache and not cache.exists():cache.write_text(json.dumps(observations),encoding='utf-8')
    speech_cache=Path(utterance_cache) if utterance_cache else None
    utterances=json.loads(speech_cache.read_text()) if speech_cache and speech_cache.exists() else utterance_nodes(src,wave,duration)
    if speech_cache and not speech_cache.exists():speech_cache.write_text(json.dumps(utterances),encoding='utf-8')
    nodes=avb_nodes(observations,utterances);results=[];trace=[];seen=set()
    for step in range(0 if variant=='ledger' else max_steps):
        raw=None
        try:
            action,planning=avb_next_action(question,nodes,results,variant=='linked')
            raw=json.dumps(dict(action=action,planning=planning))
            op=action.get('op');source=action.get('source')
            if op=='stop':trace.append(dict(raw=raw,action=action));break
            if op not in ('locate','read') or source not in ('audio','video','both'):
                raise ValueError('Unsupported evidence operation/source')
            signature=json.dumps(action,sort_keys=True)
            if signature in seen:trace.append(dict(raw=raw,action=action,error='repeated operation'));break
            seen.add(signature)
            if op=='locate':
                query=action.get('query','').strip()
                if not query:raise ValueError('locate requires a description')
                output=avb_locate(src,nodes,source,query,step)
            else:output=avb_read(src,nodes,results,action,step,duration)
            results.append(output);trace.append(dict(raw=raw,action=action))
            print('AVBIND_TOOL',json.dumps(dict(step=step,action=action,matches=output['matches'])),flush=True)
        except (ValueError,KeyError,TypeError) as exc:
            trace.append(dict(raw=raw,error=str(exc)))
            # Invalid plans are visible; never silently route to audio-first.
            break
    memory=avb_ledger(nodes,results,relations=variant=='linked')
    prompt=AVB_READOUT+memory+'\nUSER QUESTION: '+question
    if answer_options is None:
        answer=clean_observation(text_generate(prompt+'\nAnswer concisely and cite source IDs. If evidence is insufficient, say so.',192))
        scores=prediction=None
    else:
        scores=text_choice_scores(prompt,answer_options);prediction=int(np.argmax(scores));answer=answer_options[prediction]
    return dict(answer=answer,prediction=prediction,scores=scores,nodes=nodes,tool_results=results,
                controller_trace=trace,memory=memory,variant=variant,
                latency_sec=round(time.time()-started,2),duration=round(duration,3),
                dataset_or_task_used=False,answer_options_used_by_controller=False)

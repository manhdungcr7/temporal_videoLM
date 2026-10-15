"""Robust referent parsing, audio provenance verification, option entailment.

The model expands choices into propositions with the original question only,
then checks them against a compact measured event graph. There is no task name,
label offset, or fixed Yes/No/Past/Present/Future mapping in inference.
"""


def parse_reference_plan(raw,nodes):
    parsed=[]
    try:
        parsed=json.loads(raw[raw.index('['):raw.rindex(']')+1])
    except Exception:
        for match in re.finditer(r'"event"\s*:\s*"([^"\n]+)"\s*,\s*"audio_ids"\s*:\s*([^}]+)',raw):
            parsed.append(dict(event=match.group(1),audio_ids=[int(j) for j in re.findall(r'\d+',match.group(2))]))
    events=[]
    for x in parsed[:2]:
        ids=x.get('audio_ids',[])
        if isinstance(ids,(int,float,str)):
            ids=[ids]
        valid=[]
        for j in ids:
            try:
                j=int(j)
                if 0<=j<len(nodes):valid.append(j)
            except (ValueError,TypeError):continue
        event=str(x.get('event','')).strip()
        if valid and event:
            events.append(dict(event=event,proposed_ids=valid))
    return events


def verify_audio_reference(event,nodes):
    margins=[]
    for node in nodes:
        q=(f'Spoken words: "{node["words"]}"\n'
           f'Do these words refer to this event: "{event}"? '
           'Judge the meaning of the words, not whether the event happens in the video.')
        pair=text_only_scores(q,['Yes','No'])
        margins.append(round((pair[0]-pair[1])/2,3))
    best=int(np.argmax(margins))
    return dict(nodes[best],reference=event,audio_grounding=margins,selected_audio_id=best)


def measured_graph(checks):
    lines=['Source-separated measurements of referenced events. Speech and video have independent event times.']
    for j,c in enumerate(checks):
        lines.append(f'Event E{j}: {c["reference"]}.')
        lines.append(f'Audio source: speaker says "{c["words"]}" during {c["audio_interval"]} seconds.')
        if c['supported']:
            lines.append(f'Visual source: independently verified visible event "{c["reference"]}" '
                         f'at {c["visual_interval"]} seconds.')
            lines.append(f'The visible occurrence of E{j} {c["relation"]} the spoken mention of E{j}.')
        else:
            lines.append('Visual source: no positive direct support for this event in the checked video chunks.')
    return '\n'.join(lines)+'\n'


def proposition_for_choice(stem,choice):
    q=(f'Question: {stem}\nAnswer choice: {choice}\n'
       'Rewrite the question with this answer as one complete factual statement. '
       'Resolve relative terms using the reference point in the question. '
       'Do not use video evidence or decide whether the statement is true. '
       'Output only one sentence.')
    return clean_observation(text_generate(q,80)).split('\n')[0]


def grounded_choice_scores(memory,stem,options):
    propositions=[proposition_for_choice(stem,x) for x in options]
    margins=[]
    for statement in propositions:
        q=(memory+'\nStatement: '+statement+'\n'
           'Is this statement supported by the measured event times and sources? '
           'Use the observed visual event time for a visible occurrence and the audio time for speech. '
           'Do not replace a visible event with its spoken description.')
        pair=text_only_scores(q,['Yes','No'])
        margins.append(round((pair[0]-pair[1])/2,3))
    return margins,propositions


def robust_item(i,row):
    src,wave,dur,win=prep_clip(row,50)
    stem,options=parse_question(row['question'])
    observations=json.loads((OUT/f'cq_physical_memory_{i}.json').read_text())
    previous=[json.loads(line) for line in (OUT/f'cq_diagnostic_round7_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in previous if x['idx']==i and 'error' not in x)
    nodes=saved['utterances']
    references=parse_reference_plan(saved['plan'],nodes)
    checks=[]
    grounded=[]
    for ref in references:
        node=verify_audio_reference(ref['event'],nodes)
        grounded.append(node)
        old=next((x for x in saved['checks'] if x['reference']==ref['event']),None)
        if old is None:
            check=cross_source_checks(src,observations,[node])[0]
        else:
            check=dict(old,words=node['words'],audio_interval=[node['start'],node['end']])
            vi,ai=check['visual_interval'],check['audio_interval']
            check['relation']='precedes' if vi[1]<=ai[0] else 'follows' if vi[0]>=ai[1] else 'overlaps'
        checks.append(check)
    memory=measured_graph(checks) if checks else hierarchical_memory(observations,nodes,False)
    rec=dict(dur=round(dur,3),references=references,audio_grounding=grounded,checks=checks,memory=memory)
    rec['cq_verified_graph']=text_only_scores(memory+stem,options)
    rec['cq_verified_choices'],rec['propositions']=grounded_choice_scores(memory,stem,options)
    rec['cq_verified_reason'],rec['reasoning']=reason_scores(memory,stem,options)
    print('VERIFIED_GRAPH',i,json.dumps(checks),flush=True)
    print('CHOICE_PROPOSITIONS',i,json.dumps(rec['propositions']),json.dumps(rec['cq_verified_choices']),flush=True)
    return rec


CONDITIONS=('cq_verified_graph','cq_verified_choices','cq_verified_reason')
R=run_loop(ITEMS,f'cq_diagnostic_round8_s{LAB_SERVER}.jsonl',robust_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND8_RESULT',json.dumps(RESULT),flush=True)

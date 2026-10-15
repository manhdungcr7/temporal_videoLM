"""Query-referenced event alignment from physical CQ observations.

Generic referent resolution -> isolated per-chunk verification -> interval
graph -> original-question reasoning. No TA/TF or option-to-relation rules.
Requires physical CQ helpers loaded by the same job before this body.
"""


def resolve_references(memory, stem):
    prompt = (memory+'\nQuestion: '+stem+'\nIdentify up to two concrete visible events '
              'whose time or presence the question refers to. Resolve indirect references '
              'using the AUDIO observations, keeping quoted speech separate from visual facts. '
              'Return only a JSON array of short action descriptions, e.g. ["opening the door"]. '
              'Do not answer the question, choose an option, or describe a time relation.')
    raw = text_generate(prompt, 100)
    match = re.search(r'\[[\s\S]*?\]',raw)
    try:
        events = json.loads(match.group()) if match else []
        events = [x.strip() for x in events if isinstance(x,str) and x.strip()][:2]
    except Exception:
        events = []
    return events,raw


def align_events(src, observations, events):
    alignments=[]
    for event in events:
        margins=[]
        for t,x in enumerate(observations):
            chunk=cut(src,x['start'],x['end'],f'cq_phys_{int(x["start"]//2)}')
            media=prepare_video(chunk)
            question=('Considering only what is visibly being performed in this short video, '
                      f'is this event actually occurring: "{event}"? '
                      'Holding tools, preparing, or showing a finished result alone does not establish execution.')
            pair=score(media,question,['Yes','No'],mode='vision')
            margins.append(round(pair[0]-pair[1],3))
            del media
        best=int(np.argmax(margins))
        # Keep every score and an explicit uncertainty; no forced best=truth.
        top=observations[best]
        alignments.append(dict(event=event,margins=margins,best_interval=[top['start'],top['end']],
                               supported=margins[best]>0))
    return alignments


def interval_relation(a,b):
    if a[1] <= b[0]:return 'precedes'
    if b[1] <= a[0]:return 'follows'
    return 'overlaps'


def graph_memory(observations, alignments):
    memory=source_memory(observations)
    lines=['Independent checks of question-referenced visual events:']
    voiced=[x for x in observations if x['audio']!='silence']
    for result in alignments:
        interval=result['best_interval']
        lines.append(f'Event "{result["event"]}" | strongest VIDEO check {interval}s | '
                     f'positive evidence: {result["supported"]} | all chunk support margins: {result["margins"]}')
        if result['supported']:
            for x in voiced:
                audio_interval=[x['start'],x['end']]
                lines.append(f'VIDEO event "{result["event"]}" at {interval}s '
                             f'{interval_relation(interval,audio_interval)} AUDIO at {audio_interval}s: {x["audio"]}')
    return memory+'\n'+'\n'.join(lines)+'\n'


def aligned_item(i,row):
    src,audio,dur,win=prep_clip(row,50)
    stem,options=parse_question(row['question'])
    observations=json.loads((OUT/f'cq_physical_memory_{i}.json').read_text())
    memory=source_memory(observations)
    events,plan=resolve_references(memory,stem)
    aligned=align_events(src,observations,events)
    graph=graph_memory(observations,aligned)
    answer,reasoning=reason_scores(graph,stem,options)
    rec=dict(dur=round(dur,3),references=events,plan=plan,alignments=aligned,graph=graph,
             cq_event_graph=answer,reasoning=reasoning)
    print('EVENT_GRAPH',i,json.dumps(aligned),flush=True)
    print('GRAPH_ANALYSIS',i,reasoning,flush=True)
    return rec


CONDITIONS=('cq_event_graph',)
R=run_loop(ITEMS,f'cq_diagnostic_round5_s{LAB_SERVER}.jsonl',aligned_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND5_RESULT',json.dumps(RESULT),flush=True)

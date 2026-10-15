"""Audio-referent -> visual chunk checks -> a provenance interval graph.

All utterance references are checked without question/options. Original QA is
performed by the model from the graph, not by a TA/TF answer-mapping rule.
"""


def resolve_grounded_references(observations,nodes,stem):
    context='\n'.join(f'A{j} [{n["start"]},{n["end"]}]s: {n["words"]}' for j,n in enumerate(nodes))
    prompt=(context+'\nQuestion: '+stem+'\nResolve up to two concrete event references needed by the question. '
            'Use the spoken words to resolve indirect references. Combine sentences that refer to the same event. '
            'Do not invent an action from a number or infer an event not stated. '
            'Return only JSON: [{"event":"short action description","audio_ids":[0]}]. '
            'Do not give an answer, option letter, or temporal relation.')
    raw=clean_observation(text_generate(prompt,192))
    try:
        parsed=json.loads(raw[raw.index('['):raw.rindex(']')+1])
        events=[]
        for x in parsed[:2]:
            ids=[int(j) for j in x.get('audio_ids',[]) if 0<=int(j)<len(nodes)]
            if not ids or not str(x.get('event','')).strip():continue
            group=[nodes[j] for j in ids]
            events.append(dict(start=min(n['start'] for n in group),end=max(n['end'] for n in group),
                               words=' '.join(n['words'] for n in group),reference=x['event'],audio_ids=ids))
    except Exception:
        events=[]
    return events,raw


def cross_source_checks(src,observations,nodes):
    checked=[]
    for node in nodes:
        reference=node['reference']
        q=('Is the event "'+reference+'" actually visible in this short video? '
           'Judge the visible event itself, not a spoken description, title, or a finished result.')
        margins=[]
        for x in observations:
            chunk=cut(src,x['start'],x['end'],f'cq_phys_{int(x["start"]//2)}')
            media=prepare_video(chunk)
            pair=score(media,q,['Yes','No'],mode='vision')
            margins.append(round((pair[0]-pair[1])/2,3))
            del media
        best=int(np.argmax(margins))
        peak=observations[best]
        interval=[peak['start'],peak['end']]
        heard=[node['start'],node['end']]
        relation='precedes' if interval[1]<=heard[0] else 'follows' if interval[0]>=heard[1] else 'overlaps'
        # Preserve the full distribution and support flag, rather than
        # asserting that an event exists merely because one chunk ranks first.
        checked.append(dict(words=node['words'],reference=reference,audio_interval=heard,
                            visual_interval=interval,margins=margins,supported=margins[best]>0,
                            relation=relation))
    return checked


def checked_memory(observations,nodes,checks,compact=False):
    memory='' if compact else hierarchical_memory(observations,nodes,False)
    lines=['Cross-source event identity checks, made on visual-only chunks:']
    for c in checks:
        lines.append(f'Audio words "{c["words"]}" at {c["audio_interval"]}s describe "{c["reference"]}". '
                     f'Strongest direct visual match: {c["visual_interval"]}s. '
                     f'Positive visual support: {c["supported"]}. '
                     f'That visual interval {c["relation"]} the audio interval.')
    return memory+'\n'+'\n'.join(lines)+'\n'


def checked_item(i,row):
    src,wave,dur,win=prep_clip(row,50)
    stem,options=parse_question(row['question'])
    observations=json.loads((OUT/f'cq_physical_memory_{i}.json').read_text())
    previous=OUT/f'cq_diagnostic_round6_s{LAB_SERVER}.jsonl'
    cached=[json.loads(line) for line in previous.read_text().splitlines() if line.strip()]
    saved=next((r for r in cached if r['idx']==i and 'error' not in r),None)
    nodes=saved['utterances'] if saved else utterance_nodes(src,wave,dur)
    references,plan=resolve_grounded_references(observations,nodes,stem)
    checks=cross_source_checks(src,observations,references)
    memory=checked_memory(observations,nodes,checks)
    compact=checked_memory(observations,nodes,checks,True)
    rec=dict(dur=round(dur,3),utterances=nodes,checks=checks,memory=memory,plan=plan,references=references,compact=compact)
    rec['cq_cross_source']=text_only_scores(memory+stem,options)
    rec['cq_cross_source_grounded']=text_only_scores(memory+GENERIC_REASON+stem,options)
    rec['cq_cross_source_reason'],rec['reasoning']=reason_scores(memory,stem,options)
    rec['cq_cross_source_compact']=text_only_scores(compact+GENERIC_REASON+stem,options) if checks else rec['cq_cross_source']
    print('CROSS_SOURCE_CHECKS',i,json.dumps(checks),flush=True)
    print('CROSS_SOURCE_ANALYSIS',i,rec['reasoning'],flush=True)
    return rec


CONDITIONS=('cq_cross_source','cq_cross_source_grounded','cq_cross_source_reason','cq_cross_source_compact')
R=run_loop(ITEMS,f'cq_diagnostic_round7_s{LAB_SERVER}.jsonl',checked_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND7_RESULT',json.dumps(RESULT),flush=True)

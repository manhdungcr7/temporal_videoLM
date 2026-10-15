"""Question-dependent semantic grounding of choices, no label mapping.

The frozen model maps each choice into a generic interval predicate without
seeing evidence. Native QA reads the original choice plus that interpretation.
No task-specific parser, keyword matching, or answer-execution rule is used.
"""
REL_MEANINGS={
    'precedes':'the visible event has finished before the spoken reference interval begins',
    'overlaps':'the visible event is occurring for at least part of the spoken reference interval',
    'follows':'the visible event begins after the spoken reference interval has finished',
    'not_precedes':'the visible event is not entirely earlier than the spoken reference interval',
    'not_overlaps':'the visible event is not occurring at any time within the spoken reference interval',
    'not_follows':'the visible event is not entirely later than the spoken reference interval',
    'unsupported':'this choice does not express a temporal relation between these two referents'
}
SEMANTIC_CACHE={}


def interpret_choice(stem,choice):
    key=(stem,choice)
    if key in SEMANTIC_CACHE:return SEMANTIC_CACHE[key]
    definitions='\n'.join(f'{op}: {meaning}' for op,meaning in REL_MEANINGS.items())
    q=(definitions+'\nQuestion: '+stem+'\nAnswer choice: '+choice+'\n'
       'Interpret this choice using the question only, not any video evidence. '
       'Which predicate expresses the statement implied by giving this answer? '
       'Return only one predicate name from the list above.')
    raw=clean_observation(text_generate(q,48))
    match=re.search(r'\b(not_precedes|not_overlaps|not_follows|precedes|overlaps|follows|unsupported)\b',raw)
    op=match.group(1) if match else 'unsupported'
    SEMANTIC_CACHE[key]=(op,raw)
    return op,raw


def choice_meanings(stem,options):
    parsed=[interpret_choice(stem,x) for x in options]
    annotated=[label if op=='unsupported' else label+' (meaning: '+REL_MEANINGS[op]+')'
               for label,(op,raw) in zip(options,parsed)]
    return annotated,parsed


def semantic_item(i,row):
    records=[json.loads(line) for line in (OUT/f'cq_diagnostic_round11_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in records if x['idx']==i and 'error' not in x)
    stem,options=parse_question(row['question'])
    annotated,meanings=choice_meanings(stem,options)
    memory=saved['memory']
    rec=dict(set=saved['set'],checks=saved['checks'],memory=memory,interpretations=meanings,annotated_options=annotated)
    rec['cq_semantic_graph']=text_only_scores(memory+stem,annotated)
    rec['cq_semantic_reason'],rec['reasoning']=neutral_reason_scores(memory,stem,annotated)
    print('SEMANTIC_CHOICES',i,json.dumps(meanings),flush=True)
    print('SEMANTIC_REASONING',i,rec['reasoning'],flush=True)
    return rec


DEV_MANIFEST=json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
control_path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not control_path.exists():control_path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(control_path.read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in DEV_MANIFEST['selected']+CONTROL_MANIFEST['selected']]
CONDITIONS=('cq_semantic_graph','cq_semantic_reason')
R=run_loop(ITEMS,f'cq_diagnostic_round12_s{LAB_SERVER}.jsonl',semantic_item,show=CONDITIONS,min_left=0)
RESULT={split:{c:{'accuracy':acc([r for r in R if r.get('set')==split],c)[0],
                 'n':acc([r for r in R if r.get('set')==split],c)[1]} for c in CONDITIONS}
        for split in ('development','control')}
print('ROUND12_RESULT',json.dumps(RESULT),flush=True)

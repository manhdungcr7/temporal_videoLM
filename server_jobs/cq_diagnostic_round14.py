"""Separate lexical relation meaning from source-role grounding."""
SEMANTIC_CACHE={}
ABSTRACT_RELATIONS=[
    'Earlier / before the reference time',
    'At the same time / during the reference period',
    'Later / after the reference time',
    'No temporal relation is specified'
]


def interpret_choice(stem,choice):
    key=(stem,choice)
    if key in SEMANTIC_CACHE:return SEMANTIC_CACHE[key]
    q=(f'Question: {stem}\nAnswer choice: {choice}\n'
       'Interpret the temporal meaning of this answer, ignoring whether it is factually correct. '
       'Before considering affirmation or denial, identify the base temporal relationship '
       'referenced by the answer and question. Which relationship is meant?')
    family_scores=text_choice_scores(q,ABSTRACT_RELATIONS)
    selected=int(np.argmax(family_scores))
    family=('precedes','overlaps','follows','unsupported')[selected]
    polarity_scores=None
    op=family
    if family!='unsupported':
        q=(f'Question: {stem}\nAnswer choice: {choice}\n'
           f'Base temporal relationship: {ABSTRACT_RELATIONS[selected]}. '
           'According to the question, does this answer affirm or deny that relationship? '
           'Interpret language only; do not infer evidence.')
        polarity_scores=text_choice_scores(q,['Affirms that relationship','Denies that relationship'])
        if int(np.argmax(polarity_scores))==1:op='not_'+family
    trace=dict(family=family,family_scores=family_scores,polarity_scores=polarity_scores)
    SEMANTIC_CACHE[key]=(op,trace)
    return op,trace


def abstract_semantic_item(i,row):
    records=[json.loads(line) for line in (OUT/f'cq_diagnostic_round11_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in records if x['idx']==i and 'error' not in x)
    stem,options=parse_question(row['question'])
    annotated,meanings=choice_meanings(stem,options)
    memory=saved['memory']
    rec=dict(set=saved['set'],checks=saved['checks'],memory=memory,interpretations=meanings,annotated_options=annotated)
    rec['cq_abstract_graph']=text_only_scores(memory+stem,annotated)
    rec['cq_abstract_reason'],rec['reasoning']=neutral_reason_scores(memory,stem,annotated)
    print('ABSTRACT_CHOICES',i,json.dumps(meanings),flush=True)
    return rec


DEV_MANIFEST=json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
control_path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not control_path.exists():control_path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(control_path.read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in DEV_MANIFEST['selected']+CONTROL_MANIFEST['selected']]
CONDITIONS=('cq_abstract_graph','cq_abstract_reason')
R=run_loop(ITEMS,f'cq_diagnostic_round14_s{LAB_SERVER}.jsonl',abstract_semantic_item,show=CONDITIONS,min_left=0)
RESULT={split:{c:{'accuracy':acc([r for r in R if r.get('set')==split],c)[0],
                 'n':acc([r for r in R if r.get('set')==split],c)[1]} for c in CONDITIONS}
        for split in ('development','control')}
print('ROUND14_RESULT',json.dumps(RESULT),flush=True)

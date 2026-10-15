"""Constrained neural choice interpretation, rather than free-form parsing."""


def text_choice_scores(prompt,choices):
    k=len(choices)
    assert k<=4
    ids=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'][:k]
    totals=np.zeros(k)
    for perm in rotations(k):
        question=prompt+' '+ ' '.join(f'({letter}) {choices[j]}' for letter,j in zip('ABCD',perm))
        messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
                  {'role':'user','content':[{'type':'text','text':question}]}]
        chat=PROC.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        inp=PROC(text=chat,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
        with torch.no_grad():
            logits=MODEL_(**inp,use_cache=False).logits[0,-1].float()
        probs=torch.log_softmax(logits[ids].cpu(),-1).numpy()
        for slot,j in enumerate(perm):totals[j]+=probs[slot]
    return totals.round(3).tolist()


SEMANTIC_CACHE={}
def interpret_choice(stem,choice):
    key=(stem,choice)
    if key in SEMANTIC_CACHE:return SEMANTIC_CACHE[key]
    query=(f'Question: {stem}\nOne possible answer: {choice}\n'
           'Interpret the meaning of this answer without using video evidence. '
           'Ignoring affirmation versus negation, which base relation does this answer concern?')
    family_choices=[REL_MEANINGS[x] for x in ('precedes','overlaps','follows','unsupported')]
    family_scores=text_choice_scores(query,family_choices)
    family=('precedes','overlaps','follows','unsupported')[int(np.argmax(family_scores))]
    polarity_scores=None
    op=family
    if family!='unsupported':
        query=(f'Question: {stem}\nOne possible answer: {choice}\n'
               f'Base relation: {REL_MEANINGS[family]}. '
               'Does giving this answer affirm that relation or deny it? Judge the answer meaning only, '
               'not whether any event actually occurs.')
        polarity_scores=text_choice_scores(query,['The answer affirms the base relation',
                                                   'The answer denies the base relation'])
        if int(np.argmax(polarity_scores))==1:op='not_'+family
    trace=dict(family=family,family_scores=family_scores,polarity_scores=polarity_scores)
    SEMANTIC_CACHE[key]=(op,trace)
    return op,trace


def constrained_item(i,row):
    records=[json.loads(line) for line in (OUT/f'cq_diagnostic_round11_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in records if x['idx']==i and 'error' not in x)
    stem,options=parse_question(row['question'])
    annotated,meanings=choice_meanings(stem,options)
    memory=saved['memory']
    rec=dict(set=saved['set'],checks=saved['checks'],memory=memory,interpretations=meanings,annotated_options=annotated)
    rec['cq_grounded_graph']=text_only_scores(memory+stem,annotated)
    rec['cq_grounded_reason'],rec['reasoning']=neutral_reason_scores(memory,stem,annotated)
    print('CONSTRAINED_CHOICES',i,json.dumps(meanings),flush=True)
    print('CONSTRAINED_REASONING',i,rec['reasoning'],flush=True)
    return rec


DEV_MANIFEST=json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
control_path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not control_path.exists():control_path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(control_path.read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in DEV_MANIFEST['selected']+CONTROL_MANIFEST['selected']]
CONDITIONS=('cq_grounded_graph','cq_grounded_reason')
R=run_loop(ITEMS,f'cq_diagnostic_round13_s{LAB_SERVER}.jsonl',constrained_item,show=CONDITIONS,min_left=0)
RESULT={split:{c:{'accuracy':acc([r for r in R if r.get('set')==split],c)[0],
                 'n':acc([r for r in R if r.get('set')==split],c)[1]} for c in CONDITIONS}
        for split in ('development','control')}
print('ROUND13_RESULT',json.dumps(RESULT),flush=True)

"""Neutral evidence reasoning, without repeated negative source warnings.

Use exactly the measured episode graphs from round9; change only readout.
Both diagnostic cells and opposite-class controls are evaluated identically.
"""


def neutral_reason_scores(memory,stem,options):
    prompt=(memory+'\nQuestion: '+stem+'\n'
            'Read the stated measurements as facts. Identify the observations and intervals needed '
            'for this question, then explain the conclusion in two short sentences. '
            'Use only the stated facts and do not introduce new observations.')
    reasoning=clean_observation(text_generate(prompt,128))
    final=text_only_scores(memory+'\nEvidence analysis: '+reasoning+'\nQuestion: '+stem,options)
    return final,reasoning


def neutral_item(i,row):
    previous=[json.loads(line) for line in (OUT/f'cq_diagnostic_round9_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in previous if x['idx']==i and 'error' not in x)
    stem,options=parse_question(row['question'])
    answer,reasoning=neutral_reason_scores(saved['memory'],stem,options)
    rec=dict(set=saved['set'],checks=saved['checks'],memory=saved['memory'],
             cq_episode_neutral=answer,reasoning=reasoning)
    print('NEUTRAL_REASONING',i,reasoning,flush=True)
    return rec


DEV_MANIFEST=json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
control_path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not control_path.exists():control_path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(control_path.read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in DEV_MANIFEST['selected']+CONTROL_MANIFEST['selected']]
CONDITIONS=('cq_episode_neutral',)
R=run_loop(ITEMS,f'cq_diagnostic_round10_s{LAB_SERVER}.jsonl',neutral_item,show=CONDITIONS,min_left=0)
RESULT={split:{c:{'accuracy':acc([r for r in R if r.get('set')==split],c)[0],
                 'n':acc([r for r in R if r.get('set')==split],c)[1]} for c in CONDITIONS}
        for split in ('development','control')}
print('ROUND10_RESULT',json.dumps(RESULT),flush=True)

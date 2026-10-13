# OPTIONAL DIAGNOSTIC: run once for the currently selected sample.
# This diagnoses input dependence; it is not the prompting method's accuracy test.
import re
assert 'LAB' in globals() and LAB.is_alive(), 'Load the model and select a sample first.'
assert 'MEDIA' in globals() and MEDIA['item_id']==ITEM_ID, 'Run the sample selection cell.'
if not LAB.request({'cmd':'ping'}).get('diagnostic_support'):
    print('Updating the worker for modality checks. The model reloads once.',flush=True)
    LAB.close()
    for name,source in DIAGNOSTIC_RUNNER_SOURCES.items():
        (ROOT/name).write_text(source,encoding='utf-8')
    SOURCES=DIAGNOSTIC_RUNNER_SOURCES
    LAB=PromptLab(PYTHON,ROOT,OUT)
    MEDIA=LAB.request({'cmd':'sample','item_id':ITEM_ID,'question':QUESTION,'path':str(VIDEO_PATH)})
    BASELINE_RESULT=None

assert '(A) Yes (B) No' in QUESTION, 'This diagnostic requires A=Yes, B=No in the original question.'
description='Describe the ingredients and actions identifiable from the images in this clip. Report visual observations; do not use narration to name ambiguous items.'
swapped=QUESTION.replace('(A) Yes (B) No','(A) No (B) Yes')
probes=[
    {'name':'av_original','mode':'av','prompt':QUESTION,'map':{'A':'Yes','B':'No'}},
    {'name':'vision_original','mode':'vision','prompt':QUESTION,'map':{'A':'Yes','B':'No'}},
    {'name':'audio_original','mode':'audio','prompt':QUESTION,'map':{'A':'Yes','B':'No'}},
    {'name':'av_swapped_options','mode':'av','prompt':swapped,'map':{'A':'No','B':'Yes'}},
    {'name':'av_description_no_target','mode':'av','prompt':description,'map':None},
    {'name':'vision_description_no_target','mode':'vision','prompt':description,'map':None},
    {'name':'vision_higher_resolution','mode':'vision','prompt':QUESTION,'map':{'A':'Yes','B':'No'},
     'max_pixels':2*MEDIA['max_pixels_per_frame']},
]

def diagnostic_choice(raw, complete, mapping):
    if not complete or mapping is None:return None
    match=re.match(r'^\s*(?:(?:The\s+)?(?:answer|option)\s*(?:is|:)\s*)?([AB])(?=$|[\s().:—-])',raw,re.I)
    if match:
        if re.search(r'\bor\s+[AB]\b',raw,re.I):return None
        meaning=re.match(r'[\s().:—-]*(Yes|No)\b',raw[match.end():],re.I)
        semantic=mapping[match.group(1).upper()]
        return semantic if not meaning or meaning.group(1).lower()==semantic.lower() else None
    match=re.match(r'^\s*(Yes|No)\b',raw,re.I)
    return match.group(1).capitalize() if match else None

DIAGNOSTIC_RESULTS=[]
for index,probe in enumerate(probes,1):
    print(f"{index}/{len(probes)} {probe['name']} ...",flush=True)
    record={'item_id':ITEM_ID,'probe':probe['name'],'media_mode':probe['mode'],
            'option_map':probe['map'],'prompt':probe['prompt']}
    request={'cmd':'diagnostic','media_mode':probe['mode'],'prompt':probe['prompt'],'max_new_tokens':192}
    if 'max_pixels' in probe:request['max_pixels']=probe['max_pixels']
    try:
        result=LAB.request(request)
        record.update(result=result,semantic_answer=diagnostic_choice(result['raw'],result['complete'],probe['map']))
        display(Markdown('### '+probe['name']+'\n\n'+result['raw']))
        print('Meaning:',record['semantic_answer'],'| complete:',result['complete'],
              '| audio:',result['audio_provided'],'| images:',result['video_provided'],
              '| input tokens:',result['input_tokens'],'| seconds:',round(result['latency_s'],2))
    except Exception as error:
        record['error']=str(error)
        print('Probe failed:',str(error)[-1800:])
    DIAGNOSTIC_RESULTS.append(record)

path=OUT/(ITEM_ID+'_mechanism_check.json')
path.write_text(json.dumps({'item_id':ITEM_ID,'original_question':QUESTION,'gold_ui_only':GOLD,
    'scope':'One-sample modality, wording and resolution checks; no attention maps or causal attribution claimed.',
    'results':DIAGNOSTIC_RESULTS},ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved:',path)
answers={r['probe']:r.get('semantic_answer') for r in DIAGNOSTIC_RESULTS}
print('Compare meanings (Yes/No), not letters:',answers)
if answers.get('av_original')=='Yes' and answers.get('vision_original')=='No':
    print('Removing audio changed Yes to No: evidence of audio-dependent answering on this sample.')
if answers.get('av_original')=='Yes' and answers.get('vision_original')=='Yes':
    print('The Yes answer survives removing audio: audio is not necessary for this decision; inspect visual recognition and question dependence.')
if answers.get('av_original') and answers.get('av_swapped_options') and answers['av_original']!=answers['av_swapped_options']:
    print('Swapping answer order changed the semantic answer: this decision is sensitive to option order.')
if answers.get('vision_original') and answers.get('vision_higher_resolution') and answers['vision_original']!=answers['vision_higher_resolution']:
    print('Increasing visual resolution changed the answer: the decision is sensitive to preprocessing resolution.')
print('Description omissions do not prove absence. These checks do not validate the benchmark label or reveal specific attention heads.')

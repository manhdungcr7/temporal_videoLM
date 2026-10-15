"""Actual-media free-form reverse queries; qualitative, no benchmark labels."""
indices=[1454,2348] if LAB_SERVER==1 else [2139,2124]
rows=json.loads(fetch('samples.json').read_text())
requests=(['What words, if any, are spoken while the person is pouring tomatoes into the pan?',
           'What does the person say while pouring water into the container?'] if LAB_SERVER==1 else
          ['What does the person say while putting the chopped ingredients into the pan?',
           'What is heard while the person is stirring the contents of the pan?'])
records=[]
for i,question in zip(indices,requests):
    src=cut(fetch(rows[i]['video_path'].replace(BS,'/')),0,10**4,'full')
    result=AVB_NS['avbind_answer'](src,question,None,OUT/f'cq_physical_memory_{i}.json',
                                  OUT/f'avbind_utterances_{i}.json',max_steps=3,variant='linked')
    rec=dict(idx=i,question=question,result=result,qualitative_only=True)
    records.append(rec)
    print('REVERSE_QUERY',json.dumps(dict(idx=i,question=question,answer=result['answer'],
                                        actions=[x.get('action') for x in result['controller_trace']])),flush=True)
(OUT/f'avbind_reverse_query_s{LAB_SERVER}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
RESULT=dict(n=len(records),qualitative_only=True)

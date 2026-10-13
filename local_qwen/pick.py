import json,random,av,sys
from pathlib import Path
R=Path(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark');rows=json.loads((R/'samples.json').read_text(encoding='utf-8'))
def dur(p):
    with av.open(str(p)) as c:return float(c.duration)/1e6
random.seed(0);by={}
for i,r in enumerate(rows):
    if r['data_type']!='semantic':continue
    t=r['task_type_abbr'].replace('CSMB','CMSB');by.setdefault((t,r['answer']),[]).append(i)
sel=[];seen=set()
for k,v in sorted(by.items()):
    random.shuffle(v);n=0
    for i in v:
        src=rows[i]['video_path'].replace(chr(92),'/').split('/')[2]
        if src in seen:continue
        d=dur(R/rows[i]['video_path'])
        if d<=20:sel.append(i);seen.add(src);n+=1
        if n==12:break
print(len(sel),{k:len(v) for k,v in by.items()})
json.dump(sel,open('pool72.json','w'))

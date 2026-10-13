"""Usage: python try.py prompt.txt idx1,idx2,... [max_new]   (prompt 'none' = baseline). Sends a job and waits."""
import json,sys,time,re
from pathlib import Path
D=Path(__file__).parent
R=json.loads(open(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark\samples.json',encoding='utf-8').read())
prompt=None if sys.argv[1]=='none' else Path(sys.argv[1]).read_text(encoding='utf-8')
ids=[int(x) for x in sys.argv[2].split(',')];mx=int(sys.argv[3]) if len(sys.argv)>3 else 8
(D/'job_out.json').unlink(missing_ok=True)
(D/'job.json').write_text(json.dumps(dict(prompt=prompt,ids=ids,max_new=mx)),encoding='utf-8')
while not (D/'job_out.json').exists():time.sleep(2)
out=json.loads((D/'job_out.json').read_text(encoding='utf-8'))
def norm(t):
    m=re.search(r"(?:Final answer|Answer):\s*\[?\(?([AB])\b",t,re.I) or re.match(r"\W*(A|B|Yes|No)\b",t)
    return {'A':'A','Yes':'A','B':'B','No':'B'}.get(m.group(1)) if m else None
ok=0
for r in out:
    p=norm(r['text']);ok+=p==r['gold']
    print(r['idx'],R[r['idx']]['task_type_abbr'],'gold',r['gold'],'pred',p,'OK' if p==r['gold'] else 'XX',r['sec'],'s |',r['text'][:70].replace('\n',' / '))
print('correct',ok,'/',len(out))

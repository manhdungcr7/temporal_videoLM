import json,re,sys
R=json.loads(open(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark\samples.json',encoding='utf-8').read())
def norm(t):
    m=re.match(r"\W*([AB])\b",t) or None
    if m:return m.group(1)
    m=re.match(r"\W*(Yes|No)\b",t,re.I)
    return ('A' if m.group(1).lower()=='yes' else 'B') if m else None
rs=[json.loads(l) for l in open(sys.argv[1])]
ok=0
for r in rs:
    p=norm(r['text']);r['p']=p;ok+=p==r['gold']
print(len(rs),'acc',ok)
for g in 'AB':
    s=[r for r in rs if r['gold']==g];print('gold',g,'recall',sum(r['p']==g for r in s),'/',len(s),'invalid',sum(r['p'] is None for r in s))
for r in rs:
    if r['p']!=r['gold']:
        x=R[r['idx']];print(r['idx'],x['task_type_abbr'],'gold',r['gold'],'pred',r['p'],'|',x['question'][:110],'|',r['text'][:30])

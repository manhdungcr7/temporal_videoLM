import collections,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'results/yann_temporal_research'
m=json.loads((D/'data/perception48_manifest.json').read_text(encoding='utf-8'));out=[]
for r in m['rows']:
    def first(es):
        x={}
        for e in es:
            if e['label'].strip()=='Muted':continue
            if e['label'] not in x or e['timestamps'][0]<x[e['label']][0]:x[e['label']]=e['timestamps']
        return x
    a=first(r['sounds']);v=first(r['actions'])
    close=[dict(audio=al,visual=vl,gap_sec=(at[0]-vt[0])/1e6) for al,at in a.items() for vl,vt in v.items() if .5<abs(at[0]-vt[0])/1e6<2.5]
    out.append(dict(video_id=r['video_id'],split=r['split'],valid_sound_labels=len(a),has_near_pair=bool(close),examples=close[:3]))
res=dict(rows=out,counts={s:dict(n=sum(r['split']==s for r in out),near_pairs=sum(r['split']==s and r['has_near_pair'] for r in out),valid_audio=sum(r['split']==s and r['valid_sound_labels']>0 for r in out)) for s in ('train','validation','diagnostic')})
(D/'data/intervention_informativeness.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps(res['counts'],indent=2))

"""Freeze human-annotation interval pairs before native verifier outputs."""
import hashlib,json,math,random
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/native_interval_verification';D.mkdir(exist_ok=True)
source=R/'results/yann_temporal_research/data/perception48_manifest.json'
m=json.loads(source.read_text(encoding='utf-8'));selected=[];skips=[]
W=2.;GAP=.15
def merged(ivs):
    out=[]
    for a,b in sorted(ivs):
        if not out or a>out[-1][1]:out.append([a,b])
        else:out[-1][1]=max(out[-1][1],b)
    return out
def covered(w,ivs):return sum(max(0.,min(w[1],b)-max(w[0],a)) for a,b in merged(ivs))
for i,row in enumerate(m['rows']):
    if row['split']=='train':continue
    for modality,key in (('audio','sounds'),('vision','actions')):
        labels={}
        for ev in row[key]:
            label=ev['label'].strip()
            if modality=='audio' and label in ('Muted','Other','Other:background sound','Object:sound'):continue
            a,b=[float(t)/1e6 for t in ev['timestamps']]
            assert 0<=a<b<=row['duration']+.15,(row['video_id'],a,b)
            labels.setdefault(label,[]).append([max(0.,a),min(row['duration'],b)])
        candidates=[]
        for label,ivs in sorted(labels.items()):
            # Keep every instance in exclusion/coverage, not just the chosen positive occurrence.
            ev=max(ivs,key=lambda iv:iv[1]-iv[0])
            a=max(0.,min(row['duration']-W,(ev[0]+ev[1])/2-W/2));pos=[a,a+W]
            if covered(pos,ivs)<.25:continue
            grid=[k*.25 for k in range(1+int(math.floor((row['duration']-W)/.25)))]
            negs=[[a,a+W] for a in grid if all(a+W<=b0-GAP or a>=b1+GAP for b0,b1 in ivs)]
            if not negs:continue
            rng=random.Random(20261017+int(hashlib.sha256((row['video_id']+label).encode()).hexdigest()[:8],16))
            neg=rng.choice(negs)
            candidates.append(dict(label=label,all_label_intervals=ivs,positive=pos,negative=neg,positive_overlap_sec=covered(pos,ivs)))
        if not candidates:
            skips.append(dict(video_id=row['video_id'],modality=modality,reason='no label with >=.25s positive support and2s nonoverlapping negative (+.15s guard)'));continue
        rng=random.Random(20261017+int(row['video_id'].split('_')[-1])+(0 if modality=='audio' else 100000))
        chosen=rng.choice(candidates)
        # Ontology formatting only: remove namespace; preserve all label content.
        desc=chosen['label'].split(':',1)[-1].strip().replace('/',' or ')
        selected.append(dict(pair_id=f"{row['video_id']}_{modality}",video_id=row['video_id'],split=row['split'],modality=modality,description=desc,
            duration=row['duration'],zip=row['zip'],server=1+(i%2),**chosen))
out=dict(protocol='Fixed2s human-annotation event interval verification; no SVHalluc QA',source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    window_sec=W,negative_guard_sec=GAP,pairs=selected,skipped=skips,
    scope='16 previously selected public train origins allocated to internal validation/diagnostic; reused development, not unseen test. Annotation-negative intervals are proxies, not independently verified true absence.')
blob=(json.dumps(out,indent=2)+'\n').encode();(D/'manifest.json').write_bytes(blob)
print(json.dumps(dict(n_pairs=len(selected),n_origins=len({r['video_id'] for r in selected}),by_modality={k:sum(r['modality']==k for r in selected) for k in ('audio','vision')},skips=skips,sha256=hashlib.sha256(blob).hexdigest()),indent=2))

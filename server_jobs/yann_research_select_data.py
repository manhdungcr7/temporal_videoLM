"""Freeze an independent human-annotated development pool before running model scores."""
import collections,hashlib,json,random
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/yann_temporal_research/data'
def load(n):return json.loads((D/n).read_text(encoding='utf-8'))
qa=load('mc_question_train.json');a=load('action_localisation_train.json');s=load('sound_localisation_train.json');cut=load('cut_frame_mapping_train.json')
idx={Path(x['name']).stem:x for x in load('train_video_zip_index.json')}
eligible=[]
for vid,row in sorted(qa.items()):
    if vid not in a or vid not in s or vid not in idx:continue
    m=row['metadata'];dur=m['num_frames']/m['frame_rate'];ad=m['audio_samples']/m['audio_sample_rate']
    if dur>45 or ad>45 or dur<8 or m['is_cup_game'] or idx[vid]['bytes']>12_000_000:continue
    qas=[q for q in row['mc_question'] if all(str(t).strip().lower() not in ('na','n/a') for t in q['options'])]
    acts=a[vid]['action_localisation'];sounds=[e for e in s[vid]['sound_localisation'] if e['label'] not in ('Other','Other:background sound','Object:sound')]
    if not qas or not acts or not sounds:continue
    eligible.append(dict(video_id=vid,metadata=m,duration=dur,audio_duration=ad,zip=idx[vid],questions=qas,actions=acts,sounds=sounds,cut_mapping=cut.get(vid)))
rng=random.Random(20261016);rng.shuffle(eligible)
assert len(eligible)>=48,len(eligible)
selected=eligible[:48]
for i,r in enumerate(selected):r['split']='train' if i<32 else 'validation' if i<40 else 'diagnostic'
result=dict(source='google-deepmind/perception_test official train',seed=20261016,eligible=len(eligible),n=48,
    prefilter='8–45s audio+video; non-cup; <=12MB raw video; human MC questions/actions/sounds available; no invalid na options',
    selection_note='Size/non-cup filters are compute/protocol constraints, not a representative random draw of the whole dataset.',
    rows=selected,total_raw_bytes=sum(r['zip']['bytes'] for r in selected),
    qa_counts={split:sum(len(r['questions']) for r in selected if r['split']==split) for split in ('train','validation','diagnostic')})
blob=json.dumps(result,ensure_ascii=False,indent=2).encode();(D/'perception48_manifest.json').write_bytes(blob)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));print('manifest_sha256',hashlib.sha256(blob).hexdigest())

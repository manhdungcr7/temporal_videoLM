"""Move exactly the one not-yet-archived ID after server1's tunnel vanished."""
import hashlib,json
from argon_remote import ROOT,upload,submit

folder=ROOT/'results/presentation_resample40'
manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
rows=[json.loads(x) for s in (1,2) for x in (folder/f'results_s{s}.jsonl').read_text().splitlines()]
missing=[r for r in manifest['selected'] if r['idx'] not in {x['idx'] for x in rows}]
assert len(missing)==1
idx=missing[0]['idx'];print('RECOVER_EXACT_ID',idx)
originals=[json.loads(x) for s in (1,2) for x in (ROOT/'results/argon_ccld40'/f'results_s{s}.jsonl').read_text().splitlines()]
original=next(r for r in originals if r['idx']==idx)
refs=folder/'recovery_reference_row.jsonl';refs.write_text(json.dumps(original)+'\n')
samples=folder/'recovery_sample.json';samples.write_text(json.dumps(missing))
upload(2,samples);upload(2,refs)
rotation_files=[]
for rotation in range(len(original['audit'])):
    path=ROOT/'results/argon_ccld40/rotations_s1'/f'{idx}_rot{rotation}.json'
    upload(2,path,'pc_recovery_'+path.name);rotation_files.append(path.name)
wrapper="""PC_SERVER=2
PC_LIMIT=None
_recovery_source=(OUT/'uploads/presentation_resample40.py').read_text()
assert _recovery_source.count("OUT/'argon_ccld40'")==2
_recovery_source=_recovery_source.replace("OUT/'argon_ccld40'","OUT/'presentation_resample_recovery_refs'")
exec(compile(_recovery_source.split('RESULT=pc_run()',1)[0],'presentation_same_math_recovery','exec'),globals())
_ref_dir=OUT/'presentation_resample_recovery_refs';_ref_dir.mkdir(exist_ok=True)
(_ref_dir/'results_s2.jsonl').write_bytes((OUT/'uploads/recovery_reference_row.jsonl').read_bytes())
SL_SAMPLES=json.loads((OUT/'uploads/recovery_sample.json').read_text())
for _rname in ROTATION_FILES:
    (_ref_dir/_rname).write_bytes((OUT/'uploads'/('pc_recovery_'+_rname)).read_bytes())
RESULT=pc_run()
print('RECOVERED_FIXED_ID',json.dumps(RESULT),flush=True)
""".replace('ROTATION_FILES',repr(rotation_files))
path=folder/'recovery_wrapper.py';path.write_text(wrapper)
job=dict(server=2,idx=idx,**submit(2,wrapper,'presentation_exact_missing_id_recovery'))
(folder/'recovery_job.json').write_text(json.dumps(job,indent=2))
(folder/'recovery_protocol.json').write_text(json.dumps(dict(
    reason='server1 tunnel offline ERR_NGROK_3200 at19/20 local rows',idx=idx,replacement=False,server=2,
    wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
    math_source_sha256=hashlib.sha256((ROOT/'server_jobs/presentation_resample40.py').read_bytes()).hexdigest(),
    references_from_archived_native=True),indent=2))
print(json.dumps(job))

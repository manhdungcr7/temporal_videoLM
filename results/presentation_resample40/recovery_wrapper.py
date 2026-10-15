PC_SERVER=2
PC_LIMIT=None
_recovery_source=(OUT/'uploads/presentation_resample40.py').read_text()
assert _recovery_source.count("OUT/'argon_ccld40'")==2
_recovery_source=_recovery_source.replace("OUT/'argon_ccld40'","OUT/'presentation_resample_recovery_refs'")
exec(compile(_recovery_source.split('RESULT=pc_run()',1)[0],'presentation_same_math_recovery','exec'),globals())
_ref_dir=OUT/'presentation_resample_recovery_refs';_ref_dir.mkdir(exist_ok=True)
(_ref_dir/'results_s2.jsonl').write_bytes((OUT/'uploads/recovery_reference_row.jsonl').read_bytes())
SL_SAMPLES=json.loads((OUT/'uploads/recovery_sample.json').read_text())
for _rname in ['1898_rot0.json', '1898_rot1.json', '1898_rot2.json']:
    (_ref_dir/_rname).write_bytes((OUT/'uploads'/('pc_recovery_'+_rname)).read_bytes())
RESULT=pc_run()
print('RECOVERED_FIXED_ID',json.dumps(RESULT),flush=True)

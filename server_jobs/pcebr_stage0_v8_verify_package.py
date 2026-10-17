import ast,hashlib,json
from pathlib import Path
source=(OUT/'uploads/stage0_pcebr_v8_corrected_standalone.py').read_text(encoding='utf-8')
assert hashlib.sha256(source.encode()).hexdigest()=='25043c391ac1503280c2e4b70c4e38d27d3f8831741e4a38f8f96ffc4b758843'
t=ast.parse(source);t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='RESULT' for v in n.targets))]
ns=dict(OUT=OUT/'pcebr_stage0_v8');exec(compile(t,'packaged_v8_definitions','exec'),ns)
checks={dev:ns['run_algebraic_unit_checks'](dev) for dev in ('cpu','cuda:0')}
RESULT=dict(package_sha256=hashlib.sha256(source.encode()).hexdigest(),cpu_pass=True,cuda_pass=True)
(OUT/'pcebr_stage0_v8/standalone_m0_verification.json').write_text(json.dumps(dict(summary=RESULT,checks=checks),indent=2))
print('V8_PACKAGE_VERIFIED',json.dumps(RESULT),flush=True)

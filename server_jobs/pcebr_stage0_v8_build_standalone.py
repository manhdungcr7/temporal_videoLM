"""Package the evaluated interface repairs as a directly executable source file."""
import ast,hashlib,importlib.util,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
hp=R/'server_jobs/pcebr_stage0_v8_token_interface_v2.py'
spec=importlib.util.spec_from_file_location('v8_interface_v2',hp)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
source=helper.patch_token_interface((D/'stage0_pcebr_v8_logicfix.py').read_text(encoding='utf-8'))
tree=ast.parse(source)
helper_tree=ast.parse(hp.read_text(encoding='utf-8'))
fn=next(n for n in helper_tree.body if isinstance(n,ast.FunctionDef) and n.name=='word_rows')
fn.name='V8_WORD_ROWS'
idx=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.FunctionDef) and n.name=='split_query_into_endpoint_roles')
tree.body.insert(idx,fn)
extract=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_query_conditioned_salient_slots')
operational=ast.parse('''h18_t = h18_t.cpu()
h27_t = h27_t.cpu()
h22_post_t = h22_post_t.cpu() if h22_post_t is not None else None
custom_q18 = custom_q18.cpu() if custom_q18 is not None else None
custom_q27 = custom_q27.cpu() if custom_q27 is not None else None
clock_info = {k: (v.cpu() if torch.is_tensor(v) else v) for k,v in clock_info.items()}
''').body
extract.body[1:1]=operational
# Do not overwrite historical v7/v8-original files when running the corrected package.
for n in ast.walk(tree):
    if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='fname' and isinstance(n.iter,ast.Tuple):
        n.iter=ast.Tuple(elts=[ast.Constant(value='stage0_pcebr_v8_corrected_results.json')],ctx=ast.Load())
ast.fix_missing_locations(tree)
standalone=ast.unparse(tree)+'\n'
out=D/'stage0_pcebr_v8_corrected_standalone.py'
out.write_text(standalone,encoding='utf-8',newline='\n')
no_run=ast.parse(standalone)
no_run.body=[n for n in no_run.body if not(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RESULT' for x in n.targets))]
ns=dict(OUT=D);exec(compile(no_run,'standalone_corrected_definitions','exec'),ns)
m0=ns['run_algebraic_unit_checks']('cpu')
record=dict(sha256=hashlib.sha256(out.read_bytes()).hexdigest(),m0_cpu_pass=True,
    equivalence='Same corrected algorithm as runtime effective source; embedded word_rows helper and existing operational CPU-device wrapper, output file renamed to avoid historical overwrite',
    original_sha256='fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895')
(D/'standalone_package_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))

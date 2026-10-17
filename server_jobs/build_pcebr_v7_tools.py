"""Prepare transport/observation wrappers; never modify the supplied algorithm."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
S=R/'server_jobs'
def convert(name):
    return (S/name).read_text(encoding='utf-8').replace('v6','v7').replace('V6','V7').replace('PCV6','PCV7')
manage=convert('pcebr_stage0_v6_manage.py').replace("ROOT / 'results/pcebr_stage0_v7_final_preflight/stage0_pcebr_v7_final_original.py'","ROOT / 'results/pcebr_stage0_v7/stage0_pcebr_v7_final_original.py'")
(S/'pcebr_stage0_v7_manage.py').write_text(manage,encoding='utf-8')
(S/'pcebr_stage0_v7_prepare.py').write_text(convert('pcebr_stage0_v6_prepare.py'),encoding='utf-8')
run=convert('pcebr_stage0_v6_run.py').replace('6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515','d27eb87a7014e941bd7a7a920dbe3ce074338cf616757608cc476712d7c8ee80')
run=run.replace("keys = ('V18_raw'", "keys = ('clc_dir_fwd', 'clc_dir_rev', 'h22_supp', 'h22_foil', 'h22_rev', 'dh_ev', 'dh_ord', 'V18_raw'")
run=run.replace("raw = {k: v7_cpu(v[k]) for k in keys}","raw = {k: v7_cpu(v[k]) for k in keys if k in v}")
run=run.replace("'hook_option_disjoint=", "'hook_option_disjoint=")
# Existing v6 helper puts the callback before only the last return. V7 has an earlier return_model branch.
start=run.index('for node in ast.walk(V7_TREE):')
end=run.index('ast.fix_missing_locations(V7_TREE)',start)
run=run[:start]+'''class V7Observe(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name == 'eval_linear_probe':
            class ProbeReturns(ast.NodeTransformer):
                def visit_Return(self,ret):
                    call=ast.parse('V7_CAPTURE_PROBE(locals())').body[0]
                    return [call,ret]
            node=ProbeReturns().visit(node)
        return self.generic_visit(node)
    def visit_For(self,node):
        self.generic_visit(node)
        body=[]
        for statement in node.body:
            body.append(statement)
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
                c=statement.value
                if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='rows_out' and c.func.attr=='append':
                    body.append(ast.parse('V7_CAPTURE_M3(locals())').body[0])
        node.body=body
        return node
V7_TREE=V7Observe().visit(V7_TREE)
''' +run[end:]
start=run.index('    # Independent validation only;')
end=run.index("    v7_write('progress.json', V7_STATE)",start)
run=run[:start]+run[end:]
old="""    V7_STATE['M2']=V7_NS['run_stage0_m2_connected'](V7_BUNDLES['avbind_pooled_pairs'])
    v7_write('progress.json',V7_STATE)
    print('PCV7_M2_COMPLETE',json.dumps(V7_STATE['M2']),flush=True)
    V7_STATE['M3']=V7_NS['run_stage0_m3_bivector_vo_tangent_routing']()"""
new="""    V7_STATE['M2'],V7_PROBE,V7_BRIDGE=V7_NS['run_stage0_m2_connected'](V7_BUNDLES['avbind_pooled_pairs'])
    torch.save(v7_cpu(dict(probe={k:(x.state_dict() if k=='clf' else x) for k,x in V7_PROBE.items()},bridge=V7_BRIDGE.state_dict())),V7_DIR/'artifacts/connected_m2_models.pt')
    v7_write('progress.json',V7_STATE)
    print('PCV7_M2_COMPLETE',json.dumps(V7_STATE['M2']),flush=True)
    V7_STATE['M3']=V7_NS['run_stage0_m3_bivector_vo_tangent_routing'](m1_aligner=V7_BUNDLES['avbind_pooled_pairs']['m_c6'],m2_probe_bundle=V7_PROBE,hop2_bridge=V7_BRIDGE)"""
assert old in run
run=run.replace(old,new)
run=run.replace("        v7_write('progress.json',V7_STATE)","        torch.save(v7_cpu({k:x.state_dict() for k,x in bundle.items() if k.startswith('m_')}),V7_DIR/'artifacts'/f'{V7_PREFIX}_model_states.pt')\n        v7_write('progress.json',V7_STATE)")
(S/'pcebr_stage0_v7_run.py').write_text(run,encoding='utf-8')
for name in ('pcebr_stage0_v7_manage.py','pcebr_stage0_v7_prepare.py','pcebr_stage0_v7_run.py'):
    compile((S/name).read_text(encoding='utf-8'),name,'exec')
print('V7 transport and observation wrappers ready')

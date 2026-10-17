"""Observe every original M0 assertion after the unchanged M0 failed."""
import ast
import json
import torch

v7_m0_checks=[]
def v7_m0_check(ok,line,message,scope):
    keys=('bimod_l1_w','bimod_std','psi_l1_diff','psi_sync','psi_bimod',
          'j_aug_row_marg_err','j_aug_col_marg_err','row_marg_err','col_marg_err',
          'real_block_consistency_err','aug_dust_mass_err','spurious_fact_mass','spurious_joint_mass')
    values={k:(v7_cpu(scope[k]).tolist() if torch.is_tensor(scope[k]) else scope[k]) for k in keys if k in scope}
    row=dict(line=line,passed=bool(ok),message=message,values=values)
    v7_m0_checks.append(row)
    print('PCV7_M0_ASSERT',json.dumps(dict(line=line,passed=bool(ok),message=message)),flush=True)

v7_m0_ast=next(n for n in ast.parse(V7_ORIGINAL).body if isinstance(n,ast.FunctionDef) and n.name=='run_algebraic_unit_checks')
class V7M0Assertions(ast.NodeTransformer):
    def visit_Assert(self,node):
        call=ast.Expr(value=ast.Call(func=ast.Name(id='V7_M0_CHECK',ctx=ast.Load()),
            args=[node.test,ast.Constant(value=node.lineno),node.msg if node.msg is not None else ast.Constant(value=ast.unparse(node.test)),
                  ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[]))
        return ast.copy_location(call,node)
v7_m0_ast=V7M0Assertions().visit(v7_m0_ast)
v7_m0_ns=dict(V7_NS,V7_M0_CHECK=v7_m0_check)
exec(compile(ast.fix_missing_locations(ast.Module(body=[v7_m0_ast],type_ignores=[])),'m0_assertion_diagnostic','exec'),v7_m0_ns)
v7_m0_output=v7_m0_ns['run_algebraic_unit_checks']('cuda:0')
v7_write('m0_diagnostic.json',dict(original_m0_status='failed',assertions=v7_m0_checks,observed_original_return=v7_m0_output,
    note='Assertions observed without raising solely to inspect all toy checks; not an M0 pass and no algorithm feature changed.'))
print('PCV7_M0_DIAG_DONE',json.dumps(dict(failed_lines=[x['line'] for x in v7_m0_checks if not x['passed']],observed_return=v7_m0_output)),flush=True)

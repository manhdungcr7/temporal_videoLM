source=(OUT/'uploads/avbind_eval_structural.py').read_text()
source=source.replace('avbind_structural_holdout.json','avbind_structural_holdout_r2.json')
source=source.replace('avbind_structural_eval.jsonl','avbind_structural_eval_r2.jsonl')
exec(compile(source,'avbind_eval_structural_r2.py','exec'),globals())

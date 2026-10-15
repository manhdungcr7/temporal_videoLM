source=(OUT/'uploads/avbind_eval_structural.py').read_text().replace('avbind_structural_holdout.json','avbind_structural_holdout_r3.json').replace('avbind_structural_eval.jsonl','avbind_structural_eval_r4_address.jsonl')
exec(compile(source,'avbind_eval_structural_r4.py','exec'),globals())

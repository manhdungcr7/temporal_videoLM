source=(OUT/'uploads/avbind_neural_pilot.py').read_text().replace('avbind_neural_pilot_s','avbind_native_fast_r3_s').replace("NEURAL_NS['avbind_neural_answer']","NEURAL_NS['avbind_native_answer']")
exec(compile(source,'avbind_native_fast.py','exec'),globals())

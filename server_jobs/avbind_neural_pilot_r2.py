source=(OUT/'uploads/avbind_neural_pilot.py').read_text()
source=source.replace('avbind_neural_pilot_s','avbind_neural_pilot_r2_s')
exec(compile(source,'avbind_neural_pilot_r2.py','exec'),globals())

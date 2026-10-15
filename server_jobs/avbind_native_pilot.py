"""Same raw AV and memory, with/without adapter; never route by task."""
source=(OUT/'uploads/avbind_neural_pilot.py').read_text()
source=source.replace('avbind_neural_pilot_s','avbind_native_pilot_s')
source=source.replace("NEURAL_NS['avbind_neural_answer']","NEURAL_NS['avbind_native_answer']")
exec(compile(source,'avbind_native_pilot.py','exec'),globals())

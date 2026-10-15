"""Generic mandatory raw-evidence check; same samples and no task routing."""
source=(OUT/'uploads/avbind_pilot.py').read_text()
source=source.replace('avbind_pilot_r1_s','avbind_pilot_r3_s')
exec(compile(source,'avbind_pilot_r3.py','exec'),globals())

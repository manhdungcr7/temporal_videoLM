"""Repeat the same diagnostic samples with valid neural tool decisions."""
source=(OUT/'uploads/avbind_pilot.py').read_text()
source=source.replace('avbind_pilot_r1_s','avbind_pilot_r2_s')
exec(compile(source,'avbind_pilot_r2.py','exec'),globals())

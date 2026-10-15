"""Counterexample corpus with randomized event rank, duration and concurrence.

R3 is an experimental textual adapter, not learned raw-media alignment.
"""
source=(OUT/'uploads/avbind_train_structural_r2.py').read_text()
source=source.replace('2026101404','2026101405')
source=source.replace('_r2','_r3')
# The corpus is the same across both servers. Server 2 is a matched-budget
# answer-loss-only ablation, not an extra benchmark-tuned candidate.
base=(OUT/'uploads/avbind_train_structural.py').read_text()
base=base.replace('0.25*address_loss+0.1*null_loss',
                  "float(globals().get('TRAIN_ADDRESS_WEIGHT',1.0))*address_loss+float(globals().get('TRAIN_NULL_WEIGHT',0.25))*null_loss")
base=base.replace("training_seed=TRAIN_SEED,training_size=TRAIN_SIZE,steps=STEPS,",
                  "training_seed=TRAIN_SEED,training_size=TRAIN_SIZE,steps=STEPS,address_weight=float(globals().get('TRAIN_ADDRESS_WEIGHT',1.0)),null_weight=float(globals().get('TRAIN_NULL_WEIGHT',0.25)),")
# Give the wrapper this modified base without overwriting its audited source.
temporary=OUT/'uploads/avbind_train_structural_r3_base.py'
temporary.write_text(base,encoding='utf-8')
source=source.replace("uploads/avbind_train_structural.py","uploads/avbind_train_structural_r3_base.py")
exec(compile(source,'avbind_train_structural_r3.py','exec'),globals())

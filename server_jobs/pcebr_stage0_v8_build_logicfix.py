"""Generate explicit, auditable endpoint-to-physical-modality routing corrections."""
from pathlib import Path
import ast, hashlib, json
R=Path(__file__).resolve().parents[1]
D=R/'results/pcebr_stage0_v8'
s=(D/'stage0_pcebr_v8_original.py').read_text(encoding='utf-8')
assert hashlib.sha256(s.encode()).hexdigest()=='fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895'
old="""    v_slot_ids = slot_meta['v_token_slot_ids']
    a_slot_ids = slot_meta['a_token_slot_ids']
"""
new="""    # Endpoint 1/2 are not physical audio/video labels. Project their slot
    # evidence and order profiles to each physical modality using inferred rho.
    # Preserve the original A->V limit exactly at rho1A=1, rho2A=0.
    rho1a, rho2a = slot_meta.get('endpoint_modality_weights_A', [1.0, 0.0])
    rho1v, rho2v = 1.0-rho1a, 1.0-rho2a
    def endpoint_to_physical(e2, e1):
        return ((rho2v*e2+rho1v*e1)/max(1.0,rho1v+rho2v),
                (rho2a*e2+rho1a*e1)/max(1.0,rho1a+rho2a))
    # Apply projection after the original endpoint formulas, before scatter.
    # Branch-specific profile controls inherit the same projection.
    log_ev_v, log_ev_a = endpoint_to_physical(log_ev_v, log_ev_a)
    slot_ord_v, slot_ord_a = endpoint_to_physical(slot_ord_v, slot_ord_a)
    slot_meta = dict(slot_meta)
    slot_meta['token_innov_v'] = min(1.0,rho1v+rho2v)*slot_meta['token_innov_v']
    slot_meta['token_innov_a'] = min(1.0,rho1a+rho2a)*slot_meta['token_innov_a']
    v_slot_ids = slot_meta['v_token_slot_ids']
    a_slot_ids = slot_meta['a_token_slot_ids']
"""
assert s.count(old)==1
s=s.replace(old,new)
# For deictic same-modality questions, inactivity of an irrelevant other stream
# must not zero both endpoint evidence gates. Cross-modal limit is unchanged.
old='        fg_mod_gate = mod_gate_v * mod_gate_a'
new='        fg_mod_gate = (rho_1_V*mod_gate_v+rho_1_A*mod_gate_a) * (rho_2_V*mod_gate_v+rho_2_A*mod_gate_a)'
assert s.count(old)==1
s=s.replace(old,new)
ast.parse(s)
(D/'stage0_pcebr_v8_logicfix.py').write_text(s,encoding='utf-8',newline='\n')
record=dict(original_sha256=hashlib.sha256((D/'stage0_pcebr_v8_original.py').read_bytes()).hexdigest(),
    logicfix_lf_sha256=hashlib.sha256(s.encode()).hexdigest(),
    changes=['project endpoint slot/order profiles into physical audio/video token banks using rho; suppress innovation of irrelevant modality',
             'deictic foreground gate uses endpoint modality weights rather than unconditional audio*visual product'],
    status='Exploratory logic correction, frozen before observing micro8 outcome; original run retained separately',
    fixed_hyperparameters=True)
(D/'logicfix_spec.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))

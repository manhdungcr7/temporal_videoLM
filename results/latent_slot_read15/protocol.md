# Readout-stage ablation, no latent refit

Recorded2026-10-15 before this phase starts. Historical repo audits found an
evidence-reading window around blocks15–24, with some late-only interventions
ineffective. The slot optimizer fits block22 features and initially makes memory
visible only from block23. Test whether this is too late for QA.

Reuse the EXACT saved slot parameters from latent_slot_opt40. No new optimizer,
teacher, target, source/time layout, norm constraint, memory bias or question
change. Only change the first memory-visible block23->15. E0 native; E1 raw
unoptimized memory read15; E2 fitted memory read15; E3 time-shuffled target memory
read15. E4/E5 are copied matched read23 references, not rerun candidates.

Before comparing, reconstruct read23 logits for all three memories on one
rotation per worker and require max log-probability error<=.06. Earlier memory
features were not explicitly supervised at block14; an early-stage improvement
or regression cannot establish that block22 alignment itself transfers to every
earlier representation. Source/time supervision and temporal reasoning remain
different claims.

Use the same fixed40<=45s and every cyclic rotation. No task router, answer-word
clock, VAD window or modality handoff. Report every configuration and per-cell
correct/wrong, with paired comparisons to native and read23. Never refit or
select parameters from a gold label. This is repeatedly inspected development.

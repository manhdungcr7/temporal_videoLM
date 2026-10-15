# Question-conditioned continuous feedback, prospective development40

Recorded2026-10-15, before Q jobs run, after inspecting CCLD partial traces.
Run after the symmetric controls on each server. Same frozen40, five per cell,
<=45s, all cyclic answer rotations, native AV caches, labels only for reporting.

Motivation: Argon's global thought ranks generic visual/audio novelty, without
conditioning on the question. Its recurrence also includes choice/format rows.
Test a question-conditioned continuous prototype instead of a generic scene
prototype. This is a new experiment here, not a claim that weighted embeddings
or activation steering are novel primitives.

Capture native block22 states at the exact question stem, before choices.
Pool that stem under AV and under a neither-source decoder control. Map each
pooled state through the existing norm/lm_head/top48/temperature.65 expected
embedding bridge. Q1 uses Argon's global prototype; Q2 the AV question prototype;
Q3 the normalized difference between AV and neither prototypes; Q4 adds the
same-layer projected AV interaction from the preceding experiment; Q5 uses a
fixed dimension permutation of Q3, preserving norm. All thoughts enter only
stem rows at blocks0/1/2, gain.20. Q4's block22 delta is norm bounded at.15 and
never inflated. Q0 is the matched native baseline.

No speech window, Yes/Past/Present/Future vector, task classifier, fixed source
order, external captions or gold calibration. Both sources remain available.
The neither control retains the structural clock, but prevents content paths
from either source to non-media rows. Prototype subtraction is not an identified
causal decomposition. A small difference can still be noise; Q5 tests direction.

Prefix KV may be reused only when native replay error<=.03; otherwise full
execution. On the first eligible rotation per worker, compare every cached
intervention with full execution and use full scores for failures or argmax
disagreement. Native replay tolerance against saved encoding<=.12. All errors
and fallbacks saved. Report every condition, per-cell correct/wrong, TF counts,
paired rescue/regression and controls; no held-out or CVPR claim from this pilot.

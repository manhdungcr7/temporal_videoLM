# Same-layer symmetric latent correction — prospective protocol

Recorded2026-10-15 while CCLD is queued/running. Execute only after Argon's
eight configurations on each server. Same frozen40, every cyclic answer order,
same native AV decoder cache, no sample replacement and no label-dependent rule.

Motivation: Argon's visual-only difference is asymmetric, transports block22
states to earlier blocks, and normalizes every difference to a fixed fraction
of query norm even when its original magnitude is negligible. Earlier controls
already tried interaction correction on output logits without reliable gains.
This experiment tests a different intervention location and geometry.

At blocks18/22, collect query states under native AV, visual-only, audio-only
and neither-source causal masks. Define interaction I=Hav-Hv-Ha+H0. This algebraic
contrast is NOT asserted to be pure synergistic information or an identified
causal decomposition. Project I orthogonally to the span of (Hv-H0,Ha-H0) per
query row using rank-aware SVD; this is a geometric hypothesis, not proof that
all nuisance semantics have been removed.

S0 native. S1 visual contrast injected back at its own block22 (diagnostic).
S2 interaction at block22. S3 projected interaction at block22. S4 projected
interaction at its own blocks18 and22. Each delta is clipped to at most.15 of
the current query norm (.10 per block for S4) and is never amplified. Original
media states, full AV access, causal order and native RoPE remain unchanged in
the final pass. No speech window, generated captions, answer-word vector, PMI,
question-type router, or fixed audio->visual/visual->audio sequence.

All parameters shared across questions. Report every condition, TF prediction
counts, paired rescues/regressions, native replay and projection retention.
Potential harms: a useful unimodal signal can be removed; content interaction
does not guarantee correct relative-time reasoning. Free-form and broader
contexts remain unvalidated. No claim of CVPR novelty from this pilot.

Runtime optimization recorded before either S shard starts: final interventions
only touch query rows, so the causal native media prefix is invariant and its
KV can be reused. Counterfactual branches still recompute their entire media
context. Validate native tail replay on every rotation (max error<=.03) and
validate every intervened tail against full execution on the first rotation
per worker, including identical argmax. Record all errors; fail rather than
silently switch implementations. Native baseline is still a full native pass.

Additional controls recorded before either S shard starts: CCLD treats the whole
tail after the last media token as query rows, including choices and assistant
format tokens. S5 restricts the S3 intervention to the exact question stem;
S6 uses Argon C3's block22 visual delta and original cross-layer norm scaling,
but restricts it to that same stem. S6 is asymmetric and diagnostic, not the
general candidate. The stem is located by an exact lexical embedding sequence
match; no keyword, task or label classifier is used. Refuse a mismatched stem.
S1–S5 retain the no-amplification rule; only S6 reproduces Argon's rescaling.

Implementation-v1 diagnostic: the third S1-worker sample had native-prefix
replay error.03125, exceeding the prospectively fixed.03 tolerance. That job
stopped as specified; two completed diagnostic rows and the error are archived
under implementation_v1. No scientific threshold or intervention parameter is
relaxed. V2 keeps the same.03 check and uses a full decoder pass for a rotation
when the prefix replay fails; if the first intervention parity check fails,
use that full reference for that condition. Record every fallback. Start both
shards in separate remote symmetric_latent40_v2 artifacts to avoid mixing runs.

Relevant prior art:
- RED https://arxiv.org/abs/2610.02976 already decomposes audio/video and residual
  interaction contributions at decoding; an interaction formula alone is not new.
- Activation Steering Decoding https://aclanthology.org/2025.acl-long.634/ already
  uses hidden-state interventions to mitigate visual hallucination.
- Information-Theoretic Decomposition for Multimodal Interaction Learning,
  CVPR2026, https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Information-Theoretic_Decomposition_for_Multimodal_Interaction_Learning_CVPR_2026_paper.pdf
  studies decomposition through training; geometric state subtraction is not PID.

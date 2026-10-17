# CLRV v8 updated source — preflight review

Date: 2026-10-17

## Verdict

The new Downloads source is a real code revision. Its raw SHA-256 is `fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895`, exactly matching the hash in the user's pasted implementation note. The connector-free split now genuinely searches contiguous cuts using combined hidden-state and input-embedding features, and the new gate, row/column invariance, four-route branches, timestamp disclaimers and rollout controls are present.

However, the supplied proposal artifact was **not updated with this revision**. Both `proposal_stage0_v8_clrv_to_yann_lecun.md` and the attachment in this turn still advertise the previous hash `47a695...` and the older line map. This is a material provenance/spec mismatch: the prose says the updated source and proposal are synchronized, while the proposal file actually identifies the previous code. The detailed review below compares the new source to the attached proposal and the implementation note. No M3/GPU experiment was run here; the claimed M0 output was not independently executed in this audit.

The code copy reviewed here is [stage0_pcebr_v8_download_candidate.py](stage0_pcebr_v8_download_candidate.py). The existing proposal copies are in Downloads and remain at the old revision/hash.

## Material code/proposal mismatch: proposal is stale

The supplied proposal says the locked code SHA is `47a69542388aaec66b909deea09dd3bc945bc92c8c6f0343d3111a6bc6aaf8d6` (proposal line 13); the downloaded Python source hashes to `fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895`. The proposal also describes the earlier endpoint and M0 tests, while the new source has changed function signatures, line numbers, tests, and metrics. The user-provided implementation note says a revised proposal was synchronized, but the attached proposal and both Downloads proposal files are byte-identical 18,089-byte old copies. Therefore “Code–Proposal 100% synchronized” is not currently verifiable and is false for the supplied artifacts. Update the proposal from the new source, recalculate its citations/line numbers and hash, then freeze both artifacts together.

## New contiguous split: implementation present, semantic claim remains unproven

The old midpoint bug is fixed in code. `split_query_into_endpoint_roles` now builds normalized token features from centered input embeddings and optionally centered layer-22 hidden states, then searches all contiguous cuts for minimum within-cluster SSE (`:1083–1123`). That is consistent with the implementation note's described objective.

But the M0 asymmetric 4+2 test is deliberately constructed: it injects four copies of one hidden vector and two copies of a shifted vector (`:3048–3053`) for the phrase “pour boiling water pot lemon zest,” then checks that the cut is 4+2 (`:3061–3067`). This establishes that the optimizer can recover the planted split. It does not establish that hidden/embedding SSE cuts correspond to semantic event boundaries in naturally phrased questions. The algorithm also always makes a contiguous two-way partition when there are at least two specific tokens and no recognized connector; it has no reject/uncertain option or estimate that the question actually denotes two events. Queries listing one event with modifiers, e.g. “person pours boiling water into a pot,” can be split into two pseudo-endpoints. This can corrupt modality routing and temporal pairing. Keep the claim to “a heuristic contiguous two-cluster cut” until tested against independently annotated endpoint spans and one-event/ambiguous queries.

## Four modality regimes: routing branches exist, regime detection is still heuristic

The new source has endpoint-specific banks and same-modality scoring branches (`:1430–1450`, `:1519–1530`). It fixes the prior A×V-only implementation. Regime assignment still uses a finite English cue vocabulary plus max-affinity difference between audio and visual manifolds (`:1402–1429`). For unresolved/deictic clauses it imposes cross-modal complementarity (`:1412–1420`). M0's four prompts explicitly name “audio,” “video,” “speech,” or “shown” (`:3044–3063`), so passing them verifies those constructed cases, not broad natural-language coverage. The current code may support each regime structurally; robustly inferring the correct regime for implicit, ambiguous, negated, multi-clause, repeated-event, or non-English questions is a separate untested claim.

## Gate: code now exposes per-query false activations, but the sample is tiny and hand-authored

The 0.65 floor is removed. M0 now exposes each of the four absent-query gates plus max and threshold rates (`:3038–3042`, `:3111–3118`); M3 evaluates all four fixed wrong-event templates per sample (`:2008–2070`, `:2626–2632`). This is a useful correction over reporting only the mean. The four templates remain manually chosen, task-shaped English negatives, and M0 contains synthetic controls. `FPR` over four prompts or 8×4 prompt/sample combinations is not an estimate of deployment false-positive rate, especially when prompts share templates and sample origins. Treat it as a diagnostic threshold-sensitivity table, not calibrated presence detection. Independent event-absence labels are still required.

## Double-centering and null score

The unweighted double-centering implementation is mathematically invariant to adding arbitrary row/column effects (`:532–536`, `:279–283`) and the added algebraic controls directly check that invariance. This resolves the previous additive-gauge counterexample. The proposal correctly calls `P_null` a heuristic rather than an absence probability. The synthetic monotonic compatible/neutral/incompatible test does not show that real-event null scores are calibrated or discriminate true absent events.

## Prompt positions and clock provenance

The source now preserves original processor position IDs for media plus retained question stem, and shifts only the suffix after removal of options (`:1833–1851`). It correctly records that the option-free prompt is a deliberate transform (`:2646–2649`). This is materially improved, though removal of option/interstitial tokens still means the resulting assistant context differs from native full prompt; retain a native-position/control comparison.

The source also says its temporal clock is a uniform-grid interpolation, not verified presentation timestamps (`:821–884`, report field at `:2648`). Thus temporal lag is an estimate on that grid, not physical timestamp ground truth. Prior v7 audits documented clock/grid/cache discrepancies on the micro set. Do not claim physically exact synchrony without validating frame/audio PTS.

## Existing evidence against the hypotheses (not a v8 performance falsification)

There is no v8 result in the repository yet, so old results cannot falsify v8 accuracy. They do challenge predecessor assumptions:

- v7 micro8 native, CAOR-only and full all scored 5/8; the joint branch had zero audio gate and zero Ψ on those samples. This motivates v8's new audio/manifold logic but does not validate it. [v7 report](../pcebr_stage0_v7/report.md)
- v7 two-hop roll scored 57/156 (36.5%) full, 61/156 (39.1%) no-Ψ, and 58/156 (37.2%) matched 6-D Φ; factorized/no-Sinkhorn had identical predictions. This weakens causal claims for the v7 Sinkhorn/Ψ design, not the revised v8.
- v7 semantic-wrong gate averaged 0.9640 against 0.4032 true-query gate. The input-embedding gate is a meaningful new hypothesis, but hand-authored absent prompts cannot establish general calibration.
- The historical freeform output for micro ID 2150 stated the correct “right after the speech” relation while MC remained wrong. That motivates a rollout bridge; it does not establish a bridge gain. The v8 order-reversal, option-permutation and trace-only controls are useful, but require blinded/independent semantic grading and fresh evaluation to rule out answer-format exploitation. See [Raman et al. (PMLR 2026)](https://proceedings.mlr.press/v306/raman26a.html).
- Qwen2.5-Omni uses TMRoPE for synchronized multimodal positions, so prior unchanged argmax outputs do not establish that the backbone lacks temporal information ([technical report](https://arxiv.org/abs/2503.20215)).

## Recommendation

The previous spectral/midpoint implementation defect has been addressed, and the latest source is internally more careful about limits. Do not run or report it as a proposal-locked experiment until the actual proposal file is regenerated to match SHA `fc0a...`. After that, M3 can be run as a diagnostic, but report it as small-sample mechanism evaluation, not proof of generality or CVPR readiness. The 4+2 M0 check is a planted synthetic recovery test; endpoint accuracy, modality routing, gate calibration and QA improvement still need independent natural examples across A→V, V→A, A→A and V→V, with repeated and absent events.

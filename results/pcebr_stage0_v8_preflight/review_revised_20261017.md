# CLRV v8 revised — preflight review

Date: 2026-10-17

## Verdict

The revised source attachment is complete and its normalized-LF SHA-256 is `47a69542388aaec66b909deea09dd3bc945bc92c8c6f0343d3111a6bc6aaf8d6`, matching the hash in Yoshua's response. The five previously identified implementation defects have been addressed in this version: all four modality-pair branches exist, the 0.65 gate floor is removed, pair compatibility is double-centered, the option-free prefix preserves original TMRoPE positions for the retained stem, and three bridge controls were added.

I found one material proposal/source mismatch still present and two consequential limits on what the new M0 checks establish. No GPU experiment was run for this review. The source is saved as [stage0_pcebr_v8_revised_candidate.py](stage0_pcebr_v8_revised_candidate.py); the previous `review_20261017.md` reviewed the earlier source and is superseded here.

## Remaining material mismatch: connector-free split is sequential, not spectral

The proposal says that, without a recognized connector, the code uses a hidden-state/spectral split to find a cut minimizing cross-clause cosine similarity (proposal section 2.1). The implementation does not do that. In `split_query_into_endpoint_roles`, when `h_ref_t` is available it sets `mid_s = len(specific_rows) // 2`, chooses the token at that midpoint as `pivot_row`, then partitions by token index (`stage0_pcebr_v8_revised_candidate.py:1073–1084`). There is no PCA, spectral objective, or cosine minimization in that branch; `h_ref_t` is only used as a non-None condition.

The code does guarantee disjoint token sets in that branch, but disjointness is not evidence that the two sets denote the two event endpoints. A query with multiple entities, modifiers, or a different word order can be cut through one event description. This affects endpoint identity, modality inference, slot selection, and therefore all downstream temporal scores. Either implement the claimed representation-based split and test it against semantic endpoint annotations, or revise the proposal to call it a midpoint lexical partition and narrow the claim.

## Four regimes are implemented, but the M0 test does not establish general query coverage

The new implementation now constructs endpoint-specific banks and a pair matrix, including same-modality cases (`:1430–1450`, `:1519–1530`). This fixes the earlier unconditional A×V matrix. However, the regime classifier comes from lexical cue counts plus the difference between each endpoint query's maximum similarity to the audio and visual manifolds (`:1047–1091`, `:1360–1385`). If both are weak or ambiguous, code forces an opposite-modality default (`:1370–1375`). Thus support for A→A/V→V is a capability of the routing branches, not evidence that arbitrary user questions will be assigned to the right branch.

M0 checks four hand-written English prompts with explicit modality words such as “audio”, “video”, “speech”, and “shown” (`:2978–2989`). It confirms those prompts hit the expected labels; it does not test implicit references, paraphrases, negation, multiple clauses, repeated events, multilingual questions, or modality ambiguity. The connector-free example is a bag of event words, and the assertion checks only `disjoint_endpoint_roles`, not whether either partition refers to the intended event. This remains a substantive generality risk given the requirement to support open-ended questions.

## Pair compatibility: algebraic fix is valid, interpretation still needs care

The prior row/column additive counterexample is addressed: the code double-centers the compatibility matrix (`:532–536`, `:279–283`) and tests invariance under added row/column terms. That is a valid invariance to additive endpoint-only effects. It does not calibrate the resulting dustbin score as an event-absence probability; the proposal now correctly calls it a heuristic null score. Keep that terminology in the report. The M0 monotonic compatible/zero/incompatible result is a constructed probe, not evidence that real absent events receive calibrated scores.

## Gate controls: improvement, but the stated absent-query result is only an average

The 0.65 floor is gone (`:1493–1515`), and the code can return zero. The attached response reports absent-event gates `[0.0, 0.0, 0.1534, 0.0]`, mean `0.0384`. The mean is low, but one of four is not near zero; reporting only the mean can hide a false activation. Show per-query gates and thresholded false-positive rate, and test on independent absent-event annotations. The M0 checks use four hand-written absent-event questions, so they do not establish general open-vocabulary presence calibration.

## TMRoPE and temporal clock

The revised option-free prepass preserves source `position_ids` for the retained media prefix and question stem, then shifts the trailing assistant segment to close the removed options gap (`:1833–1851`). This resolves the earlier concern about overwriting question-stem positions. The sequence still removes option/interstitial tokens and changes assistant positions by design, so the no-options prompt is a transformed input; parity to that intended transform is not full equivalence to the original prompt. Report the transform and retain the native-position control.

The event clock remains explicitly a uniform-grid interpolation rather than frame timestamp oracle (`:821–884`). Therefore lag and synchrony are grid-calibrated estimates, not verified physical AV timestamps. The earlier v7 report documented clock/grid/cache discrepancies on the micro set. This does not invalidate a controlled experiment, but physical-time claims need timestamp validation.

## What existing results actually challenge

There is no v8 GPU result yet, so the repo cannot falsify v8's performance. Existing v7 data challenge predecessor mechanisms and motivate stronger controls:

- v7 micro8: native, CAOR-only, and full all scored 5/8, with no rescued answers. The joint branch had zero audio gate/zero Ψ on those eight samples, so v7 did not demonstrate its proposed end-to-end joint mechanism. This is not a test of v8's new intrinsic audio salience or four-regime routing. See [v7 report](../pcebr_stage0_v7/report.md).
- v7 two-hop roll: full scored 57/156 (36.5%), no-Ψ scored 61/156 (39.1%), matched 6-D Φ scored 58/156 (37.2%); factorized/no-Sinkhorn had identical predictions to full. This weakens claims that v7's chosen Ψ/Sinkhorn design caused a gain, but does not refute v8's changed method.
- v7 wrong-context gate mean was 0.9640 versus 0.4032 for true queries. That is evidence against the old gate; v8's clean input-embedding/manifold gate is a meaningful new hypothesis, still requiring real-data validation.
- In the earlier micro8 freeform output, sample 2150 expressed the correct “right after the speech” relation while the MC answer was wrong. This supports testing rollout-to-choice as a hypothesis. It does not show that the bridge improves accuracy or avoids option/format exploitation. The new order-reversal, option-permutation, and trace-only controls are useful; report them alongside the final score, and use independent semantic grading. Recent work on reasoning traces and test exploitation reinforces that requirement ([PMLR 2026](https://proceedings.mlr.press/v306/raman26a.html)).
- Qwen2.5-Omni uses TMRoPE for synchronized multimodal positions, so prior unchanged argmaxes do not prove temporal information is absent from hidden states ([technical report](https://arxiv.org/abs/2503.20215)).

## Recommendation

The revised version is materially better specified than the previous v8 attachment, and its source/proposal are largely aligned. Before presenting it as a general joint AV temporal method, fix or accurately describe the connector-free midpoint partition and test endpoint/regime assignment on diverse, independently annotated questions. In any GPU result, separate algebraic M0 checks from M3 QA outcomes; report per-sample gates, regime decisions, temporal clock provenance, and all bridge controls. No current evidence justifies a CVPR acceptance claim; that requires independent benchmarks and comparisons to temporal grounding work such as [TimeLens (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.html).

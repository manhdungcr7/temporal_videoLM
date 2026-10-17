# CLRV v8 proposal/code preflight

Date: 2026-10-17

## Verdict

**Do not run the attached source as CLRV v8.** The source attachment is byte-for-byte the archived Stage-0 v7 source after newline normalization. The proposal describes substantial v8 changes that are absent from that source. The attached file therefore cannot test the proposed method.

The prose claims SHA-256 `3f49e9f38e01c901342f02b6b8ea7e2e905ab7beff9ef2cbf2b79a4db001ef56`. The code attachment instead normalizes to LF SHA-256 `d27eb87a7014e941bd7a7a920dbe3ce074338cf616757608cc476712d7c8ee80` and CRLF SHA-256 `cdfdda61626c81b3b32f78017908bc9fd2865f1f695d30fe02b9f23280028cb8`. Those are the hashes already recorded for the archived v7 source in `results/pcebr_stage0_v7`; the code attachment is not the claimed v8 artifact.

## Material specification/source mismatches

1. **The alleged v8 query-role and clean-gate code is absent.** The attached source still has the old `STOPWORDS` set and `partition_stem_into_content_and_scaffold`; when no token qualifies, it falls back to the entire stem (`results/pcebr_stage0_v7/stage0_pcebr_v7_final_original.py:2313`, `:2323`). It has no `DEICTIC_META_WORDS`, `split_query_into_endpoint_roles`, or `compute_unified_gate_from_token_matrix`. For the TA wording, all lexical query words are stopwords, so the fallback is triggered. For CMTB wording, `action` is not in the stopword set, so the proposal's claim that every CMTB token is excluded is too broad; the remaining one-word “content” representation is still not a reliable endpoint parser.

2. **Audio salience remains directly gated by text-to-audio hidden similarity.** The source computes `joint_gate = alpha1 * alpha2` (`:447`) and derives audio scores from the same query-conditioned hidden-space similarities. It contains no proposed layer-22 vocabulary-manifold projection or intrinsic audio novelty term. The prior v7 run measured `alpha_audio=0`, `joint_gate=0`, and all 14-D `Psi=0` on all eight micro samples. Thus the new joint-pair branch was inactive on those cases.

3. **The promised log-space conditional normalization is absent.** The code still computes `M_act = J_real.sum().clamp(min=1e-12)` and `J_cond = J_real / M_act` (`:307–313`). It does not normalize `log_J_real` with `logsumexp`. The old actual-input audit found `J_cond` marginal errors as large as 0.467 in the low-mass regime. This is a correctness issue, not a wording difference.

4. **Pair compatibility does not change dustbin cost as described.** The code forms the dustbin score from its existing helper and sets `act_target = joint_gate * sigmoid(-z_dust)`; there is no `z_eff = z_dust - 1.25 * joint_gate * p1^T Omega p2` term. Pair compatibility enters the real block, but the proposal's claimed compatibility-dependent null calibration is missing. The old audit also showed that, under the prescribed marginals, the real/null masses were determined by the scalar active target; the new M0 assertions in the prose are not present in this source.

5. **The claimed freeform-first causal rollout and MC bridge are absent.** The code predicts `mc_pred_routed_fwd` directly from the routed MC prompt (`:3337`); it generates freeform text later for heuristic auditing (`:3272`), without scoring choices against that rollout or combining `p_mid_direct` with `p_bridge`. More seriously, the CLC direction is derived from the MC prompt and then reused for option-free generation, so the main freeform path is not choice-independent end to end. The source contains no `prefix_bank`/`DynamicCache` optimization and rebuilds the prompt each generation step.

6. **The advertised training-free scope conflicts with this source.** `run_all_stage0_v7` trains/runs M1 and M2 and passes their fitted aligner, probe, and six-scalar bridge to M3 (`:3432–3440`). This is a learned multi-stage method on a frozen backbone, not a wholly training-free method. If “training-free” means only no backbone fine-tuning, that narrower term should be used and the fitted auxiliary components disclosed.

7. **Foil-suppressed CLC is not sink-matched in this source.** The proposal says `kappa_sink=0.60` is held fixed, but `mode == 'foil_suppressed'` sets `kappa_sink=0.0` (`:2719–2723`). The resulting contrast changes both event bias and sink correction, so it cannot isolate the event-mask effect.

8. **The v8 freeform/continuous-thought path is absent.** There is no `hidden_to_manifold_thought_batch`, no thought injection to layers 0–2, no role-specific anchor/target distributions, no separate direct-versus-rollout MC outputs, and no single-pass-prefix KV cache. The source remains the v7 CAOR/CLC implementation.

## Evidence in the repository against the proposed causal story

- On the v7 micro8, native, CAOR-only, and full JEP-CLC all scored **5/8**; the full method rescued **0** and harmed **0**. Because the joint gate and `Psi` were zero on all eight, this does not test a functioning v8 joint mechanism, but it does rebut any claim that the attached source improved QA.
- The v7 gate was reversed on the semantic-wrong-query diagnostic: mean true gate **0.4032**, wrong-context gate **0.9640**. That directly rejects the old claim that content-excess gating solved query relevance. It motivates a clean v8 gate, but the attached source still uses the failed logic.
- In the two-hop roll suite, full v7 scored **57/156 (36.5%)**, the no-Psi control **61/156 (39.1%)**, the matched 6-D Phi control **58/156 (37.2%)**, and the factorized/no-Sinkhorn control had the same predictions as full. These outcomes challenge the claim that the 14-D joint Sinkhorn feature was responsible for gains. They do not falsify all joint coupling methods; v7's zero gate means the micro8 test did not activate its joint branch.
- A model-generated freeform answer on micro ID 2150 already said the event happened “right after the speech” while its MC prediction remained wrong. This is evidence that temporal information can be represented in a verbal continuation while answer selection/readout still fails. It supports testing a separately specified rollout-to-choice bridge, but it does not validate the reported 73.3%: that result must be reproduced on a fixed, leakage-audited set with a frozen scorer and independent semantic grading.
- The blanket premise that temporal information is absent from value states is not established by the cited architecture. Qwen2.5-Omni explicitly uses TMRoPE to encode synchronized audio/video time; a failure of a few attention interventions to change an argmax does not prove all positional information has vanished. The Qwen technical report describes TMRoPE and interleaved audio/video organization: https://arxiv.org/abs/2503.20215.

## Conceptual risks to test if a real v8 source is supplied

- “Top-48 vocabulary manifold” is a heuristic projection, not a guarantee of modality-independent event semantics. It may favor frequent lexical labels, discard concepts outside the selected vocabulary, and make audio-to-visual matching sensitive to prompt wording. Compare against full-vocabulary/logit and random-vocabulary controls, and measure unseen event categories.
- Parsing endpoints by a small set of relation connectors can fail on implicit references, multiple clauses, negation, repeated events, and questions whose anchor is not stated as a separate noun phrase. The task-general claim needs A→V, V→A, A→A, and V→V with repeated and absent events, plus compositional/unseen question forms.
- The reported `tf_bridge` gain is promising but not enough to establish the mechanism. CoT can change multiple-choice scoring and can be unfaithful; use a preregistered scorer, options revealed only after freeform generation, control rationales that preserve wording but reverse temporal relations, and human/independent semantic grading. Related evaluation work specifically studies how reasoning traces and option exposure affect MCQA: https://proceedings.mlr.press/v306/raman26a.html and https://proceedings.mlr.press/v306/arcuschin26b.html.
- CVPR novelty cannot rest on interval distributions, Sinkhorn, hidden-state interventions, or temporal grounding individually. TimeLens (CVPR 2026) already studies temporal grounding and finds data quality/design central; position the contribution against that literature and benchmark across independently sourced data: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.html.

## Next required artifact

Before running v8, obtain a source file whose normalized hash is the claimed `3f49e9...`, then compare it to the proposal. The currently attached code is the old v7 and cannot answer whether the v8 edits are correct. No GPU run was started for this preflight.

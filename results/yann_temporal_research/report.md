# Yann — autonomous temporal research, 2026-10-16

**Chưa tìm được phương pháp mạnh. R1–R4 chưa cải thiện micro8. R5 oracle-trained readout cho tín hiệu cơ chế, nhưng R6/R7 retrieval chưa chuyển được lợi ích đó sang suy luận tự tìm bằng chứng. Xem các vòng bổ sung bên dưới.**

Mọi micro dùng cùng8 IDs, rotation0; chưa phải full40 với110 rotations. Bộ40 đã là development qua nhiều vòng. Các public Perception Test splits chỉ là train/validation/diagnostic nhỏ theo video, không phải subject-disjoint final test. Không có CVPR/Oral acceptance claim.

## R1 — media causal topology

| Condition | Đúng/sai/8 | Cứu/hại so native |
|---|---|---|
| native | 5/3/8 | []/[] |
| explicit_native_mid | 5/3/8 | []/[] |
| bidir_all_mid | 5/3/8 | []/[] |
| bidir_cross_mid | 4/4/8 | []/[1449] |
| clock_causal_mid | 5/3/8 | []/[] |
| clock_causal_early | 5/3/8 | []/[] |

| Task/gold | N | native | explicit_native_mid | bidir_all_mid | bidir_cross_mid | clock_causal_mid | clock_causal_early |
|---|---:|---|---|---|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 1/0 | 0/1 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Native/explicit-mask parity0. Opening future media edges can change logits without improving QA; do not infer causal mechanism solely from sensitivity. Native media clock is TMRoPE metadata, not independent event ground truth.

## R2 — question-only feedback

| Condition | Đúng/sai/8 | Cứu/hại so native |
|---|---|---|
| native | 5/3/8 | []/[] |
| question_feedback_mid | 5/3/8 | []/[] |
| question_feedback_early | 5/3/8 | []/[] |
| format_feedback_mid | 5/3/8 | []/[] |
| question_feedback_visual_mid | 5/3/8 | []/[] |
| question_feedback_audio_mid | 5/3/8 | []/[] |

| Task/gold | N | native | question_feedback_mid | question_feedback_early | format_feedback_mid | question_feedback_visual_mid | question_feedback_audio_mid |
|---|---:|---|---|---|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

New feedback edges exclude MC options; native media-media order and tail causal structure remain. No candidate changes an argmax. Format control has a different edge budget, so no clean semantic attribution claim.

## R3 — learned memory/readout transfer

| Condition | Đúng/sai/8 | Cứu/hại so native |
|---|---|---|
| native | 5/3/8 | []/[] |
| matched_content_memory | 5/3/8 | [2054]/[1449] |
| TAM_Lite_time | 5/3/8 | [2054]/[1449] |
| TAM_Lite_time_address_shuffle | 5/3/8 | [2054]/[1449] |
| TAM_Lite_time_zero_memory | 5/3/8 | [2054]/[1449] |

| Task/gold | N | native | matched_content_memory | TAM_Lite_time | TAM_Lite_time_address_shuffle | TAM_Lite_time_zero_memory |
|---|---:|---|---|---|---|---|
| TA/A | 1 | 1/0 | 0/1 | 0/1 | 0/1 | 0/1 |
| TA/B | 1 | 0/1 | 1/0 | 1/0 | 1/0 | 1/0 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Time/content models selected only by8 public validation human-QA token NLL, both step50.1,350,528 trainable parameters; frozen backbone version checks pass. Same144 prepared cases/48 videos,32train/8validation/8diagnostic.400steps, fixedseed, matched data/budget.

| Variant | Validation human QA mean NLL | Diagnostic human QA mean NLL | Diagnostic localization top1 |
|---|---:|---:|---|
| matched_content_memory | 1.402234 | 2.717732 | 3/16 |
| TAM_Lite_time | 1.400446 | 2.710713 | 3/16 |

NLL and teacher-forced token accuracy are not semantic QA accuracy. A zero-content bank still retains time/modality and query features; call it a content ablation, not complete removal of every memory-related signal. No changed MC outcome under address shuffle is evidence against the tested added-address claim, not proof that the entire trained model never uses sensory input.

### Actual option-free generated text,32-token cap

**1449 — TA/A**

- native: Yes, it is. So, what do you think about the video?
请回答：你对这个视频有什么看法？


- matched_content_memory: No, the narrated event doesn't happen at the same time in the video. If you have any other questions about the video or the narration, feel free

- TAM_Lite_time: No, the narrated event doesn't happen at the same time in the video. If you have any other questions about the video or the narration, feel free

**2150 — TF/C**

- native: The narrated event happens right after the speech. So, if the speech is at 182.1 seconds, the event would be at 1

- matched_content_memory: The narrated event happens right after the speech in the video. If you have any other questions about the video or anything else, feel free to ask!

- TAM_Lite_time: The narrated event happens right after the speech in the video. If you have any other questions about the video or anything else, feel free to ask!

**1929 — TF/C**

- native: The narrated event happens right after the onions are stirred in the video. So, it's right after that moment. If you have any other questions about the

- matched_content_memory: The narrated event happens right after the onions are stirred in the video. So, it's right after that moment in the video. If you have any other

- TAM_Lite_time: The narrated event happens right after the onions are stirred in the video. So, it's right after that moment in the video. If you have any other

**2054 — TA/B**

- native: Yes, it is. So, what do you think about the cooking process?
请回答：你对这个烹饪过程有什么看法？


- matched_content_memory: No, the narrated event doesn't happen at the same time in the video. If you have any other questions about the video or the narration, feel free

- TAM_Lite_time: No, the narrated event doesn't happen at the same time in the video. If you have any other questions about the video or the narration, feel free

**1956 — CMTB/A**

- native: The person is adding flour to the pot. So, it's like they're pouring it in while the speech is going on. What do you think about that

- matched_content_memory: In the video, when the speech is heard, the person is adding flour to the pot and then whisking it in. So, that's the visual action

- TAM_Lite_time: In the video, when the speech is heard, the person is adding flour to the pot and then whisking it in. So, that's the visual action

**1513 — TF/A**

- native: The event happens right after the speech. So, if the speech is at 18.02 seconds, the event would start at 18.

- matched_content_memory: The event in the video happens right after the speech. So, if the speech is at 363.333333333

- TAM_Lite_time: The event in the video happens right after the speech. So, if the speech is at 363.333333333

**2264 — CMTB/B**

- native: Well, when the speech is heard, it's like the person is whisking the ingredients together. You can see the whisk moving around in the bowl. It

- matched_content_memory: Well, when the speech is heard, the visual action in the video is that the person is whisking the mixture in the bowl. It's like the speech

- TAM_Lite_time: Well, when the speech is heard in the video, the visual action is that the person is whisking the mixture in the bowl. It's a pretty clear

**1821 — TF/C**

- native: The narrated event happens right after the speech. So, it's right after the moment when the speech is heard. If you have any other questions about the

- matched_content_memory: The narrated event happens right after the speech in the video. If you have any other questions about the video or the narration, feel free to ask!

- TAM_Lite_time: The narrated event happens right after the speech in the video. If you have any other questions about the video or the narration, feel free to ask!

No post-hoc regex semantic scores are used. Native MC and option-free generation use different prompts and can disagree. Oversized generated timestamps are visible failures, not correct localization.

## R4 — physical-pair trained memory transfer

| Condition | Đúng/sai/8 | Cứu/hại so native |
|---|---|---|
| native | 5/3/8 | []/[] |
| matched_ordinary_temporal_memory | 5/3/8 | [2054]/[1449] |
| TAM_binding_dependent | 5/3/8 | [2054]/[1449] |
| binding_address_shuffle | 6/2/8 | [1513]/[] |
| binding_uniform_attention | 5/3/8 | []/[] |

| Task/gold | N | native | matched_ordinary_temporal_memory | TAM_binding_dependent | binding_address_shuffle | binding_uniform_attention |
|---|---:|---|---|---|---|---|
| TA/A | 1 | 1/0 | 0/1 | 0/1 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 1/0 | 1/0 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 | 1/0 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

## Data/ontology and execution corrections

The initial public-data eligibility filter failed to exclude the sound label Muted. That is a mute-state annotation, not an audible event. R3 includes invalid derived sound-grounding questions and must not be called clean event supervision. Preserve all original scores and checkpoints; do not rewrite them after ontology correction. R4v2 excludes Muted endpoints, keeps valid previous pairs and records modality fallbacks/skips before fitting. Original selected videos and benchmark40 are unchanged.

R3 half-second support for visual mean tokens underestimates native temporal tubelet support (2/native sampled fps). R4 packaging preserves original spans and corrects visual support from actual sampled-frame count and source duration before fitting. This is a known metadata correction, not a QA-dependent parameter choice.

Preparation guard100 tokens rejected a102-token multi-instance target. Resume increased only that guard to768, keeping full targets/samples. First R4 construction also rejected an assumed exact4s/frame-grid shift: native sampling fps is often1.91–1.99, not exactly2. The corrected transform uses an even number of native frames and matching audio sample delay, preserving video tubelet pairing; source waveform/frame arrays are bitwise retained inside padding. All failed job logs remain.

## Next hypothesis and limits

R4 tests a binding-dependent residual identity and actual O/A/V/AV translations with human-derived endpoint-order targets. Paired supervision should make memory necessary under representation interchange; this remains a hypothesis. See round4_preflight.md. Uniform retrieval identity is a mathematical implementation contract, not semantic calibration or an accuracy result. Related IIT/causal abstraction, SyncRA, GroundFormer and LatentOmni constrain novelty claims.

Artifacts: flow_s*.json, questionflow_s*.json, trainprobe_s*.json, probe_eval_s*.json, frozen public selection manifest, public feature archives with entry SHA checks, validation-selected checkpoints, unit checks, all launch/job/error logs, literature_and_falsification.md.

## R4–R7 — independent physical intervention choices

31 train recordings;8 validation;7 diagnostic. Same original public48 selection, no SVHalluc training. Questions are derived from human category/interval annotations and ask first onset; captions/class names can be coarse. Four O/A/V/AV versions and multiple questions are dependent observations. Report strict recording success and mixed flip-pattern success, not only individual accuracy.

| Round / condition | Selected step | Train choices | Diagnostic choices | Diagnostic strict recordings | Diagnostic mixed-pattern recordings |
|---|---:|---|---|---|---|
| trainbinding/matched_ordinary_temporal_memory | 200 | 68/124 | 25/48 | 0/7 | 0/4 |
| trainbinding/TAM_binding_dependent | 200 | 73/124 | 27/48 | 1/7 | 0/4 |
| oraclelearn/oracle_true_address | 400 | 121/124 | 42/48 | 3/7 | 1/4 |
| oraclelearn/oracle_address_shuffle | 100 | 79/124 | 30/48 | 0/7 | 0/4 |
| distillretrieval/grounding_only_retrieval | 200 | 117/124 | 21/48 | 0/7 | 0/4 |
| distillretrieval/reader_distilled_retrieval | 1600 | 122/124 | 25/48 | 0/7 | 0/4 |
| cosineretrieval/media_conditioned_cosine_retrieval | 0 | 89/124 | 25/48 | 0/7 | 0/4 |
| cosineretrieval/question_only_cosine_retrieval | 0 | 89/124 | 20/48 | 0/7 | 0/4 |

R5 oraclelearn uses annotation endpoints at input and O context throughout: privileged mechanism diagnosis, not deployment accuracy. Its native O-context diagnostic baseline is28/48, strict0/7. True-address oracle-trained reader reaches42/48, strict3/7, versus shuffled-address-trained30/48, strict0/7. Ordinary/oracle conditions differ from R4 ordinary source context; avoid treating every count as the same native baseline.

R4 full selected checkpoint improves mostly human-QA token NLL; validation binary temporal NLL is worse than untrained. Diagnostic endpoint role0 top1 only1/48, role1 16/48; forcing correct endpoint attention at inference changes27→28/48. R5 separately trains the reader with correct endpoints and instead obtains42/48, showing why inference-only oracle is not an oracle-trained upper bound. R6 freezing that reader and learning retrieval produces21/48(grounding-only) or25/48(answer-distilled); oracle remains42/48. R7 cosine/media-query and cosine/question-text-query both select step0, with25/48 and20/48: no trained retrieval improvement.

R4 micro8 ordinary/full both5/8, rescue2054 and harm1449; address-shuffle6/8, uniform-attention5/8. This does not establish temporal gain. Micro evaluation initially rejected two cached visual-support guesses below.8s; the final separately named clockfix uses each archived spg and extends the native horizon to cover all native cells. Native cache video clocks can exceed container duration (e.g.2150:15.52s start versus12.167s container); this is the known native decoder-clock mismatch, not PTS-calibrated physical time. Original cache/failed logs remain. R3 micro old half-second spans can have negative widths at beyond-container native timestamps; do not interpret it as a clean temporal-address evaluation.

Completion audit checks same input prefix with alternate teacher-forced answer:240/240 query/media features have max difference0. This is finite-precision prefix independence, not proof of independent semantic ground truth. Geometry audit includes duplicate original/extra questions(56 diagnostic versions versus48 after fit dedup); exact-event-center order51/56 versus start-order truth, earliest oracle-cell support56/56. These counts do not prove that the full learned Fourier representation lacks onset information.

## R8 — dense native-feature diagnostic

| Feature head | Diagnostic within-recording macro AUC | Diagnostic macro top1 | Time-only AUC | Independent recordings |
|---|---:|---:|---:|---:|
| linear native feature probe | 0.8182 | 0.4917 | 0.5521 | 7 |
| 128-hidden native feature probe | 0.8030 | 0.4536 | 0.5521 | 7 |

34 training category names; every valid sound/action interval provides cell supervision. Exclude Muted and specified background-other labels. R8 initial max-overlap approximation was corrected to exact merged-interval union before the reported run; old source/job logs retained. Known-category localization is a privileged semantic-label diagnostic, not arbitrary-question QA. Stronger than time-only does not prove all visual/audio semantics or every layer is sufficient.

## R9 — arbitrary-text dense scorer, held category names

| Condition | Selected step | Diagnostic seen AUC/top1 | Diagnostic held AUC/top1 | Held recording/task count |
|---|---:|---|---|---|
| true_text_semantics | 300 | 0.8169/0.4810 | 0.5854/0.0000 | 3/12 |
| seen_label_text_shuffle_control | 100 | 0.6577/0.2990 | 0.7663/0.3333 | 3/12 |

Text alignment helps seen-label localization versus semantic-corruption training, but held-label top1 is0 for the true-semantic branch and its held AUC is below the control on only3 origins/12 tasks. Therefore no open-vocabulary generalization claim is justified. Seen-label selection never uses unseen-label scores. Learned weighted-BCE outputs are not calibrated absence probabilities.

## R10 — native upper-decoder reading path feasibility

Server1: `{"server": 1, "video_id": "video_363", "prefix_tokens": 3448, "tail_tokens": 48, "hidden_max_error": 0.125, "hidden_mean_error": 0.002735137939453125, "logit_max_error": 0.01953125, "strict_absolute_parity_pass": false, "relative_hidden_l2": 0.0010558970971032977, "teacher_token_agreement": 1.0, "native_ce": 1.1931017637252808, "replayed_ce": 1.1931968927383423, "gradient_finite": true, "gradient_norm_sum": 0.4135553534142673, "adapter_parameters": 409600, "prefix_kv_bytes": 28246016, "backbone_versions_unchanged": true, "peak_allocated": [7160742912, 4358995456]}`

Server2: `{"server": 2, "video_id": "video_6300", "prefix_tokens": 2762, "tail_tokens": 20, "hidden_max_error": 0.1875, "hidden_mean_error": 0.0034656524658203125, "logit_max_error": 0.015625, "strict_absolute_parity_pass": false, "relative_hidden_l2": 0.0013108921702951193, "teacher_token_agreement": 1.0, "native_ce": 5.556992053985596, "replayed_ce": 5.544541358947754, "gradient_finite": true, "gradient_norm_sum": 8.099129289388657, "adapter_parameters": 409600, "prefix_kv_bytes": 22626304, "backbone_versions_unchanged": true, "peak_allocated": [7007535616, 4283224576]}`

Initial absolute hidden parity threshold failed. The later named audit reports relative/functional deviations and gradient feasibility separately; it does not retroactively pass the original criterion. Native frozen-prefix versus adapted text-tail decoding must match between training/deployment. This is a decoder-adaptation baseline, not a novelty or accuracy result.

## Additional execution and interpretation notes

Both initial R4 gradient runs stopped before any optimizer update because an overly strict finite-gradient guard prevented normal AMP overflow handling. R4 reruns use standard GradScaler overflow skipping with initial scale1024; both actual runs record0 skipped updates. Loss/data/seed/400-step budget unchanged. All original failures are retained.

Preflight specifications, exact scripts, selected checkpoints, per-query predictions, physical feature archives and hashes, all job/status logs and separate mechanism diagnostics are saved in this directory and server_jobs. Training/validation/diagnostic reuse must be treated as development, with eventual fresh recordings/domains required. No successful QA40 method, novelty guarantee or CVPR/Oral acceptance has been established.

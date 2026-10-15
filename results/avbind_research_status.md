# Joint AV binding research log — 2026-10-14

## Scope and acceptance criterion

Inference must accept arbitrary question text and optional answer choices. No
task label, ground truth, question-template router, fixed audio-first resolver,
or conversion from a temporal relation into a benchmark option is permitted.
Passing development examples does not establish deployability or CVPR novelty.

The 400-sample temporal reserve remains unrun (50 per each of 8 task/label cells;
200 per server). The repeatedly inspected 16 examples are development data, not
held-out evidence. Six Daily-Omni questions are only a small cross-domain probe.

## Rejected approaches

* The older provenance pipeline reached 15/16 on development data but has a
  fixed audio-first structure. It was rejected as the general solution.
* Symmetric textual memory/readback R1/R2: 4/10 on the ten-example pilot versus
  6/10 native base. Factorizing controller choices did not establish useful
  evidence retrieval. Six Daily-Omni probes: memory 3/6 versus native base 5/6.
* Learned reader R1: exact textual structural holdout 19/38 versus 22/38 for
  the same memory without reader. This is a failure, not AV grounding evidence.
  First-hop retrieval produced 10 true-positive, 147 false-positive and 28
  false-negative addresses on this holdout. Native transfer was incomplete.
* R2 ran on the prior notebooks but its completed state was not archived before
  both notebooks reset. It has no reportable final result in this log. Its
  supplemental generator was also audited and found to always place the actual
  event in the middle, with biased durations and nonconcurrent narration.

## Active R3 experiment

1. Independent visual/audio 2-second observations and speech spans form a bank
   of records `(content, source, [start,end])`. Captions remain imperfect; they
   are not treated as raw-media representations or exact event boundaries.
2. Content features are encoded independently with the frozen backbone's
   layer 28. The query sees content/source before the explicit clock table and
   before choices. This separates semantic lookup from added numerical clocks;
   native AV query features can still encode implicit temporal information.
3. A 1.31M-parameter, two-read adapter uses independent support scores and a
   learned null gate; it retains separate occurrences rather than forcing a
   single best event. The first address key excludes time. The next read can
   use source, relative clocks and pairwise endpoint relationships.
4. Clock encoding uses a shared minimum start as coordinate origin. The reader
   module is invariant to uniform clock shifts for fixed query/content features;
   this is not an end-to-end invariance claim about the backbone.
5. A gated residual enters layer 28; native AV is kept in the answer context.
   Comparison is native base, native AV + identical memory, and that same input
   plus the reader. The auxiliary bank is still contextual text, not raw states.

Training: frozen Qwen2.5-Omni-3B; 512 exact constructed textual scenes, 768
steps, no SVHalluc labels or recordings. The repaired corpus randomizes event
rank, all durations, simultaneous/non-simultaneous references, tense, IDs and
options; null and reverse-direction questions are included. Server 1 uses
answer + address + null losses; server 2 is the same-budget answer-loss-only
ablation. Their initial native pilots use different subsets and must NOT be
combined as a single candidate's score without cross-running checkpoints.

Local structural tests: 29 passing, including forward/gradient equivalence of
GQA and explicit KV repeats. They verify code properties, not QA accuracy.
Both R3 variants finished training: textual holdout 39/62 for native same-memory
base, supervised reader, and answer-only reader. No accuracy improvement.
Supervised first-hop address precision/recall: .170/.958; second hop:
.162/.629. Answer-only address precision/recall collapsed to zero on both
reads. Supervision improves retrieval but leaves many false positives.

Native trials encountered a PowerShell cache serialization wrapper (fixed,
bad copies preserved) and T4 SDPA falling back to quadratic-memory grouped-query
attention. A GPU equivalence test for explicit KV repeats passed (FP16 maximum
absolute difference .001953125 on the small test), without media/mask changes.
Both workers were restarted; archived checkpoints were restored. Valid native
R3 reruns now use the SAME supervised checkpoint on both servers.

R4 is a queued follow-up: cache frozen contextual features, then optimize
semantic-address ranking + calibrated independent support/null losses before
answer adaptation. Its training data remains the same exact constructed corpus;
no benchmark labels are read. This is not yet an accuracy improvement claim.

Valid R3 TA completion: native base TA/B 4/5 and TA/A 0/2; added-memory and
added-memory+reader both TA/B 0/5 and TA/A 2/2. All seven predictions become A.
The reader changes score values but not decisions. The added context includes
both captions AND extra format/instructions; their individual causal effects
have not been isolated. This configuration is rejected as a solution.

R5 follow-up removes all generated captions from the answer context, retaining
the exact original native question/options prompt. Its bank contains layer-28
latent states from physically isolated 2-second audio/visual crops with a
constant factual instruction, no generation, QA or timestamps in the content
keys. Source and interval remain separately attached. Reader weights are still
transferred from constructed textual supervision; this does not establish
learned native AV alignment. Initial S1 uses the pending R4 address-pretrained
reader; S2 uses R3. These distinct-weight probes must not be combined into one
method's aggregate result until weights are cross-run.

R5 completed: S1 (R4 weights) TA/B 4/5, TA/A 0/2, identical to native base;
S2 (R3 weights) TF/C 3/5, TF/A 1/2, TF/B 2/2, also identical to base. Removing
caption text avoids the new TA/A bias, but the original errors remain.

R6 reads the same isolated raw slots through token-count-normalized biases on
native raw-media keys at decoder layers 29–36, only for question/answer rows.
Both servers use the same R4 checkpoint. Fixed gain 2 was chosen before the
pilot, without label-specific settings. No prompt or position change. First
real-clip zero-gain parity test exactly matches native scores and confirms the
temporary attention registry/hook is restored. Native media hidden states can
already contain cross-modal fusion; bank isolation is not a purity guarantee
for those boosted native values. This is an explicit remaining limitation.

R6 completed on all 16 development questions: base and addressed attention both
10/16 (TA/B 4/5, TA/A 0/2, TF/C 3/5, TF/A 1/2, TF/B 2/2). No decision improved.

R7 follow-up makes the external KV values truly modality-isolated: raw content
keys exclude clocks; a separate raw encoding with factual source/observation
interval text forms value states. No caption is generated. Late question/answer
rows read these external records in addition to original native media. Their
semantic-key scores remove the question's RoPE phase; source/time are carried
by values rather than imposed as content-key similarity. Native/raw and external
memory share the softmax, with null-aware token-count accounting. This is an
experimental integration with frozen projections, not trained native AV
alignment. A two-GPU indexing bug was fixed before valid smoke evaluation.
Both zero-gain parity/registry-restoration checks passed. Neither initial hard
sample is corrected at gain 1; full controls are pending. No gain was selected
per task, label or individual sample.

Resume audit (2026-10-14): both workers idle, no queued work. R7 full runs
8488204454 / b2715feb01 completed, 7 + 9 records, no errors. R7 matches native
base exactly on decisions: 10/16, with TA/B 4/5, TA/A 0/2, TF/C 3/5, TF/A 1/2,
TF/B 2/2. It does not solve the target failure.

R8 forms content keys from factual isolated crops and matched static-video /
silent-audio controls, LN(2*factual-control), with fixed coefficient 1. It
retains R7 factual values and original AV/question inputs. Only smoke jobs
8cea2842d1 (1454) / 82b8b9dde0 (2097) were completed at audit: neither correct.
The resumed pilot recovers those actual results, reuses existing native
baselines, and executes only the remaining 6 + 8 development questions.
The question-span Unicode handling was also repaired before R8; any change
cannot automatically be attributed solely to the differential keys. These
are development tests, not independent generalization evidence. Reserved
400 samples remain untouched.

R9 is a diagnostic intervention, NOT the proposed final method: equalize
non-empty source prior budgets while preserving within-source occurrence
ratios and total null-aware mass. R7's seven TA records assign 99.706--99.916%
of reader prior to audio. These priors are not actual final attention weights;
R9 separately measures joint-softmax native/external/source shares at the last
selected text row of each late layer and option rotation. It tests whether
the transferred source calibration is a bottleneck, without adding labels,
question-template routing, media, generated captions or new reader training.
R9 queued after R8, using its completed differential keys and factual value
caches. It evaluates a NEW condition only; native baselines are reused.

R8 resume completed 7/7 + 9/9, no errors, exactly the base decisions: TA/B
4/5, TA/A 0/2, TF/C 3/5, TF/A 1/2, TF/B 2/2, total 10/16. The original two
smokes were recovered, not re-inferred. R9 early actual readout audits exposed
external domination (e.g. 1454 late-layer last text row: external 93.87% in
first affected layer, 99.99% in the last); equal priors do not bound readout.

R10 tests a fixed null-aware convex readout budget <=0.25 with separately
normalized native/external attention, retaining R9 within-source ratios and
R8 keys/factual values. It removes the native-token-count multiplier before
computing that budget. This is a calibration diagnostic, not a claim of new
trained AV reasoning. No threshold/budget is selected by task or gold label.
Both servers queued it after earlier work, reusing all observation caches and
base scores. CPU tests now 58/58 including exact null fallback, causal-mask
preservation, budget cap, source/order equivariance and archive de-duplication.

## Reproducibility and next gates

2026-10-14 continuation after user renewed both tokens: independent native
TIMELINE_NS loaded on fresh workers. R11 input fps/tensor metadata issues and
an over-broad attention dispatch touching the audio encoder were repaired;
those failures are unscored. Plain smoke results were recovered from persistent
job state before a worker-only CUDA recovery restart, then only missing isolated
conditions were run. Encoder attention now bypasses the decoder-only registry.
R11 completed: plain and isolated both 7/16. Plain cells TA/B2/5, TA/A1/2,
TF/C1/5, TF/A1/2, TF/B2/2; isolated3/5,1/2,1/5,0/2,2/2. Reject both.

R12 tests shared physical source clocks in rotary coordinates while preserving
within-record temporal/spatial geometry. Content, observation headers, decoder
mask and backbone stay fixed; no answer-template logic. Pure coordinate tests
verify source concurrency, within-record geometry and independent clock shifts
(not end-to-end QA equivalence). R12 was queued after R11; per-sample results
are archived by the watcher.

R13 preserves continuous original-source encoders and all their native feature
rows, instead of cutting speech into separate 2s input files. Only decoder
records are split/isolated; encoders may carry other-time context of their own
modality. Matched serial vs physical rotary-clock conditions retain identical
features, token IDs and masks. This directly tests the segmentation confound
before native AV training. These inference-time representations are research
probes, NOT established novelty or broad deployment success. Neither source
clocks nor contrastive decoding by itself is a credible novelty claim.

New R11 continuation: native raw source/clock-addressed records, with matched
plain causal vs locally isolated prefix attention. Common system prefix and
each record's own modality/address are visible during encoding; arbitrary
question/answer rows see every earlier record. No generated captions, text-
trained reader, labels or question-type routing. CPU mask/GQA/registry tests
pass. GPU S2 smoke 5f88656bb3 failed before any QA score at a varying-fps input
check; repair now preserves each crop's effective fps in video_second_per_grid.
S1 went ngrok offline before upload; S2 then went offline before repaired
upload. There are NO valid R11 QA results. Plan and resume targets are saved in
avbind_r11_plan.json; per-condition checkpoints avoid losing or re-running
finished conditions. Fresh workers can load a clean TIMELINE_NS directly from
the backbone helpers without restoring old task resolvers or reader training.
Reconnection is required for GPU validation. Earlier R3--R10 remain untouched.
Latest health recheck: both endpoints now return HTTP401 instead of ngrok3200;
saved credentials no longer authorize either worker. Current tokens are needed.

Latest continuation complete: R8, R9 and R10 each have 16/16 successful
development records, no errors and no decision improvements (all 10/16;
target TA/B + TF/C 7/10, controls 3/6). Both workers are idle. R9 actual external
readout reaches 99.9996%; R10 correctly caps it below 25%, yet QA is unchanged.
No unfinished pilot cell remains for these conditions. Broader CMTB/native
generalization and the 400 reserve are still untested for these variants.
Native AV supervision/readout training is the next substantive research gate,
not completed work. Full resume audit, counts, caveats and hashes are in
`avbind_resume_20261014_report.md`. CPU tests: 59/59, including an executed
resume test that recovers the smoke, skips successes, retries missing/failed
samples and performs zero inference on a second completed invocation.

`server_jobs/avbind_progress.ps1 -Download` shows live per-group correct/wrong
counts. `avbind_watch.ps1` downloads partial results and recoverable training
checkpoints. Active job IDs are in `avbind_active_jobs.json`; credentials are
only in ignored connection files.

Reject a candidate that merely flips TA/TF bias while damaging other labels.
Next gates: matched-budget checkpoint cross-runs, semantic-address accuracy,
independent source-time interventions, reverse free-form questions, broader
contexts, and perception-vs-binding error separation. Only then freeze a method
and run the 400 reserve. A caption-bank adapter alone is not enough evidence of
a general raw-AV reasoning contribution.

## Closest-work checks

[Mirasol3B (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Piergiovanni_Mirasol3B_A_Multimodal_Autoregressive_Model_for_Time-Aligned_and_Contextual_Modalities_CVPR_2024_paper.html)
already uses snippet-wise AV processing and compact representations.
[SyncRA](https://arxiv.org/abs/2609.34363) studies learned temporal
correspondence. Neither compression nor alignment alone is a novelty claim.
[Relevant Evidence Decoding, posted October 2, 2026](https://arxiv.org/abs/2610.02976)
already selects question-relevant audio/video/interaction contributions and
uses PMI-based decoding, including SVHalluc evaluation. Generic
modality-contribution boosting is therefore an especially weak novelty claim.
Our proposed research distinction would need to be empirically demonstrated
semantic-event/occurrence-clock binding under independent source-time changes,
not just a renamed evidence selector or an accuracy increase on ten examples.

## Latest resume audit (2026-10-14, supersedes stale status notes above)

Both renewed connections work. R11, R12, R13 and R14 are complete, not queued
or missing. No old condition was rerun in this continuation. R12: 5/16.
R13 serial: 10/16 (target 9/10, controls 1/6), physical: 6/16; reject both.
R14 load jobs 681627fcd9 / 8dd8ded425 and pilots 05c9a62156 / cb89e70a1c
all finished successfully. R14 cells: TA/B4/5, TA/A0/2, TF/C5/5,
TF/A0/2, TF/B1/2. Total10/16, target9/10, controls1/6: reject.
Question routing entropy gates average about .05-.16; these are retrieval
priors, NOT measured final attention or causal evidence of correct grounding.

Native pair extraction is already complete: 252 crop/source pairs from 20
recordings; 16 training and 4 held-out origins. Alignment trained 512 steps
on 202 pairs; 50 held pairs improve pseudo-description retrieval from0/50
to7/50. No benchmark questions or labels were used in that training.
Neither the extraction nor training should be repeated on resume.

Next R15 is a matched readout diagnostic: native AV + option-blind evidence
draft versus R14 guided records + the same draft instruction. All original
media remain available for final answering; drafts are explicitly fallible,
not additional observations. No task-specific/source-order routing or label
rules. A draft is cached before final scoring; each successful condition is
skipped on resume. This is NOT a claimed CVPR contribution. It checks whether
first-letter readout is a bottleneck; broader/source-time causal evaluation
still required. The 400 reserved evaluation samples remain untouched.

## Final snapshot of this continuation: R15-R18 complete

This section supersedes the pending R15 note above. All16 development samples
completed for each new inference condition, no errors or missing groups.
R15 native+draft6/16 (target3/10, controls3/6); guided+draft7/16
(target5/10, controls2/6): reject. Drafts fabricate times/absence of speech or
repeat observation IDs. R16 preserves native inputs and only biases late Q/answer
attention:10/16, identical decisions to base. Measured layer36 media attention
increases about2.57-9.18 percentage points per clip, without better QA.
R17 routes from actual layer28 native multimodal Q/current-answer states instead
of text-only queries:10/16, still identical decisions. R18 pools native media
states instead of assistant-prefix states, reuses252 annotations/text vectors
and splits, trains a NEW separate512-step projector, and runs the same native
grounded readout:10/16, target7/10, controls3/6, identical base decisions.
No original feature extraction, R14 training, baseline or finished round was
rerun. R18 training is a new representation condition, not a repeat of R14.

R18 noisy held pseudo-description retrieval:unadapted2/50, adapted1/50
versus R14 adapted7/50. This fails the semantic retrieval quality gate; it is
not temporal QA or untouched final validation.95 CPU tests pass. The R18
coordinator finished; both inference jobs completed. Reserved400 remains unrun.
Scientific objective still unmet: do not claim solved bias, generalization or
CVPR readiness. Next hypotheses require stronger multi-slot semantic occurrence
supervision and a trained content-clock readout with source-time interventions,
not task/label-specific thresholds. Full counts, caveats, recovery and provenance
are in avbind_resume_r14_r18_report.md. Live TIMELINE_NS now uses R18 pooling/
checkpoint; restore a prior loader before running any older frozen variant.

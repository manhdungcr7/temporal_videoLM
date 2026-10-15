# Continuation audit: R14 through R18 (2026-10-14)

## State preserved and completed work reused

At the initial audit both renewed servers were accessible. R14 load and pilot
jobs were already done: S1 pilot05c9a62156, S2 pilotcb89e70a1c. All16
development samples had valid scores, no missing cells or inference errors.
R11-R13 were also complete. None of those jobs, original annotation generation,
recording selection, R14 training, or native baseline evaluation was repeated.
Existing dirty user files were not reverted or edited by this continuation.

Four new rounds were implemented after the audit. All inference paths retain
raw audio/video and accept arbitrary questions; no TA/TF category routing,
answer-meaning parser, hardcoded audio-first/visual-first plan or gold-dependent
gain is used. Names/gold in the pilot scripts are evaluation bookkeeping only.
All16 samples are repeatedly used development data, NOT a held-out benchmark.
The reserved400 questions and their parent recordings are excluded from native
pair training and remain unrun.

## Completed results (correct / evaluated)

| Group | Native base | R14 | R15 native draft | R15 guided draft | R16 native layout | R17 grounded query | R18 pooled content |
|---|---:|---:|---:|---:|---:|---:|---:|
| TA/B |4/5|4/5|2/5|4/5|4/5|4/5|4/5|
| TF/C |3/5|5/5|1/5|1/5|3/5|3/5|3/5|
| TA/A |0/2|0/2|2/2|0/2|0/2|0/2|0/2|
| TF/A |1/2|0/2|1/2|2/2|1/2|1/2|1/2|
| TF/B |2/2|1/2|0/2|0/2|2/2|2/2|2/2|
| Target10 |7/10|9/10|3/10|5/10|7/10|7/10|7/10|
| Other-label controls6 |3/6|1/6|3/6|2/6|3/6|3/6|3/6|
| Total |10/16|10/16|6/16|7/16|10/16|10/16|10/16|

Reject all these variants. R14's9/10 target score hides control-label collapse.
R16/R17/R18 do not change any native decision. R18 QA completed16/16 on both
servers, no errors; archived results were checked for unique IDs and all cells.
No missing development condition remains for R14-R18. Neither server has a
remaining stage of this pipeline; the research success criterion is NOT met.

## What was actually changed and measured

R15 generates an80-token maximum option-blind evidence draft and then re-reads
the full media to answer, with the draft explicitly marked fallible. Native
and R14-guided draft variants use the same instruction. Drafts are saved before
answer scoring, and successful conditions are skipped on resume. Failure cases:
1454 native draft incorrectly says no speech;2139 invents precise utterance
times including30.08s beyond the27.0977s clip;2348/2141 guided drafts copy and
repeat observation IDs instead of providing actual content. A correct letter
following an unreliable draft is not evidence of correct temporal grounding.

R16 keeps the original native token sequence, rotary positions and prefix
attention: no caption, draft, record serialization or isolation. Learned R14
keys supply a soft density-corrected prior over existing native media tokens,
at layers29-36, for question rows and the last assistant row. Text-only question
features determine the prior; options are not included in those query features.
Measured final-row media attention increases by about2.57-9.18 percentage points
per clip (mean over rotations at layer36), yet no decisions improve. These
before/after distributions hold the current hidden states fixed at that read;
they are not a global causal attribution or a proof of correct retrieval.

R17 changes only routing query states: capture native multimodal hidden states
at layer28, after BOTH original sources are available. Each question token and
each current generated answer token receives its own refreshed occurrence
prior. Question rows are causally before choices. This avoids a text-only query
being asked to resolve an implicit reference such as 'the narrated event'.
The independently encoded keys, R14 projection, gain2, layers29-36, original
native layout and all physical media tokens stay fixed. This change still
does not improve QA, so the query-source mismatch was not demonstrated to be
the sole cause of failure.

R18 tests content representation rather than another gain adjustment. New keys
pool layer28 hidden states of ONLY native media padding tokens from a physically
isolated2s source crop; system, instruction and assistant tokens are excluded.
Global clocks and question/options are absent. This still averages potentially
different events and is not a proof of sufficient fine-grained content capacity.
All252 pseudo descriptions/text features, splits and original crops are reused;
only the new media hidden-state representation is extracted (122S1+130S2).
No annotation regeneration or original-feature re-extraction was performed.
Source calibration has the same rank64, seed and512-step budget as R14.

Training uses202 pairs from16 recording origins;50 validation pairs come from4
disjoint origins. All development/reserved parent origins are excluded. On the
same pseudo-description retrieval protocol: R14 unadapted0/50, adapted7/50;
R18 pooled unadapted2/50, adapted1/50. These are noisy teacher-description matches,
NOT event-time labels or benchmark QA. The poor result is a warning against
claiming a reliable semantic address space. Validation has now been inspected
for several representation choices, so it must not be presented as an untouched
final test. Training excludes benchmark questions and answers.

## Runtime, recovery and limitations

R15 jobs: S1ae83b61a6a, S2e2b5890c9d. R16: S19409c71734,S25fc9fa6014.
R17: S15d89ccd26e,S2f21b319939. All completed16/16, no inference errors.
R18 extraction2bf75cbdb9/a131ed2981 and trainingaf74c1c204 completed.
R18 QA jobs2b39ea6c13/129da56433 completed and are saved in avbind_active_jobs.json.
The coordinator avbind_r18_continue.ps1 downloads each stage and logs live
per-cell correct/wrong/errors/remaining; successful pilot conditions are skipped.
Do not submit completed rounds again. R18 model/feature checkpoints are separate
from R14; neither old checkpoint nor old feature caches were overwritten.
The live TIMELINE_NS switches to pooled keys/R18 calibration for R18; old named
APIs therefore require their original loader/config restored before any future
matched comparison. Merely calling an old function name is NOT a frozen variant.

CPU tests:95 passed, including causality/GQA, zero-gain identity, untouched media
rows and encoders, registry/hook cleanup, native answer-route refresh, physical
source pooling, draft cache provenance and condition-level resume idempotence.
These tests verify implementation contracts, not scientific performance.

No variant here establishes CVPR novelty, broad question coverage or deployment
quality. Future training must demonstrate reliable semantic occurrence retrieval,
then content-clock readout under independent source-time shifts and repeated
events, without gold-dependent calibration. A stronger multi-slot representation
and a trained source/time-aware readout are hypotheses, not completed methods.
Independent reverse/free-form, CMTB and broader-context evaluation is still
required before freezing a method and touching the400 reserve.

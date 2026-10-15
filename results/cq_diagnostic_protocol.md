# CQ bias diagnostic development

Date: 2026-10-14. Model: Qwen2.5-Omni-3B, the two existing Kaggle servers.

The previous 80-sample jobs were stopped at the user's request. A fixed
development set has five TA/B and five TF/C clips, each at most 50 seconds.
Selection uses seed 20261014, then duration/readability only; no prediction is
used for selection. Manifests contain every selected ID and source path.

This set is used repeatedly for method development, so its accuracy is not an
unbiased test result. A useful development target is at least four correct in
each group. A method must not implement a label-specific offset, select a
method based on task/gold, or always produce No/Future. Separate small control
manifests are frozen before method selection (seed 20261015), with TA/A and
TF/A,B. They are only used once the candidate is fixed to detect prior reversal.

All answer scores rotate option positions and sum per-content log probabilities.
This corrects letter-position preferences, not semantic Yes/Present priors.
All methods and raw scores are retained, including failures.

Round 1:

- Base and previous CQ (ordered aligned probe positions, incomplete phrases).
- Local CQ positions: each independent branch anchors to its own media clock;
  question follows the largest media/probe position, without cursor displacement.
- Source isolation: media rows cannot import the other modality through previous
  decoder hidden states or intervening text delimiters. Fusion occurs in Q/A.
- Balanced routing: allocate fixed total attention budgets (0.2 probes, 0.4 raw,
  0.4 remaining context) in layers 13–27, separately normalized per modality
  and per query/head. Retain a 10% uniform floor over time.
- Paired calibration: local CQ + alpha(local CQ - prompt-identical no-probe-media
  path), alpha in {1,2}. Raw Q/A access to media is retained in both paths.

Round 2, if round 1 fails:

- Materialize twelve-token CQ completions with isolated modality/chunk branches.
  Branches never see question/options or other branches; captions are cached.
- Compare Q/A reading completed CQ, final content tokens only, text memory,
  and memory with generic source/timestamp grounding instruction.

Research interpretation: none of these improvements alone establishes CVPR
novelty or acceptance. A paper needs a mechanism claim supported by ablations,
untouched balanced evaluation, multiple natural AV contexts, and runtime data.
Contrastive decoding already appears in CVPR 2024 (Leng et al.,
https://arxiv.org/abs/2311.16922). Qwen's native time alignment is TMRoPE
(https://arxiv.org/abs/2503.20215); interventions must preserve its media clock.

Observed round 1 (all ten valid): base TA/B 4/5, TF/C 3/5. Original CQ,
local positions, decoder source isolation, and fixed-budget routing all 0/5
in both cells. Partial-null calibration yields TA/B 0/5, TF/C 2/5. The failure
is semantic collapse, but its direction is not always Present: local CQ picks
Past on all five TF clips. This rules out treating a specific word as the sole
cause and does not establish any one causal explanation yet.

Additional candidate: contrastive residual CQ. The baseline Q/A path cannot
read the probe scaffold and keeps media-relative positions. In layers 13–27,
Q/A receives media-dependent endpoint values (real - matched null), with an
edge-specific RoPE shift from each extraction clock to a common reader clock.
Gains 0.3 and 1.0 and residual-key retrieval are compared to zero gain. The
decoder-only guard is necessary: an earlier implementation erroneously indexed
encoder tensors and triggered a CUDA device assertion. Those executions have
no valid accuracy and are excluded; affected GPU processes are restarted.

If extraction still fails, physical CQ uses independent native assistant
contexts on modality-isolated two-second crops and a signal silence floor
(RMS < 1e-4), then generic evidence reasoning. Query-referenced event checks
and interval graphs are prepared as a subsequent general method, with no
benchmark-specific answer mapping.

Round 2 observed (10/10 valid): completed CQ direct readout TA/B 1/5, TF/C
5/5; endpoints only 0/5 and 2/5; text memory 5/5 and 3/5; text memory with
generic source instruction 5/5 and 0/5. Generated probes often hallucinate
speech in silent chunks, so the memory's accuracy alone is insufficient.

Dual-reader candidate: score = completed-CQ score + beta * text-memory score.
The same beta applies to all options, questions, tasks and clips. Replaying
actual measured scores yields 9/10 for beta 0.5 and 1, and 10/10 for beta 2.
This is development-set selection, not test performance. Beta 2 is frozen
before running the six control samples. If controls expose No/Future collapse,
the candidate is rejected, and no weight is retuned on those controls.

Residual CQ completed on TA/B: native path 4/5; residual gain 0.3, gain 1,
and residual-key variant all 4/5. Same hard predictions as base, with changed
margins. In particular, the erroneous Yes prediction on item 1454 becomes
more confident with residual gain. There is no current evidence that residual
features solve the missing temporal grounding; scaffold bypass alone recovers
the base behavior. TF residual results have not been measured on a clean GPU.

Frozen dual-reader controls: TA/A 0/2; TF/A 1/2; TF/B 0/2 (overall 1/6).
Base on the same controls: 0/2, 1/2, 2/2 (overall 3/6). The dual reader is
rejected as an anti-bias candidate despite its 10/10 development accuracy.
This illustrates why scores on only No/Future clips are insufficient.

Physical context CQ: completed TA/B 5/5 (plain), 3/5 (grounding instruction),
4/5 (evidence analysis); completed TF/C 0/5, 0/5, 1/5. Audio generation
sometimes continues fabricated Human dialogue and confuses background sound
with speech. The next stage cleans that material and adds speech utterance
nodes via Silero VAD 6.2.3. Each parent utterance is audio-only and independent
of question/options; non-speech sound nodes remain available.

Hierarchy TF/C: 2/5 plain, 0/5 interval graph, 0/5 reasoning. Inspection shows
the model choosing the beginning of the video instead of locating the queried
event, sometimes claiming every visual node overlaps a single speech interval.
Next stage resolves concrete event references from speech+question (no answer
options), verifies each event against physical visual-only crops, and retains
both matched visual and original audio intervals. All classes are still read
by the model from the unchanged question, with no label mapping in code.

Server1 ngrok returned ERR_NGROK_3200 repeatedly during hierarchy execution.
Server2 remains available. Four downloaded physical memories from server1
were transferred to server2; the missing fifth is re-extracted there. The
original frozen IDs, seed, duration cap and model are preserved. Original
server1 hierarchy results are not claimed as complete or counted twice.

The user provided a new token for server1; it was saved only in the ignored
remote_config.json. The notebook is fresh, so caches were restored from local
artifacts and server2. Queue rebalancing canceled redundant TA fallback work.
Both groups resumed parallel execution of cross-source checks.

Round7 (before parser repair): TA/B 4/5 plain, 0/5 grounding, 4/5 reasoning;
TF/C 2/5 plain, 0/5 grounding, 0/5 reasoning. Many TA checks were empty because
the planner produced scalar audio_ids while the parser expected arrays. Some
plans also contained malformed JSON or an event linked to the wrong utterance.

Round8 repairs these failure paths, checks an event against all speech
transcripts to anchor its mention, and separates only the referenced matched
events into a compact provenance graph. No class or task determines inference.
The model performs three readouts, all reported: plain, automatically expanded
option propositions, and evidence reasoning followed by original-question
choice scoring. TA/B is 5/5 for all three. TF/C is 4/5 plain, 0/5 propositions,
and 5/5 reasoning. The proposition expansion often retains the vague words
Past/Present/Future rather than resolving them, so it is not selected.

Frozen candidate: verified-graph reasoning, 10/10 diagnostic development.
The API cq_provenance_answer accepts video path, question, options, and optional
observation cache, never gold or task. The same six bias controls are now
evaluated with fresh physical CQ extraction and this unchanged candidate.
These are diagnostic controls already seen in the prior candidate's analysis,
not an untouched final test set. No control labels are fed to inference.

## Final outcome

Later rounds explicitly tested the opposite-class collapse and replaced
peak-only timing with whole-event episodes and measured overlap. Round15
adds model-predicted lexical constraints to unchanged options, without CoT.
Final frozen evidence/readout: TA/B 5/5; TF/C 4/5; TA/A 2/2; TF/A 2/2;
TF/B 2/2. Total 15/16; the requested ten are 9/10. All sixteen were used
during development: neither subset is an unbiased held-out final test.
No task/gold enters the packaged inference API. See
[the final report](cq_diagnostic_final_report.md) for trajectory, limitations,
remaining error, raw artifacts and reproduction instructions.

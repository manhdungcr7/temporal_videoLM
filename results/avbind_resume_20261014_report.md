# Resume audit and completed follow-up diagnostics

## State preserved

At audit, both workers were idle (queue 0). Existing R3/R5/R6/R7 native
artifacts contain 7 + 9 unique completed samples with no errors. R4 training
and structural evaluation had already completed. No backbone restart, reader
retraining, manifest reselection, revert, or completed-condition rerun was
performed in this continuation.

R8 had only two completed smokes: S1 job `8cea2842d1`, index 1454; S2 job
`82b8b9dde0`, index 2097. Their actual results were recovered, not re-inferred.
Only the missing 6 TA and 8 TF samples were evaluated. Native baseline scores
were reused from the existing R7 run with the identical native prompt.

## Results

All cells below are development data, repeatedly used for method diagnosis.
Numbers are correct / evaluated. There is no independent generalization claim.

| Group | Native base | R7 independent KV | R8 differential keys | R9 source-budget diagnostic | R10 bounded readout |
|---|---:|---:|---:|---:|---:|
| TA/B | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 |
| TA/A | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| TF/C | 3/5 | 3/5 | 3/5 | 3/5 | 3/5 |
| TF/A | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| TF/B | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| Total | 10/16 | 10/16 | 10/16 | 10/16 | 10/16 |

No inference errors, no changed decisions. Target TA/B + TF/C remains 7/10;
other-label controls remain 3/6. None of these variants passes the research
success gate. CMTB and broader native tests of these variants are not completed;
the reserved 400-question, video-separated evaluation has not been run.

## What the diagnostics establish

- R7's seven TA records place 99.706--99.916% of normalized reader prior on
  audio. This measures transferred reader calibration, not final attention
  or proof that audio caused an answer error.
- R9 equalizes available source prior budgets while preserving within-source
  occurrence ratios and total non-null mass. It changes no prompt, question
  position, media, training weight, or question-type route.
- R9 measures actual joint-softmax shares at the last selected text row of each
  affected layer and option rotation. Across 16 samples these external shares
  range from 0.01784 to 0.999996. For 1454, the first affected read is 0.93866
  external; the last-layer read in the final rotation is 0.99991 external.
  Equal source priors alone therefore do not bound memory domination.
- R10 normalizes native/external attention separately and mixes them with a
  fixed, null-aware external budget capped at 0.25. Measured external shares
  range from 0.03088 to 0.249658. Despite this successful calibration change,
  the QA decisions do not improve.

R9/R10 are **diagnostic ablations**, not final novel methods. R8 also includes
the earlier Unicode-safe question-span fix; causal attribution to differential
keys alone would require a matched control. The reader was trained on
constructed textual supervision, not real native AV alignment. A plausible
remaining issue is the domain mismatch and insufficient learned content-clock
readout from compressed states. These experiments do not distinguish that
issue from perception errors, nor prove it is the sole cause.

## Next research gate

The next substantive direction is native AV occurrence supervision and a
content/source/clock-preserving readout, rather than more task-specific prompts
or label-specific gain tuning. Training clips must exclude all development
and reserved-evaluation videos. Tests should include source-time interventions,
repeated events, reverse visual-to-audio queries, free-form answers, and broader
scenes. A good development score alone would still not justify CVPR novelty or
general deployability. This AV training stage has **not** been run here.

## Reproducibility / continuation

- R8 jobs: S1 `a8af4a109d`, S2 `0aa35991e5`; completed 7/7 + 9/9.
- R9 jobs: S1 `d8929a90f9`, S2 `d6799fdc25`; completed 7/7 + 9/9.
- R10 jobs: S1 `5691c547a8`, S2 `6e4027a443`; completed 7/7 + 9/9.
- Both R9 loaders: SHA256 `0a1e73fcff70726662460d98c0a8e374fbca617d7a723d0126e68e83ffd226ec`.
- Both R10 loaders: SHA256 `5007c2abfe7b6ae2bcd46a32195f9e854e710da30265f7c06e9432771106ee29`.
- Both servers retain the same R4 reader and completed perception caches.
- Local artifacts: `r8-counterfactual-resume_s{1,2}.jsonl`,
  `r9-source-calibration-audit_s{1,2}.jsonl`, `r10-bounded-readout_s{1,2}.jsonl`.
- `server_jobs/avbind_progress.ps1 -Download -NameLike 'r10*'` shows per-cell
  correct/wrong/errors/remaining/target, including groups with no outputs.
  Completed successful retries are de-duplicated. Watchers save partial logs.
- `server_jobs/avbind_result_audit.py` provides a read-only distinct-index audit
  and explicitly separates reader priors from measured readout shares.
- 59/59 CPU tests pass, including executed resume/idempotency, null fallback,
  causal-mask and bounded-budget tests. These are not QA performance evidence.
- No credentials appear in this report; ignored connection files are unchanged.

# Fixed40 continuation: evidence use, not just attention mass

Frozen manifest: `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.
The same40 IDs remain in scope, five per cell, checked <=45s before selection.
The original baseline is complete and preserved. No completed method is rerun.

## Completed controls before the new response critic

Entries are correct answers out of five per cell. All rows below use all40 IDs.

| Method | TA/A | TA/B | TF/A | TF/B | TF/C | CMTB/A | CMTB/B | CMTB/C | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native baseline | 5 | 3 | 0 | 1 | 4 | 3 | 2 | 1 | 19/40 |
| Count-normalized attention | 5 | 3 | 0 | 1 | 4 | 3 | 2 | 1 | 19/40 |
| argon symmetric soft handoff | 5 | 3 | 0 | 1 | 4 | 3 | 2 | 1 | 19/40 |
| argon text PMI | 5 | 1 | 0 | 1 | 4 | 3 | 2 | 1 | 17/40 |
| Raw-AV option-blind CoT | 4 | 0 | 5 | 0 | 0 | 3 | 2 | 0 | 14/40 |
| Independent raw-source KV | 5 | 2 | 0 | 1 | 4 | 3 | 2 | 1 | 18/40 |
| Independent raw-source KV + AV boost | 5 | 3 | 0 | 1 | 4 | 3 | 2 | 1 | 19/40 |
| Faithful source memory + raw AV | 5 | 0 | 2 | 1 | 4 | 4 | 2 | 3 | 21/40 |
| Interval graph + raw AV | 5 | 0 | 4 | 0 | 3 | 4 | 2 | 2 | 20/40 |
| Verified graph v1 + raw AV | 4 | 1 | 5 | 0 | 3 | 4 | 2 | 2 | 21/40 |
| Verified graph v1 + CoT | 1 | 5 | 5 | 0 | 0 | 4 | 2 | 2 | 19/40 |
| Verified graph v2 + raw AV | 5 | 0 | 3 | 0 | 4 | 4 | 2 | 1 | 19/40 |
| Verified graph v2 + text reader | 5 | 0 | 4 | 0 | 2 | 4 | 3 | 1 | 19/40 |
| Verified graph v2 + goal-preserving CoT | 3 | 3 | 5 | 0 | 0 | 3 | 3 | 0 | 17/40 |
| Verified graph v2 + confidence fusion | 3 | 3 | 3 | 0 | 2 | 4 | 3 | 0 | 18/40 |
| Proposition direct verifier | 1 | 4 | 3 | 0 | 0 | 4 | 2 | 1 | 15/40 |
| Proposition prior-calibrated verifier | 5 | 0 | 0 | 2 | 3 | 3 | 3 | 1 | 17/40 |
| Proposition proof verifier | 2 | 2 | 4 | 0 | 0 | 2 | 1 | 2 | 13/40 |
| STCA raw-AV verifier | 5 | 1 | 3 | 1 | 0 | 4 | 2 | 1 | 17/40 |
| STCA text verifier | 0 | 5 | 5 | 0 | 0 | 4 | 4 | 0 | 18/40 |
| Matched frozen text verifier | 0 | 5 | 5 | 0 | 0 | 3 | 4 | 0 | 17/40 |
| Qwen3-4B direct evidence reader | 5 | 2 | 4 | 1 | 0 | 4 | 3 | 0 | 19/40 |
| Qwen3-4B question-only control | 5 | 0 | 5 | 0 | 0 | 2 | 2 | 2 | 16/40 |
| Qwen3-4B option-blind thinking reader | 2 | 4 | 5 | 2 | 2 | 4 | 3 | 0 | 22/40 |

The initial custom-attention-name run is excluded because causal mask dispatch
was not preserved. Initial source-memory runs are excluded because generation
included invented subsequent turns. They remain saved as implementation-fault
diagnostics, not valid evidence of an improved method.

## What the traces actually establish

- Attention redistribution changed inspected attention statistics but did not
  improve decisions. This does not prove that attention is irrelevant; it rules
  out these particular uniform interventions as a sufficient fix.
- CoT can replace one class bias with another. Getting5/5 TF/A while TF/B and
  TF/C are0/5 is not a solution to temporal reasoning.
- Raw source isolation alone is insufficient (18/40). Preventing cross-source
  mixing does not teach the reader to identify events or use their time bounds.
- VAD word-fragment ASR lost semantic context and sometimes changed languages.
  Merging utterances fixes concrete transcript fragments, not necessarily QA.
- A ranked visual segment can be the best of uniformly negative evidence.
  Such a ranking must not be reported as a detected occurrence.
- Some new analyses assign speech intervals to the visual event, or contradict
  their own interval comparison. For idx2150, even negative visual probes do not
  prevent a simultaneous-event explanation. For idx1660, the analysis lists
  overlapping bounds yet calls the visual event later. These are observed reader
  failures, not proof of a specific neural mechanism.
- Visual descriptions themselves are imperfect. Old cached descriptions can
  contain unsolicited assistant prose. Dense sampling or a better captioner
  would require a separate controlled perception experiment, not a silent change
  to the baseline inputs.

## Verified graph v2 and response critic

V2 merges speech regions into meaningful episodes, retains word timestamps,
permits references to planned/hypothetical events, adds a null visual alternative,
and repeats the original question at final readout. The first two submissions
failed before any sample because the older source-contrast job assigned an array
to the global name `av`, shadowing PyAV. Restoring the import and renaming that
score variable fixes the implementation; original results are untouched.

The response critic is a general QA-response verifier, not a task router:

1. Expose independent silent-video observations and ASR word-time evidence in a
   compact representation; keep the original raw AV accessible.
2. Assess each proposed response independently, without answer letters or other
   candidates. Option labels are only an evaluation wrapper.
3. Compare direct verification, evidence-minus-question-only log odds, and a
   claim/proof/judgement readout. Store the exact claim and proof for auditing.
4. Repeat the original question and response at judgement to preserve direction
   and referents. No TA/TF/CMTB-specific extraction or rule is used.

This architecture can verify an open-ended generated response, but effectiveness
outside this MC development set is NOT established. Source perception is still
speech-heavy and lacks explicit speaker identity and non-speech sound memories;
raw AV availability alone is not evidence of broad practical coverage.

Related prior art:
- [CVPR2025 video-grounded entailment](https://arxiv.org/abs/2501.05069): candidate
  statement verification is established prior art. This critic is a diagnostic
  adaptation, not a reproduction of the full entailment tree or a novelty claim.
- [LatentOmni](https://arxiv.org/abs/2605.22012): learned latent alignment and
  synchronization distinguish its method from arbitrary untrained clock vectors.
- [Controlled verifiable-reward study](https://arxiv.org/abs/2610.03792): higher
  task accuracy alone need not establish temporal-order grounding. A future
  learned adapter needs counterfactual time/order tests and broad retention tests.

All trials and selection on these40 examples must be disclosed. Improvement here
is development evidence only; neither generalization nor CVPR acceptance follows.
Job IDs are recorded in `research40_jobs.json`; live per-cell correct/wrong counts
are available via `server_jobs/research40_monitor.ps1`.

## Next learned control: STCA pilot

`research40_stca.py` implements rank8 query-tail Q/V adapters in layers
18,20,...,34, with the original encoders, raw-media KV and original weights frozen.
64 synthetic counterfactual pairs and16 separately seeded synthetic validation
pairs cover relative order, simultaneous observation, reverse-source quotation,
durations, completion, and source identity in varied everyday contexts. There are
no benchmark question strings, clip contents, development labels, task routers,
answer-zone vectors, or event-to-clock rules at inference. Source/time truths are
used only to generate synthetic training labels.

This is supervised evidence-reader adaptation, NOT raw AV alignment training.
The synthetic construction is necessarily limited; broader audio, spatial,
causal, multi-speaker, and open-ended coverage is not established by this pilot.
Server1 will train once, and server2 must use that exact checkpoint. Frozen and
adapted readers are compared with the same evidence and question prompts. A
successful synthetic control or development score would not establish novelty;
counterfactual media tests and unseen-video evaluation remain necessary.

STCA completed:128 optimizer steps,64.06 seconds for training plus synthetic
evaluation. Separate synthetic validation increased18/32 to21/32. The source-ID
control accounts for two of the net three added correct answers; several temporal
families remain2/4. Thus this is not evidence of strong learned clock reasoning.
All40 real examples are complete using the same1,856,391-byte checkpoint on both
servers. Raw verifier improved15->17 versus its matched frozen direct control,
but is still below native baseline19/40. Text verifier improved17->18. Large
class failures remain. No original model weights were updated.

The stronger-reader diagnostic uses public Qwen3-4B (revision recorded per row)
and exactly the existing source observations. It compares question-only priors,
direct reading, and option-blind thinking. It is explicitly a different-reader
control, not a same-backbone gain or a paper contribution by itself. The added
model is released after the job, preserving the original Omni runtime.

Direct4B reading is complete19/40, not a sufficient fix. Thinking is complete
22/40, still0/5 CMTB/C and below the development criterion. Several traces use the whole speech episode as the relevant
time reference despite the word clocks; others reject an event because a free
caption omits its object, even when an event-conditioned silent-video probe is
positive. All generated analysis traces preserve `hit_budget`;512 tokens may
truncate analysis and that limit must not be hidden in conclusions.

Next queued method: atomic provenance. A question/option-blind extractor proposes
up to six event references with exact contiguous quotes. Deterministic matching
to ASR words supplies mention clocks; invalid quotes receive no fabricated time.
Each neutral event is searched on every2s silent-video window with a null option
retained. Evidence stores occurrence support separately from mention time, with
computed temporal relations as features rather than a relation->QA-answer rule.
Compare original raw-AV reader and4B evidence readers. Original observations and
non-event utterances remain accessible; no fixed source-first QA path is used.
This preprocessing is still speech-heavy, and neither non-speech sound coverage
nor broad open-ended effectiveness has been established.

A read-only sampling audit is queued after atomic provenance. Official Decord
sampling uses rounded linspace source indices, whereas earlier chunk slices use
floor/ceil at the returned nominal sampler fps. These are not identical clocks;
the size of any outside-window excursion must be measured before claiming exact
chunk isolation. The audit records source frame PTS and first decoded audio PTS,
and exports native-frame storyboards for1449,1528,2054. No clips or baseline input
are modified. Event observation windows are not exact event onset/duration labels.

Atomic v1 early traces:1449 retained a long compound utterance;2054 resolved the
affected entity in later clauses as egg wash instead of chicken. These are parser
failures in the source references, before the final answer decoder. Atomic v2 is
queued as a separate version, not an overwrite. It adds general predicate/argument
extraction with non-benchmark coreference examples, whole lexical-token exact
quotation matching, all clocks for repeated phrases, exact native source-frame
PTS membership, and a sparse uncertain semantic-search index. A visual-window ->
heard-words co-occurrence index permits reverse lookup without forcing an
audio-first question policy; co-occurrence is explicitly not semantic equivalence.
The source extractor remains question/option/label blind. Only existing source
perception settings are used; no development gold calibrates the retrieval.

The synchronizer now validates downloads before replacing local artifacts and
retains local completed IDs if a restarted remote has fewer records. Use `-Phase`
to download/report only the active method family; `-LocalOnly` reads saved results.

## Timing audit: a more fundamental input-clock concern

Both20-clip timing audits completed. Relative to Decord's reported source PTS,
the nominal2s crop rule has maximum early excursions5.7840s (server1) and4.7760s
(server2), affecting240 and253 windows respectively. This is much larger than
ordinary one-frame rounding. First decoded audio PTS is0 on both shards. Native
frame storyboards1449,1528,2054 expose negative initial video timestamps and
highly nonuniform presentation intervals. Example1584 has68 native frames at
~1.998fps despite a29.104s audio/container clip, with initial video PTS~-4.922s.

Do NOT infer that Decord's timestamp API is authoritative solely from this audit.
Cross-decoder validation and an independent presentation-clock rendering control
are needed. The queued PTS-clock experiment decodes PyAV frame PTS, compares them
to Decord PTS, renders a regular~2fps grid on the shared clip clock, and preserves
audio PTS/gaps rather than concatenation assumptions. It keeps the same spatial
budget and question prompts, no ASR/question router, and the original baseline.
Corrected encodings are cached separately, with distinct native/source-KV cache
keys. Compare native decoding, late AV boost, and source-independent KV under the
corrected clock. No accuracy claim is made before the run completes.

Atomic v1 hit OOM when the4B reader prefills an expanded dense graph. Only7/40
were saved at the first failure. Bounded256-token causal prefill and greedy decode
are now supplied as a helper; no source evidence is truncated. A synthetic
full-vs-chunked numerical control runs before resume. Resume jobs process only
missing IDs and reuse cached source extraction/probes. V2 remains a separate run.

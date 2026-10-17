# CAOR-WEMD Stage-0 v6 — execution protocol, 2026-10-16

User authorized execution on the two freshly supplied Kaggle/ngrok runtimes, following the discussion that the last two preflight questions can be resolved locally. No proposer response is required to clarify those reporting/testing claims.

Original attachment is preserved byte-for-byte in `stage0_pcebr_v6_final_original.py`. LF-normalized SHA256: `6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515`. The wrapper removes only the automatic RESULT assignment, invokes the original phase functions in their original sequence, and inserts observation callbacks for checkpoints, per-query score matrices, episodes, probe logits, and M3 intermediates. No tuning by QA labels.

M0: original checks unchanged; a separate audit compares the interior-crossing W+/W− example against analytic values [1/96,25/96] and dense numerical quadrature. This completes the validation absent from the original M0; it is not described as an original-source check.

M1/M2: same four cached embedding files (252 pseudo description/media pairs, 202 train from 16 origins, 50 held-out from 4 origins already seen), original three M1 seeds 20261015/16/17, original training steps/optimizers. Same cache on both servers creates replicas, not independent additional data. Pseudo descriptions and null filters are not independently annotated event/time/absence ground truth.

M3: fixed first four samples of each manifest shard, rotation0 only. Server1 IDs 1449,2150,1929,2054; server2 IDs 1956,1513,2264,1821. Decoder caches rebuilt from the same frozen prompts/video preprocessing and actual duration<=45s selection. Native replay checked within the current runtime. The full research40 and reserve400 are not part of this Stage-0 request.

Legacy v5: original legacy branch unchanged. A separate audit also invokes the archived v5 extractor, clocks, VO, bivector and original nested routing function on the same inputs/runtime. Report numerical parity with that replay; do not presume bytewise identity with a historical hardware/runtime run.

Inference uses the source's two-pass CAOR path. The model backbone is frozen; M1 aligners/M2 bridges/probes are trained separately. The source's M3 CAOR path does not automatically become a learned M1/M2 QA integration. Freeform scoring remains heuristic and will be read alongside generated text.

GPU failures, wrapper errors, or any necessary operational fixes will be recorded separately before retries. Source changes affecting algorithmic choices require explicit disclosure and will not be hidden in mainrun results.

Completion: both mainruns, all original-control accuracy audits and archives completed. M0/M1/M2 replicas match. Native/v5/full CAOR and every listed MC control obtain5/8 with identical original choices. See report.md and completion_audit.json.

The first attempt hit a wrapper-only JSON Tensor serialization error after completing M1. The continuation fixes metadata serialization and reruns M2 with the same fixed seeds/budget before M3. Failed-attempt bridge/episode files are retained. The raw archives preserve the actual original runner plus continuation. The current server_jobs/pcebr_stage0_v6_run.py also incorporates that serialization fix for future launches; runner_jsonfix_for_future_runs.py is its local snapshot, not a claim of a separate complete end-to-end rerun. Algorithm source SHA remains unchanged.

After mainrun, the three fixed random-seed controls and irrelevant-slot control already present in the source were replayed only to record their accuracy, since the original mainrun reports their TV. Full replay error0 verifies reconstruction. No generation or training was repeated in that audit. Each control also obtains5/8.

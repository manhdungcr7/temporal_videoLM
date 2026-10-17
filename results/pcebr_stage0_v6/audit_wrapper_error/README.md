# Observation-wrapper error and recovery

The first execution completed M0 and both M1 cache variants on both servers. At the first M2 episode capture, `v6_write('instrumentation.json', V6_INFO)` encountered a Tensor inside the `media_rec_feats` keyword argument. This produced a JSON serialization TypeError. It was an audit-wrapper error, not a failure of the submitted algorithm.

The source algorithm remains unchanged. The continuation adds a JSON default encoder recording tensor shape/dtype/device in metadata; full tensors remain in raw PT artifacts. Completed M1 results/models are reused. Three bridge fits and the interrupted episode artifact from the failed M2 attempt are retained in the remote `audit_wrapper_error` directory before rerunning the original M2 phase with its original fixed seeds and budget. The original job logs and wrapper are kept locally here. M3 had not begun when this error occurred.

The raw archive includes the interrupted attempt and the continuation script. This is disclosed separately from the final completed mainrun.

The primary local runner now includes the same JSON encoder fix for future reruns. The executed wrapper preserved in the raw archives remains the original runner plus continuation, so its source need not have the same hash as the subsequently repaired local runner. The algorithm attachment is unchanged in all cases.

# CLRV v8 — code, protocol, and artifacts

- `stage0_pcebr_v8_original.py`: original supplied source, SHA-256 `fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895`.
- `stage0_pcebr_v8_corrected_standalone.py`: packaged corrections, SHA-256 `25043c391ac1503280c2e4b70c4e38d27d3f8831741e4a38f8f96ffc4b758843`. M0 passes CPU and CUDA on both servers. It needs the existing server globals `MODEL_`, `PROC`, `OUT`, and `LAB_SERVER`, the frozen40 manifest and decoder caches; it evaluates the original four-ID micro shard at rotation0.
- `report.md`: all three micro8 variants and component ablations.
- `report40.md`: fixed-scale CLC-only, CAOR-only and full-direct confirmation on frozen40, without the rollout bridge. Reports the 32 IDs outside micro8 separately.
- `feedback_to_yoshua_bengio.md`: findings and research recommendations.
- `all_runs_artifacts_s*.tar.gz`: JSON logs, input cache hashes, all saved per-sample tensor snapshots, actual effective sources and controls. Archives and entries are hash-checked; view-backed tensors were compacted with bitwise equality checks, preserving the original file hashes in the archive audit.
- `verification.json`: hashes, unchanged backbone parameter versions, identical algorithm source across shards, and exact replay of stored direct probabilities for component ablation.

The corrections are operational/device handling, whole-word labels for BPE pieces, relation-clause boundary priority, general modality adjectives, endpoint-to-physical-modality profile projection, modality-aware deictic foreground gate, and honoring prefix caching on a zero-gate bypass. No CLRV/gate coefficients were tuned and no weights were trained. The output filename of the packaged source is changed to avoid overwriting historical v7/v8-original results.

The original and both repair stages remain separate. `logicfix` is the first repaired interface; `corrected` also fixes initial relative-clause boundaries and modality adjectives. Effective runtime sources are saved as `effective_corrected_s1.py` and `effective_corrected_s2.py`. The standalone package embeds the helper and the CPU-device wrapper into the same algorithm; AST formatting differs from those snapshots.

Reproduction drivers are in `server_jobs/pcebr_stage0_v8_*.py`. `pcebr_stage0_v8_run.py` records the original micro protocol and additional bridge baselines. `pcebr_stage0_v8_component_ablation.py` isolates final decode CAOR, CLC and continuous thought, using saved latent directions and verifies probability replay. `pcebr_stage0_v8_clc40.py` removes rollout, wrong-query and legacy passes, while keeping the same option-free counterfactual direction construction and fixed CLC scale .042. It resumes partial shards by ID, retaining all 40 fixed samples.

This is development evaluation, not a new held-out test. The 40 questions were already used in previous research rounds, micro8 is a subset, and all tests here use rotation0. The reserve400 set was not opened. Heuristic regex freeform scores are not treated as independent semantic annotations. No external message was sent to Yoshua; the feedback file is ready to share.

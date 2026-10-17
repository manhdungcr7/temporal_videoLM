# Autonomous temporal research — 2026-10-16

Objective: find a general AV temporal QA improvement with falsifiable mechanisms and matched controls. No task router, benchmark answer-word steering or fixed VAD anchor. CVPR/oral acceptance is a target, never an experimental result. Existing research40 is repeatedly viewed development data, not a clean final test.

R1 hypothesis: serial causal media flow may restrict joint evidence access. Counterevidence: Causality Matters and V-CORE report useful temporal information from causal order, so unrestricted bidirectionality may hurt. The diagnostic retains native RoPE and text decoding; changes only media-media edges at fixed zero-based blocks0–11 or14–25. Six conditions fixed before outcomes: native; explicit native-mask parity; all-media bidirectional middle; cross-modality bidirectional middle; TMRoPE-clock-causal middle; TMRoPE-clock-causal early. Clock is processor/model timing, not annotated physical event truth. No gold-dependent inference.

Micro first4/shard, rotation0, exact IDs1449/2150/1929/2054/1956/1513/2264/1821. Source manifest prefilters audio/video duration<=45s. Diagnose cheaply, then promote only a defensible candidate with matched controls to all40 and rotations. Do not select per-task variants or retune hyperparameters against these8 gold labels. Source/logits/seconds/mask edge counts are saved. No test400 or held-out labels opened for training.

Further research must separate evidence retrieval, temporal binding and final decoder use. Cache reconstruction/latent sensitivity without QA gain is already a negative result; repeated heuristic centering/gating is not a new method.

R1 results: native5/8; explicit native5/8 parity0; bidir-all5/8; bidir-cross4/8; clock-causal-mid5/8; clock-causal-early5/8. No rescue; cross-only harms1449. Do not promote these topology-only candidates or claim all attention methods impossible.

R2 hypothesis: media features may lack question intent before final readout. Preserve native media-media causal topology, open only media→known question-stem edges during prefill; no MC choices are exposed by new edges. Fixed blocks0–11 versus14–25. Controls: native, format-only feedback, audio-only and video-only feedback. Format control is not matched edge-count, so it cannot isolate content perfectly. This is a mechanism diagnostic with precedents in question-conditioned grounding, not a novelty claim. Strict token offset roundtrip and option exclusion assertions required. Same8 IDs/rotation0.

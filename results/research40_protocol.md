# Fixed development40 protocol

Dataset: public SVHalluc temporal; native Qwen2.5-Omni-3B.

Seed 2026101419. Manifest SHA256: `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.
Five distinct-video samples per TA(A/B), TF(A/B/C), CMTB(A/B/C).
All media container/stream durations and decoded audio durations were checked <=45 seconds before selection was frozen. Each server owns20 samples. No inference-time duration skipping or replacement samples.

Baseline uses original raw AV, native frame sampling, bounded spatial resolution, and all cyclic answer-order rotations. Each rotation's decoder inputs and baseline logits are cached, so later methods reuse the same encoded media and do not recompute baseline. Labels appear only in selection and evaluation.

Development criterion: at least30/40 correct, at least3/5 in every cell, and an improvement over the fixed baseline. This is a practical development target, not evidence of CVPR acceptance or held-out generalization. Repeated method selection on40 examples must be disclosed. Prior large validation overlaps will be recorded and excluded from subsequent held-out evaluation.

Related work inspected:
- [SVHalluc](https://arxiv.org/abs/2606.02642): separates speech-mentioned event content, occurrence time, and actual visual evidence.
- [VAR](https://arxiv.org/abs/2503.03321): visual attention redistribution is head dependent; large attention alone does not establish useful evidence.
- [Latent Visual Reasoning](https://arxiv.org/abs/2509.24251): visual latent states use explicit training; arbitrary hidden-state feedback is not an equivalent implementation.
- [LatentOmni](https://arxiv.org/abs/2605.22012): latent AV reasoning uses feature supervision, synchronization, and training data. A training-free probe cannot claim reproduction.
- [Daily-Omni](https://arxiv.org/abs/2505.17862): eventual evaluation must cover more varied everyday AV questions.

Argon proposal assessment:
- Key scaling below1 can increase negative logits; use additive biases for explicit suppression.
- Claims about95% negative logits,99% sinks, exact prior cancellation, and guaranteed no OOM are unverified.
- Fixed audio-to-video handoff, single speech-window gating, and Yes/Past/Present/Future feature injection conflict with the requested question/context generality. Those components will not be treated as a general method.
- Experiments using argon's text-only calibration or adapted attention handoff carry `argon` in their names.

First experiments: `argon_text_pmi` (base minus0.75 text-only log probability); additive count correction with unchanged combined AV attention mass; symmetric soft source handoff; general raw-AV evidence reasoning. Hyperparameters are shared across every task and label. Methods are reported individually, including regressions.

Implementation audit: initial attention jobs used a custom Transformers attention implementation name. This can bypass the causal-mask registry. Initial outputs (`research40_attention_s*.jsonl`) are excluded from method comparisons. Corrected v2 jobs replace the existing sdpa registry entry temporarily and preserve native mask dispatch; baseline and frozen samples are unchanged.

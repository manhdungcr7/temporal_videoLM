# Source/time latent-slot optimization — prospective development protocol

Recorded2026-10-15 after failed directional-feedback families and the successful
gradient feasibility probe. The probe optimized8 visual slots for8 steps,
reducing internal reconstruction loss.2823->.2311, using6.12/4.16GiB peak across
twoT4s. This is a technical observation, not QA accuracy or a factuality result.

New experiment: freeze every backbone parameter; optimize only small continuous
input slots per recording, without question/options/labels in the target. Two
source-only teacher passes provide block22 sensory states. Pool each source in
fixed2s groups, keeping source and native time positions separate; do not treat
simultaneous sound and image as semantically identical. Initialize slots from
pooled native encoder embeddings. Each source's slots read its own source-only
prefix and cannot access omitted-source keys. Fit block22 states for32Adam
steps, lr.05, cosine+0.1within-source contrastive loss (temperature.1), with.001
input regularization and a.5initial-norm trust region. Backbone remains unchanged.

Export optimized slots' per-layer KV; merge them in chronological order into
the native AV prefix. Original AV keys and query RoPE positions are retained.
Only query reading from block23 onward sees the added memory; earlier blocks
mask it out. Count-normalized memory bias log(native media count/slot count) is
shared for every question. No speech window, answer-word vectors, task router,
hand-coded relation readout or fixed modality handoff.

L0 matched native; L1 unoptimized memory with identical layout/bias; L2 fitted
source/time memory; L3 the same32step budget with within-source target-time
permutation, seed42. This last control preserves the marginal features but
breaks their assigned times. Run all40 fixed<=45s samples and110 cyclic answer
rotations. Optimize once per clip, reuse across rotations, never calibrate from
gold. Native replay and zero-memory identity tolerance<=.06 recorded BEFORE QA
execution (FP16 cache paths can differ by a small number of ULPs); fail rather
than replace sample on larger error. CPU causal/source checks and a one-example
GPU smoke precede full launch. Restore all parameter requires_grad flags.

Interpretation: source-only teacher states are internal model features, not
ground-truth events. Reconstruction alone may not improve reasoning. L2 must
beat L1 and L3 before attributing any gain to source/time alignment. Native
sampler clock limitations remain, and no CVPR novelty/generalization claim is
made from development40.

Prior art: Unsilencing Visual Latents https://arxiv.org/html/2605.02735v1 already
optimizes frozen-backbone latent tokens at test time; LatentOmni
https://arxiv.org/html/2605.22012v1 uses sensory alignment and temporal latent
representations with training. Here the experimental change is source-isolated,
time-indexed reconstruction and KV readout, not the invention of latent
optimization or weighted embeddings. Earlier local raw-source KV/reader controls
did not optimize native slots; L1 explicitly tests whether optimization helps.

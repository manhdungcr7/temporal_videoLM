# Reversible evidence focus: prospective development protocol, 2026-10-15

Reuse the exact duration-prefiltered random development40, manifest SHA256
0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459.
Five samples per TA/A,B; TF/A,B,C; CMTB/A,B,C. No replacements, all cyclic
answer rotations, same frozen encoded native AV as Argon M0/M2. This is heavily
used development data, not a held-out test. Historical M2=26/40 and predicts
Present on all TF; historical M0=18/40. Labels enter only sampling/evaluation.

Primary question: does retaining global evidence access avoid M2's temporal
collapse? Mechanistic audit: D1 removes clock from M2; D2 restores native full
attention after block23; D3 restores global query attention. These diagnostics
retain old speech selector / vocabulary-derived components and are not claimed
to satisfy the requested generality.

General candidates G1/G2/G3 use no speech window, answer-word clock, vocabulary
projection, task router, answer-text semantics, fixed modality order, or PMI.
G1 penalizes existing causal cross-source media edges farther than3s by3.5;
query tokens retain all original AV keys. G2 removes that penalty from block24
onwards. G3 replays native AV while adding a same-layer contrast between the G2
and native question states at blocks18 and22, magnitude0.20 relative to native
query norm. Native RoPE and all causal exclusions remain intact in G methods.
G3 is a latent feedback hypothesis, not a reproduction of trained latent models.

All methods and fixed parameters are recorded before execution; no cell-specific
parameter selection. Development target30/40 and at least3/5 per cell. Report
every candidate, replay error, predictions, paired rescue/regression and TF
prediction distribution. Novelty and CVPR suitability remain unproven. If a
general candidate succeeds, freeze it and test independent questions and
contexts, modality/time counterfactuals, and untouched data before paper claims.

Related primary sources read:
- https://arxiv.org/html/2605.22012v1: LatentOmni uses training, feature grounding
  and temporal synchronization; arbitrary Qwen feedback is not equivalent.
- https://arxiv.org/html/2605.02735v1: Unsilencing Visual Latents uses contrastive
  and confidence-progression optimization, not just weighted soft embeddings.
- https://arxiv.org/html/2606.02642v1: SVHalluc distinguishes mentioned event
  content from visible occurrences and their relative timing.
- https://arxiv.org/html/2503.03321v1: VAR motivates inspecting attention heads;
  large attention weight alone is not evidence of useful grounding.

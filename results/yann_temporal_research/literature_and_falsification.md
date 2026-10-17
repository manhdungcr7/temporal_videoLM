# Literature and falsification log — 2026-10-16

Primary sources checked in this run; do not attribute our prototype performance to them.

| Source | Relevant evidence | What it does not establish here |
|---|---|---|
| [Causality Matters](https://ojs.aaai.org/index.php/AAAI/article/download/37856/41818) | In its video models, causal frame interactions and query pathways matter for order sensitivity. | No universal impossibility theorem for changing RoPE, attention or Omni3B. |
| [V-CORE](https://arxiv.org/abs/2601.01804) | Uses a causal temporal projector and reports order/causal gains. | Opening all future media edges is not automatically beneficial. |
| [FarSight](https://arxiv.org/html/2505.16652v1) | Attention-register decoding modulates causal normalization; future values remain excluded. | Not a test of our actual bidirectional media mask. Attention normalization is prior art. |
| [SyncRA](https://arxiv.org/html/2609.34363v1) | Intermediate within-video correspondence training is combined with answer supervision. | Its reported7B/MoE gains do not validate our3B/cache probes or guarantee narrated events coincide. |
| [LatentOmni](https://arxiv.org/html/2605.22012v1) | Trains sensory-grounded latent reasoning with feature targets and positional synchronization. | Memory, feature supervision or latent tokens alone are not novel contributions. |
| [GroundFormer](https://arxiv.org/abs/2608.15708) | Conditions visual grounding on question intent via communication tokens. | Question feedback alone is not a new research claim; Omni temporal transfer still needs evidence. |
| [QViC-MF](https://arxiv.org/abs/2603.15167) | Question-guided compression and memory feedback for long videos. | A feedback/memory diagram alone does not establish novelty. |
| [Perception Test official repository](https://github.com/google-deepmind/perception_test) | Human action/sound intervals and QA; public training videos. | Coarse labels/intervals are not perfect semantic endpoints, and our tiny split is not subject-disjoint. |

R1 preregistered mask interventions do not rescue any wrong micro answer. Cross-modality bidirectionality harms one. Thus topology-only candidates tested here are not promoted. This is consistent with the risk that indiscriminate fusion blurs temporal order, but the result does not identify that causal explanation on its own.

R2 all question-only feedback conditions remain5/8. Media representations and logits do change, so lack of an executed hook is not the explanation. This weakens the idea that exposing query intent alone suffices, without ruling out a trained query-conditioned encoder.

R3 probes whether independently supervised retrieval and answer-trained readout help together. The public train/validation pool is frozen before outcomes. TAM-Lite differs from full TAM: the small residual is applied at the frozen vocabulary head, not through trained intermediate decoder adapters. Interpret a null result at that scope; do not present it as an exhaustive disproof of temporal memory. Equally, do not claim a reduction in teacher-forced NLL is semantic QA improvement.

Outstanding novelty hurdle: prior art already includes grounding, memory, question feedback, temporal contrast and equivariance. A strong contribution would require an experimentally identifiable content–time–source binding mechanism that improves real generated reasoning under independent temporal interventions and transfers across modalities/contexts. No such result has yet been obtained in this run.

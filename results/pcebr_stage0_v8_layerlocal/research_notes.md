# Research and self-critique, 2026-10-17

## Why test layer-local directions

[Tuned Lens, Belrose et al.](https://arxiv.org/html/2303.08112v4), Sections2–3: directly applying final unembedding to intermediate states can be biased; learned layer-specific affine translators reduce this on their tested language models. This supports testing representation transport, but neither proves failure in Qwen2.5-Omni nor licenses calling intermediate slot logit outputs calibrated event probabilities. Our round uses no translator training.

[CAA, Rimsky et al.](https://aclanthology.org/2024.acl-long.828/) already builds residual steering directions from contrastive activations. Our layer-local experiment is a diagnostic refinement of v8, not evidence of publication novelty.

Repo counterevidence considered before running: v5 per-layer LN/VO bank correction stayed5/8, including random and reversed controls. Thus per-layer coordinates alone are not an established solution. The present round measures differences from native counterfactual decoder passes at each target layer, unlike v5's synthetic content/order vectors. It still inherits v8 endpoint grounding errors and treats routing reversal as a perturbation, not a certified semantic inversion.

## Why another verifier/contrast name is insufficient

[SEASON](https://arxiv.org/html/2512.04643v1), Section3, already uses temporal homogenization and attention-divergence-weighted contrastive decoding. Its visual temporal corruption is different from proving joint AV event binding. A new weighted contrast of original/corrupt logits would need a contribution beyond that pattern and must distinguish sensitivity from correctness.

[VideoMind](https://arxiv.org/html/2503.13444v1), Section3.3, already proposes moments, expands each candidate, and uses an SFT-trained verifier's yes/no scores to rerank. Merely replacing v8's parser with a planner/grounder/verifier pipeline is not a novelty claim. Critically, a trained verifier cannot be assumed equivalent to prompting frozen Omni to self-verify.

[On the Consistency of Video LLMs](https://arxiv.org/html/2411.12951v1), Sections4–5, reports poor holistic verification consistency among tested open models and introduces event/temporal verification tuning. This is counterevidence to assuming that a correct-looking grounded span plus a fluent rationale necessarily gives reliable verification. It motivates measuring that capability independently before building a reranking method around it.

[When Vision Speaks for Sound](https://arxiv.org/html/2605.16403v1) uses mute/shift/swap diagnostics and intervention-based post-training. Their results separate synchronization, audio existence and sound consistency: improvements in one do not automatically transfer to the others. Our previously tested Perception48 shift-training rounds likewise failed to transfer learned retrieval to benchmark QA; copying a shift loss with a new name would not address that evidence.

[AVTrace](https://arxiv.org/html/2609.19991v1) explicitly studies observable behavior rather than internal mechanisms, with silver-standard grounding and reasoning categories. It cautions against equating text overlap with temporal localization. The earlier proposal's description of AVTrace as proving a particular counterfactual hidden-state interaction mechanism is stronger than this paper supports.

## Next question if this round fails

Measure whether frozen native decoding can distinguish a queried event in a true interval from a same-recording wrong interval on independent, existing human interval annotations, with source modality masked separately. Compare native verifier odds, conditional description likelihood, raw intermediate-manifold affinity, and shuffled interval controls. This is a diagnostic proposal, not a validated method or new idea. Use fixed public development origins, remove invalid mute-state labels, track repeated event instances, and preserve a source-disjoint evaluation split. Do not use SVHalluc QA gold to produce or rank intervals.

If this verifier diagnostic fails, a system built on untrained self-verification has lost its key premise; change the grounding objective/data rather than add a downstream gate. If it works, measure whether correct event-instance evidence causally improves a general answerer before constructing an event graph. Only then can a stronger method be specified and tested; the desired QA improvement remains unachieved at this stage.

# Literature check supporting the feedback

Checked 2026-10-16 against the original conference source.

[Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time, ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8071_ECCV_2024_paper.php), [paper, section3.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08071.pdf).

The paper already combines optimal transport for local audio/image alignment with cross-attention consistency supervision, followed by an LLM interface for grounding and AV tasks. Its transport cost concerns cross-modal embedding alignment; it is not evidence that the current 1D temporal feature or CAOR inference intervention works.

Implication for our proposal (inference): “add OT plus cross-modal attention” alone cannot establish novelty. A successor would need a specific distinction, such as preserving query-conditioned event-pair and temporal uncertainty under interventions, together with measured improvement in temporal QA. This single related-work check is not an exhaustive novelty review.

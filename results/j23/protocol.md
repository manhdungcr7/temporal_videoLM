# J23 — pre-registered protocol (written 2026-10-16, before any J23 item was run)

Method under test: **gated_anchor** (training-free, internal; no VAD/ASR/cropping).
- Anchor: question-row attention mass per 2-s audio chunk, layers 12–35, PMI vs. the neutral question "What is happening in the video?", rel = p·(p/p0)², window = chunks with rel ≥ 0.5·max around argmax.
- Gate: normalised entropy of rel < **0.8** (fixed; not tuned).
- Gate open: in layers 13–24, question/answer rows cannot read audio tokens; video tokens get bias 4·log(rel/max rel) of their chunk. Gate closed: unmodified model.

Items: SVHalluc, seed 2030, excluding all 55 J20/J21 items. Per (task, gold): TA 10, TF 10, CMTB 15, GSA/FGSA/CSMB 8 (fewer if a cell runs out of usable clips). All option rotations.

Primary outcome: rescued vs damaged (gated_anchor vs base) over all items, two-sided exact sign test. Success = p < 0.05 and rescued > damaged.
Secondary: semantic balanced accuracy must not drop by more than 2 points; per (task, gold) table; gate-open rate per task; ungated anchor_video reported for reference.
Validity guard: null path must equal base (max score diff ≤ 0.15) or the item errors.
No other variant will be reported as the J23 result.

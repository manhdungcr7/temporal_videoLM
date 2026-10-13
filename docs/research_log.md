# Nhật ký nghiên cứu (log theo thời gian, nguyên văn)

> Ghi chép liên tục trong quá trình làm, giữ nguyên để tra cứu. Bản tổng hợp có cấu trúc nằm trong README.md.


On 2026-10-10 the user switched from semantic prompting ([[qwen-omni-prompting-local-status]]) to the SVHalluc **temporal** tasks: TA (368 items, Yes/No), TF (276 items, Past/Present/Future), CMTB (339 items, 3-way). Small training is now acceptable. The user wants a "wow" story, a new method and clearly superior results for an A* venue.

**Verified numbers (2026-10-10):**
- SVHalluc paper Table 3: Qwen2.5-Omni TA 50.27 / TF 31.52 / CMTB 53.10; Qwen3-Omni 51.11 / 30.60 / 61.75; Gemini 2.5 Pro 85.17 / 53.89 / 69.25.
- Table 4: adding a Whisper transcript does not help TA/TF and hurts CMTB.
- Qwen3-Omni localizes speech well (mIoU 0.88 vs Silero-VAD).
- Temporal labels come from YouCook2 start/end annotations: t_visual vs t_speak, "close" means Present.
- RED (arXiv 2610.02976, 2026-10-02) appendix Table I: every training-free decoding method (VCD, ICD, CODE, SID, AVCD, ASD, MAD, RED) stays near chance on temporal.
  - Qwen2.5-Omni-3B base: TA 48.1, TF 21.4, CMTB 49.0. RED: 50.0 / 29.0 / 53.4. Best TF is MAD at 30.1.
  - Qwen2.5-Omni-7B: best TA 53.3, best TF 35.5, best CMTB 51.9.
  - So the temporal bar is open.

**Claude's proposal (opinion, untested):**
- Frame TA/TF as Reichenbach's tense relation between S (speech time) and E (event time). Searches found no Reichenbach plus video/AV-LLM work.
- Hypothesis 1, "tense shortcut": the model reads time from the grammar of the narration rather than from the video.
- Hypothesis 2: speech leaks into the visual grounding (seen on semantic items 158 and 978).
- Method: estimate S from audio, then E from a speech-blind, chunked video search, then compute the relation symbolically. A small trained head is optional.
- Close prior work: Temporal Insight Enhancement (2401.09861, CLIP/BLIP timestamps, Charades). The narration–action asynchrony literature (TAN 2204.02968, WYS² 2301.02307) is related.

**How to apply:** run the decisive pilot first: data audit, oracle upper bound, tense-shortcut correlation, and a component diagnostic. Then build the method. Never promise "absolute SOTA" or a best paper.

**2026-10-11 Kaggle result (CMTB, Qwen2.5-Omni-3B FP16, n=55 valid; 5 clips of 47-52 s ran out of memory, so the sample is clips of 44 s or shorter):** each condition scored under 3 cyclic option orders. Files: anchor_kaggle_results/anchor_results.jsonl, analyzed by local_qwen/analyze_anchor.py.
- base_av (whole clip + audio, original question) 52.7%; vision_full (whole clip, audio removed) 41.8%; speech_win (only the audio-active window, audio removed) 96.4%; silent_win (muted window, control) 5.5%. Paired speech_win vs base_av: 24 wrong→correct, 0 correct→wrong, +43.6 points, CI [+30.9, +56.4].
- Reading: the model recognizes the action and tells windows apart (silent_win is far BELOW chance, so it correctly sees the other-time action there), but it cannot use "when the speech is heard" to select the visual content inside the full clip.
- Benchmark structure audit (all 397 temporal clips, analysis only): the non-speech half is digitally silent (-120 dB in all 12 audited clips) and the speech window is about half the clip. Speech position alone gives TF accuracy 66.7% with zero video: every Future item has speech in the first half (92/92), every Past item in the second half (92/92), and Present items fall in either (40/52). TA is not predictable from position.
- Caveats: the speech window comes from the digital-silence construction, so localization is trivial here. A natural-video test is needed before claiming a method. TA/TF were not run with the anchor pipeline yet. Qwen2.5-Omni-3B only.

**2026-10-11 later:**
- Kaggle CMTB diagnosis: of the 26 items the whole-clip baseline got wrong, 24 picked the same option the model chose on the muted (other-time) window. Reading: the model is pulled toward the narrated action that happens at another time.
- The 2 speech_win failures were near-miss or short hand actions (zip-lock seal vs put meat in the bag; brief sauce pour).
- Method (a) "speech-anchored attention gating": local_qwen/speech_gate.py builds a 4D additive mask from the TMRoPE time ids (25 per second) so that positions cannot read video tokens outside the RMS speech window. Single pass, no crop.
  - Local NF4 sanity: with no gating, the logits match the plain forward (max diff 0.04).
  - On the 3 shortest CMTB clips, gating fixed 2/3 (2312, 1962); 1917 stayed wrong. Gating only the text positions behaved like gating all positions, so leakage through media tokens is NOT supported.
- Full study notebook kaggle_svhalluc_temporal_study.ipynb, built by build_kaggle_temporal_study_notebook.py from kaggle_temporal_study.py and temporal_study_report.py. It has three parts:
  - main: TA/TF/CMTB with conditions base, gate_hard, gate_soft, gate_text, and lac (look-and-compare across windows);
  - loc: does the model know when the speech is;
  - noise: digital silence replaced by noise.
- The notebook has not been run yet.

**2026-10-11 internal-anchor probe (local NF4, eager attention, 24 short clips: 16 CMTB + 8 TA, scripts internal_anchor_probe.py and analyze_internal_anchor.py):**
- Per head, I measured the attention of the post-media positions onto video tokens inside vs outside the audio-active window, normalised per token.
- Result: no reliable head-level preference for the speech window. The median log-focus is -0.39 and only 23% of (item, head) entries are positive. The best heads (L24 H00 and H09, positive on 75-79% of items) are about what chance would give among 576 heads. Heads chosen on half the items score -0.02 on the other half.
- Caveats: only 24 clips; the NF4 model; the measure sums the template tokens as well as the question content; and "no evidence" does not mean absent.
- Consequence: for Qwen2.5-Omni-3B, a self-derived anchor from attention (the idea in 2605.21954) is not supported. The anchor has to come from the audio activity or from an explicit localization step. This narrows the claim of generality: the clean case is questions anchored on an audible event.
- 2605.21954 (arXiv "MLLMs Know When Before Speaking", read via the abstract only) reports that TG-heads focus on the target interval during prefill, in vision-only models (Qwen3-VL, MiMo-VL, Molmo2), and it fixes this by crop-and-rerun for timestamp output. It is the closest prior work for the gating idea.

**2026-10-11 evening: position-bias lead (the user wants a GENERAL video-LM fix that also keeps other abilities, not a benchmark-specific trick):**
- Re-analysis of the Kaggle CMTB anchor data (recency_check.py): the model prefers the EARLY part of the clip.
  - Video-only full clip: 61.5% when the correct action is in the first half (n=26) vs 24.1% when it is in the second half (n=29).
  - Audio+video: 61.5 vs 44.8.
  - Crop: 96 vs 97.
  - When the speech is late, 16/16 base errors pick the early-half action.
  - A recency hypothesis (RoPE decay toward the late frames) was REJECTED; the data show primacy.
- Literature (read via abstracts/HTML):
  - VideoRoPE (ICML 2025 oral, 2502.05173): blames temporal frequency allocation, V-NIAH-D 74.67 to 87.11, needs about 704 A100 hours of fine-tuning.
  - PAS (2511.10979): training-free phase offsets on the query stream, Qwen2.5-VL-7B only, MVBench 67.2 to 69.5, TempCompass 71.5 to 73.3, no primacy analysis, no audio.
  - Revisiting MPE (ICLR 2026, 2510.23095): MHRoPE/MRoPE-I; VideoRoPE/HoPE hurt DocVQA/ChartQA.
  - VRoPE (EMNLP 2025, 2502.11664).
  - Video-LevelGauge (2508.19650): 27 LVLMs show head/neighbor preference, no fix.
  - Qwen3.5-Omni report: absolute-time TMRoPE ids are too sparse for long inputs.
- Queued locally: segment_swap_probe.py (same content, swapped segment order, av and vision) as the causal test of position bias; gate_audio_ablation.py (audio blocked vs window gate).

**2026-10-11 related-work survey** saved at literature/svhalluc_related_2026-10-01/temporal_related_work_2026-10-11.md. The key prior work it covers:
- DTR (2604.12582): a training-free decoder-side temporal rebalancing against "anchor-frame dominance" (content-independent, often first/last frame). Video-only models (Qwen2.5-VL, Qwen3-VL, InternVL3.5, LLaVA-NeXT-Video). Qwen2.5-VL-7B EventHallusion 68.22 to 73.84.
- SyncRA (2609.34363): trains a contrastive alignment of AV representations. On the swap diagnostic, base Qwen2.5-Omni-7B AllFour scores 2.25%. It does not run on SVHalluc.
- AVTrace (2609.19991): all omni models fall below the majority baseline on sync verification.
- Nobody has improved SVHalluc temporal yet (RED only).
- Planned positioning: cue-conditioned temporal routing (training-free, AV) vs unconditional rebalancing (DTR/PAS) vs training (SyncRA).

**2026-10-12 temporal lab notebook (staged, not full-benchmark):** kaggle_svhalluc_temporal_lab.ipynb, built by build_kaggle_temporal_lab_notebook.py from kaggle_temporal_lab.py (+ speech_gate.py, temporal_lab_report.py).
- The user criticised an earlier 8-shard full-benchmark design, so it was changed to a stage-gated pilot: PER_LABEL=8 gives 64 items on ONE account (about 3 h), and the decision criteria were written before any result.
- Conditions: base, ts05/ts02 (compressed TMRoPE time ids), gate_all, gate_late (layers 18+), audio_block/gate_audio/crop (CMTB only), and segment-swap variants.
- The trace (shard 0, 2 items per task) writes trace/trace_<idx>.json with preprocessing facts, token runs with TMRoPE ranges, logits for each condition, a per-layer logit lens, and per-layer/head attention.
- Verified locally on one CMTB item:
  - explicit ids + 4D mask with kappa=1 reproduces the plain forward (max diff 0.02);
  - the last logit-lens layer equals the plain logits;
  - attention was captured for 36 layers x 16 heads.
- Bug found and fixed: the swapped-clip speech window was mis-detected as the whole clip, so it is now computed from the known segment lengths.

**2026-10-12 RESULTS (Kaggle: study 200 orig + 80 noisy + 40 loc; lab pilot 64 items + 6 traces; extracted under kaggle_results/):**
- Localization (n=40): the 3B model is poor at knowing WHEN the speech is. The timestamp IoU with the audio-active window has median 0.44 (only 45% above 0.5), and the first/second-half forced choice is 57.5% (chance 50). The paper's 0.88 is for Qwen3-Omni, not this model.
- Attention gating on original clips (rotated options): TA 53.5 to 76.7 (n=43, CI +9.3 to +37.2; Yes-recall 10.5 to 63.2, No-recall unchanged at 87.5). CMTB 54.8 to 69.4 (n=62, CI +4.8 to +24.2, no label collapse). TF unchanged (+6.0, CI 0 to 13.4).
- Pilot (n=19 CMTB): gate_audio = crop = 94.7%, gate_all 63.2%, audio_block alone 31.6% (it hurts). So BOTH the window gate and blocking the audio content are needed, single pass, no crop. Tiny n.
- lac (look and compare across windows): TF 88.1% (n=67) vs the speech-position-only rule at 65.7%. Present recall 87%, so the present vs not-present decision uses the video. But the Past/Future direction comes free from the clip structure. In the noisy control, TF lac is 69.2% vs rule 57.7% (n=26), and TA lac shows no gain (n=15).
- Artifact control (digital silence replaced by noise, 66 matched items): the speech windows still match (IoU 0.98). The gate's gain on the same items shrinks from +13.6 pts (45.5 to 59.1) to +1.5 pts (50.0 to 51.5), and it changes 16 answers (orig) vs 8 (noisy). So part of the gate gain depends on the digital-silence artifact. NOT yet shown artifact-free.
- Negative: compressing TMRoPE ids (ts05/ts02) is inconsistent. ts02 on TF collapses onto Present (predicted share 0.95), which is a mirage. The segment-swap causal test shows no clear position effect (n=10-15). So the positional-embedding/primacy story is NOT supported behaviourally. gate_late (layers 18+) is weaker than gate_all.
- Attention traces (6 items): the last-position attention to video is front-loaded in 6/6 (second/first-half ratio 0.61-0.93). Apparent "focus on the speech window" appears only when the window is in the first half, so it is explained by position, not by speech.
- Next stage idea: artifact-free evaluation of gate_audio and lac on more items, then Qwen2.5-Omni-7B and MiniCPM-o, plus a general AV benchmark (no-harm check).

**2026-10-12 stage 2 prepared (not yet run on Kaggle):** kaggle_svhalluc_temporal_stage2.ipynb, built by build_kaggle_temporal_stage2_notebook.py from kaggle_temporal_stage2.py and temporal_stage2_report.py.
- Design: every item runs on the original clip AND a noisy clip (digital silence replaced by noise) with the conditions base / gate / audio_block / gate_audio (a 2x2: time gate x audio block) plus crop (CMTB). For TA/TF it also runs hear (the model itself says which action the speaker describes), look_* with audio hidden, htl (hear-then-look), and lac.
- Decision criteria pre-registered in the report:
  - C1: CMTB noisy gate_audio >= base+15 and beats gate alone by >=10.
  - C2: TA htl >= base+10 on both versions.
  - C3: TF noisy Present-vs-not >= 75% and htl beats the position rule by >=10.
- Local single-item test (TA 2310, NF4): hear returned "Dicing potato"; the look margin for the speech window was +6.5 vs -0.73 (orig) and +6.3 vs -1.7 (noisy), against the lac margins of 0.64/0.36 and 0.5/0.7, and htl was correct on both versions while lac flipped on the noisy one. One item only, anecdotal.
- Prior-work check for novelty:
  - OmniAgent (2512.23646) is training-free and audio-guided, with coarse-to-fine localization via tools and an agent (abstract only; no SVHalluc mention seen).
  - TFO (2609.04242) pipes Whisper timestamps into a frozen VLM.
  - AVCD, MAD and RED are decoding methods.
  - Not found: attention-level time gating combined with audio blocking inside one forward pass of an omni model. This is unverified; more reading is needed (OmniAgent full text, CAD, FATE 2608.01310).

**2026-10-12 human-vs-architecture analysis + binding lab:**
- Qwen2.5-Omni facts (config: seconds_per_chunk=2, 25 time ids per second; traces):
  - each 2-s chunk is ordered [video ~288 tokens, audio ~50 tokens] under a causal mask, so video never reads the same-moment audio, while audio reads the video of ALL earlier moments;
  - there are about 144 video vs 25 audio tokens per second;
  - there are no chunk separators.
- Key paper, 2606.10147 "From Senses to Decisions" (Qwen2.5-Omni-3B/7B):
  - information flows modality -> question tokens (early-mid layers) -> last token; there is NO direct modality-to-last-token flow;
  - cross-modal A<->V knockout matters mostly for speech recognition and speaker detection;
  - it proposes no method and does not analyse the within-chunk order.
  - This is consistent with our results (gate_text ~ gate_all; gate_late weaker).
- Other related work:
  - 2605.10815 (cross-modal sink hubs, ASD decoding, VGGSound/AudioSet);
  - OmniVinci (ICLR 2026: Temporal Embedding Grouping and Constrained Rotary Time Embedding, trained);
  - Qwen3-VL text timestamps "<3.0 seconds>" (trained);
  - OmniForcing (bidirectional within block, generation, trained);
  - OmniRefine (cross-modal evidence crosses chunk boundaries).
- New masks in local_qwen/speech_gate.py:
  - binding_mask (cross-modal attention only within the same chunk +-delta; optional video->same-chunk-audio);
  - balance_mask (audio keys get +log(nV/nA) for question rows).
- Local check on 1917: the no-change version matches the plain forward; the binding masks change logits by at most 0.1, so the effect is tiny on that item.
- Notebook kaggle_svhalluc_binding_lab.ipynb (builder build_kaggle_binding_lab_notebook.py; files kaggle_binding_lab.py and binding_lab_report.py):
  - 5 items per (task, label) over temporal + semantic, about 2 h;
  - screening criterion: +5 on at least 2 temporal tasks and no semantic task below -3;
  - smoke-tested locally on 1 item (all 6 conditions ran).

**2026-10-12 binding lab result (58 valid items, 5 per task-label, kaggle_results/binding):** the binding masks (bind0, bind0_bi, bind1_bi, bi_only) have essentially NO effect. The predictions are identical on almost every item, and bind0 costs -16.7 on TA (n=6) and -10 on GSA. So the "fix how audio and video of the same moment bind" idea is NOT supported. This is consistent with 2606.10147 (the decision is read at the question tokens).
- balance (audio keys +log(nV/nA) at the question rows) was flagged "PROMISING" by the screening rule, but every delta rests on 1-2 flipped items. On TF it pushes the predictions toward Present (Present recall 100%, Past 0%). Treat it as noise or a label shift until tested at larger n.

**Stage 2 interim (my hand tally of the user's log at 62/96, before the final report):**
- CMTB n=17: base 47% (orig) / 47% (noisy); gate_audio 94% (orig) / 88% (noisy). So the audio-decoupled gate SURVIVES the noise control, unlike gate alone in the earlier study (+1.5).
- TF n=25: base 20% / 16%; htl 88% / 84%. The position rule still has to be compared.
- TA n=10: base 60% / 70%; htl 80% / 60%, so C2 looks like it fails on noisy.

**2026-10-12 root-cause synthesis:** research_topics_learning/14_root_causes_and_general_method.md, with a 16-row cause table. Status of the main causes:
- CONFIRMED main causes: #5 speech localization is poor in 3B; #9 the reading at question tokens is not time-selective; #10 source confusion (heard -> seen); #13 late-layer decision bias.
- REJECTED or unsupported: #6 within-media binding, #7 PE/position, #15 token balance.
- UNMEASURED: #4 speech content accuracy, #11 relation reasoning from facts, #12 question pragmatics, #14 representation-action gap (probe); plus a per-error cause decomposition.
- Method proposal STMR (Source- and Time-Monitored Reading): a model-internal router (evidence source + temporal anchor) feeds readout masks at the question tokens (time window plus an audio block when the question asks about what is SEEN), htl handles relations, and the default is a no-op.
- The generality test is to apply the audio block to the semantic FGSA/CMSB tasks (also heard->seen errors) and to AVHBench, with a no-harm check on general benchmarks.
- Cognitive anchor: source monitoring (Johnson 1993); auditory information is more prone to source misattribution.
- Supporting literature: 2608.27785 (prior dominance at late layers; alignment changes bias, not conflict resolution), 2605.13737 (representation-action gap; probe-guided logit adjustment), 2604.16902 (visual preference in omni models emerges mid-late).

**2026-10-12 root-cause completion (the user said: identify ALL root causes on temporal FIRST; no generalization experiments yet):**
- Local activation patching on CMTB 1713 (NF4; patching_debug.py; patching_1713.json): there is an evidence-reading window in layers ~13-27.
  - Plain question states inserted into the clean run break the answer from layer 15 on.
  - Clean question states inserted into the plain run fix it only from layer 27 on.
  - Rotated-option smoke test: same, 27/15.
- Local synthetic text-only relation test (NF4, n=90): when given explicit times in text, the model answers Past for 89/90 TF questions (34%), while TA same-time Yes/No is 94%. So relation mapping to Past/Present/Future is a ROOT CAUSE (#11), not just perception. htl's TF success came from our coded relation step.
- 1713 extras: vision_only is right; audio_only gives the same wrong answer as base; audio block alone fixes it.
- Notebook kaggle_svhalluc_rootcause.ipynb (builder build_kaggle_rootcause_notebook.py; kaggle_rootcause.py, rootcause_report.py) runs 2 shards (SHARD_ID 0/1), PER_LABEL=20. It measures:
  - paraphrases, content-free prior, vision/audio-only;
  - the CMTB 2x2+crop decomposition;
  - hear vs Whisper;
  - hidden states for probes;
  - patching on the fixable CMTB items;
  - the synthetic tense x relation test (TF Past/Present/Future wording, before/during/after wording, TA);
  - eager attention for 24 items.
- Smoke-tested locally on 1713 (all parts ran; the report works).
- Extended local synthetic test (NF4, text-only facts, n=30 per tense):
  - With the Past/Present/Future wording the model answers Past almost always: 33-37% overall, rising only to 50% for present-tense speech.
  - With the before/during/after wording it scores 77/70/67% (future/neutral/present tense) but only 37% when the speech is in the past tense.
  - Same-time Yes/No scores 93-100%.
  - Root cause #11 therefore splits in two: (a) the Past/Present/Future framing collapses onto Past (label-semantics failure); (b) speech tense biases temporal ordering (a tense shortcut, seen in the before/after wording).
  - To be confirmed in FP16 on Kaggle shard 0.

**2026-10-12 STAGE 2 FINAL (73 valid of 96; kaggle_results/stage2; temporal_stage2_report.py):**
- C1 PASS. CMTB n=28: base 39.3 / gate 67.9 / gate_audio 92.9 / crop 92.9 on orig. On noisy: 39.3 / 71.4 / 89.3 / 89.3. gate_audio gives 14-15 w2c and 0 c2w, makes the same predictions as crop, and every label's recall stays at or above base.
- C3 PASS. TF n=30, noisy: htl 83.3 and lac 83.3 vs the position rule at 63.3; Present-vs-not 83.3 for htl. On orig: lac 90 / htl 86.7 vs rule 73.3. gate does not help TF (+6.7).
- C2 FAIL. TA n=15: htl is 73.3 (orig) but 53.3 (noisy). gate alone helps TA (+33.3 orig, CI +13 to +60; +13.3 noisy). gate_audio does NOT help TA, as expected since TA needs the speech content. lac scores 80/86.7.
- Inconsistency to resolve: in the earlier study's noisy set, gate_hard on CMTB was identical to base (n=25), while here gate gives +32 on noisy CMTB. The paired stage-2 design is cleaner. The cause is unknown (both used RMS windows and CFR).

**2026-10-12 remote GPU on Kaggle (user idea):** the notebook kaggle_svhalluc_gpu_server.ipynb (builder build_kaggle_gpu_server_notebook.py; server kaggle_gpu_server.py) keeps Qwen2.5-Omni-3B FP16 loaded and is exposed through ngrok. The pattern comes from the user's own repo/example_GPU_encode_service.py: the Kaggle Secret NGROK_AUTHTOKEN must have Attach enabled, and the notebook prints a random URL and token.
- Client: F:\A_star_paper_NLP_ACL\remote_client.py (commands: ping, info, run FILE, upload, get, restart). Its config is remote_config.json {"url", "token"}, which the user pastes in after starting the notebook.
- The server runs Python jobs inside the namespace of kaggle_rootcause.py (the MODEL_, PROC, scores, masked_forward, patch_sweep, run_item, cut, fetch helpers). It has a queue, a /restart endpoint, upload/get restricted to OUT, and a token check.
- Tested locally over loopback (NF4): ping, info, bad token (401), a job with RESULT, an error traceback, upload + import, file download, both path-escape guards, and a real media job on CMTB 1917 (same scores as before). The ngrok part is NOT tested.
- Use a secondary Kaggle account. Whether tunnelling is allowed by the Kaggle terms is unverified.

**2026-10-12 remote GPU is LIVE (ngrok, 2x Tesla T4; remote_config.json written; URL changes every notebook start).** FP16 confirmation of cause #11 with the text-only tense x relation test (n=30 per tense):
- Past/Present/Future wording: 33-43% (66.7% only for present-tense speech). Future recall is 0% in EVERY tense, so the model collapses onto Past.
- Before/during/after wording: 93.3% (future-tense speech), 86.7% (neutral), 96.7% (present), but only 56.7% for PAST-tense speech. That is a tense shortcut worth about 30-40 points.
- Same-time Yes/No: 97-100%.
- This confirms the NF4 finding and the tense-shortcut hypothesis for the first time.

**2026-10-12 remote jobs queued (serial):**
- tf_framing.py: 72 TF items, Past/Present/Future vs before/during/after wording, av and vision.
- tense_intervention (tense_job.py): TTS speech with future/present/past/orig tense at the same onset, video fixed. Analysed by analyze_tense_intervention.py.
- method_dev_job.py: routing in all layers vs layers 13-27 only; time vs time+source; htl_mid; the bda wording; a router accuracy test on 240 questions. Results go to OUT/method_dev.jsonl and OUT/router_dev.json.
- Fixed: the old ffmpeg on that Kaggle image lacks -fps_mode. The server was patched live, and kaggle_rootcause.ffmpeg() now retries with -vsync.

**Method (user insists: root causes AND a new method):** each component maps to a measured cause.
- Anchor (#5).
- Evidence routing in layers 13-27 at the question tokens (#9).
- Source monitoring, i.e. block audio for SEEN queries (#10).
- Canonical relation wording (#11a).
- Relation from compared evidence margins rather than grammar (#11b).
- Permutation averaging plus content-free calibration (#13).
- A model-internal router for generality.

**2026-10-12 TF framing on real video (remote, n=68 balanced, kaggle_results/tf_framing.jsonl):**
- ppf (original Past/Present/Future) scores 29.4% with Past recall 0.
- bda (before/during/after wording) scores 42.6%, +13.2 with CI +1.5 to +25.0 (14 w2c, 5 c2w).
- Removing audio does not help: ppf_vision is 30.9% with Future recall 0.
- The position-only rule scores 69.1%.
- Conclusion: label framing is a REAL cause on video, but only a partial one. My earlier interim claim that "removing audio helps TF" was wrong; it came from the missing Future items.

**General-benchmark candidate:** Daily-Omni (HF liarliar/Daily-Omni: qa.json + Videos.tar 3.7 GB; 1197 4-choice questions; 647 of them on 30 s videos).
- Question types: Event Sequence 306, AV Event Alignment 238, Context 193, Reasoning 175, Inference 154, Comparative 131.
- Ideal for "helps temporal/AV types, no harm on the others". The baseline job is remote_tests/dailyomni_baseline.py (30 s subset; base + vision_only).
- Related training-free frame selection: Q-Frame (ICCV25), AKS (CVPR25), MDP3, FOCUS. Attention steering: VGA (CVPR26), HAS, gaze heads.

**SOTA table, SVHalluc temporal (verified from the PDF by word positions, 2026-10-12):**

| Model | TA | TF | CMTB |
|---|---|---|---|
| Gemini 2.5 Pro (SOTA; average about 69.4) | 85.17 | 53.89 | 69.25 |
| Qwen3-Omni | 51.11 | 30.60 | 61.75 |
| Qwen2.5-Omni | 50.27 | 31.52 | 53.10 |
| video-SALMONN 2 | 50.00 | 33.33 | 48.58 |
| video-SALMONN | 50.00 | 32.81 | 37.39 |
| VideoLLaMA 2 | 50.00 | 32.25 | 45.72 |
| RED (Qwen2.5-Omni-3B) | 50.0 | 29.0 | 53.4 |

- Best of the 8 decoding methods in the RED paper: TA 53.3, TF 35.5, CMTB 53.4.
- Only RED cites SVHalluc with results (Semantic Scholar citations checked on 2026-10-12).
- Our subset signals (3B, ≤42 s clips, rotated scoring): CMTB about 93%, TF 83-90% (the direction partly comes from structure), TA 80-87%.
- To claim SOTA we need: all 983 items (handle clips over 42 s), both protocols (generation+parse and rotated scoring) with the baseline matching the published numbers, an artifact-free check, and the GENERAL router-driven method.

**2026-10-12 root-cause shards DONE (130 valid; kaggle_results/rootcause; summary in doc 14, section 6):**
- #4: the model hears the speech correctly (96% content overlap with Whisper) -> REJECTED.
- #12: wording is stable for TA/CMTB; TF answers never agree across wordings.
- #13: CONFIRMED for TA. The content-free answer is always "No", 85% of TA errors equal it, and vision_only (60%) is about base (62.9%), so video is barely used.
- CMTB pooled (n=72): base 51.4, gate 80.6, audio_block 52.8, gate_audio 95.8, crop 97.2. Of the errors, 63% are time-only, 20% need both, 14% are audio-only, 3% are perception.
- #8: attention is front-loaded (second/first-half ratio 0.44 in layers 24-35) and does not prefer the speech window.
- Patching (10 clean cases): the switch happens in layers 15-24, exactly at layer 21 in 5 of them.
- #14: a TF probe reaches 71% vs the output's 27.5%. That may just be the speech-position information (the rule gives 68.6%).
- Benchmark shortcut 2: CMTB scores 65.9% with a black video plus audio, above base.

**method_dev (dev; ORACLE routing chosen by task, so not a method claim; 52 valid; kaggle_results/method_dev.jsonl):**
- CMTB n=18: base 38.9; route_all (time+source, all layers) 94.4; route_mid (layers 13-27 only) 100 (94% agreement with route_all); time-only routing in mid layers 55.6. So routing only the evidence-reading layers suffices when audio is also blocked.
- TA n=13: base 46.2; time gate on all layers 69.2; time gate on mid layers 53.8; htl_mid 92.3.
- TF n=21: base 14.3; bda 33.3; htl_mid 81.0 vs the position rule 71.4.
- The router part crashed (LETTERS had only A-C for 4 options). It was rerun as remote_tests/router_dev.py on server 2.

## 2026-10-12 late: TTS tense intervention on real video (n=18 TF, server 1) — CAUSAL tense shortcut
Video and timing identical, only the speech tense re-synthesised (mms-tts): with Past/Present/Future wording the answer follows the tense (future-tense speech -> P(Future)=0.80, past-tense -> P(Past)=0.55; P(Future) diff +0.74 CI [+0.56,+0.88]; answers differ on 14/18). With Before/During/After wording the effect vanishes (diff ~0.05, CI includes 0). So TF's label-framing + tense shortcut is confirmed causally on real clips; accuracy stays ~chance (11-39%). TA hardly moves (3/12 differ).
Next: 5 diagnostic notebooks written to repo/kaggle (nb1..nb5 + jobs/ + build_notebooks.py): J1 perception vs binding, J2 position/noise causal, J3 prior+logit lens, J4 official-protocol reproduction incl. >42 s clips, J5 internal speech localization via attention. J1 smoke-tested locally OK. Daily-Omni job on server 3 had the 4-option LETTERS bug; fixed and resubmitted (job 2d7ec076c3). Server 1 now runs timing_diagnostics; server 2 anchor_text then router_dev.

## 2026-10-12 night: timing diagnostics + text anchor (servers 1,2)
- timing_diag (TA n=19, TF n=29): moving the speech to the mirrored half changes the answer in only 21% (TA) / 14% (TF) of items, reversing the video in 11% / 7%; accuracy against the expected flipped answer 58% / 41% / 38% (~chance). The model barely reacts to the relative order of speech and event, i.e. it largely does NOT use relative timing. rope_reversed changes 16% / 31%.
- anchor_text (n=65): telling the model the speech window as text does nothing (CMTB base 50 -> anchor 50; TA 66.7 -> 66.7; TF 25 -> 29 / bda+anchor 33). Attention routing with the same window: CMTB 92.3. So the missing ingredient is not knowledge of the time: it is that the model cannot USE time as text; the fix has to act on attention/evidence selection.
- router_dev server job done (router accuracy 100% on 6 task types: separates tasks automatically, but still needs treating as task routing, not a general method).

## 2026-10-12 night: natural-data validation launched (decision after reflection)
- Biggest open risk: are the CMTB root cause (no selection of speech-time evidence) and the routing fix SVHalluc artefacts (speech in one half, digital silence)? Daily-Omni AV Event Alignment / Context questions with a quoted utterance + visual target (162 items, e.g. "what is he doing when he says 'Are you ready to rock?'") are the natural CMTB counterpart.
- remote_tests/daily_anchor.py on server 1 (job 19143c842a, ~155 s/item, 30-s clips first): automatic anchor = Whisper word timestamps + fuzzy quote match (fixed component); conditions base / anchor_text / crop (TFO-style, arXiv 2609.04242 = closest prior work: Whisper timestamps -> crop frames for frozen VLMs) / crop_av / route_t / route_ta (layers 13-27). Output OUT/daily_anchor.jsonl. Reference: Qwen2.5-Omni(7B) Daily-Omni AV Event Alignment 51.26 (OmniVideo-100K table).
- server 2: scale7b.py (job 5051f4b246) = Qwen2.5-Omni-7B, base/prior/vision_only/audio_only (+gate_audio, crop for CMTB), clips <= 28 s, ~80 s/item. Server holds 7B afterwards (restart to get 3B back).
- tense_patch.py failed: fix_ffmpeg.py deletes tense_intervention.jsonl on the server; re-upload the local copy (kaggle_results/tense_intervention.jsonl) before rerunning.

## 2026-10-12: user correction — no full-benchmark runs before a validated general method
User rejected J4 (full 983-item baseline reproduction): "chưa có phương pháp mới tổng quát hiệu quả mà đã chạy full benchmark?". Replaced by J4 postdiction (repo/kaggle/jobs/j4_postdiction.py): causal-mask hypothesis — video tokens of an event BEFORE the speech can never read the later speech; test v2a_all / v2a_early (layers 1-14) / media_bidir masks + per-chunk hidden states. Binding lab had only tested LOCAL (same/±1 chunk) edges. j4_official_protocol.py kept for later only.

## 2026-10-12 late night: method convergence (STMR) — key findings
- Reasoning ceiling (text only, perfect label-built timeline, n=96 TA/144 TF/36 CMTB): no timeline 50/33/19; ABSOLUTE times only 50/33/36 (model cannot compare seconds -> explains why text time anchors fail); RELATIVE named segments (before/while/after the speech) TA 100 (all tenses), CMTB 81, TF only 39-46 (Past 25, Future 15): TF bottleneck is the language of the relation, not the video.
- STMR pilot v1 (server 3, automatic, stmr_pilot.py, 16 items): verification step mostly CORRECT (TF segments right 3/3, TA 2/2); failures from (a) final text reasoning on TF labels, (b) binarising per-content argmax for CMTB (several options "yes" in one segment); using the max option score in the speech segment would give CMTB 4/5 vs 2/5.
- Natural data (Daily-Omni quoted-speech visual questions, n=21 partial): base 57.1 / anchor_text 61.9 / crop (TFO-style) 61.9 / route_t 66.7 / route_ta 71.4 (+14.3, CI [-4.8,+33.3]; fixed 4, broke 1).
- 7B (n=33 partial): same failure pattern (TA 58, TF 18, CMTB 30) and CMTB gate_audio 100 / crop 100 (n=10): root cause not a small-model artefact; fix transfers.
- Kaggle plan given to user: run nb1 (J1), nb4 (J4), nb5 (J5) now; hold 2 accounts for STMR v2 pilots; J2/J3 deferred.

## 2026-10-13 early: STMR v2 composition + honest natural-data update
- STMR v2 replay on real pilot evidence (54 items, automatic): base 41 -> plain 67 (TA 92/TF 29/CMTB 90) -> cot letter-scored 80 (TA 54/TF 86/CMTB 90). Composition = symmetric contrast marking (each option to its best segment, each segment keeps top-contrast option; CMTB evidence exact 14/15) + 'list' timeline + reasoning. Claim localisation TA/TF 20/21.
- Text-level SOURCE CONFUSION: plain CoT says the narrated event "is also seen" because it is HEARD in that segment -> TA Yes-collapse. Evidence-first prompt (cot2) fixes the reasoning text, but letter scoring after runaway generations ("Human: ...") was broken -> stmr_cot_parse.py parses the answer from the reasoning and votes across orders (job c7636cb27b, server 3).
- Option grounding (model maps options to before/while/after) FAILS (maps Present->before, Yes->while): dropped. Synonym bridges in the timeline hurt (TA 100->50): dropped.
- 7B final (n=69): TA 57 / TF 28 / CMTB 48; CMTB gate_audio 100 / crop 100 (n=23).
- Daily-Omni natural n=73: base 47.9 / anchor_text 52.1 / route_t 54.8 / route_ta 56.2 / crop 57.5 / crop_av 60.3. Routing helps (~+8, CI touches 0) but does NOT beat TFO-style crop -> novelty cannot rest on the routing operator alone.
- Held-out STMR verification run on server 2 (seed 1234, 10/label, job 5a0eb22d3e) for confirmation of the final composition.

## 2026-10-13: J5 (Kaggle) — the model knows WHEN the speech is (internal anchor)
n=77 valid (<=34 s). ~300 heads (question rows -> audio chunks) correlate r>0.5 with speech activity, ~100 with r>0.7, and they survive pink noise at 10 dB. Window from 16 heads chosen on one half, evaluated on the other: IoU with true speech chunks median 0.88 (clean) / 0.83 (noisy), as good as the external RMS detector on noisy audio (0.83). => fully automatic anchor without external VAD is feasible.
Accuracies: CMTB base 53.6 / gate_audio 96.4 / noisy + re-detected window 92.9; TA 61.1 -> 72.2; TF 45.2 -> 29.0 (gate hurts TF, needs whole timeline); noise does not change base.
STMR held-out (server 2, v1 composition, n=74): TA 55.6->83.3, TF 26.7->26.7, CMTB 57.7->61.5 (v1 composition known broken; v2 replay queued).
Parsed-CoT on pilot timelines (54): plain 66.7 / cot 75.9 / cot2 70.4 (TA/TF/CMTB detail in stmr_cot_parse.json).
User request (2026-10-13): no more Kaggle accounts for now -> run everything on servers; friend asks for per-method FAILURE ANALYSIS (where and how each method fails). J7 (2-s chunk queries, friend's spec) running on server 2 (job be7e25bbd9).

## 2026-10-13: J4 postdiction (Kaggle, n=137) — hypothesis REJECTED
Letting video tokens read all (later) audio changes the answer on only 6.6% of items (v2a_all), 5.8% (layers 1-14); on event-before-speech items accuracy 32.3 -> 32.3 (CI [0,0]). Fully bidirectional media changes 24.8% but no gain (TA 48.6->45.7, TF 32.1->34.0, CMTB 57.1->53.1). Causal order is NOT a root cause. CMTB speech-first/second asymmetry not replicated here (56 vs 58; earlier 70 vs 50 at n=44) -> the position asymmetry claim is weak; TF asymmetry (50 vs 15) is label-distribution + Past failure.
STMR failure analysis (pilot 54): perception/verification wrong in only 3/54; errors are reasoning: (a) TF label mapping in plain reasoning (15/21), (b) SOURCE CONFUSION in CoT over the timeline (TA 6/13: HEARD read as SEEN). v3 = narration declared outside the timeline (stmr_v3.py, server 3 job 2d26443e26, pilot + held-out).
Server namespace bug: kaggle_rootcause sets PER_LABEL=1 from argv -> server jobs must set PER_LABEL explicitly (J6/J7 first runs only 8 items; full reruns queued: J7 3226f71b89 on S2, J6 3e664e7d40 on S3).
Kaggle CLI works (~/.kaggle/access_token): use `kaggle kernels output <user>/<slug> -p DIR --file-pattern '<zip>'` (without pattern it downloads the whole /kaggle/working incl. venv, GBs).
- J7 local smoke (4 items): the question's attention over the chunk queries rises monotonically with query position (0.10 ... 0.25 for the last) = RoPE RECENCY bias, not content; top chunk has speech 25% vs chance 47%. Fix added: cqp* conditions = PARALLEL positions for the isolated queries (all share the same RoPE ids, like Parallel Context Windows). Server 2 job 8d4edd8561 (OUTNAME j7_par.jsonl) after J7 full (3226f71b89).

## 2026-10-13 morning: J1 (Kaggle, n=218) — PERCEPTION IS FINE, failure is binding/readout (decisive)
TA: base 46.5 -> model's own hear + per-half look (sound off) picks the right half 90.7% (whisper transcript 95.3); automatic hear-then-look rule 90.7. TF: base 29.5 -> perception 89.4 (whisper 92.4); rule 89.4. Perception by margin tercile 70 / 99 / 100 (confidence usable). Position-controlled probes: event half decodable within speech-position strata at L21 74.5 / 83.3 vs majority 65.3 / 54.2 -> weak-to-moderate internal encoding, output at chance (partial representation-action gap; earlier n=51 "no encoding" was underpowered).
All 3 servers went offline (sessions ended ~07:00 2026-10-13): J6/J7/v3/held-out-v2/Daily-Omni tail results are only in the server notebooks' outputs (other Kaggle accounts; local CLI token is account dngomnh).

## 2026-10-13: in-model method ACAR + notebooks round 2 (user runs everything on Kaggle)
Per user preference (see [[method-in-model-intervention]]): main method = ACAR (Anchor-Conditioned Attention Re-allocation): pass 1 eager prefill, question rows -> audio-token attention per 2-s chunk (all heads, layers 12-35 or 1-12, label-free) = anchor profile; pass 2 layers 13-27, question rows get lambda*log(a(c)/max a) on video tokens of chunk c, optional audio block. Plus in-model "look" per window (before/during/after anchor) for TA/TF. jobs/j8_acar.py (local smoke OK: anchor peaks on speech chunks, IoU 0.75). jobs/j9_nonspeech_anchor.py: procedurally synthesised bell/beep/clap/whistle mixed into the non-speech half; questions about the sound vs the speech vs neutral -> does the anchor follow the question. Notebooks: nb8 (J8 pilot), nb9 (J8 held-out), nb10 (J9), nb6/nb7 (J7 pilot/held-out incl. parallel-position cqp*). Priority order in repo/kaggle/README.md.

## 2026-10-13: J9 (Kaggle, n=43) — question-conditioned anchoring WORKS with PMI (non-speech sounds)
Synthetic bell/beep/clap/whistle in the non-speech half. Raw question->audio attention (layers 12-35): top chunk = speech for every question (q_sound 7% sound). PMI vs neutral question, rel = p*(p/p0)^alpha: alpha=2 -> q_sound lands on the sound 86% (alpha=4: 91%) while q_speech stays on speech 100%. Per sound (alpha 2): whistle 100, bell 94, beep 89, clapping 60. q_sound raises sound mass over neutral in 91% of clips (median x1.65). Early layers 1-12 are question-independent (useless for anchoring) -> use layers 12-35. J8 already uses mid band + ALPHA=2.

## 2026-10-13 afternoon: J8 pilot (Kaggle, n=46) + local mechanism debugs -> J10 (full automatic method)
J8 pilot: CMTB base 52.9 / oracle_route 88.2 / acar_a_l4_ab (attention anchor from audio, video re-weight, audio blocked) 94.1 / acar_a_pmi 88.2 / qtar_av_pmi (both modalities, no gate) 58.8 / _sg 64.7. TA 54.5 -> <=63.6, TF 22 -> 33. Failure analysis: EVERY error has the top relevance chunk inside the speech (selection right); errors are reading: qtar without gate keeps the base answer on 7/7 CMTB errors (source confusion); TA/TF relational reading unsolved.
PMI alpha: top chunk in speech 100% for alpha 0..2 (mass on speech ~0.8); window IoU drop was an artefact of window extraction -> alpha 2 fine for speech AND non-speech anchors.
Local debug item 1713 (CMTB): bias moves 100% of question-row video attention to the selected chunks but the answer does not change while audio is kept (speech content read); attention-mass modality PMI picks the wrong target (audio x0.97 vs video x0.88) -> unusable; the model's own TEXT self-query ("seen / heard / both") picks "seen" correctly; QTAR + audio blocked in layers 13-27 -> correct 0.922 (oracle 0.856); blocking only layers 21-27 insufficient (0.381). Scene anchoring (red square drawn in a known window): video relevance raw and PMI put the top chunk on the square.
Self-query on question types: CMTB seen 10/10, TA both 10/10, TF heard 10/10 (wrong for TF), Daily-Omni mixed/sensible.
J10 (jobs/j10_auto.py, extra j8_acar.py; notebooks nb11 pilot, nb12 held-out): auto policy seen->block audio, heard->block video, both->layers 13-20 both then 21-27 sound off; with force_v / force_both / no_gate controls. Local smoke OK.

## 2026-10-13: dry-run debug of the TA/TF "look" step (local, items 1472 TA-Yes, 1485 TA-No, 1513 TF-Past)
- J8 look (question rows read audio in every layer, video window): wrong 3/3. Split "hear in layers 0-12, look sound-off in 13-27": right 2/3; but margins only ~0.5 logit and a no-audio control is also 2/3 -> the latent template is WEAK; logit-lens trajectories of the windows almost identical across layers.
- Explicit INNER SPEECH (the model's own generated phrase of what it heard, e.g. 'Pour sauce over potatoes', + vision-only crop of the window): right 3/3 with margins 2-18 logits.
- Cognitive framing: verbal working memory / phonological loop (Baddeley); inner speech works for any sound ("a dog barks"), unlike ASR.
- Next test (debug_look2.py): same explicit template, but the look is done by in-model routing (audio blocked, video window in layers 13-27 or all layers) vs crop.
- J7 pilot (Kaggle, n=54) conclusive negative: cq 38.9 vs base 40.7, iso = noiso, select picks the LAST query 100% (recency), top-has-speech 37% < chance 50%; cqp not in that run (old notebook).
- User decision: nb12 (J10 held-out) NOT to be run until the method is final; nb11 (J10 pilot) yes.

## 2026-10-13: J8 held-out (n=73) + J7 held-out (n=88) + routed inner-speech look + J11
- J8 held-out: CMTB base 73.1 / oracle 92.3 / acar_a_l4_ab 92.3 / acar_a_pmi_l4_ab 92.3 (auto == oracle); TF 29.6 -> 48.1 (a_pmi); TA flat (45-55). Top relevance chunk in speech: a 100%, a_pmi 97%, av_pmi 88%.
- J7 held-out: cq 47.7 vs base 46.6; noiso 39.8 (isolation matters only relative to non-isolated queries); select = last query (pos 1.00), speech 32% < chance 48%. Chunk-query idea closed (training-free).
- debug_look2 (local, 5 items): explicit inner-speech template + IN-MODEL routed look (audio blocked all layers, video window layers 13-27) picks the right window 5/5 with margins equal to or above crop (e.g. 6.64 vs 5.66).
- Oracle-timeline CAD (evidence minus no-evidence) does NOT fix TF (flips to always Future): TF needs explicit step-by-step reasoning.
- J11 (jobs/j11_full.py, notebook nb13): orient (attention PMI anchor) -> rehearse (model's inner speech, general audio prompt) -> look (routed, sound off) -> remember (source-separated timeline v3 + symmetric contrast marks) -> reason (plain / CoT parse-vote); plus base and acar.
- Server 4 (remote_config4.json, user-provided 2026-10-13): J11 v1 on dev2 (seed 555) exposed 3 bugs: (1) anchor window too narrow (PMI peak, "above mean" rule; speech 0-8 s -> anchor 2-6 s) -> fixed with grow_window: seed = PMI argmax, extent = contiguous raw attention >= 0.3*max (J8 data n=119: IoU median 0.50 -> 0.83, coverage 0.84, spill 0.08); (2) contrast marking put never-seen options (all scores < 0) into the timeline -> now only positive scores; (3) inner speech run-on ("... If you have any other questions") and sound description instead of speech content -> first sentence only, prompt asks speech-action first. J11 v2 running on server 4 (job 8377c4059d, file j11v2_dev2.jsonl). Kaggle nb13 (user started earlier) runs J11 v1 (bugged) -> treat as v1 diagnostic.
- 2026-10-13 in-pass debug (repo/kaggle/jobs/inpass.py, remote_tests/debug_inpass.py; items 1713, 1772 CMTB, 1472 TA): relevance measured inside the same forward (layers 12-20, eager toggled per layer) correlates 0.95-0.99 with the two-pass profile, BUT applying the bias only from layer 21 fails (1713: two-pass 13-27 0.922 vs in-pass 21-27 0.239; 1772 same) -> evidence must be routed from layer ~13; next: measure at ~8-12, apply 13-27.
- J11 v2 dev2 partial (n=24): base 37.5 / acar 54.2 / plain 54.2 / cot 66.7 (CMTB cot 70 < acar 80).
- User asked (2026-10-13) to package everything into GitHub repo https://github.com/manhdungcr7/temporal_videoLM with a full Vietnamese README for the friend (exclude secrets: remote_config*.json, kaggle tokens).

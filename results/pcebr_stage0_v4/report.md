# CQ-TSPR / PC-EBR Stage-0 v4 — kết quả, 2026-10-15

**V4 chạy xong trên cả hai server với nguyên công thức của đề xuất. C3 cải thiện retrieval; bản M2 đầy đủ vẫn thua đối chứng tốt nhất trên cả ba suites. M3 có lỗi bất biến khi đảo thời gian: hai vector gần như giống nhau, nên phép thử sensitivity không đo được hướng thời gian như dự định. Chưa đủ bằng chứng chuyển A0–A7.**

Hai runtime T4 mới đều khỏe; M1/M2 trùng toàn bộ giữa hai replicas. Có 252 pairs: 202 train/16 origins và 50 held-out/4 origins; ba seeds cố định 20261015/16/17. M1 có 30 fits/server (5 conditions × 3 seeds × 2 representations), 180 steps/fit. M2 có 21 probes/server, 200 steps/probe. Chỉ 8 ID micro, 4/shard, rotation 0, cùng các lượt trước. Chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.

Bản gốc 1515 dòng được lưu cùng proposal; alias trong server_jobs có SHA giống bản gốc. Wrapper chỉ chèn quan sát để lưu checkpoints, scores, episodes, probabilities và generated texts; không đổi công thức, nhãn, seeds hoặc siêu tham số. Không cần devicefix/autograd repair ở mainrun v4. Prepare thêm input_ids vào cache và dùng T4 SDPA explicit-KV-repeat như v3.

Native pairs là post-media readout block index 27; pooled là mean media tokens cùng block. Đây là pseudo descriptions từ model, không phải annotation event/time độc lập. 50 queries đến từ 4 origins đã xem nhiều lần; ba seeds và hai replicas không tăng số nguồn held-out độc lập. Nonnull có 31 queries theo substring filter; không gọi đó là 31 event ground truths.

## M0 — ổn định số

```json
{
  "basis_test_diag_output": [
    2.0,
    0.0,
    0.0,
    0.0
  ],
  "basis_test_lowrank_output": [
    2.0,
    2.0,
    2.0,
    2.0
  ],
  "yann_tiny_noise_1e4_v3_max_weight": 1.0,
  "yann_tiny_noise_1e4_v4_max_weight": 0.2,
  "yann_tiny_noise_1e4_v4_gate": 1.0456810622201575e-14,
  "yann_tiny_noise_1e4_v4_phi_max_abs": 2.2593699300772195e-21,
  "strong_signal_v4_gate": 0.9922,
  "strong_signal_v4_phi_cdiff": 1.5918
}
```

Shrinkage sửa được counterexample nhiễu 1e-4, giữ peak support ≈.2 và relational feature rất nhỏ. Đây là unit check ổn định số, chưa xác nhận confidence calibration hoặc correctness khi logits nhọn nhưng chọn nhầm event.

## M1 — đầy đủ số đúng qua ba seeds

Các cột Global/Exact/Multipos có mẫu số 50; Nonnull có 31. Sai = mẫu số trừ số đúng của từng seed. Đây là retrieval, không phải số câu QA40.

### Native readout

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_baseline | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.00/32.00/42.00/32.30 |
| C2_ctx_within_loss_diag | [11, 8, 8] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 18.00/32.00/42.00/32.30 |
| C3_whitened_ctx_diag | [15, 15, 15] | [20, 20, 20] | [26, 26, 26] | [12, 12, 12] | 30.00/40.00/52.00/38.70 |
| C4_loc_only_lowrank_v3_ref | [4, 5, 5] | [11, 12, 10] | [15, 16, 14] | [5, 6, 5] | 9.33/22.00/30.00/17.20 |
| C8_shuffled_within_rec_ctrl | [5, 8, 10] | [10, 14, 11] | [11, 16, 13] | [6, 8, 7] | 15.33/23.33/26.67/22.60 |

### Pooled media

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_baseline | [16, 16, 16] | [19, 19, 19] | [23, 23, 23] | [14, 14, 14] | 32.00/38.00/46.00/45.20 |
| C2_ctx_within_loss_diag | [16, 16, 16] | [20, 20, 20] | [24, 24, 24] | [14, 14, 14] | 32.00/40.00/48.00/45.20 |
| C3_whitened_ctx_diag | [22, 21, 21] | [21, 20, 20] | [24, 23, 23] | [15, 14, 14] | 42.67/40.67/46.67/46.27 |
| C4_loc_only_lowrank_v3_ref | [2, 2, 2] | [11, 9, 9] | [15, 13, 13] | [6, 6, 5] | 4.00/19.33/27.33/18.30 |
| C8_shuffled_within_rec_ctrl | [13, 15, 23] | [18, 18, 20] | [19, 19, 24] | [14, 14, 14] | 34.00/37.33/41.33/45.20 |

C3 native: Global 11→15/50, Exact 16→20/50, Multipos 21→26/50, Nonnull 10→12/31 ở cả 3 seeds. Pooled: Global 16→[22,21,21]/50; Exact 19→[21,20,20]/50. Đây là tiến bộ đáng giữ làm baseline. Tuy nhiên C3 đổi đồng thời standardization và within-rec loss weight 0→.20; C2 dùng .25, nên chưa cô lập đóng góp whitening. Code chỉ chuẩn hóa diagonal mean/std, không tính full covariance whitening. Pooled shuffle Exact [18,18,20] gần C1 [19,19,19]; C8 Global seed 3=23 còn hơn C3. Chưa có xác nhận localization/generalization ổn định trên nguồn mới.

## M2 — mọi probe và từng lớp

Natural có 232 held-out episodes (58/lớp), Reversal 576 (144/lớp), Roll 156 (39/lớp). Roll v3 có 752 episodes, nên v3/v4 không cùng evaluation set. Tất cả vẫn phụ thuộc 4 held-out origins. 3-way là argmax riêng ba relation logits trên nonfoil subset; không phải lọc 4-way predictions. Paired metric chỉ trên các cặp còn cả fwd/rev sau balancing/truncation.

### PartB1_Natural_Unshifted_Balanced

| Probe | Đúng/sai/tổng4way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng (%) |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_baseline | 109/123/232 | 47.0 | 63.8 | 41/58 | 26/58 | 21/58 | 21/58 | — |
| B2_coord_shuffle_control | 65/167/232 | 28.0 | 36.8 | 25/58 | 14/58 | 8/58 | 18/58 | — |
| B3_wrong_event_query_control | 65/167/232 | 28.0 | 45.4 | 10/58 | 21/58 | 24/58 | 10/58 | — |
| B4_uniform_retrieval_sim0_control | 58/174/232 | 25.0 | 33.3 | 58/58 | 0/58 | 0/58 | 0/58 | — |
| B5_pcebr_phi_only_no_null | 102/130/232 | 44.0 | 57.5 | 29/58 | 40/58 | 29/58 | 4/58 | — |
| B6_pcebr_null_only_no_phi | 56/176/232 | 24.1 | 32.2 | 21/58 | 3/58 | 15/58 | 17/58 | — |
| B7_shrinkage_calibrated_full_c1_connected | 88/144/232 | 37.9 | 51.7 | 25/58 | 36/58 | 19/58 | 8/58 | — |

### PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit

| Probe | Đúng/sai/tổng4way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng (%) |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_baseline | 249/327/576 | 43.2 | 60.2 | 94/144 | 12/144 | 89/144 | 54/144 | 128/216 (59.3) |
| B2_coord_shuffle_control | 149/427/576 | 25.9 | 36.6 | 44/144 | 22/144 | 35/144 | 48/144 | 49/216 (22.7) |
| B3_wrong_event_query_control | 148/428/576 | 25.7 | 40.7 | 30/144 | 50/144 | 34/144 | 34/144 | 85/216 (39.4) |
| B4_uniform_retrieval_sim0_control | 144/432/576 | 25.0 | 33.3 | 0/144 | 0/144 | 0/144 | 144/144 | 0/216 (0.0) |
| B5_pcebr_phi_only_no_null | 234/342/576 | 40.6 | 53.7 | 62/144 | 108/144 | 62/144 | 2/144 | 116/216 (53.7) |
| B6_pcebr_null_only_no_phi | 134/442/576 | 23.3 | 34.5 | 20/144 | 38/144 | 18/144 | 58/144 | 21/216 (9.7) |
| B7_shrinkage_calibrated_full_c1_connected | 238/338/576 | 41.3 | 53.7 | 57/144 | 92/144 | 57/144 | 32/144 | 116/216 (53.7) |

### PartB3_CrossModal_Only_TwoHop_Roll

| Probe | Đúng/sai/tổng4way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng (%) |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_baseline | 34/122/156 | 21.8 | 34.2 | 26/39 | 6/39 | 2/39 | 0/39 | — |
| B2_coord_shuffle_control | 45/111/156 | 28.8 | 36.8 | 26/39 | 4/39 | 5/39 | 10/39 | — |
| B3_wrong_event_query_control | 40/116/156 | 25.6 | 41.9 | 1/39 | 4/39 | 15/39 | 20/39 | — |
| B4_uniform_retrieval_sim0_control | 39/117/156 | 25.0 | 33.3 | 0/39 | 39/39 | 0/39 | 0/39 | — |
| B5_pcebr_phi_only_no_null | 33/123/156 | 21.2 | 29.9 | 10/39 | 10/39 | 13/39 | 0/39 | — |
| B6_pcebr_null_only_no_phi | 42/114/156 | 26.9 | 33.3 | 3/39 | 6/39 | 21/39 | 12/39 | — |
| B7_shrinkage_calibrated_full_c1_connected | 33/123/156 | 21.2 | 24.8 | 5/39 | 6/39 | 14/39 | 8/39 | — |

Full B7 đạt37.9/41.3/21.2%; đối chứng tốt nhất47.0/43.2/28.8%; gap−9.1/−1.9/−7.6pp. Reversal paired full53.7% còn dưới mean-time59.3%. During full62.1/63.9/15.4%; tăng During ở hai suites đầu không đồng nghĩa thắng tổng thể. Phi-only Natural44.0% còn hơn full37.9%, trong khi mean-time vẫn47.0%. Uniform25.0/33.3% nhưng có thể dự đoán một lớp cố định: During100% ở uniform Roll là constant-class, không phải năng lực temporal. Theo gain10pp/During35 của v3 làm mốc tham chiếu, cả3suite đều không đạt; v4 không được xem là một preregistration gate mới.

Foil train chỉ5/16origins qua filter;11/16fallback. Held-out0/4qua,4/4fallback. Điều này trái claim100%strict/no fallback trong proposal. Kể cả qua cosine/Jaccard cũng chỉ là khác mô tả văn bản, không xác nhận event vắng mặt trong recording. C1sim_mat được dùng thật ở Natural/Reversal; Roll dùng C1encoder cho truy vấn hợp thành rồi khử projection của hop1, chưa phải learned multi-hop reasoning.

## M3 — micro-probe và generation

| Shard/ID | Task/gold | Gate | Cosfwd/rev | SwapTVlate/mid | Routed/nativeTV | IrrelevantTV | RandomTV | MCnative→routed |
|---|---|---:|---:|---|---:|---:|---:|---|
| 1/1449 | TA/A | 0.9801 | 1.0 | 0.0/0.0 | 0.2051 | 0.0636 | 0.2994 | A→B |
| 1/2150 | TF/C | 0.9746 | 1.0 | 0.0/0.0024 | 0.2235 | 0.2273 | 0.3155 | A→B |
| 1/1929 | TF/C | 0.9621 | 1.0 | 0.0018/0.0 | 0.3348 | 0.1628 | 0.2251 | C→B |
| 1/2054 | TA/B | 0.981 | 1.0 | 0.0006/0.0 | 0.1366 | 0.1784 | 0.2857 | A→A |
| 2/1956 | CMTB/A | 0.9731 | 1.0 | 0.0001/0.0016 | 0.3401 | 0.2356 | 0.0532 | A→A |
| 2/1513 | TF/A | 0.9764 | 1.0 | 0.001/0.0021 | 0.12 | 0.0388 | 0.2155 | B→B |
| 2/2264 | CMTB/B | 0.9721 | 1.0 | 0.0/0.0006 | 0.3017 | 0.1355 | 0.0992 | B→B |
| 2/1821 | TF/C | 0.9812 | 1.0 | 0.0019/0.0021 | 0.3647 | 0.2423 | 0.2103 | C→B |

MeanTVswaplate=0.000675,mid=0.001100;meanrouted=0.253312,irrelevant=0.160537,random=0.212987. Max|vecfwd−vecrev|=8.46385956e-06. Khi cảswapTV gần0, tỷ sốmid/late không có ý nghĩa để suy khả năng temporal giữa tầng. Đúng hướng thời gian phải được đo bằng correctness sau biến đổi hợp lệ.

### Đúng/sai theo task và gold trên8micro,rotation0

| Task/gold | N | Native đúng/sai | Routed đúng/sai |
|---|---:|---|---|
| TA/A | 1 | 1/0 | 0/1 |
| TA/B | 1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 |
| TF/B | 0 | 0/0 | 0/0 |
| TF/C | 3 | 2/1 | 0/3 |
| CMTB/A | 1 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 |
| CMTB/C | 0 | 0/0 | 0/0 |

**Các cellsN=0 chưa được đánh giá. Bảng8micro không đại diện kết quả40mẫu hoặc score cyclic-option rotations110 của các vòng trước.**

### Văn bản24tokens

Tổng MC micro: native đúng 5/8, routed đúng 2/8. Đây là rotation 0 trên 8 mẫu, không phải kết quả đầy đủ QA40.

**Lỗi nhãn trong script gốc:** `sample.get("answer","A")` mặc địnhA vì manifest dùng`gold`. Sai5/8nhãn; bảng này chấm lại từ outputs/probabilities đã lưu bằng manifestgold, không chạy lại model và không sửa mainrun. Raworiginalaudit được giữ nguyên; so sánh nhãn/cực tính sửa ởcorrected_gold_audit.json.

**1449 — TA,goldA (Yes)**

Native: Yes, the narrated event of adding cooked chickpeas is happening at the same time in the video. You can

Routed: The scene is being shown.

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": true}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**2150 — TF,goldC (Future)**

Native: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at

Routed: The video is showing the process of adding ingredients to a pot. The person is pouring boiling water into a pot. The

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**1929 — TF,goldC (Future)**

Native: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So,

Routed: The onions are being sautéed in a pot.

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**2054 — TA,goldB (No)**

Native: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the

Routed: The person is preparing a cooking process, the person is putting the chicken's hands are shown, the hands are shown,

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": false}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**1956 — CMTB,goldA (mix in flour)**

Native: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the

Routed: The video is showing a person is adding a white substance to a pot of food.

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**1513 — TF,goldA (Past)**

Native: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after

Routed: The person is stirring a pot, and the hand is holding a spoon, and the hand is stirring a pot, and

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**2264 — CMTB,goldB (add milk and one egg whisk together)**

Native: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the

Routed: The person is adding ingredients to a bowl.

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

**1821 — TF,goldC (Future)**

Native: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the

Routed: The video shows a pot of water is being boiled. The water is being heated and the water is being heated.

Audit native: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Audit routed: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "yes_no_polarity_matches_gold": null}`

Regex chỉ phát hiện mốc vượt thời lượng, không phát hiện mọi timestamp sai hoặc chứng minh một mốc là clip-relative. Không nhắc timestamp/không trả YesNo không tự là cải thiện. Câu trả lời chung chung, thiếu nội dung hoặc bị cắt ở24tokens cần được tính vào chất lượng. Không dùng8mẫu để tunegate/strength rồi gọi là validation mới.

### Đồng hồ và cấu trúc vector

| ID | Clipduration | Audio25Hzspan | Cachedvmax | Gv | Patches/griduniform | Mappedvmax |
|---|---:|---:|---:|---:|---|---:|
| 1449 | 19.124 | 19.12 | 20.08 | 21 | 144/True | 18.669 |
| 2150 | 12.167 | 12.16 | 15.52 | 16 | 144/True | 11.787 |
| 1929 | 37.078 | 37.04 | 36.4 | 37 | 144/True | 36.577 |
| 2054 | 23.109 | 23.12 | 23.16 | 24 | 144/True | 22.628 |
| 1956 | 22.074 | 22.04 | 25.04 | 26 | 144/True | 21.649 |
| 1513 | 13.074 | 13.04 | 16.04 | 17 | 140/True | 12.689 |
| 2264 | 30.127 | 30.12 | 32.0 | 33 | 144/True | 29.671 |
| 1821 | 17.083 | 17.08 | 17.04 | 18 | 144/True | 16.608 |

Mappingg→(g+.5)T/G là giả định chia đều theo duration, chưa khôi phục framePTS. Cần lưu sampledframeindices/timestamps,temporalpatchgrouping và padding. Code không assert audio-span khớp duration hoặc countsuniform. Không nên gọi oracle hay kết luận lỗi đồng hồ model đã được chứng minh.

Với phép đảoP trên lưới đối xứng, p′=Pp,V′=PV nên tổngp′V′=tổngpV:ev_bound giữ nguyên. rel_scalar đổi dấu, chron_dir cũng đổi dấu, nên tíchrel_scalar×chron_dir giữ nguyên. Vì thế vector cuối là even/invariant dưới đảo; đây là lỗi thiết kế readout temporal, không phải bằng chứng backbone không biết thứ tự. Dữ liệu thực và toy controls ghi ởcontrols_s*.json.

Code gamma=max(.35,min(1,max(alphaV,alphaA))) có floor.35; uniform vẫninject.077norm/layer (.22×.35), trái mô tả gamma→0 khi không có evidence. Top15% cosine theo câu hỏi là salienceheuristic, chưa được xác nhận là định vị đúng event hoặc phân biệt narration với action. A/V cộng đối xứng chưa tự chứng minh conditional binding hoặc suy luậnA→A,V→V.

## Tái lập và giới hạn

Mỗi server111rawPT:30projectors,30retrievalmatrices,2bundles,24episodetensors,21probes,4M3banks;1500retrievalqueryrows/server được tính lại đúng. Archive và từngentry SHA được kiểm tra local, khôngextract. Native replay0,tokenroundtrip/hookdisjoint/freeform bỏchoices đủ8/8. Source/cachehash và metadata được lưu; secrets chỉ ởignoredconfigs.

Việc upload ban đầu bị auto-review từ chối; đã kiểm kê read-only xác nhận runtime mới trống. Sau câu trả lời rõ ràng của người dùng “Đồng ý tải lên cả hai server”, mới tải4cache,manifest vàcode. Không thay dữ liệu hoặc gửi sang đích khác.

Phản biện và đề xuất tiếp: [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md). Các định danhpaper sai và giới hạn literature: [literature_audit.md](literature_audit.md). TênYoshua/Yann là quy ước vai trò trong trao đổi, không phải xác nhận tác giả thật của các nhà khoa học.

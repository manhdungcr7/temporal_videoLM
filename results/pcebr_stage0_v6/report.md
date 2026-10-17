# CAOR-WEMD / PC-EBR Stage-0 v6 — kết quả và kiểm toán, 2026-10-16

**Micro8: native 5/8, legacy v5 5/8, CAOR full 5/8. Cứu 0, làm hỏng 0.**

M0/M1/M2 chạy lại cùng dữ liệu trên hai server T4; đây là hai replicas, không thêm video độc lập. M1 có 252 pseudo pairs: 202 train/16 origins, 50 held-out/4 origins đã xem ở các vòng trước. Native cache là post-media last-token readout block index27; pooled là media-token mean cùng block. Nhãn mô tả/event presence chưa được annotation độc lập.

Source gốc giữ nguyên; wrapper chỉ lưu observations và kiểm tra bổ sung. Xem [protocol.md](protocol.md) và [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md). M3 chỉ chạy 8 mẫu rotation0 đúng manifest; chưa chạy đủ QA40 hoặc reserve400.

## M0

Kiểm tra gốc qua cả hai server. Audit độc lập ví dụ CDF giao nhau: `{"computed": [0.010416666977107525, 0.2604166865348816], "analytic_expected": [0.010416666666666666, 0.2604166666666667], "fine_grid": [0.010416666666666664, 0.2604166666666667], "max_error_analytic": 1.986821490662649e-08, "max_error_fine_grid": 1.986821490662649e-08, "points": 240001}`.

Phi có rank số học6 trên bộ toy đã định; local non-media mass invariance và reversal checks là unit checks, không phải bằng chứng temporal QA đúng. Zero-noop ép gate1 và gọi kernel đủ6 layer mỗi mẫu; kiểm tra parity logit được báo riêng.

## M1 — số đúng qua ba seeds

Global/Exact/Multipos có mẫu số50. Global chấm mô tả top1 bằng chuỗi normalized content, không yêu cầu đúng recording/interval; Exact yêu cầu đúng index trong cùng recording/modality; Multipos chấp nhận normalized description trùng trong group. Nonnull có mẫu số31 theo substring filter, chưa phải nhãn event tồn tại. Số sai = mẫu số − số đúng. Các seed không tạo thêm held-out origins.

### M1_native

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global / Exact / Multipos / Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_w00 | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.0 / 32.0 / 42.0 / 32.3 |
| C2_ctx_diag_w20_no_whiten | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.0 / 32.0 / 42.0 / 32.3 |
| C3a_txt_std_diag_w20_v4_exact_replay | [15, 15, 15] | [20, 20, 20] | [26, 26, 26] | [12, 12, 12] | 30.0 / 40.0 / 52.0 / 38.7 |
| C3b_txt_std_diag_w20_v5_ref | [15, 12, 13] | [20, 18, 19] | [26, 24, 25] | [12, 10, 11] | 26.67 / 38.0 / 50.0 / 35.5 |
| C4_pure_mod_diag_w20_no_txt_mean | [18, 18, 18] | [18, 18, 18] | [23, 23, 23] | [9, 9, 9] | 36.0 / 36.0 / 46.0 / 29.0 |
| C5_pure_mod_zca_w20_no_txt_mean | [8, 8, 8] | [15, 15, 15] | [18, 18, 18] | [8, 8, 8] | 16.0 / 30.0 / 36.0 / 25.8 |
| C6_bilevel_global_plus_innov_w20 | [18, 17, 17] | [19, 18, 18] | [25, 24, 24] | [10, 9, 9] | 34.67 / 36.67 / 48.67 / 30.1 |
| C8a_shuffled_c3b_w20_ctrl | [13, 11, 15] | [19, 18, 17] | [21, 21, 18] | [13, 10, 13] | 26.0 / 36.0 / 40.0 / 38.7 |
| C8b_innov_only_shuffled_c6_ctrl | [11, 11, 10] | [15, 13, 16] | [19, 14, 21] | [10, 9, 9] | 21.33 / 29.33 / 36.0 / 30.1 |
| C8c_both_branches_shuffled_c6_ctrl | [8, 10, 9] | [14, 13, 15] | [18, 14, 20] | [9, 9, 8] | 18.0 / 28.0 / 34.67 / 27.93 |

### M1_pooled

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global / Exact / Multipos / Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_w00 | [16, 16, 16] | [19, 19, 19] | [23, 23, 23] | [14, 14, 14] | 32.0 / 38.0 / 46.0 / 45.2 |
| C2_ctx_diag_w20_no_whiten | [16, 16, 16] | [20, 20, 20] | [24, 24, 24] | [14, 14, 14] | 32.0 / 40.0 / 48.0 / 45.2 |
| C3a_txt_std_diag_w20_v4_exact_replay | [22, 21, 21] | [21, 20, 20] | [24, 23, 23] | [15, 14, 14] | 42.67 / 40.67 / 46.67 / 46.27 |
| C3b_txt_std_diag_w20_v5_ref | [23, 23, 23] | [23, 23, 23] | [26, 26, 26] | [16, 16, 16] | 46.0 / 46.0 / 52.0 / 51.6 |
| C4_pure_mod_diag_w20_no_txt_mean | [23, 23, 23] | [22, 22, 22] | [25, 25, 25] | [16, 16, 16] | 46.0 / 44.0 / 50.0 / 51.6 |
| C5_pure_mod_zca_w20_no_txt_mean | [16, 16, 16] | [19, 19, 19] | [19, 19, 19] | [15, 15, 15] | 32.0 / 38.0 / 38.0 / 48.4 |
| C6_bilevel_global_plus_innov_w20 | [22, 21, 22] | [22, 21, 22] | [25, 24, 25] | [15, 14, 15] | 43.33 / 43.33 / 49.33 / 47.33 |
| C8a_shuffled_c3b_w20_ctrl | [23, 23, 23] | [22, 21, 22] | [25, 24, 25] | [16, 15, 16] | 46.0 / 43.33 / 49.33 / 50.53 |
| C8b_innov_only_shuffled_c6_ctrl | [17, 18, 18] | [20, 22, 21] | [23, 26, 21] | [15, 15, 15] | 35.33 / 42.0 / 46.67 / 48.4 |
| C8c_both_branches_shuffled_c6_ctrl | [17, 18, 18] | [20, 22, 21] | [23, 26, 21] | [15, 15, 15] | 35.33 / 42.0 / 46.67 / 48.4 |

## M2 — probes và đối chứng

Các episodes là biến đổi từ cùng4 held-out origins. Reversal/Roll tác động interval coordinates/banks; tên “Physical” trong source không có nghĩa đã dịch waveform hoặc đảo frame. Acc3 dùng argmax riêng3 relation logits trên subset không foil. Paired chỉ tính cặp còn đủ thuận/đảo sau balancing.

### PartB1_Natural_Unshifted_Balanced

| Probe | Đúng/sai/tổng4 | Acc4 % | Acc3 % | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_c1_ref | 123/109/232 | 53.0 | 64.4 | 41/58 | 22/58 | 32/58 | 28/58 | — |
| B1b_unmodified_softmax_mean_time_c6_matched | 119/113/232 | 51.3 | 59.8 | 35/58 | 28/58 | 28/58 | 28/58 | — |
| B2_coord_shuffle_control | 114/118/232 | 49.1 | 39.1 | 16/58 | 22/58 | 24/58 | 52/58 | — |
| B3_wrong_event_query_control | 73/159/232 | 31.5 | 40.2 | 14/58 | 23/58 | 5/58 | 31/58 | — |
| B4_uniform_retrieval_sim0_control | 58/174/232 | 25.0 | 33.3 | 12/58 | 0/58 | 46/58 | 0/58 | — |
| B5_barycenter_plus_phi_emd_no_null_c6 | 114/118/232 | 49.1 | 55.7 | 26/58 | 34/58 | 26/58 | 28/58 | — |
| B6_shift_sensitive_presence_null_only_c6 | 105/127/232 | 45.3 | 33.9 | 7/58 | 28/58 | 15/58 | 55/58 | — |
| B7_barycenter_plus_presence_NO_PHI_yann_ctrl_c6 | 149/83/232 | 64.2 | 59.2 | 29/58 | 36/58 | 33/58 | 51/58 | — |
| B8_full_bary_phi_emd_presence_c6_connected | 149/83/232 | 64.2 | 60.3 | 29/58 | 39/58 | 31/58 | 50/58 | — |

### PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit

| Probe | Đúng/sai/tổng4 | Acc4 % | Acc3 % | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_c1_ref | 248/328/576 | 43.1 | 60.0 | 95/144 | 32/144 | 81/144 | 40/144 | 125/216 |
| B1b_unmodified_softmax_mean_time_c6_matched | 222/354/576 | 38.5 | 54.6 | 78/144 | 33/144 | 66/144 | 45/144 | 113/216 |
| B2_coord_shuffle_control | 306/270/576 | 53.1 | 42.1 | 53/144 | 44/144 | 71/144 | 138/144 | 65/216 |
| B3_wrong_event_query_control | 205/371/576 | 35.6 | 42.8 | 37/144 | 47/144 | 52/144 | 69/144 | 90/216 |
| B4_uniform_retrieval_sim0_control | 144/432/576 | 25.0 | 33.3 | 0/144 | 120/144 | 24/144 | 0/144 | 60/216 |
| B5_barycenter_plus_phi_emd_no_null_c6 | 256/320/576 | 44.4 | 55.8 | 58/144 | 80/144 | 55/144 | 63/144 | 116/216 |
| B6_shift_sensitive_presence_null_only_c6 | 275/301/576 | 47.7 | 37.5 | 50/144 | 76/144 | 11/144 | 138/144 | 44/216 |
| B7_barycenter_plus_presence_NO_PHI_yann_ctrl_c6 | 351/225/576 | 60.9 | 53.5 | 67/144 | 84/144 | 67/144 | 133/144 | 114/216 |
| B8_full_bary_phi_emd_presence_c6_connected | 363/213/576 | 63.0 | 56.0 | 66/144 | 96/144 | 66/144 | 135/144 | 121/216 |

### PartB3_CrossModal_Only_TwoHop_Roll

| Probe | Đúng/sai/tổng4 | Acc4 % | Acc3 % | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_c1_ref | 40/116/156 | 25.6 | 35.9 | 8/39 | 13/39 | 4/39 | 15/39 | — |
| B1b_unmodified_softmax_mean_time_c6_matched | 31/125/156 | 19.9 | 36.8 | 9/39 | 5/39 | 1/39 | 16/39 | — |
| B2_coord_shuffle_control | 63/93/156 | 40.4 | 37.6 | 8/39 | 2/39 | 16/39 | 37/39 | — |
| B3_wrong_event_query_control | 43/113/156 | 27.6 | 33.3 | 24/39 | 4/39 | 1/39 | 14/39 | — |
| B4_uniform_retrieval_sim0_control | 39/117/156 | 25.0 | 33.3 | 0/39 | 0/39 | 39/39 | 0/39 | — |
| B5_barycenter_plus_phi_emd_no_null_c6 | 46/110/156 | 29.5 | 36.8 | 5/39 | 4/39 | 12/39 | 25/39 | — |
| B6_shift_sensitive_presence_null_only_c6 | 57/99/156 | 36.5 | 34.2 | 1/39 | 18/39 | 1/39 | 37/39 | — |
| B7_barycenter_plus_presence_NO_PHI_yann_ctrl_c6 | 54/102/156 | 34.6 | 35.9 | 6/39 | 9/39 | 2/39 | 37/39 | — |
| B8_full_bary_phi_emd_presence_c6_connected | 58/98/156 | 37.2 | 32.5 | 6/39 | 8/39 | 7/39 | 37/39 | — |
| B8a_teacher_forced_bridge_ctrl | 57/99/156 | 36.5 | 35.9 | 5/39 | 9/39 | 8/39 | 35/39 | — |
| B8b_oracle_hop1_anchor_upper_bound | 59/97/156 | 37.8 | 31.6 | 7/39 | 6/39 | 8/39 | 38/39 | — |

Foil selection và hop diagnostics trước/sau balancing lưu đầy đủ trong main_results_s1/s2.json. Foils vẫn dựa vào caption/embedding disjointness, không phải annotation event vắng mặt. “Oracle hop1 anchor” thay vector context cho bridge, vẫn giữ lg1/p1 trong time features; chưa phải oracle sửa toàn bộ hop1 localization hoặc một upper bound toán học.

## M3 — micro8

| ID | Task/gold | Native | V5 | Full | No-desink | Order-sign đảo | Slot-profile đảo | Rel-only | Order-only |
|---|---|---|---|---|---|---|---|---|---|
| 1449 | TA/A | A | A | A | A | A | A | A | A |
| 2150 | TF/C | A | A | A | A | A | A | A | A |
| 1929 | TF/C | C | C | C | C | C | C | C | C |
| 2054 | TA/B | A | A | A | A | A | A | A | A |
| 1956 | CMTB/A | A | A | A | A | A | A | A | A |
| 1513 | TF/A | B | B | B | B | B | B | B | B |
| 2264 | CMTB/B | B | B | B | B | B | B | B | B |
| 1821 | TF/C | C | C | C | C | C | C | C | C |

| Task/gold | N | Native đúng/sai | V5 đúng/sai | Full đúng/sai |
|---|---:|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo |

Cứu: []; làm hỏng: []; thay đổi prediction: [].

| ID | Gate true / permuted / semantic wrong | TV full/native | TV random/native | TV mid sign swap / slot profile swap | Kernel no-op error |
|---|---|---:|---|---:|
| 1449 | 0.9562 / 0.9513 / 0.9649 | 0.0309 | 0.0283 | 0.0309 / 0.0461 | 0.0 |
| 2150 | 0.9849 / 0.6383 / 0.9682 | 0.0191 | 0.0087 | 0.0114 / 0.0138 | 0.0 |
| 1929 | 0.8366 / 0.9602 / 0.9303 | 0.0173 | 0.0171 | 0.0027 / 0.0249 | 0.0 |
| 2054 | 0.8886 / 0.7385 / 0.8552 | 0.0229 | 0.0025 | 0.0 / 0.0229 | 0.0 |
| 1956 | 0.9262 / 0.66 / 0.9484 | 0.0037 | 0.0051 | 0.001 / 0.0068 | 0.0 |
| 1513 | 0.7401 / 0.9577 / 0.9859 | 0.0175 | 0.0164 | 0.0065 / 0.0302 | 0.0 |
| 2264 | 0.9705 / 0.7412 / 0.9675 | 0.0017 | 0.0011 | 0.0004 / 0.0033 | 0.0 |
| 1821 | 0.9401 / 0.8708 / 0.9494 | 0.0173 | 0.0113 | 0.0031 / 0.0078 | 0.0 |

Trung bình: `{"confidence_gate_gamma_true_query": 0.9054, "confidence_gate_gamma_wrong_query": 0.81475, "confidence_gate_gamma_semantic_wrong_query": 0.946225, "tv_late_kslot_swap": 0.0018124999999999999, "tv_mid_kslot_swap": 0.007, "tv_mid_slot_profile_swap": 0.019475, "tv_mid_routed_vs_nat": 0.0163, "tv_mid_irrel_slot_vs_nat": 0.0151375, "tv_mid_rand3seed_vs_nat": 0.0113125}`.

Legacy parity được kiểm tra bằng original v5 source trên cùng inputs/runtime; không mặc định trùng bit với lần chạy lịch sử khác runtime. Kết quả từng mẫu lưu trong m3_audit_s1/s2.json.

Accuracy của các control có sẵn, chạy lại cùng source/hệ số/seeds để ghi prediction: {"irrelevant_slot": 5, "random_seed_101": 5, "random_seed_202": 5, "random_seed_303": 5}. Full replay probability error=0 ở8/8. Các control khớp L2 norm của token-bias profile theo modality; random/irrelevant tắt row de-sinking và audio damping, nên không coi là toàn bộ attention differential cùng norm.

Audit signed transport: max |(gated W+−W−)−gated normalized mean lag| = 8.94e-08. Xem [math_note_signed_transport.md](math_note_signed_transport.md); điều này áp dụng riêng term W trong M3, không suy toàn bộ M2/M3 chỉ có mean-time.

### Sinh tự do32 tokens

Điểm heuristic_clause_aligned chỉ là rule-based audit, không phải semantic accuracy được annotation. Đọc trực tiếp văn bản, quan hệ được nói tới và reference event. {"native": 1, "routed": 1}. Ví dụ2150 và1821 nói rõ event sau speech nhưng bị flag generic/false; không suy chất lượng QA từ số heuristic này.

**1449 — TA/A (Yes)**

Native: Yes, the narrated event of adding cooked chickpeas is happening at the same time in the video. You can see the chickpeas being added to

CAOR: Yes, the narrated event of adding cooked chickpeas happens at the same time in the video. You can see the chickpeas being added to the

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": true, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": true, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`

**2150 — TF/C (Future)**

Native: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

CAOR: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

**1929 — TF/C (Future)**

Native: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So, it's likely that the narration starts a

CAOR: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So, it's likely that the narration starts a

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

**2054 — TA/B (No)**

Native: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

CAOR: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": false, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": false, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

**1956 — CMTB/A (mix in flour)**

Native: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. It's a

CAOR: Well, when the speech is heard, it seems like the person is adding flour to the pot. You can see the flour being poured in. And then,

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

**1513 — TF/A (Past)**

Native: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, it's right

CAOR: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, it starts right

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

**2264 — CMTB/B (add milk and one egg whisk together)**

Native: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisking action in the video. So

CAOR: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisking action in the video. So

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

**1821 — TF/C (Future)**

Native: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

CAOR: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

CAOR audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

## Artifacts

Hai archive được kiểm tra SHA256 và từng entry local, không extract. Mỗi server có 198 raw PT files; 3000 query rows được recompute; 29 linear probes được tính lại từ logits. Backbone parameter version counters giữ nguyên, hook count trước/sau bằng nhau, CUDA khỏe. Không suy novelty/CVPR acceptance từ Stage-0.

Lỗi JSON serialization ở observation wrapper được giữ tại audit_wrapper_error; source thuật toán không đổi. M0/M1 đã hoàn thành được dùng tiếp; M2 chạy lại sau khi sửa ghi metadata, M3 chạy sau đó. Audit thêm về signed-Wasserstein/mean-lag và teacher/oracle feature differences lưu riêng; không dùng chúng để thay phương pháp mainrun.

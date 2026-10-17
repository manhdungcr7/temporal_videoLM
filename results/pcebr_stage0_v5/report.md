# OB-TVOR / PC-EBR Stage-0 v5 — kết quả và kiểm toán, 2026-10-16

**V5 sửa đúng parity và uniform gate. M2 full C5 đạt 67,7/59,9/34,6% trên Natural/Reversal/Roll; bỏ Phi vẫn đạt 67,2/57,3/34,0%. M3 giữ nguyên toàn bộ MC predictions: native và routed đều 5/8. Chưa xác nhận cải thiện temporal QA để chuyển A0–A7.**

Kết quả của cấu hình gốc được báo cáo riêng với các đối chứng exploratory bổ sung. Xem đánh giá và hướng cải tiến ở [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md).

Hai runtime mới, cùng 252 pseudo pairs: 202 train từ 16 origins và 50 held-out từ 4 origins đã xem trong các vòng trước. Ba seeds 20261015/16/17. M1: 36 fits/server × 180 steps; M2: 2 bridge fits × 120 steps, 27 probes × 200 steps. Hai replicas trùng kết quả M1/M2, không tăng số origins độc lập. Chỉ 8 mẫu micro cũ, 4/shard, rotation 0; chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.

Source gốc 1939 dòng và alias có cùng SHA. Wrapper chỉ lưu observations/checkpoints; không đổi công thức, seeds, nhãn hoặc ngân sách mainrun. Native pairs là post-media readout block index 27; pooled là mean media states cùng block. Pseudo descriptions và substring Nonnull filter chưa phải event/time ground truth độc lập.

## M0 — đại số

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
  "yann_tiny_noise_1e4_v5_max_weight": 0.2,
  "yann_tiny_noise_1e4_v5_gate": 1.0456810622201575e-14,
  "yann_tiny_noise_1e4_v5_phi_max_abs": 2.2593699300772195e-21,
  "uniform_zero_evidence_gate": 0.0,
  "uniform_zero_evidence_effective_scale": 0.0,
  "toy_9cases_bivector_reversal_audit": [
    {
      "K": 5,
      "seed": 101,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.108919,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 2e-08,
      "rel_scalar_fwd": 0.587838,
      "rel_scalar_rev": -0.587838
    },
    {
      "K": 5,
      "seed": 202,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.098441,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 0.0,
      "rel_scalar_fwd": -0.016774,
      "rel_scalar_rev": 0.016774
    },
    {
      "K": 5,
      "seed": 303,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.100886,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 0.0,
      "rel_scalar_fwd": -1.280854,
      "rel_scalar_rev": 1.280854
    },
    {
      "K": 6,
      "seed": 101,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.10016,
      "content_even_max_err": 0.0,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 0.0,
      "rel_scalar_fwd": 0.881498,
      "rel_scalar_rev": -0.881497
    },
    {
      "K": 6,
      "seed": 202,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.091207,
      "content_even_max_err": 0.0,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 3e-08,
      "rel_scalar_fwd": 0.070109,
      "rel_scalar_rev": -0.070109
    },
    {
      "K": 6,
      "seed": 303,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.099447,
      "content_even_max_err": 0.0,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 4e-08,
      "rel_scalar_fwd": -1.613147,
      "rel_scalar_rev": 1.613147
    },
    {
      "K": 9,
      "seed": 101,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.110982,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 1e-08,
      "rel_scalar_fwd": 2.744035,
      "rel_scalar_rev": -2.744035
    },
    {
      "K": 9,
      "seed": 202,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.092221,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 0.0,
      "rel_scalar_fwd": -1.717809,
      "rel_scalar_rev": 1.717809
    },
    {
      "K": 9,
      "seed": 303,
      "vec_fwd_rev_cosine": 0.161103,
      "vec_max_abs_diff": 0.098018,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "orthogonality_max_err": 1e-08,
      "rel_scalar_fwd": -0.566304,
      "rel_scalar_rev": 0.566305
    }
  ],
  "all_9_toy_cases_content_invariant_and_order_flipped": true
}
```

Cosine thuận/đảo xấp xỉ .1611 được quyết định bởi tan(theta)=.85, với hai kênh trực giao even/odd. Đây là kiểm tra representation đổi đúng theo định nghĩa; không đo temporal correctness. Uniform gate bằng 0 sửa đúng sàn .35 của v4.

## M1 — đúng qua ba seeds

Global/Exact/Multipos có mẫu số 50, Nonnull có 31 theo substring filter. Số sai của từng seed = mẫu số trừ số đúng.

### M1_native_orthogonal_factorial

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_w00 | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.00/32.00/42.00/32.30 |
| C2_ctx_diag_w20_no_whiten | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.00/32.00/42.00/32.30 |
| C3_txt_std_diag_w20_v4_ref | [15, 12, 13] | [20, 18, 19] | [26, 24, 25] | [12, 10, 11] | 26.67/38.00/50.00/35.50 |
| C4_mod_zca_ctx_diag_w20 | [12, 12, 13] | [16, 16, 18] | [21, 21, 24] | [9, 9, 10] | 24.67/33.33/44.00/30.10 |
| C5_mod_zca_hybrid_ctx_rec_w20 | [15, 14, 15] | [17, 16, 17] | [22, 21, 22] | [9, 8, 9] | 29.33/33.33/43.33/27.93 |
| C8_shuffled_within_rec_w20_ctrl | [13, 11, 15] | [19, 18, 17] | [21, 21, 18] | [13, 10, 13] | 26.00/36.00/40.00/38.70 |

### M1_pooled_orthogonal_factorial

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_w00 | [16, 16, 16] | [19, 19, 19] | [23, 23, 23] | [14, 14, 14] | 32.00/38.00/46.00/45.20 |
| C2_ctx_diag_w20_no_whiten | [16, 16, 16] | [20, 20, 20] | [24, 24, 24] | [14, 14, 14] | 32.00/40.00/48.00/45.20 |
| C3_txt_std_diag_w20_v4_ref | [23, 23, 23] | [23, 23, 23] | [26, 26, 26] | [16, 16, 16] | 46.00/46.00/52.00/51.60 |
| C4_mod_zca_ctx_diag_w20 | [22, 22, 21] | [22, 22, 21] | [25, 25, 24] | [16, 16, 15] | 43.33/43.33/49.33/50.53 |
| C5_mod_zca_hybrid_ctx_rec_w20 | [21, 21, 21] | [21, 21, 21] | [24, 24, 24] | [15, 15, 15] | 42.00/42.00/48.00/48.40 |
| C8_shuffled_within_rec_w20_ctrl | [23, 23, 23] | [22, 21, 22] | [25, 24, 25] | [16, 15, 16] | 46.00/43.33/49.33/50.53 |

C2 và C3 cùng within-rec weight .20, giúp tách tác dụng chuẩn hóa. Tuy nhiên C3 không phải exact replay C3 v4: v5 khởi tạo mean_shift từ media đã pre-whiten, v4 dùng media normalize trước whitening. C4/C5 có ZCA riêng mỗi modality nhưng _pre_whiten_media vẫn cộng nhánh (x_in−txt_mean)*txt_inv_std trước normalize; đây là hybrid chưa loại bỏ việc trừ text mean trên media như prose tuyên bố. C8 là shuffle của txt_diag_v4, chưa phải shuffle khớp C5 ZCA/hybrid.

## M2 — mọi lớp và đối chứng

Các episode là nhiều biến đổi/cặp từ cùng 4 held-out origins; không phải video độc lập. 3-way dùng argmax riêng ba relation logits trên nonfoil subset. Paired metric chỉ xét các cặp còn đủ thuận/đảo sau balancing.

### PartB1_Natural_Unshifted_Balanced

| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_baseline_c1 | 123/109/232 | 53.0 | 64.4 | 41/58 | 22/58 | 32/58 | 28/58 | — |
| B1b_unmodified_softmax_mean_time_c5_zca | 108/124/232 | 46.6 | 54.6 | 31/58 | 30/58 | 23/58 | 24/58 | — |
| B2_coord_shuffle_control | 126/106/232 | 54.3 | 46.0 | 30/58 | 17/58 | 23/58 | 56/58 | — |
| B3_wrong_event_query_control | 62/170/232 | 26.7 | 36.2 | 6/58 | 12/58 | 13/58 | 31/58 | — |
| B4_uniform_retrieval_sim0_control | 58/174/232 | 25.0 | 33.3 | 12/58 | 0/58 | 46/58 | 0/58 | — |
| B5_barycenter_plus_phi_no_null_c1 | 117/115/232 | 50.4 | 60.9 | 35/58 | 29/58 | 29/58 | 24/58 | — |
| B6_shift_sensitive_presence_null_only_c5 | 89/143/232 | 38.4 | 21.8 | 5/58 | 17/58 | 10/58 | 57/58 | — |
| B7_full_bary_phi_presence_c1_connected | 109/123/232 | 47.0 | 59.8 | 14/58 | 24/58 | 17/58 | 54/58 | — |
| B8_full_bary_phi_presence_c5_zca_connected | 157/75/232 | 67.7 | 60.9 | 29/58 | 37/58 | 34/58 | 57/58 | — |

### PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit

| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_baseline_c1 | 248/328/576 | 43.1 | 60.0 | 95/144 | 32/144 | 81/144 | 40/144 | 125/216 |
| B1b_unmodified_softmax_mean_time_c5_zca | 207/369/576 | 35.9 | 52.1 | 70/144 | 22/144 | 65/144 | 50/144 | 110/216 |
| B2_coord_shuffle_control | 284/292/576 | 49.3 | 42.1 | 33/144 | 35/144 | 73/144 | 143/144 | 60/216 |
| B3_wrong_event_query_control | 181/395/576 | 31.4 | 39.6 | 19/144 | 59/144 | 23/144 | 80/144 | 68/216 |
| B4_uniform_retrieval_sim0_control | 144/432/576 | 25.0 | 33.3 | 0/144 | 120/144 | 24/144 | 0/144 | 60/216 |
| B5_barycenter_plus_phi_no_null_c1 | 248/328/576 | 43.1 | 55.6 | 69/144 | 78/144 | 49/144 | 52/144 | 116/216 |
| B6_shift_sensitive_presence_null_only_c5 | 264/312/576 | 45.8 | 35.2 | 72/144 | 46/144 | 4/144 | 142/144 | 27/216 |
| B7_full_bary_phi_presence_c1_connected | 268/308/576 | 46.5 | 51.9 | 30/144 | 75/144 | 29/144 | 134/144 | 111/216 |
| B8_full_bary_phi_presence_c5_zca_connected | 345/231/576 | 59.9 | 55.8 | 60/144 | 86/144 | 59/144 | 140/144 | 120/216 |

### PartB3_CrossModal_Only_TwoHop_Roll

| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_baseline_c1 | 51/105/156 | 32.7 | 34.2 | 7/39 | 11/39 | 9/39 | 24/39 | — |
| B1b_unmodified_softmax_mean_time_c5_zca | 38/118/156 | 24.4 | 35.9 | 12/39 | 2/39 | 8/39 | 16/39 | — |
| B2_coord_shuffle_control | 48/108/156 | 30.8 | 30.8 | 8/39 | 1/39 | 5/39 | 34/39 | — |
| B3_wrong_event_query_control | 34/122/156 | 21.8 | 31.6 | 0/39 | 6/39 | 10/39 | 18/39 | — |
| B4_uniform_retrieval_sim0_control | 39/117/156 | 25.0 | 33.3 | 0/39 | 0/39 | 39/39 | 0/39 | — |
| B5_barycenter_plus_phi_no_null_c1 | 39/117/156 | 25.0 | 29.9 | 10/39 | 6/39 | 9/39 | 14/39 | — |
| B6_shift_sensitive_presence_null_only_c5 | 41/115/156 | 26.3 | 32.5 | 1/39 | 0/39 | 14/39 | 26/39 | — |
| B7_full_bary_phi_presence_c1_connected | 51/105/156 | 32.7 | 29.9 | 5/39 | 7/39 | 3/39 | 36/39 | — |
| B8_full_bary_phi_presence_c5_zca_connected | 54/102/156 | 34.6 | 39.3 | 9/39 | 4/39 | 7/39 | 34/39 | — |

### Foil proxy audit

```json
{
  "PartB1_Natural_Unshifted_Balanced": {
    "train": {
      "total_origins": 16,
      "strict_cos015_and_jaccard0_origins": 5,
      "lexical_jaccard0_origins": 10,
      "string_disjoint_fallback_origins": 1,
      "note": "Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation."
    },
    "heldout": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "note": "Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation."
    }
  },
  "PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit": {
    "train": {
      "total_origins": 16,
      "strict_cos015_and_jaccard0_origins": 5,
      "lexical_jaccard0_origins": 10,
      "string_disjoint_fallback_origins": 1,
      "note": "Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation."
    },
    "heldout": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "note": "Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation."
    }
  },
  "PartB3_CrossModal_Only_TwoHop_Roll": {
    "train": {
      "total_origins": 16,
      "strict_cos015_and_jaccard0_origins": 5,
      "lexical_jaccard0_origins": 10,
      "string_disjoint_fallback_origins": 1,
      "note": "Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation."
    },
    "heldout": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "note": "Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation."
    }
  }
}
```

Foils được phân hạng strict/lexical/string; tất cả vẫn là proxy từ mô tả, chưa xác nhận event vắng mặt. Thêm raw affinity có thể cải thiện phân biệt recording/query khác nguồn; phải đọc riêng relation-only và coordinate-shuffle trước khi quy gain 4-way cho temporal reasoning. B7 dùng C1; B8 và các controls B2/B3/B4 dùng C5. Baseline khớp B8 là B1b.

### Đối chứng exploratory bỏ riêng Phi, giữ C5 mean-time và presence

Cùng chuẩn hóa train, linear head, seed, lr=.035, weight decay=.04, 200 steps như v5. Không thay mainrun. Raw X/y/logits/checkpoint được lưu trong expl_presence_probe_*.pt.

| Suite | Mean-time+presence, không Phi: đúng/tổng | Acc4% | Acc3% | Full B8 Acc4% |
|---|---|---:|---:|---:|
| PartB1_Natural_Unshifted_Balanced | 156/232 | 67.2 | 63.8 | 67.7 |
| PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit | 330/576 | 57.3 | 51.9 | 59.9 |
| PartB3_CrossModal_Only_TwoHop_Roll | 53/156 | 34.0 | 35.0 | 34.6 |

## M3 — 8 mẫu micro

| ID | Task/gold | Native→routed/reversed | True/wrong gate | Fwd/rev cosine | Swap TV late/mid |
|---|---|---|---|---:|---|
| 1449 | TA/A | A→A/A | 0.9489/0.0126 | 0.1611 | 0.0156/0.0152 |
| 2150 | TF/C | A→A/A | 0.9403/0.0242 | 0.1611 | 0.0241/0.0446 |
| 1929 | TF/C | C→C/C | 0.9203/0.0216 | 0.1611 | 0.0194/0.0809 |
| 2054 | TA/B | A→A/A | 0.9584/0.1134 | 0.1611 | 0.0/0.0601 |
| 1956 | CMTB/A | A→A/A | 0.9414/0.4581 | 0.1611 | 0.0051/0.0131 |
| 1513 | TF/A | B→B/B | 0.9496/0.0194 | 0.1611 | 0.0057/0.0645 |
| 2264 | CMTB/B | B→B/B | 0.9487/0.002 | 0.1611 | 0.0055/0.0048 |
| 1821 | TF/C | C→C/C | 0.9626/0.0011 | 0.1611 | 0.0344/0.0748 |

Native đúng 5/8, routed đúng 5/8. Mean true gate=0.946275; wrong-dimension-permuted-query gate=0.081550. Mean TV swap late=0.013725, mid=0.044750; routed/native=0.020512, random/native=0.040837.

| Task/gold | N | Native đúng/sai | Routed đúng/sai |
|---|---:|---|---|
| TA/A | 1 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo |

Đây là micro rotation 0; không so trực tiếp với bảng 40 mẫu có cyclic rotations. Wrong-query trong code là hoán vị các chiều hidden của cùng stem, không phải một câu hỏi có nghĩa nhưng hỏi event khác. VO chỉ được tính ở block 18; cùng vector được inject vào các blocks 14/16/18/20/22/24 hoặc 28–33. Không được mô tả là VO theo đúng từng block.

### Văn bản sinh 32 tokens

Regex semantic_gold_aligned: native 6/8, routed 6/8. Đây là chấm bằng regex/keyword, chưa phải annotation semantic độc lập; các phản ví dụ ở controls_s*.json.

Ba phản ví dụ chấm nhầm được chạy lại trực tiếp từ hàm gốc tại [freeform_metric_counterexamples.json](freeform_metric_counterexamples.json): phủ định TA, phủ định action CMTB và quan hệ TF nói về sự kiện khác đều có thể được regex tính aligned.

**1449 — TA/A (Yes)**

Native: Yes, the narrated event of adding cooked chickpeas is happening at the same time in the video. You can see the chickpeas being added to

Routed: Yes, the narrated event of adding chickpeas to the pot is happening at the same time in the video. You can see the chickpeas being

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": true, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": true, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

**2150 — TF/C (Future)**

Native: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

Routed: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

**1929 — TF/C (Future)**

Native: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So, it's likely that the narration starts a

Routed: The narrated event happens right after the onions are stirred in the pot. So, it's likely that the stirring of the onions is part of the process described

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

**2054 — TA/B (No)**

Native: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

Routed: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": false, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "semantic_gold_aligned": false}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": false, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "semantic_gold_aligned": false}`

**1956 — CMTB/A (mix in flour)**

Native: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. It's a

Routed: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. And then,

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": true, "semantic_gold_aligned": true}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": true, "semantic_gold_aligned": true}`

**1513 — TF/A (Past)**

Native: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, it's right

Routed: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, if the speech

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "semantic_gold_aligned": false}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "semantic_gold_aligned": false}`

**2264 — CMTB/B (add milk and one egg whisk together)**

Native: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisking action in the video. So

Routed: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the hand moving the whisk around in the bowl

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": true, "semantic_gold_aligned": true}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": true, "semantic_gold_aligned": true}`

**1821 — TF/C (Future)**

Native: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

Routed: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

Routed audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "semantic_gold_aligned": true}`

## Đối chứng exploratory cùng cường độ .045×gate

Các nhánh này được thêm riêng sau thiết kế gốc để tách raw/VO và content/order, không chỉnh mainrun hoặc chọn siêu tham số theo QA. Chỉ thêm MC forwards, không sinh lại text.

| ID | Native | Full VO+order | Raw cùng cường độ | VO content-only | Gold |
|---|---|---|---|---|---|
| 1449 | A | A | A | A | A |
| 2150 | A | A | B | A | C |
| 1929 | C | C | C | C | C |
| 2054 | A | A | A | A | B |
| 1956 | A | A | A | A | A |
| 1513 | B | B | B | B | A |
| 2264 | B | B | B | B | B |
| 1821 | C | C | C | C | C |

Tổng đúng: native 5/8; full 5/8; raw low-strength 5/8; VO content-only 5/8. Zero-scale replay khớp native logits tuyệt đối 8/8.

Không suy bảo toàn cú pháp/ngữ nghĩa 100% từ việc giữ norm hoặc bước tiếp tuyến. Chỉ thay biên độ giữa v4/v5 cũng có thể giảm generic captioning; đối chứng cùng cường độ ở trên kiểm tra phần nào đóng góp của VO và order.

## Tái lập và artifacts

Mỗi server có 140 raw PT files, 1.800 retrieval query rows được tính lại đúng, 27 linear probe outputs được tổng hợp từ logits. Hai archives và từng entry được kiểm tra SHA local, không extract. Checkpoints ZCA lưu cả tensors ngoài state_dict. Native replay, token roundtrip, hook/option disjointness và prompt không choices đều qua 8/8; mọi intervention hooks đã tháo và CUDA khỏe.

Mainrun, controls, source gốc, proposal và các job logs được lưu riêng. Names Yoshua/Yann là quy ước vai trò; không phải xác nhận tác giả thật. Novelty, tổng quát hóa và CVPR acceptance chưa được xác nhận từ Stage-0 này.

PC-EBR Stage-0 v3 — kết quả và phản biện, 2026-10-15

**Cổng 0 v3 không qua các điều kiện đã công bố. Unit checks và group-max invariance qua; hiệu năng M1, các gate M2 và phần sensitivity của M3 không qua. Chưa chạy A0–A7 hoặc reserve400.**

Hai runtime mới; cùng4 cache pairs,202 train/50 held-out từ16/4 recording origins;3seeds. M1/M2 và unit results giống nhau giữa hai replicas. Bản gốc lỗi backward-through-graph lần2 ở M2. Repair: đóng băng feature teachers M1 trước train probes; weights được so bitwise với checkpoint và không đổi. M1 không chạy lại. Bản script standalone được sửa bằng một `.detach()` ở media feature extraction; công thức/loss/labels/seeds không đổi. M3 đủ8ID (4/shard, rotation0), không thay mẫu, cả hai GPU runtime hoàn tất bình thường.

**1. Unit checks**

Basis diag=[2,0,0,0], lowrank=[2,2,2,2]; mean-zero counterexample đổi winner0→2, Max-Zero giữ winner0. Đây là identity toán học đã được triển khai đúng; không tự chứng minh retrieval hoặc temporal grounding.

**2. M1 — số đúng theo3seeds và mean metrics**

Global/Exact/Multipos có50queries; Nonnull có31 theo NULL_PATTERNS mới (v2 là32), nên không so trực tiếp Nonnull% giữa hai phiên bản như cùng denominator. Held-out4origins đã được xem qua nhiều vòng; replicas/seeds không làm tăng số samples độc lập.

**Native post-media readout**

| Condition | Global đúng/50 qua3seeds | Exact đúng/50 qua3seeds | Multipos mean% | Nonnull mean% |
|---|---|---|---:|---:|
| C1_ctx_only_diag | [11, 11, 11] | [16, 16, 16] | 42.00 | 32.30 |
| C2_loc_only_diag | [6, 6, 6] | [12, 12, 12] | 32.00 | 19.40 |
| C3_ctx_plus_mean_zero_adv_diag | [12, 13, 13] | [17, 16, 16] | 42.67 | 30.10 |
| C4_ctx_plus_max_zero_hier_diag | [8, 8, 8] | [12, 12, 12] | 32.00 | 19.40 |
| C5_ctx_only_lowrank | [8, 12, 13] | [11, 14, 12] | 30.00 | 25.80 |
| C6_loc_only_lowrank | [4, 5, 5] | [11, 12, 10] | 30.00 | 17.20 |
| C7_ctx_plus_max_zero_hier_lowrank | [12, 14, 12] | [11, 12, 10] | 30.00 | 17.20 |
| C8_shuffled_within_rec_ctrl | [9, 7, 7] | [8, 8, 13] | 26.67 | 19.33 |

**Pooled media**

| Condition | Global đúng/50 qua3seeds | Exact đúng/50 qua3seeds | Multipos mean% | Nonnull mean% |
|---|---|---|---:|---:|
| C1_ctx_only_diag | [16, 16, 16] | [19, 19, 19] | 46.00 | 45.20 |
| C2_loc_only_diag | [5, 5, 6] | [15, 15, 16] | 38.67 | 33.37 |
| C3_ctx_plus_mean_zero_adv_diag | [17, 17, 17] | [17, 17, 17] | 42.00 | 41.90 |
| C4_ctx_plus_max_zero_hier_diag | [13, 13, 13] | [15, 15, 15] | 38.00 | 32.30 |
| C5_ctx_only_lowrank | [14, 10, 7] | [13, 15, 15] | 36.67 | 33.37 |
| C6_loc_only_lowrank | [2, 2, 2] | [11, 9, 9] | 27.33 | 18.30 |
| C7_ctx_plus_max_zero_hier_lowrank | [13, 9, 7] | [14, 13, 11] | 35.33 | 23.67 |
| C8_shuffled_within_rec_ctrl | [4, 4, 10] | [10, 14, 13] | 32.00 | 24.73 |

Group maxima error=0 và winning-group agreement100% cho C4/C7 ở mọi3seeds, cả hai representations. Nhưng native Exact C1→C4:32→24%; C5→C7:24.67→22%. Pooled:38→30% và28.67→25.33%. C7 Nonnull còn thua C8 shuffle (native17.20% so19.33%; pooled23.67% so24.73%). Vì vậy gate performance M1 fail. Pooled context-only diag C1 tốt nhất đơn giản:16/50global,19/50exact,23/50multipos,14/31nonnull. Max-Zero bảo toàn winning group, không bảo toàn đúng semantic slot: native C1global22%→C4global16% dù group giữ nguyên.

**3. M2 — gate so với đầy đủ controls**

Natural/Reverse/Roll có lần lượt232/576/752held-out episodes, cân bằng4lớp. Đây là synthetic episodes tái dùng4origins, không phải video QA độc lập. 3-way là argmax riêng3relation logits trên subset không foil; khác với chỉ bỏ foil khỏi4-way predictions.

**PartB1_Natural_Unshifted_Balanced**

| Probe | Đúng/tổng4-way | 4-way% | 3-way% | Before% | During% | After% | Foil% |
|---|---|---:|---:|---:|---:|---:|---:|
| B1_matched_std_attn_baseline | 70/232 | 30.2 | 39.1 | 27.6 | 60.3 | 27.6 | 5.2 |
| B2_coord_shuffle_control | 60/232 | 25.9 | 32.8 | 48.3 | 6.9 | 41.4 | 6.9 |
| B3_wrong_event_query_control | 74/232 | 31.9 | 40.2 | 70.7 | 22.4 | 20.7 | 13.8 |
| B4_uniform_retrieval_sim0_control | 58/232 | 25.0 | 33.3 | 100.0 | 0.0 | 0.0 | 0.0 |
| B5_pcebr_phi_only_no_null | 68/232 | 29.3 | 38.5 | 27.6 | 36.2 | 13.8 | 39.7 |
| B6_pcebr_null_only_no_phi | 63/232 | 27.2 | 31.6 | 44.8 | 0.0 | 50.0 | 13.8 |
| B7_pcebr_pos_debiased_full | 73/232 | 31.5 | 37.4 | 25.9 | 39.7 | 25.9 | 34.5 |

Uniform gate pass; full4-way gap so control tốt nhất=-0.4pp,3-way gap=-2.8pp. During=39.7%.

**PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit**

| Probe | Đúng/tổng4-way | 4-way% | 3-way% | Before% | During% | After% | Foil% |
|---|---|---:|---:|---:|---:|---:|---:|
| B1_matched_std_attn_baseline | 204/576 | 35.4 | 41.2 | 30.6 | 56.2 | 30.6 | 24.3 |
| B2_coord_shuffle_control | 166/576 | 28.8 | 36.6 | 44.4 | 7.6 | 53.5 | 9.7 |
| B3_wrong_event_query_control | 130/576 | 22.6 | 31.2 | 25.7 | 36.8 | 27.1 | 0.7 |
| B4_uniform_retrieval_sim0_control | 144/576 | 25.0 | 33.3 | 0.0 | 0.0 | 0.0 | 100.0 |
| B5_pcebr_phi_only_no_null | 191/576 | 33.2 | 40.7 | 23.6 | 38.9 | 22.2 | 47.9 |
| B6_pcebr_null_only_no_phi | 179/576 | 31.1 | 35.0 | 0.0 | 9.7 | 89.6 | 25.0 |
| B7_pcebr_pos_debiased_full | 186/576 | 32.3 | 39.8 | 27.1 | 40.3 | 27.1 | 34.7 |

Uniform gate pass; full4-way gap so control tốt nhất=-3.1pp,3-way gap=-1.4pp. During=40.3%.

**PartB3_TwoHop_Conditioned_CrossModal_Roll**

| Probe | Đúng/tổng4-way | 4-way% | 3-way% | Before% | During% | After% | Foil% |
|---|---|---:|---:|---:|---:|---:|---:|
| B1_matched_std_attn_baseline | 210/752 | 27.9 | 36.2 | 46.8 | 11.7 | 46.3 | 6.9 |
| B2_coord_shuffle_control | 212/752 | 28.2 | 36.5 | 34.0 | 31.9 | 40.4 | 6.4 |
| B3_wrong_event_query_control | 180/752 | 23.9 | 33.2 | 6.9 | 42.0 | 40.4 | 6.4 |
| B4_uniform_retrieval_sim0_control | 188/752 | 25.0 | 33.3 | 0.0 | 0.0 | 0.0 | 100.0 |
| B5_pcebr_phi_only_no_null | 240/752 | 31.9 | 36.5 | 35.1 | 20.7 | 27.7 | 44.1 |
| B6_pcebr_null_only_no_phi | 197/752 | 26.2 | 33.3 | 87.2 | 3.2 | 5.3 | 9.0 |
| B7_pcebr_pos_debiased_full | 204/752 | 27.1 | 36.9 | 35.1 | 22.9 | 36.7 | 13.8 |

Uniform gate pass; full4-way gap so control tốt nhất=-1.1pp,3-way gap=+0.4pp. During=22.9%.

Uniform25%/33.3% đáp ứng ngưỡng cụ thể; nhưng features bằng0 là tính chất được mã hóa vào operator. Uniform có thể luôn đoán foil:foil recall100% cùng0% ở3lớp khác không có nghĩa nhận ra thiếu bằng chứng. Full natural thua wrong-event31.9%; reversal thua baseline35.4%; roll thua coordinate-shuffle28.2%. Không bộ nào đạt yêu cầu +10pp. During gate chỉ pass Natural/Reverse, fail Roll22.9%.

**4. M3 — dữ liệu đầy đủ8mẫu**

| Shard | ID | Cos(fwd,rev) | TVswap late | TVswap mid | TV M1+M2/native | TV irrelevant/native | TV random/native | Native replay error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1449 | 0.4819 | 0.1860 | 0.0038 | 0.0038 | 0.0636 | 0.2347 | 0.0 |
| 1 | 2150 | 0.6902 | 0.0578 | 0.1933 | 0.3086 | 0.1663 | 0.1978 | 0.0 |
| 1 | 1929 | 0.6980 | 0.1519 | 0.1552 | 0.0149 | 0.0283 | 0.1335 | 0.0 |
| 1 | 2054 | 0.3930 | 0.0927 | 0.2878 | 0.1949 | 0.1241 | 0.1656 | 0.0 |
| 2 | 1956 | 0.8678 | 0.0091 | 0.0072 | 0.0485 | 0.0830 | 0.0393 | 0.0 |
| 2 | 1513 | 0.7672 | 0.0513 | 0.0274 | 0.0465 | 0.0500 | 0.1113 | 0.0 |
| 2 | 2264 | 0.5067 | 0.0143 | 0.0141 | 0.0282 | 0.0053 | 0.0341 | 0.0 |
| 2 | 1821 | 0.4003 | 0.1203 | 0.1490 | 0.2100 | 0.3176 | 0.1063 | 0.0 |

Gộp8mẫu: TVswap mid=0.104725,late=0.085425,ratio=1.226×<2.0. TV M1+M2/native=0.106925,irrelevant=0.104775,random=0.127825. Không vượt random control. Native replay0 trên8/8; span roundtrip đúng8/8,hook disjoint options8/8,free-form generation nonempty8/8. Không dùng các pass kỹ thuật này để thay cho gate sensitivity hoặc temporal correctness.

**Sinh tự hồi quy sau bỏ choices — nguyên văn outputs, không phải accuracy**

| ID | Generated tokens | Text |
|---|---:|---|
| 1449 | 12 | Yes, the speech is heard at the same time as the |
| 2150 | 12 | The event occurs at 1:00. It's |
| 1929 | 12 | The onions are being browned in the pan. You can |
| 2054 | 12 | Yes, the person is doing all the steps in the video |
| 1956 | 12 | You can see the onions and bacon in the pot. The |
| 1513 | 12 | The sauce mixture is poured into the bowl, which is the |
| 2264 | 12 | You're mixing flour, cornmeal, sugar, and milk |
| 1821 | 12 | The event is likely to be a boiling pot of red liquid |

**5. Audit còn quan trọng trước đề xuất kế tiếp**

- M2 không dùng các giá trị `sim_mat` hierarchical; chỉ dùng `.device` rồi tính lại media/text affinities bằng m_loc. Context m_ctx/Max-Zero không nằm trong downstream scoring path. M3 cũng chỉ dùng m_loc, không dùng toàn hierarchical model.
- Tất cả20origins train/held-out không có foil nào đạt text-cos<.55; code dùng fallback chỉ khác chuỗi caption. Đây không phải verified absence bằng annotation.
- Natural During dùng khoảng cách centers≤2.1s, bao gồm adjacent2s windows; Roll tại anchor sát biên có thể gán Before/After cho displacement1slot=2s, không nhất quán định nghĩa Natural. B3 vẫn tách role banks khi src_a==src_b.
- M3 có đường audio-first→visual cố định, và đảo logits/weights visual trong khi evidence value bank V18 giữ nguyên; chưa phải raw AV reversal có bảo toàn event identity. Nó chưa đáp ứng nhiều chiều reasoning như user yêu cầu.
- Positive-part triệt exact uniform, nhưng tiny logit peak1e-4 đã cho normalized Phi=[1,1,1,0,0,0] dù margin≈5.6e-6; M3 cộng14%norm không gate theo uncertainty/null. Không suy calibration hoặc không hallucination từ algebra.
- sec lấy từ temporal RoPE/25 khớp decoder positions, chưa phải verified PTS:idx2150 video position clock đến15.52s dù clip12.167s; idx1513 đến16.04s dù clip13.074s. Cần clock provenance từ encoder/frame metadata trước khi gọi slots là khoảng vật lý.
- Sinh được12tokens không chứng minh grounded answer. Ví dụ2150 sinh “The event occurs at 1:00” trong clip12.167s; cần kiểm tra độc lập, không coi freeform_gen_ok là temporal correctness.

**6. Artifacts và giới hạn**

143rawPTfiles/server gồm36M1checkpoints,52retrievalmatrices,2bundles,32episodefiles (8 của attempt lỗi,24 của resume),21probes. 2600retrievalqueryrows/server recompute đúng; SHA archives và mọi entry đã kiểm local, không extract. Teacher weights freeze không đổi; hooks cuối bằng0 ởcả2runtime. M1/M2 replicas trùng. Không chạy bảng40/A0–A7/reserve400. Các phần fail chỉ bác bỏ cấu hình và dự đoán cụ thể này, không bác bỏ toàn bộ latent reasoning hoặc mọi tầng muộn.

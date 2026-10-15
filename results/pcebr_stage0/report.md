PC-EBR — kết quả Cổng 0 (Stage-0), 2026-10-15

**Kết luận: H1 không đạt tiêu chí đề xuất; M2 chỉ qua kiểm tra đại số; H3 chưa được hiện thực. Chưa đủ điều kiện chuyển A0–A7.**

Hai server T4 chạy cùng cache và cùng script cho toàn bộ số đếm giống nhau. Đây là replicas, không phải hai tập dữ liệu độc lập. Bản gốc lỗi CUDA-generator/CPU-randperm trên cả hai server. Bản devicefix chỉ thêm `device=device` cho randperm; giữ seed, split, architecture, loss và budget. Logs lỗi và cả hai source được lưu nguyên. Mỗi exact run sửa device hoàn tất trong khoảng 13–14 giây.

252 pairs: 202 train/16 recording origins; 50 held-out/4 origins. Held-out này đã được xem ở R14/R18, không phải final test mới. Nhãn là pseudo descriptions do model sinh, không phải annotation sự kiện hoặc interval truth độc lập. Native cache là post-media last-token state tại block index27 (tầng28), pooled cache là media-token mean cùng tầng, không phải một extraction mới ở tầng22.

| Cache | Cấu hình | Global đúng/tổng | Within-recording đúng/tổng |
|---|---|---:|---:|
| Native post-media readout | Archived R14 replay | 7/50 | 13/50 |
| Native post-media readout | 1_unadapted_raw | 0/50 | 9/50 |
| Native post-media readout | 2_unadapted_deconfounded | 1/50 | 9/50 |
| Native post-media readout | 3_pcebr_uncentered_r24 | 13/50 | 13/50 |
| Native post-media readout | 4_pcebr_deconfounded_r24 | 6/50 | 16/50 |
| Native post-media readout | 5_pcebr_shuffled_time_ctrl | 3/50 | 9/50 |
| Pooled media | Archived R18 replay | 1/50 | 11/50 |
| Pooled media | 1_unadapted_raw | 2/50 | 12/50 |
| Pooled media | 2_unadapted_deconfounded | 4/50 | 14/50 |
| Pooled media | 3_pcebr_uncentered_r24 | 2/50 | 8/50 |
| Pooled media | 4_pcebr_deconfounded_r24 | 5/50 | 11/50 |
| Pooled media | 5_pcebr_shuffled_time_ctrl | 10/50 | 11/50 |

Native centered r24 chỉ6/50 global so R14=7/50 và r24 uncentered=13/50; within-rec16/50 so cả hai baseline=13/50 (+6 điểm phần trăm). So native raw chưa học9/50, mức tăng within là14 điểm phần trăm, chưa đạt15 điểm đề xuất. Pooled centered=5/50 global và11/50 within; R18=1/50 và11/50, tức global+8 điểm phần trăm, within không tăng. Pooled shuffle=10/50 global,11/50 within. Không thể so within-rec với global7/50 hoặc1/50 như cùng metric.

**Đối chứng exploratory ba fixed seeds — không thay tiêu chí sau kết quả**

Seeds 20261015/20261016/20261017;384 steps, cùng r24, lr/loss/hard-neg boost. Media-only center giữ nguyên query text khi inference, giúp kiểm tra phụ thuộc vào batch pseudo queries held-out. Các số dưới là từng seed và mean; mean của3 seed không phải150 event labels độc lập. Audit v2 giữ nguyên AST training gốc; các nhánh seed20261015 khớp toàn bộ mainrun. Audit v1 viết lại loss tương đương toán học gây đổi optimization trajectory được giữ riêng audit_v1 và không dùng cho bảng chính. Một lỗi truyền OUT của wrapper v2 cũng được lưu ở audit_wrapper_error trước khi sửa; không có sample nào được thay.

| Cache | Variant | Global đúng/50 từng seed | Within-rec đúng/50 từng seed | Mean global % | Mean within % |
|---|---|---|---|---:|---:|
| avbind_native_pairs | uncentered | [13, 9, 6] | [13, 9, 11] | 18.7 | 22.0 |
| avbind_native_pairs | dual_center_original | [6, 11, 1] | [16, 12, 13] | 12.0 | 27.3 |
| avbind_native_pairs | media_only_center | [9, 8, 6] | [16, 8, 13] | 15.3 | 24.7 |
| avbind_native_pairs | dual_center_shuffle | [3, 4, 3] | [9, 8, 8] | 6.7 | 16.7 |
| avbind_pooled_pairs | uncentered | [2, 5, 3] | [8, 13, 10] | 6.7 | 20.7 |
| avbind_pooled_pairs | dual_center_original | [5, 8, 7] | [11, 10, 13] | 13.3 | 22.7 |
| avbind_pooled_pairs | media_only_center | [11, 5, 9] | [11, 9, 10] | 16.7 | 20.0 |
| avbind_pooled_pairs | dual_center_shuffle | [10, 7, 2] | [11, 12, 12] | 12.7 | 23.3 |

Các comparisons paired bên dưới là exploratory, chưa hiệu chỉnh multiple tests; chỉ4 held-out origins nên không thể xác nhận generalization từ p-value.

| Cache | Control | Metric | Cứu/làm hỏng | Exact McNemar p | Source sign-flip p |
|---|---|---|---:|---:|---:|
| avbind_native_pairs | uncentered_seed20261015 | global_correct | 1/8 | 0.0391 | 0.2500 |
| avbind_native_pairs | uncentered_seed20261015 | local_correct | 10/7 | 0.6291 | 0.7500 |
| avbind_native_pairs | dual_center_shuffle_seed20261015 | global_correct | 6/3 | 0.5078 | 1.0000 |
| avbind_native_pairs | dual_center_shuffle_seed20261015 | local_correct | 14/7 | 0.1892 | 0.1250 |
| avbind_pooled_pairs | uncentered_seed20261015 | global_correct | 5/2 | 0.4531 | 0.5000 |
| avbind_pooled_pairs | uncentered_seed20261015 | local_correct | 8/5 | 0.5811 | 0.2500 |
| avbind_pooled_pairs | dual_center_shuffle_seed20261015 | global_correct | 5/10 | 0.3018 | 0.5000 |
| avbind_pooled_pairs | dual_center_shuffle_seed20261015 | local_correct | 9/9 | 1.0000 | 1.0000 |

**Candidate sets và tính mơ hồ của pseudo labels**

avbind_native_pairs: group sizes=[8, 8, 3, 3, 5, 5, 9, 9]; random uniform local top1 chance=16.0%; 4 nhóm content trùng ở nhiều interval cùng recording/source.
- i0qYuhtSQHI/audio, indices=[9, 13]: there are no speech or other audible sounds in this audio clip.
- i0qYuhtSQHI/audio, indices=[11, 15]: there's no speech in this audio.
- -ju7_ZORsZw/audio, indices=[25, 29]: there is no speech in this audio clip.
- cDYCtBwin5g/audio, indices=[33, 37, 47, 49]: there is no speech in this audio. the only sound is music playing in the background. if you have any other questions or need more details, feel free to ask.
avbind_pooled_pairs: group sizes=[8, 8, 3, 3, 5, 5, 9, 9]; random uniform local top1 chance=16.0%; 4 nhóm content trùng ở nhiều interval cùng recording/source.
- i0qYuhtSQHI/audio, indices=[9, 13]: there are no speech or other audible sounds in this audio clip.
- i0qYuhtSQHI/audio, indices=[11, 15]: there's no speech in this audio.
- -ju7_ZORsZw/audio, indices=[25, 29]: there is no speech in this audio clip.
- cDYCtBwin5g/audio, indices=[33, 37, 47, 49]: there is no speech in this audio. the only sound is music playing in the background. if you have any other questions or need more details, feel free to ask.

Nếu cùng mô tả/query feature lặp ở nhiều interval, không có thông tin trong query để chỉ đúng instance start; hard exact-start top1 có thể phạt một alternative hợp lệ. Cần multi-positive interval labels hoặc instance-disambiguating queries. Shuffle ở mã gốc tráo media-text matching trong group, không sử dụng intervals trong objective, nên là alignment corruption control chứ chưa chứng minh học đại số thời gian.

**Kiểm tra M2 và giới hạn**

```json
{
  "uniform_clock_shift_error": 0.0,
  "commutative_sum_order_sensitivity": 0.0,
  "pcebr_directed_order_sensitivity": 4.3944,
  "phi_before_vec": [
    0.0,
    0.0,
    -2.398,
    1.946,
    -2.197,
    0.0
  ],
  "phi_during_vec": [
    0.8,
    1.0,
    -1.03,
    -1.03,
    0.0,
    -0.211
  ],
  "phi_after_vec": [
    0.0,
    0.0,
    1.946,
    -2.398,
    2.197,
    0.0
  ]
}
```

Đây là kiểm tra hàm Phi trên intervals biết trước; không có audio/video được dịch, không có event retrieval, two-hop reasoning hoặc LLM QA. Phép cộng interval bằng nhau khi đảo hai hạng là identity đại số, không chứng minh attention tiêu chuẩn mù thứ tự. Counterexample với content-dependent weights [.9,.1] và centers[3,11]: weighted time3.8, swap times10.2. Pooling giao hoán theo thứ tự liệt kê cặp (content,time) khác việc tráo gán time giữa hai content.

Translation invariance error=0 trong toy check. Nhưng common scale x2 làm full Phi đổi0.646627; signed log gaps và duration ratio có offset0.1 không scale-invariant. Order categories có thể giữ nguyên dưới positive affine map, không suy full latent/readout invariant. Full Phi không antisymmetric; center-offset channel thì có. Coverage dùng inter/min(duration), là overlap coefficient, không phải coverage của anchor; anchor[0,10],target[2,4] trả1 thay vì0.2 nếu muốn đo anchor coverage.

**H3 / M3**

Không có implementation microprobe4 mẫu, middle/late routing, TV_swap, learned two-hop bank hay source shift QA trong attachment. Không ghi H3 pass/fail hoặc tự thay bằng một heuristic khác.

**Tái lập và provenance**

Bốn cache local có SHA khớp cả hai server. Pair origins không trùng38 origins của research40; không trùng 182 origins trong danh sách loại trừ được lưu. Đây là kiểm tra metadata đã archive; không mở reserve400 hoặc chạy QA. Projector r24 có 294,985 parameters, không phải~98K; R14/R18 có397,440.
Block30 native lens thực tế đồng ý 37/40 với final (IDs khác: [2109, 1513, 2281]). Block34 đồng ý40/40. Không được gọi quan sát agreement là quyết định đã khóa cứng về mặt nhân quả.

Không chạy A0–A7, không huấn luyện backbone, không chỉnh hyperparameters theo QA40. Đối chứng3seeds là extension exploratory riêng sau mainrun, không thay H1 threshold hậu kiểm. Raw main results/audits/job logs, per-query predictions, similarity matrices, sources và cache/checkpoint hashes lưu trong folder này.

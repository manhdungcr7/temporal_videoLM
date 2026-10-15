Đối chứng attention/MLP cuối

Ngày2026-10-15. Hoàn tất40/40 ID; 5 mỗi task/gold, duration<=45s, mọi110 cyclic rotations.

Đây là development đã tái sử dụng, không phải held-out. Mọi ô là **đúng/sai**; TA:A=Yes/B=No, TF:A=Past/B=Present/C=Future; CMTB là lựa chọn riêng mỗi câu.

Điểm cao nhất: **19/40**, cấu hình P3_last_mlp_amplify. Cấu hình đạt mục tiêu nội bộ30/40 và>=3/5 mọi ô: không có. Mục tiêu này không phải luật nhận CVPR.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|P0_native_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|P1_last_mlp_half|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|
|P2_last_attn_half|5/0|2/3|0/5|1/4|4/1|3/2|2/3|1/4|18/22|1/5/9|
|P3_last_mlp_amplify|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/5/9|
|P4_stem_mlp_half|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|P5_last_both_half|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|
|P6_last_mlp_random|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|


**Nhận xét từng ý tưởng**

|Method|Phù hợp về thiết kế|Đánh giá từ kết quả và cơ chế|
|---|---|---|
|P0_native_reference|Đối chứng sao chép|18 đúng/22 sai; ô<3/5: TF/A, TF/B, CMTB/B, CMTB/C. Native Round5 cùng inputs/model, không inference baseline mới toàn bộ.|
|P1_last_mlp_half|Diagnostic không task/window|17 đúng/23 sai; ô<3/5: TA/B, TF/A, TF/B, CMTB/B, CMTB/C. MLP x.5 tại last row block28–35; gain không chứng minh event grounding.|
|P2_last_attn_half|Diagnostic không task/window|18 đúng/22 sai; ô<3/5: TA/B, TF/A, TF/B, CMTB/B, CMTB/C. Attention output x.5 tại cùng row/blocks; không attention-mass oracle.|
|P3_last_mlp_amplify|Diagnostic hướng đối nghịch|19 đúng/21 sai; ô<3/5: TF/A, TF/B, CMTB/B, CMTB/C. MLP x1.5; kiểm tra chiều tác dụng, không chọn scale theo nhãn.|
|P4_stem_mlp_half|Diagnostic phạm vi token|18 đúng/22 sai; ô<3/5: TF/A, TF/B, CMTB/B, CMTB/C. MLP x.5 chỉ lexical stem; không trực tiếp can thiệp choices/format.|
|P5_last_both_half|Diagnostic cả hai nhánh|17 đúng/23 sai; ô<3/5: TA/B, TF/A, TF/B, CMTB/B, CMTB/C. Cùng giữ fullAV/causality, scale attention vàMLP. Không ablation một yếu tố soP1/P2.|
|P6_last_mlp_random|Negative control có giới hạn|17 đúng/23 sai; ô<3/5: TA/B, TF/A, TF/B, CMTB/B, CMTB/C. Một fixed random direction/layer, norm khớp local current MLP update; không trajectory matching hoàn hảo.|

**So ghép cặp với native cùng lượt**

|Method|Cứu câu native sai|Làm sai câu native đúng|Exact paired p|Holm p trong suite|
|---|---|---|---|---|
|P1_last_mlp_half|0|1|1.0000|1.0000|
|P2_last_attn_half|1|1|1.0000|1.0000|
|P3_last_mlp_amplify|1|0|1.0000|1.0000|
|P4_stem_mlp_half|0|0|1.0000|1.0000|
|P5_last_both_half|0|1|1.0000|1.0000|
|P6_last_mlp_random|0|1|1.0000|1.0000|

Paired exact McNemar hai phía; Holm chỉ hiệu chỉnh nội bộ suite, không xử lý mọi lần chọn phương pháp trong lịch sử. N nhỏ và tái dùng dev giới hạn suy luận gain ổn định.

Baseline khác lịch sử trên0/40 mẫu: [].

**Giới hạn và artifacts**

Không có task router, gold-dependent selection hay ghép oracle. Không suy ra CVPR acceptance/novelty hoặc broad deployment từ điểm40 này. Các bước mask/steering/soft thoughts và scalar gating cần đối chiếu prior art; kết quả local không tự xác lập novelty.

Protocol trước chạy: [protocol.md](protocol.md). Raw sample scores: results_s1/s2.jsonl. Chi tiết từng câu: per_sample.csv. Mọi per-rotation scores và archive audit được lưu riêng; kiểm tra trạng thái audit trước khi gọi raw archive hoàn tất.

Raw archives đã audit: 2/2 server; tổng rotations=110.

Các argmax có margin<1e-6 (tie chọn A theo np.argmax, không sửa hậu kiểm): {'P2_last_attn_half': [2109], 'P0_native_reference': [1898], 'P4_stem_mlp_half': [1898]}.

**Độ nhạy quyết định**

|Method|Đổi đáp án/40|Mean TV|
|---|---|---|
|P0_native_reference|0|0.0000|
|P1_last_mlp_half|1|0.0899|
|P2_last_attn_half|2|0.0808|
|P3_last_mlp_amplify|1|0.0779|
|P4_stem_mlp_half|0|0.0010|
|P5_last_both_half|1|0.0469|
|P6_last_mlp_random|1|0.0166|

TV dùng mean log-prob qua rotations rồi normalize trên choices. Không đổi argmax không đồng nghĩa mọi hidden state/path bị bỏ qua; đây không phải event-grounding metric.

Bộ40 có38 nguồn gốc. McNemar theo clip chỉ thăm dò khi có cùng nguồn lặp. Paired sign-flip theo source origin (chưa hiệu chỉnh multiple tests): {'P0_native_reference': 1.0, 'P1_last_mlp_half': 1.0, 'P2_last_attn_half': 1.0, 'P3_last_mlp_amplify': 1.0, 'P4_stem_mlp_half': 1.0, 'P5_last_both_half': 1.0, 'P6_last_mlp_random': 1.0}. Cả hai loại kiểm định đều chưa xử lý selection bias do tái dùng development.

TF prediction changes so native: {'P0_native_reference': [], 'P1_last_mlp_half': [], 'P2_last_attn_half': [], 'P3_last_mlp_amplify': [], 'P4_stem_mlp_half': [], 'P5_last_both_half': [], 'P6_last_mlp_random': []}. P3 chỉ sửa idx1898 CMTB/C từ A sang C; native sample này có exact score tie qua rotations, nên không xem +1 là gain grounding ổn định. P1/P2/P5/P6 làm sai idx2109 TA/B; P4 giữ nguyên mọi argmax.

Suy luận về attention/MLP muộn phải đọc cùng layer lens và control random. Tăng accuracy hoặc đổi TF distribution chưa chứng minh restored grounding; null result chưa chứng minh model không liên kết thời gian. Không đề xuất scalar gating đơn lẻ như novelty CVPR.

Cache audit: {'rotations': 110, 'max_full_condition_parity': 0.015625, 'max_native_prefix_error': 0.03125, 'max_native_reference_error': 0.0, 'full_condition_calls': 6}. Fallback full giữ ngưỡng.03; không nới để ép chạy cached. Thời gian condition có thể gồm parity reference, không dùng làm latency deployment đầy đủ.

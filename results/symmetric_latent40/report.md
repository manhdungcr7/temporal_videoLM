# Phản hồi latent đúng tầng và tương tác hai nguồn:40/40

Ngày2026-10-15. Cùng40 video cố định<=45s, năm mỗi task/nhãn, tất cả110 cyclic rotations. Đây là development đã dùng nhiều lần; không dùng nhãn trong suy luận.

**Bảng ghi đúng/sai.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động của mỗi câu.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|S0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|S1_visual_same_layer|0/5|5/0|2/3|0/5|4/1|3/2|2/3|3/2|19/21|4/1/10|
|S2_joint_same_layer|1/4|4/1|3/2|0/5|4/1|4/1|2/3|4/1|22/18|5/0/10|
|S3_orthogonal_joint|2/3|3/2|1/4|1/4|4/1|3/2|2/3|2/3|18/22|1/3/11|
|S4_two_layer_orthogonal_joint|3/2|3/2|1/4|1/4|4/1|3/2|2/3|2/3|19/21|1/3/11|
|S5_question_stem_joint|3/2|3/2|0/5|1/4|4/1|3/2|2/3|0/5|16/24|1/5/9|
|S6_question_stem_argon_delta|0/5|5/0|0/5|2/3|3/2|3/2|3/2|0/5|16/24|0/7/8|

Cấu hình có số đúng cao nhất trong lượt này: **S2_joint_same_layer: 22/40**.

**Không cấu hình nào đạt30/40 và ít nhất3/5 trong mọi ô.**

## So ghép cặp với native cùng lượt

|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p, chưa hiệu chỉnh|
|---|---:|---:|---:|---:|
|S0_native|0|0|+0|1.0000|
|S1_visual_same_layer|7|6|+1|1.0000|
|S2_joint_same_layer|9|5|+4|0.4240|
|S3_orthogonal_joint|3|3|+0|1.0000|
|S4_two_layer_orthogonal_joint|3|2|+1|1.0000|
|S5_question_stem_joint|0|2|-2|0.5000|
|S6_question_stem_argon_delta|4|6|-2|0.7539|

## Nhận xét và phạm vi

S1 là đối chứng visual-asymmetric ở đúng block22. S2 dùng tương tác đại số hai nguồn; S3 chiếu bỏ span hai chênh lệch unimodal; S4 dùng hai block18/22. S5 giới hạn S3 vào stem. S6 giữ delta/cross-layer rescaling của Argon C3 nhưng giới hạn vào stem.

S1–S5 không khuếch đại delta nhỏ; S6 tái lập rescaling của Argon để tách tác dụng của vị trí token. Hiệu hidden-state và phép chiếu không được coi là phép phân rã nhân quả đã được xác nhận.

Đối chứng này kiểm tra ba giả thuyết khác nhau: truyền sai tầng, nhiễu do tăng norm và can thiệp trực tiếp vào choices/format. Hiệu quả phải đánh giá trên cả8 ô, không chỉ tổng hoặc việc hết collapse.

V1 dừng ở mẫu thứ3 do prefix replay.03125 vượt ngưỡng.03. Hai record v1 được loại khỏi bảng chính và lưu ở implementation_v1. V2 giữ ngưỡng, dùng full execution khi native prefix check không qua; không đổi công thức/scale hoặc mẫu.

S1/S6 vẫn ưu tiên visual; S2–S5 đối xứng nguồn ở phép xây delta. Không cấu hình nào có clock từ nhãn, speech window, task router hoặc audio handoff. Phạm vi câu hỏi rộng vẫn cần kiểm chứng ngoài40 development.

Không thể suy ra phương pháp đủ mới hoặc đủ mạnh cho CVPR chỉ từ kết quả này. Các primitive counterfactual/activation steering/soft embedding đã có tiền lệ; cần đối chứng mạnh và kiểm chứng tổng quát.

## Kiểm tra triển khai

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`. Native replay max=0.00000000.

Native prefix fallback ở1/110 rotations. Số condition fallback ở các parity checks đầu worker=0.

Thời gian trung bình của toàn suite và mọi rotations: 27.1s/mẫu; không phải latency của một method.

Raw sample scores: results_s1/s2.jsonl. Summary/table/per_sample.csv và protocol.md được giữ riêng từng họ. Mọi lỗi kỹ thuật và bản triển khai cũ được giữ, không quietly skip hoặc thay sample.

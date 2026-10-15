# Tiếp nối ChronoDeSmear — đăng ký trước khi chạy, 2026-10-15

Cùng manifest40<=45s, mọi110 rotations, native cùng phiên. Chạy sau toàn bộ
Round5 trên mỗi server; không chọn mẫu/method theo gold. Mã đề xuất Round5 giữ
nguyên, lượt này thư mục riêng. Không có task router/clock từ nhãn/speech-window.

## Giả thuyết

1. Delta nhiều nhánh của Round5 có thể làm hại; H_AV-H_staticA thuần có thể tốt hơn.
2. Center options, de-smear media và thought là các yếu tố riêng, không mặc định
   kết hợp sẽ tốt hơn. N2->N3->N4->N5 giữ nguồn delta/gain cố định, thêm từng yếu tố.
3. Token low-innovation chưa được chứng minh là background. So với chọn random
   cùng cardinality từng bin (N6), không dùng event oracle.
4. Chỉ phản thực tế visual có thể bỏ qua audio events. N7 xây delta hai nguồn:
   .5*((H_AV-H_staticV,A)+(H_AV-H_V,staticA)), cùng5bins/45% cho cả hai nguồn.
   Đây là contrast đại số, không claim phân rã nhân quả hoặc giữ mọi information.
5. Phản hồi khác tầng/rescale có thể gây shift. N8 là candidate độc lập: cùng
   delta đối xứng nhưng giới hạn norm15%, không khuếch đại delta nhỏ, đưa vào
   đúng block22 và chỉ lexical stem. N7->N8 đổi nhiều yếu tố, không ablation đơn.

## Cấu hình cố định

N0 native rerun. N1 sao chép C1 de-smear cùng rotation từ Round5 (reference,
không gọi một inference độc lập mới). N2 pure delta, không center, không de-smear,
gain.24/.30/.30/.24 tại10/12/14/16. N3 nhưN2+option-center .55. N4 nhưN3+de-smear
V=.26/A=.18 tại8/10/12/14/16. N5 nhưN4+thought original .18 ở0/1/2.
N6 nhưN3 nhưng static mask random seed2027, đúng số token từng bin. N7 source-
symmetric centered delta, không de-smear/thought. N8 source-symmetric same-layer,
normcap15%, lexical-stem-only. N9 fixed layer ensemble: trung bình log-prob
các block22/26/30/34/35. N9 là diagnostic/readout baseline đã có tiền lệ,
không claim novelty layer ensemble hoặc chọn layer theo task/gold.

## Đo layer và representation

Native logit lens tại block4/8/12/16/18/22/26/30/34/35: final RMSNorm+lm_head,
chấm cùng ABC rotations. Mỗi fixed-layer có per-cell và mẫu đúng ở giữa/sai ở
cuối. Không dùng gold để chọn layer trong N9, không báo oracle thành candidate.
Logit lens chưa calibrated; intermediate accuracy/khác layer không tự chứng
minh causal late overwrite. Mọi causal claim cần activation patching hoặc
intervention độc lập nữa. Block35 phải khớp native cuối trong tolerance.

Đo mean-vector norm ratio và cosine5bin của V/A trên native4/8/12/16/18/22/
26/30/34/35. Các số đo anisotropy không phải tỷ lệ background90%, không event
grounding accuracy. Không có timestamp annotation độc lập trong lượt này.

## Literature đã đọc

[Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1), mục2.2:
AV được interleave theo chunk2s với TMRoPE. Thứ tự causal và physical timestamp
không phải một thứ; không suy ra mọi token nhìn mọi sự kiện vật lý trước nó.

[Causality Matters](https://ojs.aaai.org/index.php/AAAI/article/view/37856),
abstract: temporal cues có thể được tích hợp qua inter-frame attention rồi query.
Suy luận thiết kế của chúng tôi: trừ lịch sử có thể xóa context hữu ích, nên
không mặc định causal accumulation là lỗi. Kết quả paper không tự áp dụng cho
Qwen3B/jointAV và không chứng minh cơ chế của Round5.

## Tái lập và báo cáo

Giữ source/proposal/runtime SHA. Baseline so Round5 từng ID, raw rotations archive
local. Đúng/sai đủ8ô, TF confusion, rescue/regression và kiểm định ghép cặp.
Các yếu tố này có thể dùng rộng về cấu trúc nhưng chỉ40 dev chưa chứng minh
generalization/free-form hoặc novelty đủ cho CVPR. Nếu thất bại vẫn báo đủ;
không dùng best layer/choice đúng làm selector hoặc sửa đáp án bằng heuristic.

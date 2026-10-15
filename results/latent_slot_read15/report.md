# Đối chứng thời điểm đọc memory: block15 so với23:40/40

Ngày2026-10-15. Cùng40 video cố định<=45s, năm mỗi task/nhãn, tất cả110 cyclic rotations. Đây là development đã dùng nhiều lần; không dùng nhãn trong suy luận.

**Bảng ghi đúng/sai.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động của mỗi câu.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|E0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|E1_unoptimized_read15|4/1|3/2|0/5|2/3|2/3|3/2|2/3|0/5|16/24|1/7/7|
|E2_optimized_read15|4/1|3/2|0/5|1/4|4/1|2/3|2/3|1/4|17/23|2/3/10|
|E3_shuffled_read15|5/0|1/4|1/4|1/4|4/1|2/3|2/3|1/4|17/23|2/3/10|
|E4_optimized_read23_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|E5_unoptimized_read23_reference|4/1|3/2|0/5|2/3|4/1|3/2|2/3|1/4|19/21|0/6/9|

Cấu hình có số đúng cao nhất trong lượt này: **E5_unoptimized_read23_reference: 19/40**.

**Không cấu hình nào đạt30/40 và ít nhất3/5 trong mọi ô.**

## So ghép cặp với native cùng lượt

|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p, chưa hiệu chỉnh|
|---|---:|---:|---:|---:|
|E0_native|0|0|+0|1.0000|
|E1_unoptimized_read15|2|4|-2|0.6875|
|E2_optimized_read15|1|2|-1|1.0000|
|E3_shuffled_read15|3|4|-1|1.0000|
|E4_optimized_read23_reference|0|0|+0|1.0000|
|E5_unoptimized_read23_reference|2|1|+1|1.0000|

## Nhận xét và phạm vi

E1/E2/E3 đọc cùng memory chưa fit/đã fit/fit target bị đảo từ block15. E4/E5 là các điểm read23 được sao chép từ lượt gốc, không huấn luyện hoặc chạy lại candidate.

Các latent được giữ nguyên từng số float. Chỉ đổi tầng bắt đầu cho query đọc memory. Một rotation mỗi worker dựng lại read23 để kiểm tra parity trước khi so sánh.

Mở quyền đọc sớm có thể giúp sử dụng evidence, cũng có thể làm nhiễu readout. Các slot chỉ có loss ở block22, nên không mặc định mọi feature ở tầng sớm đã được align tốt.

Phương pháp không chọn theo task/nhãn và giữ toàn bộ AV. Tuy vậy kết quả40 development không chứng minh tổng quát, triển khai free-form hoặc chấp nhận CVPR.

Không thể suy ra phương pháp đủ mới hoặc đủ mạnh cho CVPR chỉ từ kết quả này. Các primitive counterfactual/activation steering/soft embedding đã có tiền lệ; cần đối chứng mạnh và kiểm chứng tổng quát.

## Kiểm tra triển khai

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.

Thời gian trung bình của toàn suite và mọi rotations: 7.8s/mẫu; không phải latency của một method.

Raw sample scores: results_s1/s2.jsonl. Summary/table/per_sample.csv và protocol.md được giữ riêng từng họ. Mọi lỗi kỹ thuật và bản triển khai cũ được giữ, không quietly skip hoặc thay sample.

Read23 reconstruction parity max=0.00000000. Không refit parameters.

E2 so với E4: cứu1 câu và làm sai2 câu.

## Độ nhạy quyết định với memory

Từ scores có sẵn, không chạy lại inference. TV đo trên phân phối lựa chọn: trung bình log-prob qua rotations rồi chuẩn hóa. Đây không phải sensor attention hoặc độ chính xác grounding sự kiện.

Đối chiếu với `E4_optimized_read23_reference`.

|Method|Đổi dự đoán/40|Mean TV|Median TV|
|---|---:|---:|---:|
|E1_unoptimized_read15|9|0.1365|0.1205|
|E2_optimized_read15|5|0.0915|0.0616|
|E3_shuffled_read15|9|0.1322|0.1298|

# Đối chứng lấy frame theo chỉ số và theo presentation timestamp:40/40

Ngày2026-10-15. Cùng40 video cố định<=45s, năm mỗi task/nhãn, tất cả110 cyclic rotations. Đây là development đã dùng nhiều lần; không dùng nhãn trong suy luận.

**Bảng ghi đúng/sai.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động của mỗi câu.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|P0_original_native_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|P1_pyav_frame_index|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/4/10|
|P2_pyav_presentation_time|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/4/10|

Cấu hình có số đúng cao nhất trong lượt này: **P1_pyav_frame_index: 19/40**.

**Không cấu hình nào đạt30/40 và ít nhất3/5 trong mọi ô.**

## So ghép cặp với native cùng lượt

|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p, chưa hiệu chỉnh|
|---|---:|---:|---:|---:|
|P0_original_native_reference|0|0|+0|1.0000|
|P1_pyav_frame_index|1|0|+1|1.0000|
|P2_pyav_presentation_time|0|0|+0|1.0000|

## Nhận xét và phạm vi

P0 là điểm native gốc được sao chép, không thay đầu vào. P1 dùng PyAV và chỉ số frame cách đều. P2 dùng cùng decoded pictures nhưng chọn hình đang hiển thị tại mỗi thời điểm grid native. Cả hai giữ nguyên audio, FPS, số frame và geometry của original utility.

P1 so với P0 thay decoder và số frame gốc khả dụng; không gọi là decoder-only ablation. P2 so với P1 mới kiểm tra riêng quy tắc chọn chỉ số/thời gian trên cùng decoded pictures.

PTS nhân time_base, trừ container origin. Các bước PTS không đơn điệu được stable sort cùng index ảnh, không đoán từ average FPS hoặc API Decord. Black trước frame đầu và hold sau frame cuối được ghi lại.

Đây là kiểm tra đầu vào có căn cứ từ pixel audit, không phải novel temporal reasoning, event oracle hoặc bảo đảm cải thiện. Dù có tăng QA cũng chưa đủ cho đóng góp CVPR; nếu không tăng, chỉ bác lợi ích QA của quy tắc này trên development40.

Không thể suy ra phương pháp đủ mới hoặc đủ mạnh cho CVPR chỉ từ kết quả này. Các primitive counterfactual/activation steering/soft embedding đã có tiền lệ; cần đối chứng mạnh và kiểm chứng tổng quát.

## Kiểm tra triển khai

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.

Thời gian trung bình của toàn suite và mọi rotations: 34.8s/mẫu; không phải latency của một method.

Raw sample scores: results_s1/s2.jsonl. Summary/table/per_sample.csv và protocol.md được giữ riêng từng họ. Mọi lỗi kỹ thuật và bản triển khai cũ được giữ, không quietly skip hoặc thay sample.

Native replay max=0.00000000.

Tổng native frame slots=2070, cùng chỉ số chọn giữa P1/P2=100.

Black trước ảnh đầu=0 slots; hold sau PTS cuối=135 slots; clips có bước PTS không đơn điệu=39/40.

## Độ nhạy quyết định

Từ scores có sẵn, không chạy lại inference. TV đo trên phân phối lựa chọn: trung bình log-prob qua rotations rồi chuẩn hóa. Đây không phải sensor attention hoặc độ chính xác grounding sự kiện.

Đối chiếu với `P0_original_native_reference`.

|Method|Đổi dự đoán/40|Mean TV|Median TV|
|---|---:|---:|---:|
|P1_pyav_frame_index|2|0.0167|0.0095|
|P2_pyav_presentation_time|1|0.0267|0.0161|

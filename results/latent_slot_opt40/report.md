# Tối ưu latent slots theo nguồn và thời gian:40/40

Ngày2026-10-15. Cùng40 video cố định<=45s, năm mỗi task/nhãn, tất cả110 cyclic rotations. Đây là development đã dùng nhiều lần; không dùng nhãn trong suy luận.

**Bảng ghi đúng/sai.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động của mỗi câu.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|L0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|L1_unoptimized_memory|4/1|3/2|0/5|2/3|4/1|3/2|2/3|1/4|19/21|0/6/9|
|L2_optimized_memory|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|L3_time_shuffled_target|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|0/4/11|

Cấu hình có số đúng cao nhất trong lượt này: **L1_unoptimized_memory: 19/40**.

**Không cấu hình nào đạt30/40 và ít nhất3/5 trong mọi ô.**

## So ghép cặp với native cùng lượt

|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p, chưa hiệu chỉnh|
|---|---:|---:|---:|---:|
|L0_native|0|0|+0|1.0000|
|L1_unoptimized_memory|2|1|+1|1.0000|
|L2_optimized_memory|0|0|+0|1.0000|
|L3_time_shuffled_target|0|1|-1|1.0000|

## Nhận xét và phạm vi

L1 thêm memory chưa tối ưu; L2 tối ưu input slots với source-only teachers; L3 cùng số bước và constraints nhưng đảo target theo thời gian trong từng nguồn. Backbone đóng băng và nhãn benchmark không vào loss.

Tái dựng đúng target nội bộ không đồng nghĩa định vị đúng sự kiện thật. Teacher là hidden states của chính model và dùng clock native, không phải annotation sự kiện độc lập.

L2 cần hơn L1 để chứng minh lợi ích optimization, và hơn L3 để chứng minh sự sắp xếp theo thời gian quan trọng. Nếu feature reconstruction tăng nhưng QA không tăng, chưa sửa được khoảng cách biểu diễn và quyết định.

Memory chỉ được đọc từ block23, sau layer22 được fit. Lượt read15 giữ nguyên các slot để kiểm tra riêng nút thắt thời điểm đọc.

Không speech-window, clock từ đáp án, loại câu hỏi hay thứ tự modality cố định. Đây là một thích nghi của latent optimization đã có tiền lệ, chưa phải bằng chứng novelty đủ mạnh cho CVPR.

Không thể suy ra phương pháp đủ mới hoặc đủ mạnh cho CVPR chỉ từ kết quả này. Các primitive counterfactual/activation steering/soft embedding đã có tiền lệ; cần đối chứng mạnh và kiểm chứng tổng quát.

## Kiểm tra triển khai

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.

Thời gian trung bình của toàn suite và mọi rotations: 28.8s/mẫu; không phải latency của một method.

Raw sample scores: results_s1/s2.jsonl. Summary/table/per_sample.csv và protocol.md được giữ riêng từng họ. Mọi lỗi kỹ thuật và bản triển khai cũ được giữ, không quietly skip hoặc thay sample.

Native replay max=0.03125000.

Zero-memory identity max=0.03125000.

## Tái dựng slot theo teacher nội bộ

|Memory|Video cosine|Video đúng slot|Audio cosine|Audio đúng slot|
|---|---:|---:|---:|---:|
|L1_unoptimized_memory|0.7330|113/527|0.7371|257/514|
|L2_optimized_memory|0.8585|458/527|0.9162|440/514|
|L3_time_shuffled_target|0.7919|47/527|0.7831|51/514|

Đúng slot nghĩa là nearest target của cùng nguồn trùng index thời gian nội bộ. Có thể giải mã clock hoặc self-match; không coi số này là độ chính xác event grounding thật.

## Độ nhạy quyết định với memory

Từ scores có sẵn, không chạy lại inference. TV đo trên phân phối lựa chọn: trung bình log-prob qua rotations rồi chuẩn hóa. Đây không phải sensor attention hoặc độ chính xác grounding sự kiện.

Đối chiếu với `L0_native`.

|Method|Đổi dự đoán/40|Mean TV|Median TV|
|---|---:|---:|---:|
|L1_unoptimized_memory|3|0.0822|0.0773|
|L2_optimized_memory|0|0.0463|0.0304|
|L3_time_shuffled_target|3|0.0540|0.0379|

L2 giữ nguyên mọi40 argmax native dù xác suất có thay đổi. Điều này không chứng minh model bỏ qua memory ở mọi layer; chỉ chứng minh slot fit này chưa sửa quyết định cuối.

# Protocol (viết TRƯỚC khi chạy): ChronoDeSmear Round 5 trên SVHalluc semantic

Ngày 2026-10-15. Mục tiêu: kiểm tra, không chỉnh sửa gì, các cấu hình Round 5 của bạn (R5_C0..C6, mã ở `server_jobs/chronodesmear40.py`) trên các task **semantic** của SVHalluc.
Chưa có kết quả nào khi viết file này.

## Dữ liệu
- Task: GSA (lời nói có mô tả hình không), FGSA (chi tiết hình có xuất hiện không), CSMB (sự kiện có hiện ra không). Cả ba đều **Yes/No** (A = Yes, B = No).
- 6 ô (task, đáp án đúng), **10 mẫu mỗi ô = 60 mẫu**, clip ≤ 40 s, seed `31337` (chưa dùng ở đâu khác). Chọn chỉ dựa vào seed và độ dài clip. `server_jobs/semantic_manifest_job.py`.
- Chia đều hai server theo từng ô. Mọi phép xoay đáp án (2 phép cho Yes/No).
- Đây là bộ **dev** của semantic. Nếu một cấu hình đạt ngưỡng, xác nhận trên bộ mới (seed khác, 60 mẫu mới) **trước khi** phát biểu gì.

## Cấu hình
- Tham chiếu: `R5_C0_baseline` (native cùng phiên). Ứng viên: `R5_C1`..`R5_C5`. Đối chứng ngẫu nhiên cùng norm: `R5_C6` (3 seed).
- Ứng viên chính, nêu trước: `R5_C2_stratified_dyn_cd` (tốt nhất ở temporal) và `R5_C5_chronodesmear_full` (phương pháp đầy đủ).
- Không đổi hệ số, layer, ngưỡng sau khi xem kết quả. Không chọn tập con ô.

## Điều kiện để gọi một cấu hình là "tốt trên semantic" (đủ **tất cả**)
1. Độ chính xác cân bằng trung bình (trung bình của GSA, FGSA, CSMB; mỗi task là trung bình hai ô) **cao hơn native ≥ 5 điểm**.
2. McNemar chính xác ghép cặp so với native, **Holm trên 5 ứng viên: p < 0,05**.
3. **Hơn đối chứng ngẫu nhiên R5_C6 ≥ 5 điểm** (để loại hiệu ứng "đổi đáp án bất kỳ").
4. **Không task nào giảm quá 3 điểm** độ chính xác cân bằng so với native.
5. **Không dồn đáp án:** với mỗi task, tỉ lệ dự đoán dồn vào một chữ ≤ 75%. (Đáp án đúng cân bằng 50/50 nên dồn đáp án cho độ chính xác cân bằng 50%.)
6. Phải báo đủ: đúng/sai từng ô, phân phối A/B từng task, cứu/làm sai, p và Holm, khoảng Wilson 95%.

Công suất: với 60 mẫu nhị phân, p < 0,05 ghép cặp cần khoảng ≥ 9 câu cứu so với ≤ 2 câu làm sai (hoặc tương đương). Kết quả không đạt điều kiện 2 là **không kết luận**, không phải "gần đạt".

## Sau khi có kết quả
- Nếu không cấu hình nào đạt: ghi nhận là kết quả âm tính (cùng chiều với temporal); không tinh chỉnh trên bộ này.
- Nếu một cấu hình đạt 1–6: xác nhận trên bộ mới 60 mẫu (seed khác), thêm temporal no-harm (đã có: R5 temporal C2 21/40 so với native 18/40, p=0,63, không có gain ổn định) và so với prior art (latent steering, contrastive decoding) **trước** khi nói tới paper.
- Câu chuyện paper phải dựa trên task nó thực sự cải thiện. Các cấu hình này được thiết kế để sửa lỗi **thời gian**; nếu chỉ tốt ở semantic thì đó là bài về ảo giác ngữ nghĩa, không phải bài temporal.

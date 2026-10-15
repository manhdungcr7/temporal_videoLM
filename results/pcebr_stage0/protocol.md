# PC-EBR Stage-0 — protocol trước khi chạy

Ngày 2026-10-15. Ưu tiên chạy nguyên thuật toán attachment, không chạy A0–A7 hoặc research40 trong lượt kiểm chứng này.

1. Lưu nguyên hai attachment. Dùng đúng bốn cache pair local đã archive từ R14/R18; không tạo cặp mới hoặc thay split. Upload lên hai runtime mới.
2. Chạy cùng script nguyên gốc trên cả hai server để kiểm tra tái lập; wrapper chỉ lưu RESULT, metadata/cache hashes và các thông số thực thi. Không nới ngưỡng hoặc chọn seed theo kết quả.
3. Báo cáo mọi count/denominator thực tế, split theo recording/source, trùng đoạn/content, số candidate nội video. Không gọi pseudo-description/chunk self-match là event grounding độc lập.
4. Gate H1 yêu cầu tăng ít nhất 15 điểm phần trăm theo đề xuất và vượt shuffle; global và within-rec là metrics khác nhau, không so within-rec với historical global7/50 hoặc1/50.
5. M2 là unit check đại số trên intervals do code cung cấp, không phải counterfactual QA trên mô hình. H3/M3 chỉ có docstring, chưa có implementation; không ghi là đã kiểm chứng mid-vs-late hoặc thông qua toàn Stage-0.
6. Sau exact run có thể chạy đối chứng audit riêng: metadata/label dependency, true factorial cùng kiến trúc, multiple fixed seeds, faithful R14 baseline. Mọi extension phải tách tên/kết quả khỏi code gốc và không thay tiêu chí hậu kiểm.

Trọng số backbone không được huấn luyện; projector M1 được huấn luyện đúng như attachment. Cả hai server có cùng bốn cache, hai lần chạy là replicas, không phải hai tập dữ liệu độc lập.

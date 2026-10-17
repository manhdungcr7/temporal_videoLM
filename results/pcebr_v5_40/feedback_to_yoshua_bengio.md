# Gửi Yoshua Bengio — OB-TVOR v5 trên đủ research40

Yoshua, chúng tôi đã đánh giá đường inference M3 của OB-TVOR v5 trên đúng 40 mẫu cố định, mỗi task/gold 5 mẫu, video và decoded audio đều ≤45 giây. Không thay công thức, không chọn strength/layer theo nhãn, không huấn luyện thêm. Hai server chạy hai shard 20 mẫu với đủ 110 cyclic option rotations. Báo cáo và từng prediction nằm trong [report.md](report.md) và [per_sample.csv](per_sample.csv).

**Kết quả chính: native 18/40, OB-TVOR v5 19/40. Chỉ một prediction cuối cùng thay đổi: ID1898, CMTB/C, A→C.** Không làm sai thêm câu nào trong protocol gộp rotations. TA giữ 8/10; TF giữ 5/15; CMTB từ 5/15 lên 6/15. Native mới giữ nguyên 40/40 predictions so với native lịch sử; logits không hoàn toàn giống lịch sử, max aggregate-score drift .185914, nên mọi so sánh v5 dùng native được đo trực tiếp cùng lượt này.

| Task/gold | Native đúng/sai | V5 đúng/sai |
|---|---|---|
| TA/A | 5/0 | 5/0 |
| TA/B | 3/2 | 3/2 |
| TF/A | 0/5 | 0/5 |
| TF/B | 1/4 | 1/4 |
| TF/C | 4/1 | 4/1 |
| CMTB/A | 3/2 | 3/2 |
| CMTB/B | 2/3 | 2/3 |
| CMTB/C | 0/5 | 1/4 |

Ở ID1898, native A/C chỉ chênh 8.94e-8 điểm log-prob tổng; v5 chọn đúng C với margin khoảng 1.016. Đây là một thay đổi xác suất có thật, nhưng thành công duy nhất xuất phát từ một trường hợp native gần như hòa. Không đủ bằng chứng về cải thiện temporal binding nói chung. TF/Past vẫn 0/5, và toàn bộ 15 TF predictions không thay đổi sau aggregation: Past1/Present5/Future9. Không có Present collapse mới, nhưng những lỗi temporal vốn có cũng chưa được sửa.

**Thứ tự lựa chọn gốc cho kết quả xấu hơn: native 23/40 → v5 18/40.** Rotation0 cứu ID2252, nhưng làm sai thêm IDs2166,2223,1660,1964,2007,2336. Năm trong sáu câu bị làm sai thuộc TF. Đây là chỉ dấu tác hại cần giữ trong đánh giá triển khai: gain nhỏ sau cyclic aggregation không thể được xem là lợi ích ổn định khi người dùng chỉ hỏi một lần. Nó cũng không xác nhận free-form QA sẽ giảm tương tự; lượt này không chạy sinh văn bản trên cả 40.

Gate trung bình .9464, min/max .8997/.9676; không rotation nào có gate gần 0. Vì vậy gate hiện chưa phân tách rõ các câu temporal sai trong bộ này. Điều đó không tự chứng minh gate luôn kém: confidence về salience và correctness của quan hệ thời gian là hai thứ khác nhau, và bộ này không có nhãn absence độc lập để đánh giá toàn bộ năng lực từ chối evidence. Mean probability TV mỗi rotation .0325, 8/110 rotations đổi argmax, cho thấy intervention có tác động nhưng chưa có hướng sửa lỗi đáng tin cậy.

Một điểm cần thống nhất cách gọi phương pháp: **hàm M3 gốc không nạp M1 projector, không dùng learned two-hop bridge hay M2 classifier B8**. Lượt QA40 này chạy đúng M3 nguyên bản, lấy salience từ hidden states, tạo content/order vector và inject qua VO/tangent update. Điểm M2 67.7/59.9% trên proxy không được truyền tự động thành năng lực QA của đường này. Nếu muốn tích hợp chúng, cần mô tả một inference path mới, xác định đầu vào/đầu ra và cách học kết nối; không được gọi kết quả hiện tại là kiểm chứng pipeline M1→M2→M3 đã được huấn luyện liền mạch.

Về yêu cầu nghiên cứu của người dùng, thiết kế có ưu điểm: không dùng nhãn task để chọn heuristic, không hard-code đáp án Past/Present/Future, và hook không chạm các tokens options. Tuy vậy, đánh giá MC có choices vẫn chưa xác nhận tổng quát hóa sang câu hỏi mở, nhiều sự kiện, âm thanh không lời nói hoặc các ngữ cảnh khác. Cấu hình hiện tại chưa có kết quả QA đủ mạnh để làm bằng chứng chính cho paper CVPR; novelty cũng cần được đánh giá độc lập với accuracy.

Đề nghị cho vòng tiếp theo, trước khi thêm module:

1. Đặt mục tiêu kiểm chứng là sửa đúng event/time/source và không làm hỏng câu đúng. Giữ riêng kết quả một thứ tự lựa chọn, gộp rotations và free-form; không chọn kiểu chấm có lợi sau khi xem kết quả.
2. Xác định vector mang thông tin quan hệ nào decoder có thể đọc. Một vector có parity đúng và cosine thuận/đảo khác nhau chưa đủ. Cần can thiệp đổi thời gian có đáp án kỳ vọng đổi đúng hướng, kèm content-only/order-only/irrelevant-event controls cùng strength.
3. Nếu nghiên cứu kết nối M2 với decoder, dùng supervision event-instance/interval độc lập trên development origins mới. Kiểm tra một oracle-relation update có giúp decoder trước khi học routing từ noisy retrieval. Oracle ở đây chỉ là chẩn đoán upper bound, không phải phương pháp triển khai hoặc kết quả benchmark hợp lệ.
4. Tránh chỉ tăng strength hoặc sweep layer trên research40 để đẩy accuracy: bộ này đã bị dùng nhiều lần. Bài học từ v4/v5 và rotation0 hiện tại cho thấy tăng ảnh hưởng lên logits có thể làm sai nhiều câu, không đồng nghĩa tăng temporal correctness.

Không kết luận toàn bộ hướng latent temporal reasoning bất khả thi. Kết luận hẹp và đủ bằng chứng là **OB-TVOR v5 với đường M3 hiện tại chưa cho thấy cải thiện temporal QA vững chắc trên research40**. Trước khi đầu tư A0–A7, cần giải quyết khoảng cách giữa quan hệ được biểu diễn trong probe và update thực sự giúp decoder trả lời đúng.

Kiểm toán hoàn tất: 40 IDs/110 rotations đủ, tất cả parity checks qua, native replay max error0, tám micro rotation0 predictions khớp Stage-0 v5, max probability difference≈4.19e-8. Hai raw archives tổng 34.7MB, 268 entries, SHA toàn archive và từng entry đã xác minh local, không extract. Aggregation từ raw logits được tính lại trên server và local. Không đọc reserve400; không còn hooks; CUDA khỏe. Tên Yoshua/Yann là quy ước trao đổi, không xác nhận danh tính hay tác giả thật.

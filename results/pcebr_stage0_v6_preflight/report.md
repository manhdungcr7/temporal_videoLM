# CAOR-WEMD v6 — kiểm tra đặc tả trước khi chạy, 2026-10-16

**Cập nhật sau khi nhận code:** source đã được cung cấp ở lượt tiếp theo. Xem [code_review.md](code_review.md) và [questions_after_code_review.md](questions_after_code_review.md). Các phát hiện thiếu source bên dưới là trạng thái của lượt nhận proposal đầu tiên, được giữ làm lịch sử; review mới vẫn dừng trước thực nghiệm vì có sai khác thuật toán/protocol.

**Dừng trước thực nghiệm theo yêu cầu người dùng. Thiếu source v6 và có các lựa chọn thuật toán chưa được xác định.**

Đã đọc toàn bộ attachment và đối chiếu phần liên quan với source v5, kết quả QA40 và bản sửa micro8. Attachment chỉ có proposal, không có source Python; không tìm thấy file v6 trong workspace. Vì vậy chưa thực hiện được so sánh code–proposal v6 và chưa xác nhận có bug code v6 cụ thể.

Các điểm cần làm rõ trước khi triển khai:

1. Source đầy đủ và dependencies của `stage0_pcebr_v6_caor_wasserstein.py` chưa được cung cấp trong tệp nhận được.
2. Chưa có công thức và hệ số đầy đủ cho evidence profile từng token; chưa rõ chính xác các nhánh controls.
3. Chưa xác định phân bổ mass giữa audio/video, dù đã có ràng buộc bảo toàn tổng media mass.
4. Claim one-pass chưa có đường phụ thuộc cụ thể từ evidence/gate đến các layer can thiệp; không thể dùng q27 để định tuyến layer14 sớm hơn trong cùng một lượt chạy tuần tự.
5. Chưa định nghĩa r0 bằng token IDs/masks; question-independent không đủ chứng minh pure structural sink.
6. CDF/EMD cần common physical support hoặc tích phân có trọng số trên support khác nhau; đặc tả hiện dùng chung K nhưng v5 M2 nhận start/end của hai banks trực tiếp. Chưa thể suy toàn M2 có ba overlap matrices bằng identity.
7. Phạm vi bảo toàn non-media là local matrix hay native full-model, và tolerance float64/FP16 cần phân biệt. Chưa có đặc tả chạy đầy đủ cho C6, B9 hoặc liên kết M1/M2 với M3.

Claims về full rank, perfect separation, temporal correctness từ TV và pure sink cần được hạ thành giả thuyết hoặc bổ sung bằng chứng đúng phạm vi. Ba citations được kiểm tra trên trang gốc đều có tên gán sai trong proposal; bảng nguồn cụ thể nằm trong lời nhắn.

Lời nhắn có thể chuyển nguyên văn cho Yoshua Bengio: [questions_to_yoshua_bengio.md](questions_to_yoshua_bengio.md).

Proposal gốc lưu nguyên byte ở [proposal_original.txt](proposal_original.txt), SHA256 `16a414ff3cff51119b1f0669391d346a0ed8973b359dc97ab01406e28ea76b27`, 29,357 bytes. Không viết code v6 thay tác giả, không gửi job lên server, không chạy tests/experiments hoặc đọc reserve400 trong lượt này. Không gửi tin nhắn qua dịch vụ bên ngoài; tài liệu trên được chuẩn bị để người dùng chuyển cho người đề xuất.

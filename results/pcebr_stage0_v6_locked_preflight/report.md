# CAOR-WEMD v6 locked — kiểm tra proposal/source, 2026-10-16

**Dừng trước thực nghiệm theo yêu cầu người dùng: còn hai điểm code/spec cần Yoshua Bengio chốt.**

Đã đọc proposal cập nhật và các nhánh source liên quan đến những điểm review trước. Các sửa chính đã được hiện thực: shared C6 scorer, C8b innovation-only/C8c both-shuffled, loss innovation1.0, bridge relation filter, common physical support, CAOR query-row differential, norm matching cấp token và hai-pass accounting. Không tiếp tục hỏi lại những điểm đã được code làm rõ.

Hai vấn đề còn lại:

1. Nhánh ghi “exact legacy v5” lấy đúng post-block activations và legacy Phi, nhưng gọi salience v6 với normalization, de-sinking, weights và gate khác v5. Chưa phải replay chính xác baseline cũ.
2. Proposal ghi delta attention FP32, code cast dp về dtype V trước matmul; với V FP16, đây là mixed-precision differential. Cần chốt chọn code hiện tại hay FP32 như mô tả.

Góp ý riêng về giới hạn: midpoint quadrature không luôn là exact Wasserstein khi CDF difference đổi dấu trong segment; zero-noop có thể bypass nếu gate0; full-profile reversal hiện chỉ đảo slot terms, giữ token innovation. Các điểm này được phân biệt với hai sai khác chính, không được báo như kết quả test đã chạy.

Nguồn được lưu nguyên byte: [proposal_original.txt](proposal_original.txt) và [stage0_pcebr_v6_locked_original.py](stage0_pcebr_v6_locked_original.py). Source SHA256 Windows `779e8281087d92a3c4d6085c645fc11431cebe73e1f47cd4f889d63fab686bc2`; chuẩn hóa LF khớp SHA tác giả công bố `a03c8c28ba97a373a2a0824f1555496610da183b7b810e576f234274ceab6fab`. Proposal raw SHA `966d9270b1ab0bae9d4732a93d85b542b289af5a71cf477800c4b992ab9aa80c`.

Lời nhắn có thể chuyển nguyên văn: [questions_to_yoshua_bengio.md](questions_to_yoshua_bengio.md).

Chỉ kiểm tra tĩnh và SHA: không import/execute source, không chạy M0 hoặc experiment, không gửi job lên server, không sửa phương pháp, không đọc reserve400. Không tự gửi tin nhắn qua dịch vụ ngoài; tài liệu được soạn để người dùng chuyển cho người đề xuất. Tên Yoshua/Yann là quy ước vai trò trong trao đổi.

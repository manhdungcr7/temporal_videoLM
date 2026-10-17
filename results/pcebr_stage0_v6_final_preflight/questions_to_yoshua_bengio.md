# Gửi Yoshua Bengio — v6 final preflight: còn điểm cần làm rõ

Yoshua, tôi đã đọc bản source mới và đối chiếu với các điểm đã chốt. SHA256 sau chuẩn hóa CRLF→LF là `6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515`, khớp SHA cậu công bố `6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515`.

Hai điểm trước về mixed precision và `zero_noop` đã được sửa đúng trong source: `dp_f32 @ vg.float()` được tính trước khi cast output; `zero_noop` đặt gate=1, tăng counter trong kernel và assert đủ sáu layer. Công thức tích phân từng đoạn cũng hiện diện.

Tôi chưa chạy source hay gửi job GPU. Còn một lệch giữa cam kết ở đầu tệp và phần kiểm chứng thực tế; xin cậu chốt trước khi chạy.

## 1. M0 chưa thực hiện đối chiếu fine-grid như phần mô tả tuyên bố

Phần đầu tệp nói công thức zero-crossing đã được “verified against fine-grid quadrature in M0”. Nhưng trong `run_algebraic_unit_checks`, ví dụ zero-crossing ở khoảng dòng590–597 chỉ assert `W_plus > 0` và `W_minus > 0`. Tôi không thấy code tạo lưới dày, tích phân số, tính sai số, hay so với giá trị giải tích đã biết. Các assert W disjoint-support khác không thay thế kiểm tra đó.

Cậu muốn:
- thêm đối chiếu fine-grid/giá trị giải tích vào M0 và giữ nguyên claim; hay
- sửa claim thành M0 mới chỉ kiểm tra zero-crossing cho cả hai hướng có diện tích dương?

## 2. Phạm vi của “100% exact replay”

Extractor v5 được đưa vào như hàm độc lập; phép chiếu VO và dạng tangent nhìn khớp với source v5 đã lưu. Tuy nhiên M3 mới tính `p_legacy_v5` rồi ghi prediction/gate, không đối chiếu logits hoặc xác suất với output archived v5. Vì vậy code hiện chứng minh đang chạy lại cùng pipeline được mô tả, nhưng chưa tự xác minh câu “reproduces đúng từng chữ số” hoặc “100% exact replay” theo nghĩa output số trùng archived run.

Cậu xác nhận “exact replay” chỉ có nghĩa thuật toán/nhánh control được tái hiện, hay muốn khẳng định parity số học với archive? Nếu là parity output, xin bổ sung comparison trên cùng sample/runtime và ngưỡng sai số; nếu không, nên giới hạn cách diễn đạt ở “reimplementation of the archived v5 pipeline” và báo cáo parity riêng nếu có.

## Nhận xét không chặn chạy

Công thức positive/negative area đóng trên từng đoạn có zero-crossing là hợp lý khi CDF difference tuyến tính từng đoạn; breakpoint endpoints bảo đảm tính tuyến tính đó. `torch.where` vẫn tính cả hai nhánh, nhưng denominator được clamp nên không tạo chia cho zero. `zero_noop` kiểm tra đúng sáu target-layer invocations theo counter. Cần ghi rõ đây là kiểm tra đường thực thi, còn parity số học được assert riêng bằng `kernel_noop_err`.

Tôi dừng ở preflight theo yêu cầu; chưa chạy M0/M1/M2/M3 và chưa gửi gì lên hai server. Khi cậu trả lời hai điểm trên hoặc gửi bản đã cập nhật, tôi sẽ đối chiếu phần thay đổi trước khi bắt đầu thực nghiệm.
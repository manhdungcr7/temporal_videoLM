# Bản sửa M3 theo cách hiểu VO riêng từng tầng

Source gốc: `results/pcebr_stage0_v5/stage0_pcebr_v5_original.py`. Bản chạy sửa: `server_jobs/pcebr_v5_layerfix_micro8.py`. Source gốc được giữ nguyên để replay và đối chiếu. Các hàm clock, salience, pooling policy, VO và bivector được lấy trực tiếp từ source gốc; chỉ thêm quan sát indices vào salience và xác nhận toàn bộ outputs bằng tensor gốc.

Proposal có hai mô tả khác nhau: phần Vòng1 yêu cầu `T_VO^(l)` của chính từng tầng; lời nhắn cuối lại ghi `T_VO^(18)`. Vì vậy gọi đây là kiểm chứng **cách hiểu từng tầng**, không khẳng định mọi lựa chọn còn thiếu trong proposal đều đã được xác định duy nhất.

| Nhánh | Nguồn media states | VO | Điểm tangent update |
|---|---|---|---|
| V5 gốc | Output block18 | LN/V/O block18 | Output mỗi block đích |
| F1 | Output block18 như gốc | LN/V/O của từng block đích | Output mỗi block đích |
| **F2 chính** | Input của chính block đích, trước LN | LN/V/O của chính block đó | Output mỗi block đích |
| F3 bổ sung | Giống F2 | Giống F2 | Residual sau attention, trước MLP |

F2 lấy toàn bộ media input states của mỗi block đích, áp dụng phép trừ .85×media mean và normalize như pooling gốc, rồi lấy mean đúng những patches mà salience gốc chọn. Nhờ giữ patch indices và logits salience, các nhánh có cùng gate và ngân sách tương đối `.045×gate`; thay đổi representation theo tầng không bị trộn với đổi quy tắc lựa chọn bằng chứng. F1 và F2 tạo một bivector riêng cho mỗi block, không tái sử dụng vector của block18 tại mọi tầng.

Các media banks lấy từ một native prepass, không cập nhật lại theo hidden states đã can thiệp. F3 giữ đúng các vectors F2, dùng actual residual của forward đang chạy để thực hiện tangent step. PyTorch block pre-hook có thể chạy trước khi Accelerate chuyển input sang GPU của block; vì vậy residual được chuyển về cùng device/dtype với attention output trước khi tính delta. Đây là sửa kỹ thuật cho model chia hai T4; không đổi công thức của thí nghiệm.

Nguồn salience vẫn là output H18/H27 của native prepass. Với mỗi sample, cả mười một conditions dùng cùng input rotation0, thời gian nội suy, options và labels hợp lệ. M3 không được nối với các projector M1 hoặc classifier/bridge M2: proposal và mã gốc không cung cấp đường chuyển chúng thành decoder update cho câu hỏi tùy ý. Tự thêm kết nối đó sẽ là thiết kế mới cần học/kiểm chứng riêng.

Các nhánh đảo thời gian chỉ đảo clock trong hàm tạo vector; không đảo waveform/frames. Random dùng một generator seed `idx+101` và vectors riêng cho các tầng. Generation chỉ chạy native/F2, greedy32 tokens, giữ chính quy tắc bỏ choices, position IDs và prefix hooks của v5. Quy trình giữ vector từ prepass như code gốc; không phải học lại bộ suy luận trên freeform.

Không diễn giải phép LN/V/O của vector đã pool thành exact native attention: nó không có attention weights theo query/head, và pooling/normalization không giao hoán nói chung. Giữ norm chỉ ràng buộc hình học của state tại điểm can thiệp; không bảo đảm giữ nguyên ngữ nghĩa hoặc cú pháp. Kết quả đúng/sai phải được đo thực nghiệm.

Trước khi so kết quả, runner bắt buộc native logits khớp cache tuyệt đối, replay v5 gốc khớp xác suất archive, zero-scale logits khớp native, và hooks không chạm options. Archive job tính lại VO bằng trọng số của từng block trên raw bank đã lưu, xây lại bivectors và kiểm tra exact tensor equality. Hash trọng số LN/V/O, source, runner, manifest và từng artifact được lưu để kiểm toán.

# Bản tóm tắt để quyết định phương pháp (cập nhật tối 13/10/2026)

**Bài toán của paper:** giúp video LM xác định đúng **lúc nào** sự kiện xảy ra và hiểu đúng **thứ tự trước–sau**. Phương
pháp phải đúng khi đổi cách hỏi và khi đổi bối cảnh. SVHalluc chỉ là một bài kiểm tra. Phương pháp phải **hoàn toàn mới**,
đơn giản (kiểu LayerNorm / residual), **tự động hoàn toàn** (không định tuyến theo tên task), dùng được cho mọi video LM.

## 1. Sự thật đã kiểm chứng (dùng để chọn hướng)
| # | Sự thật | Số liệu |
|---|---|---|
| 1 | Mô hình **thấy đúng** nội dung của từng đoạn ngắn | khoảng 90% (J1, n=218) |
| 2 | Mô hình **nhớ được lúc nào** của thứ nổi bật (lời nói, âm thanh, cảnh lạ) | Đọc từ biểu diễn câu hỏi 96–100%; attention 98–100% |
| 3 | Mô hình **không tìm được lúc nào** của nội dung cụ thể không nổi bật (một hành động) | Đọc từ biểu diễn 66–81%; qua attention ≈ ngẫu nhiên (P1; J13: không head nào chỉ đúng nửa video có sự kiện) |
| 4 | **Soi cô lập từng đoạn** thì tìm đúng | khoảng 90% (J1); một lượt song song khớp với soi riêng (P3: tương quan 0,92–1,00) |
| 5 | Biết đúng "lúc nào" của hai sự kiện thì **thứ tự gần như hiển nhiên** | TF 93,8% (n=16) |
| 6 | Mô hình **nhầm điều nghe thành điều thấy** | CMTB 59 khi giữ tiếng, 92–94 khi tắt tiếng |
| 7 | Thời gian dạng chữ hay số giây vô dụng; thứ tự gần như không được dùng | Báo bằng chữ 50 → 50; dời sự kiện chỉ đổi 7–21% đáp án |
| 8 | **Mù thứ tự không riêng SVHalluc** | Daily-Omni Event Sequence: đảo ngược video chỉ đổi 35% đáp án, độ chính xác 50 → 41,7 (T2, n=48) |
| 9 | **Mô-đun A gây hại ngoài SVHalluc** | Daily-Omni cả 6 loại, n=220: 66,4 → 54,5 (P4) |
| 10 | Có ~30–100 head (trên 576) chỉ đúng **chunk có tiếng nói**, nhưng **0 head** chỉ đúng chunk của sự kiện hình ảnh | J13 / J13b / J13d, kiểm chéo: 0,94–1,00 so với 0,52 (audio); 0,44–0,56 so với 0,50 (visual) |
| 11 | Câu hỏi TA/TF **không nhắc gì tới hành động** ("the narrated event"), nên tìm sự kiện là bài toán **hai bước** (câu hỏi → nội dung lời nói → chunk hình khớp) | Giải thích vì sao relevance câu hỏi → prompt visual không thể đúng |
| 12 | Thêm khối prompt dài làm mô hình **dồn về một đáp án** (TA → Yes, TF → Present) | J12 / J12b / J12d. Phần "tăng" độ chính xác chỉ là hiệu ứng mẫu |

## 2. Các hướng đã thử và kết quả
| Hướng | Kết quả | Kết luận |
|---|---|---|
| Định tuyến attention theo mốc (mô-đun A, 1 lượt) | CMTB 47 → 79 (P2), 53 → 94 (J8); TA/TF gần như không đổi; Daily-Omni **gây hại** (66,4 → 54,5, n=220) | Chỉ giải "lúc đó thấy gì"; không giải thứ tự; cần cơ chế bật/tắt |
| Soi cô lập + đọc ra theo quy tắc (P5, 70 câu mới) | 44 → 77 (TA 88, TF 79, CMTB 69) | Cơ chế đúng, nhưng bước đọc ra "nặn theo đề" |
| Soi cô lập, bản tổng quát không huấn luyện (P6 v2) | 36 → 36 (n=11) | Mô hình 3B đóng băng không tự ra quyết định tin cậy |
| Câu tóm tắt 2 s không huấn luyện (J7) | ≈ base; luôn chọn câu cuối | Token chèn thêm vô dụng nếu không học |
| **Chunk prompts (J12, J12b, J12d)**: 2 prompt cô lập mỗi chunk 2 s, tăng attention theo độ liên quan | Tăng giả (dồn về Yes/Present); relevance không chỉ đúng chunk sự kiện | Prompt "…in one word:" có nội dung thật, nhưng câu hỏi không khớp được (hai bước) |
| **Probe tokens huấn luyện (T1)**: chỉ học 8 embedding | KL 0,78 → 0,48; khớp token 65 → 73% (vẫn xa "thầy") | Có học được, nhưng sức chứa hạn chế; chưa dạy thứ tự |
| SLA (ghi "đã nhìn lúc nào" vào attention) | — | Bỏ: tiền đề sai (sự thật 2, 3) |

## 3. Việc nên quyết định tiếp
1. **Bước khớp hai bước, không huấn luyện:** lấy chunk có tiếng nói bằng các head tốt (J13), sinh vài từ từ prompt audio, so với
   prompt visual của từng chunk. Phép đo offline rẻ trên 35 câu của J13d: chunk khớp nhất có rơi vào nửa video có sự kiện nhiều hơn 50% không?
2. **Đo relevance theo từng token câu hỏi** (không trung bình) trên template cố định của SVHalluc.
3. **Tách nguyên nhân dồn về một đáp án** (thì hiện tại của prompt; văn bản chen giữa; câu hỏi đọc prompt) và **báo mọi kết
   quả theo (task, đáp án đúng)**.
4. Nếu 1–2 không có tín hiệu: huấn luyện nhẹ prompt (T1 giai đoạn 2) với dữ liệu tự sinh có nhãn thứ tự.

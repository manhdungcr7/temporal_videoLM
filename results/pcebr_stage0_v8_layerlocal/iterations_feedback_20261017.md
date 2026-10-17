# Hai vòng kiểm chứng bổ sung sau CLRV v8

Hai runtimes Qwen2.5-Omni-3B, cùng frozen40 rotation0 của báo cáo trước; 38 recording origins. Không chọn lại mẫu, không train, không đổi hệ số .042×profile gamma, không mở reserve400. Hai server xử lý hai shard khác câu, không phải hai replicas độc lập. Cả hai vòng đã hoàn tất và giữ artifacts local.

| Cấu hình | Đúng/40 | Cứu/hại so native |
|---|---:|---:|
| Native | 23 | 0/0 |
| CLC v8 cũ, shared22 + bivector, toàn tail | 19 | 4/8 |
| Shared22 contrast native, bỏ bivector | 23 | 4/4 |
| Contrast native lấy tại từng tầng, toàn tail | 18 | 0/5 |
| Từng tầng, chỉ evidence contrast | 19 | 1/5 |
| Từng tầng, đổi dấu | 20 | 2/5 |
| Random từng tầng, ba seed | 21 / 20 / 19 | xem report |
| Shared22 cũ, chỉ token trả lời cuối | 21 | 0/2 |
| Shared22 native, chỉ token trả lời cuối | 22 | 0/1 |
| Từng tầng native, chỉ token trả lời cuối | 21 | 0/2 |
| Từng tầng evidence, chỉ token trả lời cuối | 21 | 0/2 |
| Từng tầng đổi dấu, chỉ token trả lời cuối | 21 | 0/2 |
| Random chỉ token trả lời cuối, ba seed | 22 / 22 / 20 | xem report |

## Đã bác bỏ được điều gì ở protocol này

1. **Đổi sang vector đo ngay tại tầng đích chưa đủ cứu phương pháp.** Hướng local khác hẳn L22: cosine trung bình ở L14≈.014, L16≈.064, L18≈.218, L20≈.498. Dù vậy nhánh local18/40, không cứu được câu nào. Khác hệ tọa độ là câu hỏi có cơ sở để đo, nhưng không thể tiếp tục dùng nó như lời giải thích duy nhất cho thất bại v8. Kết quả không chứng minh mọi cơ chế transport/layer-local đều bất khả thi.
2. **Bỏ thành phần bivector có thể giảm tác hại, nhưng chưa tăng QA.** Shared22-native từ19 lên23/40, vẫn cứu4/hại4; ngoài micro8 là17/32 so native18/32. Không chọn riêng các câu được cứu để mô tả cải tiến. Thay đổi này còn renormalize hướng phối hợp, nên không quy toàn bộ chênh lệch cho một ý nghĩa semantic riêng của bivector.
3. **Thu hẹp can thiệp xuống token trả lời chưa đủ.** Nhánh cũ từ19 lên21; native contrast shared22 đạt22. Không nhánh last_ nào cứu một câu native sai. Giảm số vị trí cũng giảm tổng can thiệp; không thể từ đây kết luận option-token contamination là nguyên nhân duy nhất.
4. **Không có bằng chứng hướng contrast hiện tại mang lợi thế temporal so đối chứng.** Nhánh local chính18, đổi dấu20, random21/20/19. Các số này không chứng minh random tốt hơn về mặt tổng quát; chúng ngăn claim rằng normalization, sink matching và chiều contrast đã bảo đảm một vector evidence/order chính xác.

## Kiểm chứng thực thi

Hai sources effective của vòng local trùng SHA. Native/CAOR/CLC cũ replay sai số xác suất tối đa0.0; no-op0.0. Cả40 có input cache và direction artifact hashes khớp, backbone parameter versions không đổi. Hướng được xây trên prompt không chứa lựa chọn, còn QA gold chỉ phục vụ chấm kết quả.

Audit local tái tính các evidence/order differences và hướng local từ tensor đã lưu: khớp bitwise toàn40. Tái tạo normalized random vectors giữa torch local2.10/Windows và server2.11/Linux khác tối đa7.45e-09; không gọi phần random đó là bitwise. Bytes của các vector đã chạy vẫn khớp SHA. Xem [continuation_verification.json](continuation_verification.json).

Vòng readout có lỗi wrapper ban đầu: dùng gamma trước làm tròn thay vì gamma trong profile. Ngưỡng replay phát hiện sai khác .0018059 ngay ở mẫu đầu; đã giữ lỗi và sửa dùng đúng profile gamma. Lượt sau replay khớp0.0 trên40, không nới ngưỡng và không bỏ mẫu. Archives giữ initial error và completion cuối. Đây là sửa tái lập, không phải lỗi khoa học của CLRV.

Hệ số, rows/layers, protocol và seeds đã ghi trước outputs của từng vòng trong [protocol layer-local](protocol.md) và [protocol readout](../pcebr_stage0_v8_readout/protocol.md). Bảng từng task/gold, per-ID, cứu/hại, probabilities, ba seeds và uncertainty của vòng local nằm ở [report local](report.md), [summary local](summary.json), [report readout](../pcebr_stage0_v8_readout/report.md). So sánh dùng cùng native23/40 của lượt này, không trộn baseline cyclic-rotation cũ.

## Điều cần thay đổi trong hướng nghiên cứu

Tôi rút đề xuất “đổi sang contrast từng tầng” khỏi danh sách cải tiến có triển vọng trên bằng chứng hiện tại. Đó là giả thuyết đã được thực nghiệm và thất bại, không tiếp tục quảng bá chỉ vì lập luận hình học nghe hợp lý. Hướng can thiệp chỉ token trả lời cũng không đạt yêu cầu QA.

Nút thắt tiếp theo cần kiểm chứng là **độ tin cậy của grounding và verification ở cấp event instance**, trước khi xây thêm contrast. Một phép đo độc lập cụ thể: trên các public origins có interval annotations đã chọn, so khả năng frozen native decoder nhận đúng mô tả sự kiện trong interval thật với interval sai của cùng recording. Tách audio/video, giữ nhiều instance đúng và hard negatives cùng scene; so native verification odds, conditional description likelihood và manifold affinity. Không dùng QA gold của40 để chọn interval. Nếu ngay phép đo này thất bại, planner/verify loop dựa trên chính backbone cũng thiếu tiền đề và phải thay supervision hoặc scorer.

Đây chưa là phương pháp mạnh hoặc mới. [VideoMind](https://arxiv.org/html/2503.13444v1) đã dùng grounder/verifier với huấn luyện; [temporal consistency study](https://arxiv.org/html/2411.12951v1) đã cho thấy grounding không đồng nghĩa verification consistency. Vì vậy thêm bước “verify” bằng prompt không đủ tạo novelty hay bảo đảm hiệu quả. Xem [research notes](research_notes.md) cho các nguồn đã đọc và phản biện liên quan.

Mục tiêu cải thiện QA tổng quát vẫn chưa đạt. Hai vòng này hoàn thành việc loại bỏ hai giả thuyết sửa chữa cụ thể; chúng không chứng minh mọi phương pháp joint AV temporal reasoning đều thất bại và không cho phép kết luận về CVPR acceptance.

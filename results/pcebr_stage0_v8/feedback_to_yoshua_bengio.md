# Gửi Yoshua Bengio — kết quả CLRV v8, sửa lỗi và kiểm định cơ chế

Ngày 2026-10-17. Tôi dùng source gốc SHA `fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895` cùng phần tổng hợp sửa đổi cậu gửi làm mốc; không chặn thực nghiệm vì file proposal cũ. Đã tái lập M0 và chạy trên hai runtimes Qwen2.5-Omni-3B, mỗi runtime 2×T4. Hai server chia hai shard câu hỏi khác nhau; không phải hai lần độc lập của cùng QA8.

## 1. Kết quả micro8 và ablation

| Nhánh | Đúng/8 | Gate mở |
|---|---:|---:|
| Native | 5 | — |
| V8 gốc fc0a… | 5 | 0/8 |
| V8 sửa interface v1 | 3 | 6/8 |
| V8 sửa đầy đủ interface v2 | 3 | 8/8 |
| CAOR-only trên bản sửa cuối | 5 | 8/8 |
| CAOR + CLC, bỏ CT | 5 | 8/8 |
| CAOR + CT, bỏ CLC | 4 | 8/8 |
| CT-only ở lượt decode cuối | 5 | 8/8 |
| CLC-only ở lượt decode cuối | 6 | 8/8 |
| Bridge từ rollout routed | 3 | — |
| Bridge từ rollout native | 3 | — |
| Bridge không có rollout | 4 | — |

CLC-only vẫn xây hướng từ các lượt Support/Foil/Order của CAOR; tên gọi chỉ có nghĩa bỏ attention routing và CT ở lượt decode cuối. Không được mô tả nó là một forward, hay chưa từng dùng CAOR.

V8 full sau sửa làm hỏng 1449 và 1929, không cứu câu nào. CLC-only giữ các câu native đã đúng và cứu 1513 trên micro8. Đây là kết quả exploratory trong sáu ablation, không phải bằng chứng xác nhận độc lập. Tất cả replay combined-direct trong component audit khớp xác suất mainrun với max error=0.0.

## 2. Xác nhận trên frozen40 bằng cùng công thức, không chọn lại hệ số

Tiến độ: 40/40 mẫu đã lưu. Hệ số CLC .042 được giữ nguyên; không train và không chọn hệ số bằng ground truth. Bộ 40 là manifest đã chọn trước, rotation0, mỗi task/gold 5 mẫu, clip ≤45 giây. Không mở reserve400. Đây vẫn là development set đã xem qua nhiều vòng.

| Decoder cuối | Đúng/tổng toàn bộ | Đúng/tổng ngoài micro8 |
|---|---:|---:|
| Native | 23/40 | 18/32 |
| CAOR-only | 20/40 | 15/32 |
| CLC-only | 19/40 | 13/32 |
| CLRV full direct, không bridge | 16/40 | 13/32 |

CLC-only cứu các ID [1427, 1513, 1584, 1952]; làm hỏng các ID [1528, 1660, 1898, 1964, 2007, 2166, 2223, 2336]. Phải đọc cả hai nhóm, không chỉ net accuracy. Đối chiếu từng task/gold và từng câu ở [report40.md](report40.md). Kết quả 40 không bao gồm generation/bridge; bridge được kiểm tra đầy đủ trên micro8 và chưa cho tín hiệu tốt.

Kết luận trên phiên bản/protocol đã chạy: V8 full-direct và CLC-only đều chưa cải thiện baseline. Lợi thế 6/8 của CLC-only không giữ trên 32 mẫu ngoài micro: 13/32 so native 18/32. Không được dùng riêng 6/8 làm claim improvement. Trên bảng theo gold, CLC tăng TF/A và CMTB/A nhưng giảm TF/C, TF/B, TA/A và CMTB/C; đây là trade-off theo nhóm, không phải cải thiện temporal đồng đều.

## 3. Những lỗi đã sửa, không quy chúng thành thất bại khoa học

1. **M0 CUDA dùng index GPU vào buffer CPU.** CPU M0 chạy qua; CUDA bản gốc lỗi. Đã đưa đầu vào extraction/clock indices về CPU, đúng giao diện mà M3 vốn dùng. Không đổi công thức. Source đóng gói sau sửa chạy M0 CPU và CUDA qua trên cả hai server.
2. **Token BPE bị coi là entity.** Tokenizer thật tách `narrated` thành `narr` và `ated`; per-token regex làm câu quy chiếu bị chuyển sang nhánh specific rồi gate=0. M0 dùng `_UnitTok` theo nguyên từ nên che khuất lỗi này. Đã phân loại theo whole-word offsets và kiểm tra token roundtrip, giữ token indices cho mô hình.
3. **Endpoint bị gán cứng vào physical modality.** Sau khi trả về banks endpoint1/2, profile cũ vẫn gán endpoint1 vào audio, endpoint2 vào visual bất kể rho. Đã project evidence/order profiles vào từng physical token bank theo rho và xử lý deictic foreground gate theo hai endpoint. Unit check giữ nguyên giới hạn A→V; với A→A/V→V, profile evidence của modality không được chọn trở về 0 trong giới hạn rho cứng.
4. **Tách nhầm mệnh đề.** Với “Relative to the moment when …, when does …”, parser cũ lấy “Relative moment” làm endpoint1 và gộp speech+event vào endpoint2. Đã ưu tiên ranh giới dấu câu của mệnh đề mở đầu, ưu tiên relation connector hơn `and` bên trong tên sự kiện.
5. **Tính từ modality bị coi như tên event.** Đã thêm các tính từ kênh phổ quát `visual`, `auditory`, `acoustic`, `audiovisual` vào nhóm từ quy chiếu, không thêm action/đối tượng nấu ăn hay điều kiện theo task/ID/gold. Đây là sửa interface sau xem development micro8, không được trình bày như thiết kế đã được khóa trước khi xem dữ liệu.
6. **Nhánh zero-gate bỏ qua KV cache.** Nó tính lại media prefix cho từng token dù flag cache=True. Đã sửa để tôn trọng flag; vẫn giữ fallback về full pass khi sai số cache vượt ngưỡng .02 của code. Không nới ngưỡng để lấy điểm hoặc tốc độ.
7. **Giao source chạy độc lập.** Đã nhúng helper và sửa device vào `stage0_pcebr_v8_corrected_standalone.py`, hash `25043c391ac1503280c2e4b70c4e38d27d3f8831741e4a38f8f96ffc4b758843`; đổi tên output để tránh ghi đè historical v7. M0 CPU/CUDA của package đều qua. Tất cả source gốc, v1, v2 và logs được giữ riêng.

Wrapper xác nhận40 ban đầu thiếu `gc` trong namespace sau khi đã xong một mẫu/shard. Đã sửa và resume các ID còn lại, giữ nguyên mẫu đầu, không đổi mẫu/hyperparameters; logs lỗi và lần resume được lưu. Đây là lỗi của wrapper thực nghiệm, không phải lỗi thuật toán CLRV.

## 4. Phát hiện cơ chế đáng giữ lại

- Ở 1929, native đúng C; CAOR-only, CLC-only, CT-only, CAOR+CLC và CAOR+CT đều còn đúng. Chỉ ghép cả CAOR+CLC+CT mới thành A ở direct, rồi bridge thành B. Đây là tương tác bất lợi của các can thiệp, không phải bằng chứng từng thành phần riêng đều vô dụng.
- Ở 1449, native/CAOR/CLC-only/CT-only đều đúng A, nhưng CAOR+CT và full thành B. Vì vậy không nên mặc định đưa continuous thought vào các tầng 0–2 sẽ giúp bảo toàn/ngữ nghĩa.
- Bridge của V8 cho 3/8; bridge dùng rollout native cũng 3/8. Full bridge với đảo routing vẫn3/8; option permutation sau ánh xạ về nhãn gốc đạt4/8; trace-only không media đạt2/8. Không có bằng chứng bridge này đang khai thác quan hệ temporal một cách đáng tin. Thay đổi latent order routing không bảo đảm câu sinh thật sự đảo chiều; phải đọc trace thay vì suy từ TV.
- Control permutation gốc thêm tiền tố “Answer choices:” trong khi forward không có, nên không thể quy toàn bộ chênh lệch 3→4 cho hoán vị. Đã bổ sung matched-format control riêng dùng cùng trace tái-token hóa và giữ nguyên từng khung nhãn/khoảng trắng; báo cáo riêng, không thay mainrun. Kiểm tra identity re-render phải khớp nguyên chuỗi.
- Matched-format control đã chạy đủ8: forward3/8, permutation sau unmap4/8; identity re-render khớp100%, và forward từ trace tái-token hóa khớp xác suất bridge gốc với max error0.0. Vì thế ảnh hưởng của permutation vẫn tồn tại sau khi bỏ confound định dạng, nhưng một biến đổi duy nhất trên8 câu chưa đủ định lượng robustness tổng quát.
- Kiểm tra input/output embeddings cho cosine cùng token≈1.0, norms trùng. Giả thuyết lỗi do hai bảng embedding hoàn toàn khác nhau không được dữ liệu này ủng hộ. Vấn đề vẫn có thể nằm ở semantic calibration của intermediate logit lens và pooling, không ở việc hai bảng embedding không cùng tọa độ.

## 5. Những tuyên bố còn phải giảm mức mạnh

**P_null không tự trở thành absence detector.** Double-centering giải quyết đúng gauge cộng theo hàng/cột. Nhưng với marginals uniform, expected compatibility bằng0 cho mọi double-centered Omega: một kiểm tra độc lập đổi coupling L1≈1.282 nhưng P_null gần như không đổi. Cần gọi nó heuristic. Trong M3, null features được ghi vào audit; gamma routing chủ yếu lấy sharpness và excess gate, nên không được quy mọi cải thiện null diagnostic thành cải thiện QA.

**Mean lag vẫn chỉ phụ thuộc marginals.** E[t2−t1]=E[t2]−E[t1]. Channel12 không phải thông tin riêng của joint coupling như docstring nói; một số histogram/synchrony/dispersion channels mới có thể phụ thuộc coupling.

**Conjunctive gate không chứng minh cùng event instance.** Ma trận [[.8,0],[0,.8]] vẫn cho gate .45 dù hai từ chỉ khớp ở hai slots khác nhau. Cần instance-coherent event evidence, đặc biệt với repeated actions. Gate cao .99 trên query thật cũng không có nghĩa câu trả lời temporal đúng.

**Bốn nhánh routing không đồng nghĩa bốn loại reasoning đã được xác nhận.** Sau parser sửa, một số câu hỏi benchmark Joint AV vẫn được detector chọn A→A. Đây là quyết định dựa trên cue/manifold affinity, chưa có annotation endpoint/modality độc lập. Không sửa bằng việc ép tất cả câu benchmark sang A→V. Cần semantic endpoint validation trên câu hỏi/context độc lập.

**M0 4+2 là planted synthetic recovery.** Hidden states được cài bốn vector cùng nhóm và hai vector khác nhóm. M0 chứng minh optimizer tìm được cấu trúc đã cài, không chứng minh event parsing tự nhiên. Các proxy absent queries đều xa domain nấu ăn; FPR0/32 ở micro không phải calibration cho hard negatives cùng video.

## 6. Hướng tiếp theo tôi đề nghị

1. Giữ CLC-only như giả thuyết đơn giản đã được ablate, đọc kết quả40 và 32 ngoài micro trước khi thêm thành phần. Nếu lợi thế không còn, dừng claim improvement thay vì tiếp tục tối ưu hệ số trên bộ này.
2. Muốn xác định bottleneck, cần một tập nhỏ có annotation độc lập cho event identity, interval và absence. So sánh learned/heuristic grounding với oracle interval ở dạng diagnostic, rồi xem decoder có dùng đúng evidence không. Không dùng oracle làm phương pháp triển khai.
3. Làm hard negatives cùng recording, đảo quan hệ có nhãn đúng, repeated instances, và cả A→A/V→V với câu hỏi không chứa cue rõ. Không chỉ thử welder/guitarist trên cooking.
4. Tách grounding, causal-direction quality, can thiệp decoder, và answer selection thành measurements riêng. Không suy vector “evidence thuần túy” hoặc “order thuần túy” chỉ từ shared-sink subtraction/norm/reversal identity.
5. Kiểm tra option rotations và câu trả lời tự do được chấm nghĩa độc lập trước khi mở test mới. Không đưa metric regex/keyword thành semantic accuracy, không trộn kết quả rotation0 với bảng cyclic-rotation cũ.
6. Một giả thuyết cải tiến có thể kiểm tra: thay hướng CLC chung lấy ở L22 rồi inject vào nhiều tầng bằng các contrast Support−Foil và Support−Order được đo ngay tại từng tầng đích. Giữ norm/budget, so với hướng L22 chung và random controls. Điều này kiểm tra vấn đề lệch hệ tọa độ giữa các tầng; chưa có kết quả ở lượt này, chưa phải claim novelty, và không tự giải quyết event grounding sai. Cần đối chiếu các thử nghiệm cũ trước khi triển khai để tránh lặp lại.

Các sửa hiện tại giúp phương pháp được thực thi đúng hơn, nhưng không đủ chứng minh tổng quát hóa hoặc novelty ở mức CVPR. Hướng paper nên dựa vào một cơ chế đã có bằng chứng nhân quả và lợi ích ngoài các template đã xem, thay vì ghép thêm gate/OT/CT/contrast chỉ vì từng unit check qua.

## 7. Artifacts và tái lập

Các file `report.md`, `report40.md`, `verification.json`, raw archives, sources/patches, per-query predictions và ablation replay được giữ local trong folder này. Backbone parameter versions không đổi; không train M1/M2/backbone. Source algorithm effective của hai shard trùng bytewise. Raw tensor views được compact nhưng kiểm tra giá trị bằng nhau bitwise; lưu cả hash gốc lẫn compact. Chưa gửi tin nhắn ra dịch vụ bên ngoài; file này là bản trả lời để người dùng chuyển cho Yoshua.

Kiểm toán40 xác nhận đủ40 ID không trùng từ38 recording origins, mỗi task/gold đúng5 mẫu, clip dài nhất43.1s, prompt token roundtrip chính xác40/40. Native replay max error0, kernel-noop max error0;17/40 mẫu dùng fallback full-pass theo ngưỡng parity gốc. Micro8 CLC replay ở lượt40 có max probability error0. Số native23/40 ở đây là rotation0 cùng encoding hiện tại; không trộn với các baseline cũ dùng cyclic rotations hoặc protocol khác.

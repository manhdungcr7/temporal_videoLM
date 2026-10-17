# Đề xuất của Yann — Temporal Address Memory, 2026-10-16

Trạng thái: đề xuất nghiên cứu, chưa hiện thực hoặc chạy. Không phải kết quả thực nghiệm hoặc xác nhận novelty. Tên tạm TAM. Backbone Qwen2.5-Omni-3B đóng băng; huấn luyện bộ nhớ/adapter nhỏ. Đây không phải training-free.

## 1. Giả thuyết cần kiểm chứng

Một nguyên nhân khả dĩ của temporal hallucination là decoder không sử dụng đúng liên kết giữa nội dung sự kiện, thời điểm và nguồn phương thức trong câu hỏi hiện tại. Chưa có bằng chứng rằng Qwen hoàn toàn không có thông tin thời gian trong V. Không suy luận bất khả thi cho attention từ các lần can thiệp không đổi argmax.

V7 micro8: native/full cùng5/8; audio excess/gate0 toàn bộ8 mẫu làm joint_gate/Psi0; gate semantic-wrong cao hơn true8/8; J_cond low-mass normalization sum≈.01. Các kết quả này bác bỏ hiệu quả của cấu hình đang chạy, nhưng không bác bỏ một bộ nhớ có bằng chứng thời gian học cùng QA. M3 nối model calls nhưng feature format còn khác train. Xem pcebr_stage0_v7/report.md và feedback_to_yoshua_bengio.md.

TAM đặt câu hỏi cụ thể: nếu decoder được đọc các tuple nội dung–khoảng thời gian–nguồn được giữ nguyên sự gắn kết, và đường đọc được học bằng các can thiệp thời gian có nhãn hợp lệ, liệu strict paired QA có tăng so với adapter cùng budget không giữ gắn kết này?

## 2. Bộ nhớ giữ địa chỉ của bằng chứng

Trích feature audio/video ở cùng một hook đã khóa, với cùng preprocessing trong train và inference. Mỗi chunk có tuple b_i=(c_i,[s_i,e_i],m_i), trong đó c_i là content feature, [s_i,e_i] là physical interval và m_i là modality. Khi cần, giữ nhiều local tokens trong chunk; không lấy một global mean đại diện cả recording.

Dùng bộ nhớ chung B=B_A∪B_V. Query conditioner đọc câu hỏi tự nhiên, tạo một số ít soft role queries bằng cross-attention; không parser theo TA/TF/CMTB, không hardcode speech-anchor, không cần MC options. Mỗi role có thể đọc audio, video, cả hai hoặc null. Các role là latent variables cần học/kiểm tra, không mặc định tên slot1 luôn là audio và slot2 luôn là video.

Mỗi role giữ một phân phối trên nhiều chunk và null, thay vì lập tức lấy expected timestamp hoặc argmax. Với hành động lặp, một role được phép có nhiều mode. Không ép doubly-stochastic/one-to-one matching. Một relation-attention block có thể trao đổi giữa các roles qua content, modality và relative interval features. Sự kiện nào không được hỏi không bắt buộc phải có một role riêng; số role hữu hạn là giới hạn năng lực cần đo.

Adapter đọc các tuples với content projection và time/source projection được giữ riêng đến khâu kết hợp theo từng tuple. Có thể biểu diễn giá trị đọc như [W_c c_i ; E_t(s_i,e_i) ; E_m(m_i)] rồi chiếu về decoder dimension. Không cộng một tổng content và một tổng time độc lập, vì cách đó đánh mất ai gắn với thời điểm nào. Decoder nhận đường đọc bộ nhớ có residual zero-init, backbone đóng băng. Thử rank8 như một cấu hình ban đầu được khóa trước evaluation; không khẳng định rank này tối ưu.

Không dùng tích gate_audio*gate_video để bật cả module. Khi query chỉ cần visual thì visual evidence vẫn dùng được; câu hỏi cần hai endpoints có thể có một endpoint null. Null học từ semantic negatives có xác nhận, gồm hard negatives cùng chủ đề. Chỉ entropy hoặc sắc nhọn salience không được diễn giải là xác suất hiện diện.

## 3. Học bằng can thiệp vào gắn kết nội dung và thời gian

Tạo các biến đổi trên waveform/frames và tái mã hóa toàn pipeline, không chỉ sửa timestamp tensor:

1. Dịch chung audio/video mà không cắt sự kiện: localization phải dịch theo. Chỉ áp dụng answer-invariance nếu ý nghĩa câu hỏi thật sự không phụ thuộc absolute clock, boundary hoặc thứ tự first/last bị thay đổi.
2. Dịch một sự kiện hoặc một stream sao cho các event được hỏi còn nguyên và quan hệ thật thay đổi: annotation mới quyết định answer mới. Không mặc định tất cả câu hỏi đều đổi Before↔After theo một permutation.
3. Thay đúng một endpoint bằng event khác cùng chủ đề, hoặc che endpoint có kiểm chứng: học absence/uncertainty và câu trả lời mới từ annotation. Không gọi bất cứ caption disjoint nào là absence ground truth.
4. Di chuyển distractor mà giữ endpoints/quan hệ đúng: câu trả lời nên ổn định trong các case không đổi semantics.

Không reverse waveform lời nói, không cyclic-roll mù qua boundary; các thao tác đó có thể phá semantics. Chọn non-clipping translations hoặc non-overlapping event-block edits có kiểm tra biên/nền. Chia train/test theo recording origin trước khi sinh biến đổi. Một họ biến đổi từ cùng recording không tạo thêm samples độc lập.

Loss gợi ý:

L = L_answer + lambda_g L_ground + lambda_e L_equiv + lambda_n L_null.

L_answer là teacher-forcing NLL trên câu trả lời đúng của bản gốc và bản can thiệp. Không dùng native prediction làm nhãn đúng. L_ground dùng multi-positive event/interval annotation, giữ tất cả instance hợp lệ thay vì phạt alternative đúng. L_equiv so phân phối grounding của hai bản theo ánh xạ physical transform g; không tự ép mọi hidden representation invariant. L_null dùng event-existence labels độc lập. Không có temporal relation label ở inference.

Tránh tự thêm diversity penalty bắt hai role phải khác nhau: hai mô tả có thể chỉ cùng một event. Nếu cần event-role supervision, gán bằng span/endpoints hợp lệ hoặc matching permutation-invariant. Đo role collapse trực tiếp.

## 4. Điểm khác hướng cũ và các nguy cơ

- Một đường train→inference thống nhất, cùng hook/query/media features; loss QA đi qua frozen decoder để adapter học một hướng mà decoder thực sự dùng.
- Không dựa vào việc subtract hidden states sẽ tự khử sạch language prior, hoặc một Phi14D sẽ đủ biểu diễn mọi temporal query.
- Quan hệ thời gian được học từ thay đổi bằng chứng và đáp án, không từ tên task hoặc mẫu câu hỏi.
- Bộ nhớ có thể phục vụ A→V,V→A,A→A,V→V theo thiết kế. Đây là khả năng kiến trúc, chưa phải bằng chứng tổng quát hóa.
- Không bảo đảm content/time độc lập tuyệt đối trong backbone feature; branches chỉ là cấu trúc lưu trữ. Cần counterfactual audit để phát hiện content feature rò absolute position hoặc model đọc shortcut từ edit artifacts.
- Caption/query/event slots có thể sai, adapter có thể bỏ qua memory, fitting data nhỏ vẫn overfit. Có null không tự đảm bảo calibration. Bộ nhớ giới hạn và chunk resolution giới hạn suy luận đếm/định vị tinh.
- Freezing backbone không đồng nghĩa train không tốn activation memory: QA loss cần gradient qua phần decoder sau injection. Dùng sequential microbatch, checkpointing và đo thật trước khi cam kết fit2xT4.

## 5. Phép thử nhỏ trước khi xây cả hệ thống

A. Oracle-memory diagnostic: cung cấp intervals/endpoint identities đã annotation cho8 micro để tách khả năng decoder đọc đúng bằng chứng khỏi khả năng truy hồi. Đây là privileged-input diagnostic, không được báo là inference gain hợp lệ. Không tune adapter, alpha hoặc layers trên8 mẫu để chọn winner.

B. Học prototype trên recordings mới tách mọi tập đã xem, có labels endpoints/absence/QA; validation recordings riêng. Cache252 hiện tại không đủ làm ground truth mới. Khóa recipe bằng train/validation trước khi đánh giá research40. Giữ nguyên40 mẫu≤45s và bảng từng(task,gold), không thay samples theo kết quả.

C. Matched-budget controls: frozen native; generic adapter cùng parameter/QA budget; content-only memory; full memory; full nhưng shuffle time giữa contents; full nhưng shuffle modality; full không can thiệp train. Shuffle là diagnostic corruption, không phải một đối thủ tự nhiên cùng semantics. Full cũng phải thắng baseline được train matched chứ không chỉ thắng corruption control.

D. Báo strict-pair correctness: cả original lẫn transformed answer đúng; localization/multi-positive accuracy; absence precision/recall; ordinary QA; performance theo modality pair và repeated events. Chỉ changed-answer rate không phải temporal correctness. Dùng recording-level uncertainty trên tập đủ lớn; không suy significance từ8 micro hoặc số augmentation.

Nếu oracle memory không giúp thì ưu tiên đường decoder readout/training, chưa mở rộng event retriever. Nếu oracle giúp nhưng predicted memory không giúp thì tập trung endpoint retrieval/calibration. Nếu full bằng content-only hoặc memory-time-shuffle trên independent recordings thì bác bỏ giả thuyết đóng góp của temporal address trong prototype này.

## 6. Literature và giới hạn claim novelty

Qwen2.5-Omni đã có time-interleaving và TMRoPE; không thể gọi việc thêm timestamp là phát minh temporal encoding: https://arxiv.org/abs/2503.20215

Temporal Grounding Bridge đã kết hợp grounding với MLLMs: https://aclanthology.org/2024.emnlp-main.556/

Towards One-to-Many Temporal Grounding xử lý nhiều đoạn disjoint cho một query: https://proceedings.mlr.press/v306/xu26cc.html

TECL đã đề xuất grounding equivariance dưới temporal rolling: https://www.sciencedirect.com/science/article/pii/S0950705126014139 (abstract trên search index; open toàn trang bị403 trong phiên này). Không được claim temporal-equivariance loss là mới.

Đóng góp dự kiến phải được thu hẹp và kiểm chứng: question-conditioned AV provenance memory giữ gắn kết content–interval–source, được học/đánh giá bằng các can thiệp phân biệt đúng event binding với temporal prior, và cải thiện generated QA dưới backbone đóng băng. Từng thành phần memory/adapter/contrastive/equivariance đều có tiền lệ. Chưa có exhaustive novelty search, kết quả TAM hoặc bảo đảm CVPR acceptance.

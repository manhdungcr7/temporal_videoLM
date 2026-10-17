# Gửi Yoshua Bengio — kết quả CAOR-WEMD v6 và hướng nên đổi

Yoshua, tôi đã chạy bản final trên cả hai server GPU. **M0 qua; M1 chưa vượt baseline mạnh một cách nhất quán; M2 có lợi ích hạn chế từ Phi, thất bại ở two-hop; M3 giữ nguyên mọi đáp án, native=v5=CAOR=5/8.** Những điểm phần mềm chúng ta đã chốt được hiện thực và kiểm tra. Vì vậy hiện chưa có lý do quy kết quả yếu này cho lỗi FP16 trong delta hoặc nhánh legacy chưa khôi phục. Cần chuyển trọng tâm sang chất lượng event binding và cách tín hiệu quan hệ tác động vào LLM.

Trong tài liệu này, Yoshua/Yann là quy ước vai trò của cuộc trao đổi, không phải xác nhận tác giả hay cộng tác của các nhà nghiên cứu thật.

## 1. Đã chạy gì và đã xác minh gì?

Source attachment được giữ nguyên. SHA256 sau chuẩn hóa LF: `6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515`. Wrapper chỉ thêm lưu checkpoint/score/episode/logits và audit riêng. Không đổi công thức, seeds, số bước hoặc chọn hệ số theo QA.

Hai runtime đều Qwen2.5-Omni-3B FP16, 2×T4. M1/M2 dùng cùng bốn embedding caches, SHA khớp local; hai replicas trùng kết quả. Đây vẫn chỉ là 202 train pairs/16 origins và 50 held-out pairs/4 origins đã xem ở những vòng trước. M2 dùng bundle pooled của seed đầu tiên đúng source; không chọn seed tốt nhất sau khi xem kết quả. Các mô tả là pseudo descriptions, không phải annotation event/time/absence độc lập.

M1 hoàn thành 66 fits/server ×180 steps. Main M2 có 3 bridge fits ×120 steps và29 probes ×200 steps. Một lỗi JSON serialization do wrapper gặp tensor trong kwargs được sửa; M1 được giữ lại, M2 được chạy tiếp bằng original functions. Ba bridge fits và episode dở của lần lỗi được lưu riêng. Lỗi này không làm đổi thuật toán. Log và wrapper trước/sau lỗi đều có trong artifacts.

Hai câu hỏi preflight cuối đã giải quyết được bằng audit tại chỗ:

- Ví dụ CDF giao nhau cho W+ =1/96 và W− =25/96; code khớp cả giải tích và fine-grid240001 điểm với max error≈1,99e−8. Công thức mới là đúng trên ví dụ kiểm chứng này.
- Nhánh legacy được đối chiếu với **original v5 source trên cùng inputs/runtime**: vector, gate và probability error đều0 ở cả8 mẫu. Native replay và active-kernel zero-noop có logit error0, counter đủ6 layer ở8/8. Không mặc định parity bitwise với mọi lần chạy lịch sử khác runtime.

Không huấn luyện backbone. Parameter version counters trước/sau giữ nguyên; hook count0; hai CUDA devices khỏe. M3 chỉ chạy đúng micro8 rotation0; chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400 trong lượt này.

## 2. M1: tách global/innovation đúng protocol hơn, nhưng lợi ích còn yếu

Trung bình ba seeds, đơn vị%:

| Cache/condition | Global | Exact interval | Multipos | Nonnull |
|---|---:|---:|---:|---:|
| Native C3b |26,67|38,00|50,00|35,50|
| Native C6 |34,67|36,67|48,67|30,10|
| Native C8b shuffle innovation |21,33|29,33|36,00|30,10|
| Pooled C3b |46,00|46,00|52,00|51,60|
| Pooled C6 |43,33|43,33|49,33|47,33|
| Pooled C8b shuffle innovation |35,33|42,00|46,67|48,40|

Global ở đây chấm normalized caption content của top1, không yêu cầu đúng recording/time. Exact là đúng index trong group cùng recording/modality. Multipos chấp nhận mô tả trùng trong group. Đừng gọi tăng Global là đã tăng định vị thời gian.

Điểm tốt: C6 hơn matched innovation-shuffle, nhất là native. Có tín hiệu alignment thật trong phần innovation. Bilevel scorer nay nhất quán với embedding retrieval; pure modality normalization đã được tách thành conditions riêng thay cho hybrid cũ.

Điểm chưa tốt: native C6 tăng Global nhưng giảm Exact/Multipos/Nonnull so C3b. Pooled C6 giảm cả bốn metric. Với pooled, phần Exact chỉ hơn shuffle1,33 điểm phần trăm và Nonnull còn thấp hơn shuffle. Chưa có bằng chứng đây là representation tốt hơn để làm temporal QA.

Pure ZCA C5 cũng không thắng: native16% Global/30% Exact; pooled32%/38%, thấp hơn C3b. Vì vậy sửa đúng sự bất nhất giữa prose/code về whitening không đồng nghĩa phương pháp whitening đó hiệu quả hơn. Tôi không đề nghị tiếp tục đổi tên hay thêm whitening variants chỉ để tối ưu cùng4 held-out origins.

## 3. M2: đọc riêng temporal relation và foil

| Suite | Full B8 đúng/sai/tổng | Bỏ Phi B7 | Coordinate shuffle B2 | Full Acc3 |
|---|---|---|---|---:|
|Natural|149/83/232 (64,2%)|149/83/232 (64,2%)|114/118/232 (49,1%)|60,3%|
|Reversal|363/213/576 (63,0%)|351/225/576 (60,9%)|306/270/576 (53,1%)|56,0%|
|Two-hop Roll|58/98/156 (37,2%)|54/102/156 (34,6%)|63/93/156 (40,4%)|32,5%|

Uniform control trở về25% ở cả ba suite, tốt hơn protocol v2 nơi uniform thắng learned. Nhưng vượt uniform chưa đủ: Natural thêm Phi không tăng tổng số đúng; Reversal thêm12 câu, Roll thêm4 câu, tất cả vẫn đến từ cùng4 origins.

Trên Natural, full hơn matched C6 mean-time baseline30 câu, nhưng22 câu trong đó đến từ foil. Acc3 chỉ tăng59,8→60,3%; C1 mean-time reference đạt64,4%, cao hơn full60,3%. Trên Reversal, full Acc3=56,0%, vẫn thấp hơn C1 reference60,0%; paired thuận/đảo full121/216 so C1=125/216. Không thể dùng Acc4 tăng mạnh để kết luận có tiến bộ temporal reasoning tương ứng.

Two-hop là điểm không đạt rõ nhất:

- Full chỉ đúng6/39 Before,8/39 During,7/39 After, nhưng37/39 foil. Nếu bỏ foil và dùng argmax3 relations, chỉ đúng38/117=32,5%.
- Presence/null-only đã57/156, gần full58/156. Coordinate-shuffle đạt63/156, cao hơn full.
- Hop2 train276/333=82,9%, held-out21/117=17,9%; hop1 held-out48/117=41,0%. Đây là khoảng cách train/held-out lớn trên dữ liệu hiện có, không phải dấu hiệu bridge đã tổng quát hóa tốt.
- Khi hop1 top1 đúng, hop2 chỉ6/48=12,5%. Không nên đổ toàn bộ lỗi hop2 cho hop1.
- Teacher-forced bridge đạt57/156; mixture-trained đạt58/156. Các feature tensors có khác nhau và từng episode đúng/sai có thể khác, nhưng tổng điểm chưa cho thấy lợi ích đáng kể từ mixture training.
- Oracle anchor đạt59/156, gần58/156. Tuy nhiên tên “upper bound” cần giới hạn: code chỉ thay vector context đưa vào bridge bằng bank1[gold], vẫn giữ lg1/p1 trong temporal features. Đây không phải oracle sửa cả hop1 location. Audit cho thấy oracle đổi features nhưng không đổi các hop2 top1 correctness flags trên bộ này.

Các foils held-out vẫn có0/4 origins qua strict cosine+lexical filter,2 lexical và2 string fallback. Đừng gọi chúng là event absence truth. Reversal/Roll ở đây thay coordinates/bank assignments, không phải đã đảo frames/dịch waveform và đánh giá câu trả lời từ AV mới.

## 4. M3: có tác động lên phân phối, chưa sửa được QA

Tất cả native, legacy v5, full CAOR, no-desink, order-sign-reversed, slot-profile-reversed, rel-only, order-only, irrelevant-slot và cả ba random seeds đều **5 đúng/3 sai**. Mọi lựa chọn đáp án của full giữ nguyên native.

| Task/gold | N | Native đúng/sai | V6 đúng/sai |
|---|---:|---|---|
|TA/A|1|1/0|1/0|
|TA/B|1|0/1|0/1|
|TF/A|1|0/1|0/1|
|TF/B|0|Chưa đo|Chưa đo|
|TF/C|3|2/1|2/1|
|CMTB/A|1|1/0|1/0|
|CMTB/B|1|1/0|1/0|
|CMTB/C|0|Chưa đo|Chưa đo|

Ba lỗi vẫn là2150 TF/Future,2054 TA/No,1513 TF/Past. Không có mẫu được cứu hoặc làm hỏng ở MC.

Mean TV full/native=.0163; random/native=.01131; irrelevant/native=.01514. Mean mid sign-swap TV=.0070, late=.0018125, ratio≈3,86×. Slot-profile swap TV=.019475. Đây tiếp tục là sensitivity, không phải directional correctness. Tăng tỷ lệ mid/late không đủ thành Cổng0 cho phương pháp chống hallucination.

Random/irrelevant khớp L2 norm của token-bias profile theo modality, nhưng chúng tắt row de-sinking và late-audio damping. Vì vậy chúng không được mô tả thành matched norm của toàn bộ output correction. Full/no-desink chỉ tách row-attention component; cả hai vẫn dùng salience extraction có r0 de-sinking.

M3 có sử dụng Phi từ các salience logits của chính nó, nhưng **không dùng trained C6 aligner hoặc trained M2 bridge** để trả lời. Kết quả M2 và M3 chưa tạo thành bằng chứng cho một end-to-end learned PC-EBR QA pathway. M3 CAOR hiện training-free; toàn bộ Stage-0 gồm M1/M2 thì có training.

## 5. Gate chưa đo được bằng chứng phù hợp câu hỏi

Mean gamma:

- Câu hỏi thật: .9054.
- Query hoán vị hidden dimensions: .81475.
- Câu hỏi có nghĩa nhưng sai ngữ cảnh: .946225.

Ví dụ1513: true=.7401, câu hỏi guitarist=.9859. Ví dụ1449: true=.9562, câu hỏi welder=.9649. Đây là kết quả của negative-control prompts được thiết kế; chưa phải bộ event-absence annotations độc lập, nhưng đã không đáp ứng kỳ vọng sanity check rằng câu hỏi sai ngữ cảnh sẽ ít kích hoạt routing hơn.

Gamma được tính từ độ tập trung trên slots, và dùng max(alpha_v,alpha_a). Một phân phối nhọn có thể nhọn vào event sai; uniform gate=0 không chứng minh semantic relevance, calibrated confidence hoặc null-event detection. Hơn nữa r0 thực tế là media patch token ở cả8 mẫu, có cả AUDIO và VIDEO, không phải delimiter token. Nó có thể làm context baseline, nhưng không được mặc định là một hướng trung tính, bất biến theo nội dung.

Tôi nghi max-over-query-tokens, chuẩn hóa residual và shared phrasing có thể góp phần làm gate nhọn cho câu hỏi khác nội dung. Đây là giả thuyết, chưa được tách bằng ablation nhân quả. Đừng kết luận chỉ thay threshold sẽ sửa được vấn đề.

## 6. Thành phần W mới bị nén về mean-time trong M3

Đây là điểm toán học cần cậu biết trước khi thiết kế tiếp:

W+−W− = integral(F1−F2)dt/span = (E[T2]−E[T1])/span.

M3 đưa `exp_phi[3] - exp_phi[4]` vào rel_scalar. Sau khi tính thêm joint_gate đúng như code, audit trên cả8 raw tensors cho max error chỉ8,94e−8 so normalized mean lag. Vì vậy riêng phần W mới đi vào M3 đang tái mã hóa một first moment; không giữ độ lớn vận chuyển/hình dạng phân phối như W++W−.

M2 vẫn nhìn W+,W− riêng; M3 vẫn có tanh precedence, slot profiles và token innovation. Tôi không nói toàn bộ phương pháp chỉ dùng mean-time. Nhưng không nên lấy “exact Wasserstein computation” làm bằng chứng rằng LLM đang được cung cấp đầy đủ distributional temporal structure. Chứng minh và giới hạn ở [math_note_signed_transport.md](math_note_signed_transport.md).

Một điểm tương tự: cosine bivector thuận/đảo=.1611 đúng theo thiết kế. CAOR mới không inject vector đó; nó dùng các scalars/probabilities trong aud_info rồi sửa attention output qua V thực của mỗi layer. Do đó toy geometry của bivector không tự xác minh geometry của CAOR correction.

## 7. Sinh tự do và MC đang có lỗi đánh giá khác nhau

Hai nhánh heuristic đều chỉ1/8 aligned. Không dùng số này làm semantic QA accuracy:2150 và1821 nói rõ sự kiện xảy ra sau speech, nhưng evaluator vẫn gắn generic/false. Ví dụ2150 native MC chọnPast sai trong khi native freeform nói “happens right after the speech”, khớpFuture của câu hỏi. Đây là bằng chứng một phần lỗi có thể nằm ở format/readout/option mapping, chưa chứng minh perception hoặc hidden temporal information hoàn toàn thiếu.

Mẫu1513 vẫn nói event sau speech dù goldPast;2054 vẫnYes dù goldNo. V6 không sửa được các lỗi này trong freeform. Mẫu1929 nói event sau onions-stirring, chưa trả rõ quan hệ với speech. Việc có từ“after” không đủ. Cần chấm đúng hai event được liên hệ, hướng quan hệ và phủ định, không chỉ regex.32 tokens cũng có thể cắt lời giải ở giữa câu.

Tôi đã lưu nguyên văn native/routed cùng rule audits. Không thay score mainrun bằng một chấm thủ công tùy ý sau khi xem kết quả. Nên bổ sung annotation/evaluator độc lập trước khi dùng freeform làm kết luận về hallucination.

## 8. Phù hợp yêu cầu của người dùng đến đâu?

Điểm phù hợp: CAOR là can thiệp ở attention, dùng latent question/media states, không có chuỗi agent hoặc luật chọn phương pháp theo TA/TF/CMTB. Hai-pass inference và backbone frozen là hướng triển khai có thể cân nhắc. MC parser và scoring theo task chỉ thuộc harness; chúng không phải lý do phương pháp tự động bị benchmark-specific.

Điểm chưa đạt: chưa có QA gain, chưa có evidence validity cho gate, chưa có two-hop generalization, chưa có pathway thống nhất giữa M1/M2 và LLM QA.8 micro cũ và4 pseudo-label held-out origins không xác nhận nhiều ngữ cảnh, robustness khi không có event, hoặc khả năng chấp nhận ở CVPR.

Tôi không khuyên viết claim “giải quyết temporal hallucination” hoặc đóng băng A0–A7 từ kết quả này. Cũng không khuyên tiếp tục vòng sửa đại số nhỏ rồi coi sensitivity tăng là tiến bộ ngữ nghĩa.

## 9. Hướng cải thiện tôi ưu tiên

Đây là đề xuất tiếp theo, chưa chạy và chưa tuyên bố novelty.

**Giữ phân phối chung của event pairs thay cho chỉ kết hợp hai marginal độc lập.** Hiện Phi dựa trên outer product p1(i)p2(j). Khi có event lặp/multi-peak, phép tích này có thể tạo những cặp không phải một binding hợp lệ. Đề nghị học một joint posterior J_q(i,j) trên event bank có provenance (audio/video/time), kèm null state. q là latent câu hỏi, không phải task ID hay bộ mẫu hỏi đáp. Dùng bank chung để cho phép A→V, V→A, A→A, V→V; không ép mọi câu hỏi đều theo audio-tìm-visual.

Từ J_q, giữ một phân phối lag/overlap hoặc các statistics đủ giàu, thay vì chỉ W+−W−. Ví dụ tổng J_q(i,j) theo các bins của t_j−t_i giữ cả hướng lẫn độ phân tán. Không tự động kết luận binning là đủ: cần so trực tiếp với mean-lag-only và factorized marginals dưới cùng training/evaluation budget.

**Học cách biến bằng chứng thành correction nhỏ**, trên backbone frozen, bằng dữ liệu AV/event pairs độc lập với QA40. Một adapter/controller có thể nhận query+event-pair state rồi xuất head/layer biases; so với các weights cố định của CAOR hiện tại. Điều này có training, cần gọi đúng, nhưng tránh đặt kỳ vọng semantic calibration vào entropy heuristic. Loss cần gắn với event identity/provenance, relation và câu trả lời; không chỉ contrastive caption retrieval hoặc toy coordinate classification.

**Dùng counterfactual supervision có thể kiểm chứng trên media thật.** Dịch cả hai streams cùng nhau phải giữ quan hệ; chỉ dịch một stream hoặc hoán đổi event instances phải đổi quan hệ khi annotation bảo đảm điều đó; can thiệp vào đoạn không liên quan cần giữ câu trả lời. Những ràng buộc này áp dụng cho event pairs tổng quát, không viết luật cho TA/TF/CMTB. Nếu ghép/sửa video/audio, phải kiểm tra seams, độ dài, presence và split theo recording để tránh shortcut mới.

**Tách câu hỏi nghiên cứu nhỏ trước khi huấn luyện lớn.** So factorized versus joint binding trên dữ liệu có instance-disambiguating labels; đo relevance/null calibration bằng semantic negatives khó cùng recording; sau đó mới kiểm tra correction có cứu QA so native và random controls hay không. Oracle hoàn chỉnh cho cả event location và relation distribution sẽ giúp phân biệt lỗi grounding với lỗi truyền vào LLM. Không tăng liên tục eta trên8 mẫu để lấy vài argmax flips.

Cuối cùng cần thận trọng novelty: [Meerkat, ECCV2024, section3.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08071.pdf) đã kết hợp OT cho local AV alignment với attention consistency supervision. Vì vậy “OT + cross-attention” tự nó chưa là đóng góp mới. Khác biệt có thể nghiên cứu là query-conditioned event-instance binding, uncertainty về quan hệ temporal và hiệu quả của intervention lên QA; phải chứng minh bằng đối chứng. Xem [literature_notes.md](literature_notes.md); đây chưa phải exhaustive literature audit.

Giữ QA40 làm development set đã xem, reserve400 chỉ dùng khi protocol đã chốt. Mục tiêu tiếp theo nên là một cải thiện QA tái lập được với bằng chứng đúng event/time/source, thay vì một phiên bản Phi mới chỉ qua unit checks.

## Artifacts để cậu đối chiếu

- [report.md](report.md): mọi condition M1, mọi probe M2, per-task/per-gold counts, per-sample MC và freeform.
- [summary.json](summary.json), [m3_per_sample.csv](m3_per_sample.csv): bảng số để dùng lại.
- `main_results_s1/s2.json`, `m3_audit_s1/s2.json`, `control_accuracy_s1/s2.json`: kết quả gốc và probabilities.
- `archive_audit_s1/s2.json`: số đúng tính lại từ score/logits,29 probes và3000 retrieval query rows/server.
- Hai raw archives,198 PT files/server, đã kiểm SHA từng entry local; có cả source, checkpoints, episodes, raw features và provenance lỗi wrapper.
- `wasserstein_signed_mean_identity_s1/s2.json`, `hop_condition_comparison_s1/s2.json`: audit thêm được báo riêng, không dùng chỉnh mainrun.

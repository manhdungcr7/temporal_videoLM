# Gửi Yoshua Bengio — cần làm rõ Stage-0 v6 CAOR-WEMD trước khi chạy

Yoshua, theo yêu cầu của người dùng, tôi kiểm tra đặc tả trước khi hiện thực/chạy. **Tôi dừng ở bước này: tệp nhận được chỉ có proposal, chưa có source v6; đồng thời có những lựa chọn ảnh hưởng trực tiếp đến kết quả chưa được xác định. Chưa chạy thí nghiệm v6, chưa sửa hoặc tự viết thay mã của cậu.**

Tệp đã nhận là `proposal_original.txt`, 29,357 bytes, SHA256 `16a414ff3cff51119b1f0669391d346a0ed8973b359dc97ab01406e28ea76b27`. Proposal nói source trên Canvas, nhưng attachment không chứa khối Python và workspace không có `stage0_pcebr_v6_caor_wasserstein.py`. Vì vậy hiện chưa thể kết luận code và proposal khớp hay không. Một số hàng trong bảng sáu vòng cũng kết thúc giữa biểu thức/cụm token; cần bản đầy đủ nếu phần bị cắt có chứa công thức triển khai.

## 1. Source thực tế nào cần được kiểm toán?

Nhờ cậu gửi toàn bộ file `stage0_pcebr_v6_caor_wasserstein.py` cùng các dependencies mới, cấu hình mặc định và entrypoint. Nếu đây là patch trên v5, cần exact base revision/hash và patch. Source cần chứa đầy đủ M0/M1/M2/M3, không chỉ nhắc tên module trên Canvas. Sau khi có code, tôi sẽ đối chiếu từng công thức, mask, dtype, controls và đường gọi; hiện chưa xác nhận có lỗi code v6 cụ thể.

## 2. Công thức evidence profile và cường độ CAOR chính xác là gì?

Proposal nêu `beta_ev^(m)(j)` tổng hợp từ slot distribution, token innovation và cửa sổ dịch chuyển thời gian, nhưng chưa định nghĩa hàm tổng hợp đó. Nhờ chốt rõ:

- Nguồn tạo `p_hat_m`, gate và innovation; dùng tầng nào và tensor trước hay sau attention/MLP?
- Từ xác suất trên slot sang profile từng token như thế nào, có chia số patches/tokens trong slot không?
- Cửa sổ dịch có tâm, độ rộng, dấu dịch và quy tắc xử lý biên nào? Cách chọn chiều A→V/V→A hoặc đồng thời cả hai phải dựa vào input, không dựa vào gold/task label.
- Các giá trị cố định `eta`, `kappa_sink`, epsilon, hệ số profile, gain/gate và cách normalize/clip chúng. Proposal mới cho `beta_sink=.65`.
- Công thức chính xác tách full, rel-only, order-only, reversed, irrelevant và random; random phải khớp chuẩn ở tensor nào, tính riêng hay chung theo row/head/modality?

Các lựa chọn này có thể đổi phương pháp và điểm QA; tôi sẽ không tự đoán chúng từ mô tả định tính.

## 3. Chia lại mass audio/video thế nào, và bảo toàn so với baseline nào?

Điều kiện `M_tilde_v + M_tilde_a = M_tot` chưa xác định duy nhất hai mass mới. Cậu muốn giữ riêng từng modality (`M_tilde_m = M_m`) hay chuyển mass giữa audio và video? Nếu chuyển, nhờ cung cấp hàm phân bổ, gate, smoothing, giới hạn chuyển và xử lý khi một modality có mass0/gần0. Hai cách này cho attention output khác nhau dù cùng bảo toàn tổng media mass.

Claim bảo toàn non-media cần xác định phạm vi: bảo toàn các phần tử non-media trong **cùng một attention matrix được tính từ Q/K/V hiện tại**, hay so với native forward không can thiệp xuyên suốt mọi tầng? Phương án đầu có thể hiện thực bằng phép copy. Phương án sau không được suy ra từ ràng buộc mass, vì steering ở tầng trước làm Q/K/V và attention tự nhiên của tầng sau thay đổi. Tương tự, không suy bảo toàn cú pháp/ngữ nghĩa từ bảo toàn một khối xác suất.

Ngưỡng `<1e-15` áp cho M0 float64 hay runtime FP16/float32? Nhờ chốt dtype và tolerance riêng, có zero-strength replay kiểm tra parity với backend SDPA đang dùng. Cần giữ nguyên đúng causal mask, attention scaling và các bước runtime khác khi thay backend attention.

## 4. “Một forward duy nhất” tính những bước nào?

Nếu profile/gate lấy từ q18/q27 như v5 nhưng CAOR bắt đầu ở block14, thì tại block14 trong forward đầu tiên chưa có q27. Nhờ cậu xác nhận một trong các thiết kế:

- Có native prepass lấy evidence rồi một routed pass: khử sink không cần null-prompt pass, nhưng QA vẫn cần hai passes.
- Profile được xây online chỉ từ tensors đã có ở tầng hiện tại/tầng trước; cần chỉ rõ công thức và đường phụ thuộc.
- Một cách thực thi khác có thể kiểm tra được.

Đo wrong-query bằng forward bổ sung có thể là diagnostic riêng; cần tách nó khỏi chi phí inference phương pháp chính. Tôi không tự gọi toàn pipeline là one-pass chỉ vì không dùng null_q.

## 5. r0 là token nào, và vì sao nó chỉ đo sink cấu trúc?

`last_media` trong cache là chỉ số gắn với media; chưa đủ để xác nhận chính nó là token đóng `<|vision_eos|>` đứng sau toàn bộ cả audio lẫn video. Nhờ định nghĩa chọn r0 bằng input IDs và media masks: token cuối là audio/video embedding hay delimiter, có thấy đủ cả hai modality dưới causal mask không, và xử lý AV interleaving thế nào?

Việc r0 không nhìn câu hỏi chỉ chứng minh attention ở r0 không phụ thuộc câu hỏi tương lai. Nó vẫn có thể mang nội dung media, modality/position bias và self-attention đặc thù của r0; chưa chứng minh đó là pure structural sink. Cậu muốn coi đây là giả thuyết có đối chứng, hay có kiểm tra độc lập nào để tách sink khỏi event evidence? Cần chỉ rõ cách xử lý self-token và các vị trí causal-inaccessible khi chuẩn hóa p(r0,media).

## 6. Phi_EMD dùng chung trục thời gian nào trong M2?

Trong source v5, `build_connected_m2_episodes` tạo `iv_1/iv_2` trực tiếp từ `examples[i]['start'/'end']` theo source. Đoạn đó không ép cả hai banks thành cùng lưới K bin đều, cùng số bins, cùng boundaries. Vì vậy lập luận ba ma trận đầu đều là I_K chỉ đúng dưới giả định shared equal disjoint grid; chưa thể áp mặc nhiên cho toàn M2, nhất là Roll/coordinate-shuffle. M3 cũng làm tròn interval boundaries và điều chỉnh biên cuối; claim exact zero cần gắn với grid cụ thể được dùng.

Nhờ cậu chỉ rõ v6:

- Nếu M2 chuyển sang common grid, dùng bin edges nào và đẩy khối lượng interval/retrieval lên grid bằng quy tắc gì? CDF phải cùng physical time support, không chỉ cùng candidate rank.
- Nếu giữ hai banks có interval khác nhau, tính W_plus/W_minus bằng hợp các support points và trọng số khoảng cách như thế nào?
- Trục u chuẩn hóa có bước1/K hay1/(K−1), endpoints/centers ra sao, và phần trừ uniform được tính trên chính các support ấy thế nào?
- Reversal/Roll biến đổi content, coordinate và nhãn ra sao để giữ đúng đối chứng v5?

Sáu features khác công thức chưa tự bảo đảm full rank trên mọi miền dữ liệu hoặc linear separation hoàn hảo. Khi retrieval uniform, joint gate0 vẫn làm Phi_EMD0 cho cả During bất định và Foil. Nhờ hạ claim “không bao giờ nhầm” thành giả thuyết kiểm chứng và nêu miền/tiêu chí rank cùng matched no-Phi control; không sửa tiêu chí sau kết quả.

## 7. C6, bridge và bảng ablation sẽ chạy theo protocol nào?

C6 “giữ nguyên nhánh C3b” có nghĩa đóng băng checkpoint C3b và chỉ học innovation, hay train chung hai nhánh? Nhờ chốt initialization, loss, weights, seeds, steps và trainable parameters. C8b chỉ shuffle innovation nhưng giữ global branch có semantic signal là một control có ích; cần mô tả đúng nó kiểm tra contribution innovation, không phải phá mọi semantic alignment.

Mixture-aware bridge cần cùng teacher, scale, composite-query mixture và candidate construction giữa train/test. Nhờ định nghĩa B8a teacher-forced/B8b oracle và mọi chỉ số hop1/hop2: gold instance hay multi-positive, denominator và cách xử lý ambiguous/missing event. Proposal cuối nhắc B9 nhưng chưa định nghĩa B9.

M1/M2 vẫn là probes độc lập hay sẽ dùng trực tiếp trong CAOR QA? Nếu dùng, nhờ chỉ rõ đường nạp model và đường tensor từ query tùy ý vào decoder. Không tự nối classifier bốn lớp vào mọi câu hỏi. Đồng thời ghi rõ phần nào training-free, phần nào có học.

## 8. Claims từ v5, controls ngữ nghĩa và nguồn tham khảo cần sửa thế nào?

Đo TV cao hơn trong một nhóm8 mẫu không chứng minh temporal alignment đã đúng event/time/source. Đặc biệt, đáp án CMTB là hành động đang thấy, không tự là annotation rằng narrated event và visual event đồng bộ; không gọi hai CMTB samples là synchronous nếu chưa có annotation xác nhận. Bản sửa VO từng tầng chúng tôi đã chạy cũng không giúp MC: native/F1/F2/F3 đều5/8, nên cần phân biệt sensitivity với directional correctness.

Wrong-query welder là một diagnostic có thể dùng, nhưng câu hỏi khác chủ đề không tự là absence label độc lập. Xin xác nhận nó chỉ dùng làm control, không chọn thuật toán/đáp án theo bối cảnh cooking. Với free-form, cần thêm case đúng event nhưng sai reference (ID1821 sau layerfix), không chỉ ba phản ví dụ cũ. Passing ba ví dụ không biến clause heuristic thành semantic accuracy.

Tôi đối chiếu ba links và thấy gán sai tên:

| Link trong proposal | Nội dung thực tế của trang gốc |
|---|---|
| D19-1006, gán “Attention is not not Explanation” | [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://aclanthology.org/D19-1006/) |
| arXiv2503.20215, gán “VTAR: Visual-Time Attention Reweighting” | [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) |
| arXiv2505.20444, gán “Anisotropy & Whitening” | [HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](https://arxiv.org/abs/2505.20444) |

Nhờ cậu sửa metadata và claim được nguồn hỗ trợ; tôi chưa xác minh toàn bộ các citations còn lại và không gọi các nguồn đó là đã được kiểm chứng. Đây là lỗi nguồn tham khảo, không phải bằng chứng bản thân CAOR không thể có ích.

**Nhờ gửi source đầy đủ và trả lời các lựa chọn chính ở câu2–6 trước. Sau khi code và đặc tả khớp, chúng tôi mới chạy frozen252 pairs và micro8 đúng protocol đã thống nhất. Trong lượt này chưa có kết quả v6 để báo cáo.**

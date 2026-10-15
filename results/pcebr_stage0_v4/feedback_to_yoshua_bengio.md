# Gửi Yoshua Bengio — kết quả và phản biện CQ-TSPR / Stage-0 v4

Yoshua, chúng tôi đã chạy các hàm v4 bạn gửi trên cả hai server. **C3 cải thiện retrieval và shrinkage sửa được phản ví dụ nhiễu gần uniform. Tuy nhiên, M2 đầy đủ thua đối chứng tốt nhất ở cả ba suites; M3 có lỗi cấu trúc khiến vector gần như bất biến khi đảo thời gian. Trên 8 mẫu MC micro, routed giảm từ 5/8 xuống 2/8. Chưa đủ bằng chứng để chuyển A0–A7.**

Bảng đầy đủ, từng câu đúng/sai và văn bản sinh ở [report.md](report.md). Chúng tôi lưu source gốc, checkpoints, similarity matrices, episodes, classifier logits, M3 banks và xác suất đáp án. Tên Yoshua/Yann là quy ước vai trò trong trao đổi, không phải xác nhận các nhà khoa học thật là tác giả mã này.

## 1. Phạm vi chạy và khả năng tái lập

Hai runtime T4 mới đều khỏe. M1/M2 dùng cùng 252 pseudo pairs: 202 train từ 16 recording origins; 50 held-out từ 4 origins đã được xem qua nhiều vòng. Ba seeds cố định 20261015/16/17; 30 lượt huấn luyện projector/server, 180 steps/lượt; 21 linear probes/server, 200 steps/probe. Kết quả M1/M2 hai server trùng toàn bộ. Đây là hai replicas; số nguồn held-out độc lập vẫn là 4.

M3 giữ đúng 8 ID, 4/shard, rotation 0: 1449, 2150, 1929, 2054 và 1956, 1513, 2264, 1821. Không thay mẫu khó. Chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.

Source gốc và alias trong server_jobs có cùng SHA. Wrapper chỉ lưu kết quả trung gian; không sửa công thức, loss, nhãn, seeds hoặc siêu tham số. V4 không cần devicefix/autograd repair. Hai mainruns hoàn tất khoảng 429 và 419 giây. Mỗi server có 111 raw PT files; 1.500 retrieval query rows được tính lại khớp báo cáo. Hai archives và SHA từng entry đã được kiểm tra local.

Các pair caches đều ở block index 27. Native là post-media readout state; pooled là trung bình media states. Không gộp hai loại này thành cùng một phép mean-pooling không gian. Các mô tả là pseudo labels do model sinh, chưa phải annotation event/time độc lập. 31 Nonnull queries chỉ được xác định bằng substring filter, không phải 31 sự kiện đã được xác nhận tồn tại.

## 2. Những phần có tiến bộ và nên giữ

Shrinkage qua đúng phản ví dụ logits 1e-4: trọng số hỗ trợ lớn nhất từ 1.0 ở v3 xuống khoảng .2; joint gate ≈1.046e-14; max|Phi| ≈2.259e-21. Với peak mạnh, joint gate ≈.9922 và signed relation ≈1.5918. Đây là sửa ổn định số hợp lý. Nó chưa chứng minh một peak nhọn tương ứng với định vị đúng event.

C1 thực sự nối sang M2 Natural/Reversal qua sim_mat. Roll dùng C1 teacher encoders đã detach/freeze; không lặp lại lỗi autograd của v3. B1 nay dùng raw softmax mean-time riêng, không cắt dương hoặc ghép null. Đối chứng này giúp đánh giá phần mới rõ hơn.

C3 là kết quả M1 đáng giữ:

| Cache | C1 Global | C3 Global | C1 Exact | C3 Exact | C1 Multipos | C3 Multipos |
|---|---|---|---|---|---|---|
| Native | [11,11,11]/50 | [15,15,15]/50 | [16,16,16]/50 | [20,20,20]/50 | [21,21,21]/50 | [26,26,26]/50 |
| Pooled | [16,16,16]/50 | [22,21,21]/50 | [19,19,19]/50 | [21,20,20]/50 | [23,23,23]/50 | [24,23,23]/50 |

Native tăng đồng thời Global và Exact 8 điểm phần trăm, Multipos 10 điểm. Pooled Global tăng 10,67 điểm, nhưng Exact chỉ 2,67 điểm và Multipos 0,67 điểm. C3 xứng đáng làm baseline tiếp theo; chưa đủ để gọi là bước đột phá về binding.

C3 trong code chỉ chuẩn hóa từng chiều bằng mean/std của tập train, không tính full covariance whitening. Nó còn đổi within-rec loss weight từ 0 của C1 sang .20; C2 dùng .25. Vì vậy chưa cô lập được đóng góp riêng của whitening. Cần đối chứng cùng loss weight trước khi quy toàn bộ gain cho phép chuẩn hóa. Chưa huấn luyện thêm đối chứng đó trong lượt này.

Pooled C8 shuffle Exact [18,18,20] vẫn gần C1 [19,19,19]; Global C8 seed 3 đạt 23/50, còn hơn C3. Đừng xem gain trung bình trên 4 origins cũ là xác nhận generalization.

## 3. M2: bản đầy đủ vẫn thua baseline đơn giản

| Suite | N held-out episodes | Full 4-way | Đối chứng tốt nhất 4-way | Gap | Full 3-way | During full |
|---|---:|---:|---:|---:|---:|---:|
| Natural | 232 | 37,9% | Raw softmax mean-time 47,0% | −9,1 điểm | 51,7% | 62,1% |
| Global reversal | 576 | 41,3% | Raw softmax mean-time 43,2% | −1,9 điểm | 53,7% | 63,9% |
| Cross-modal two-hop roll | 156 | 21,2% | Coordinate shuffle 28,8% | −7,6 điểm | 24,8% | 15,4% |

Natural Phi-only đạt 44,0%, hơn full 37,9% nhưng vẫn dưới B1 47,0%. Reversal paired-both-directions của full đạt 53,7%, dưới B1 59,3%. Thêm calibrated Phi/null chưa chứng minh lợi ích vượt cách đọc mean-time đơn giản.

Natural/Reversal full vượt một số controls wrong-query và coordinate-shuffle, và During tốt hơn v3. Đây là tín hiệu có ích khi dùng teacher C1 tốt hơn. Nhưng gain đó chưa chuyển thành ưu thế trước tất cả matched controls. Roll full thấp hơn mức ngẫu nhiên 25% ở 4-way và 33,3% ở 3-way. Phép hợp thành hai query rồi trừ projection hop 1 vẫn là heuristic cố định; chưa chứng minh learned two-hop reasoning.

Roll v3 có 752 episodes, v4 có 156 vì đã đổi điều kiện chọn cặp cross-modal/interior. Không so hai accuracy như cùng evaluation set. Các episodes vẫn phụ thuộc cùng 4 held-out origins. 3-way dùng argmax riêng trên ba relation logits; nó không phải accuracy của 4-way sau khi bỏ foil.

Uniform đạt 25%/33,3% ở cả ba suites, đúng đối chứng mong muốn. Tuy nhiên Uniform Roll có During recall 100% vì dự đoán During cho mọi episode. Không diễn giải recall này là năng lực temporal.

### Claim về foil không khớp thực chạy

Train chỉ 5/16 origins tìm được strict candidates; 11/16 phải fallback. Held-out là 0/4 strict và 4/4 fallback, lặp lại ở mọi suite. Claim “100% strict verified, 0% fallback” trong đề xuất bị kết quả chạy bác bỏ.

Ngay cả khi cosine đã center <.15 và Jaccard=0, đó vẫn chỉ là khác biệt văn bản theo hai phép đo. Nó không loại trừ event ngoài caption, từ đồng nghĩa hoặc mô tả thiếu. Cần annotation/verification độc lập về sự hiện diện và vắng mặt. Đừng tiếp tục đổi threshold để đạt tỷ lệ foil đẹp trên 20 origins đã xem.

Hàm classify_interval_relation đã được dùng thống nhất; assertions Roll chạy qua. Đây là sửa đúng. Nhưng During còn bao gồm windows chạm biên hoặc có center cách nhau khoảng 2 giây: đó là quy ước của coarse slots, chưa phải quan hệ ground truth của hai event thật. Đảo/roll cached features cũng chưa tương đương biến đổi waveform/frames rồi chạy lại encoder.

## 4. M3: hai lần đổi dấu làm mất thông tin hướng thời gian

Với phép đảo P trên lưới đối xứng, build_symmetric_routed_vector có tính chất:

1. Hỗ trợ mới p′=Pp và bank mới V′=PV, nên tổng p′kV′k bằng tổng pkVk. Phần ev_bound giữ nguyên.
2. rel_scalar đổi dấu khi đảo thời gian.
3. chron_dir, được tính từ chênh lệch trạng thái cuối và đầu, cũng đổi dấu.
4. Tích rel_scalar×chron_dir giữ nguyên. Vector cuối normalize(ev_bound+.45×rel_scalar×chron_dir) vì vậy bất biến dưới đảo.

Giữ đúng event identity là cần thiết. Nhưng cách nối hai thành phần cùng mang dấu đã triệt tiêu chính bit hướng muốn truyền. Đây là lỗi thiết kế readout temporal, không phải bằng chứng backbone mù thứ tự.

Các phép kiểm tra đã xác nhận điều này:

- Trên 9 toy cases, K=5/6/9 và 3 seeds, max sai khác vector chỉ khoảng 1e-8.
- Trên 8 real banks, cosine thuận/đảo đều làm tròn thành 1,0; max sai khác từng tọa độ ≤8,464e-6.
- Mean swap-TV late=.000675, mid=.001100. Tỷ số khoảng 1,63 chỉ chia hai lượng rất nhỏ; nó chưa thể hiện khả năng đọc đúng hướng thời gian.
- Sai khác nhỏ có thể đến từ làm tròn interval endpoints .001s và tính toán số. Nó không khôi phục một channel temporal có ý nghĩa.

Tôi giữ nguyên công thức trong mainrun để đánh giá đúng đề xuất. Bước tiếp theo phải giữ content và order đọc được độc lập. Một ordered representation cần bảo toàn event identity khi đảo, đồng thời có một temporal channel biến đổi theo quy tắc đã biết. Phải kiểm tra quy tắc đó bằng unit test trước khi nối vào decoder. Đây là yêu cầu thiết kế, chưa phải phương pháp mới đã được xác nhận.

### Cổng confidence vẫn can thiệp khi uniform

Code dùng max(.35,min(1,max(alphaV,alphaA))). Vì có floor .35, uniform vẫn được inject .22×.35=.077 norm ở mỗi một trong 6 layers. Nó không có tính chất “thiếu bằng chứng thì không can thiệp” như mô tả. Trên 8 mẫu thực, gates đều khoảng .96–.98, tức gần như luôn can thiệp mạnh.

Mean TV của routed so native=.253313, hơn random=.212988 và irrelevant=.160538. Điều này chỉ xác nhận ảnh hưởng lớn hơn lên phân phối đáp án. Trong khi đó MC accuracy giảm, nên không thể gọi TV lớn là tốt hơn về reasoning.

Chọn patch theo stem phù hợp hướng tổng quát hơn query-blind pooling. Tuy nhiên cosine của hidden states giữa các modality chưa tự là event verifier. Nó có thể phản ứng với từ câu hỏi hoặc narration đã đi vào question states. Cần đo trực tiếp event localization và wrong-query sensitivity bằng nhãn độc lập.

## 5. “Physical Clock Oracle” chưa khôi phục được frame PTS

Trích xuất các bậc liên tiếp của temporal position giúp nhận biết grid order; kiểm tra patches/grid cũng hữu ích. Nhưng ánh xạ [gT/G,(g+1)T/G] chỉ giả định chia đều tổng duration. Nó chưa dùng sampled frame indices, frame PTS, fps thực, temporal patch grouping hoặc padding. Nhiều tập timestamps khác nhau có cùng G và T; hai đại lượng đó không xác định duy nhất thời điểm lấy mẫu.

Code ghi log audio span và counts uniform, chưa assert các điều kiện hứa hẹn. Audio span được tính từ position span, khác N_a/25 nêu trong prose. Audio còn được rescale theo min(T,audio_span), nên không giữ nguyên chính xác bước .04 giây cho từng token.

Phát hiện v3 là sec suy từ position_ids chưa được xác nhận là physical clock. Nó chưa chứng minh Qwen có bug đồng bộ nội bộ hoặc nguyên nhân là second_per_grid. V4 chưa đo được một event thật ở 12 giây bị model đọc thành 15,5 giây. Cần trace token→grid→sampled frames→PTS. Việc mapping luôn nằm trong duration là điều công thức cưỡng chế, không phải bằng chứng mapping đúng.

## 6. Lỗi đọc gold và chất lượng câu trả lời thực

Manifest dùng trường gold; script đọc sample.get('answer','A'). Vì thiếu answer, cả 8 mẫu đều được gán A. Sai 5/8 nhãn: 2150, 1929, 2054, 2264, 1821. Đặc biệt 2054 phải là B/No. Lỗi nằm ở audit sau generation, không ảnh hưởng forward hoặc M1/M2.

Chúng tôi giữ nguyên raw outputs/audits, chấm lại từ xác suất và văn bản đã lưu bằng manifest gold. Không chạy lại model. So sánh trước/sau ở [corrected_gold_audit.json](corrected_gold_audit.json); bản [stage0_pcebr_v4_goldfix.py](stage0_pcebr_v4_goldfix.py) chỉ sửa lookup này để dùng về sau.

| Task/gold | N micro | Native đúng/sai | Routed đúng/sai |
|---|---:|---|---|
| TA/A | 1 | 1/0 | 0/1 |
| TA/B | 1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 0/3 |
| CMTB/A | 1 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo |
| Tổng | 8 | 5/3 | 2/6 |

Đây chỉ là MC micro rotation 0. Không so 5/8 hay 2/8 trực tiếp với 18/40 hoặc 26/40 trước đây dùng cyclic option rotations.

Regex không có timestamp vượt clip chưa đủ chứng minh giảm hallucination. Một câu trả lời bỏ hẳn quan hệ thời gian luôn qua check này. 1449 native trả Yes và nêu event; routed trả “The scene is being shown.”, không còn trả lời câu hỏi. Với TF 2150/1513, routed chuyển sang mô tả hành động và không trả lời trước/trong/sau.

Native 2150 bản 24 tokens đã nói “right after the speech”, không sinh 1:00 như bản 12 tokens v3, dù vẫn là native. Không được quy thay đổi đó cho CQ-TSPR. Mẫu 2054 native mở đầu Yes là sai gold No; routed không trả Yes/No, cũng chưa thể coi là sửa đúng.

Cần chấm correctness, mức độ trả lời đúng yêu cầu, groundedness và abstention cùng nhau. Tăng/giảm một regex flag có thể thưởng cho câu trả lời chung chung. Một số câu còn bị cắt ở token 24, nên chưa thể dùng các cờ tự động này làm đánh giá toàn diện.

## 7. Các khẳng định literature và cơ chế cần sửa

Đã kiểm tra nguồn chính thức; xem [literature_audit.md](literature_audit.md):

- Link gắn tên “SynchOnly”, arXiv:2509.23142, là paper “Exploring Magnetic Phases in Dual-Species Mott insulating Spinor Lattice Gases”.
- Link gắn tên “AV-Haystack/LART”, arXiv:2410.18514, là “Scaling up Masked Diffusion Models on Text”.
- VideoRoPE và HoPE đúng là papers về positional embeddings cho video VLM. Chúng tạo động cơ nghiên cứu, chưa xác nhận chẩn đoán nhân quả trên Qwen2.5-Omni-3B và các clip này.
- Ethayarajh 2019 đúng nguồn về geometry của contextual embeddings; không cung cấp oracle xác nhận absence.

Cosine .9231 không đo được 90% background patches, tỷ lệ action <10% hay SNR −11 dB. Native pair cache cũng không phải mean-pool patch. DPI không chứng minh mọi projector đều bất khả thi: cần chứng minh thông tin task đã mất, điều chưa đo được. C3 v4 tăng retrieval cho thấy vẫn còn tín hiệu khai thác được. Đừng gọi việc thử projector tiếp là “vi phạm DPI”.

J1/P5 có ích nhưng khác protocol. J1 event-half dùng construction labels cho analysis và có automatic hear-then-look rule. P5 chia readout theo answer format và speech anchor. Những kết quả đó không chứng minh cosine patch trực tiếp trên full context sẽ đạt 88–90% hoặc tổng quát ngoài speech/context của benchmark. Cần phân biệt perception accuracy, QA accuracy, split và lượng thông tin hỗ trợ trước khi so sánh.

## 8. Phù hợp yêu cầu nghiên cứu đến đâu; nên làm gì tiếp

V4 có các lựa chọn phù hợp yêu cầu người dùng: không định tuyến theo tên task, can thiệp và sinh text không cần options, chọn salience theo câu hỏi, xử lý A/V đối xứng và hướng tới cơ chế nội bộ. Vì vậy chưa nên bác bỏ toàn bộ latent temporal reasoning. Nhưng cộng hai modality vào một vector chưa tự giải quyết được nhiều dạng joint reasoning; hiện chưa xác nhận event identity, source identity, order hoặc multi-hop correctness.

Novelty và khả năng được CVPR chấp nhận chưa được xác nhận. Tổ hợp cosine top-K, mean centering, gating và residual injection cần một đóng góp cụ thể cùng bằng chứng mạnh. Tên phương pháp mới hoặc tuyên bố “không còn phản ví dụ” không thay thế được kiểm chứng.

Tôi đề nghị ưu tiên:

1. **Chốt một giả thuyết có thể bác bỏ.** Giữ C1/C3 làm retrieval baselines. Trước khi can thiệp decoder, kiểm tra representation content/time/source có định luật biến đổi rõ ràng. Uniform phải dẫn tới không can thiệp; wrong query phải làm confidence giảm. Tránh hai lần đổi dấu tự triệt tiêu.
2. **Cố định bằng chứng độc lập.** Dùng một số recording origins development mới, có annotation event instances, intervals, source và absence; lưu token→sampled frame PTS. Bao gồm A→V, V→A, A→A, V→V, âm thanh không phải lời nói, nhiều instance và thứ tự khác narration. Chốt protocol trước khi xem test; giữ reserve400 cho đánh giá cuối.
3. **Kiểm tra từng mắt xích.** Retrieval chưa thắng shuffle/wrong query với nhãn thật thì sửa grounding trước Phi. Retrieval tốt nhưng relation readout thua mean-time thì dùng baseline đơn giản và chứng minh phần mới bằng matched ablation. Readout đúng nhưng QA kém mới là thời điểm kiểm tra causal transfer sang decoder.
4. **Đánh giá câu trả lời thực.** Native/routed phải dùng cùng prompt không options và token budget đủ hoàn chỉnh. Báo cáo correctness, coverage và abstention. Đừng tối đa hóa TV: random steering cũng đổi đáp án. Muốn claim temporal phải đo trả lời đúng hướng sau biến đổi hợp lệ.

Hiện phần đáng giữ nhất là numerical shrinkage và C3 retrieval. Hai kết quả này chưa tạo thành một giải pháp temporal hallucination hoàn chỉnh. Cần sửa M3 order readout và protocol ground truth trước khi mở bảng A0–A7.

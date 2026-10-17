# Gửi Yoshua Bengio — kiểm chứng OB-TVOR v5, 2026-10-16

Yoshua, chúng tôi đã chạy mã v5 bạn gửi trên hai runtime mới và bổ sung các đối chứng để tách đóng góp của presence, Phi, VO và mức can thiệp. Kết quả đầy đủ, từng mẫu và các kiểm tra artifacts nằm ở [report.md](report.md). Đây là đánh giá cấu hình cụ thể; chưa xác nhận novelty, khả năng tổng quát hoặc CVPR acceptance.

**V5 có tiến bộ thật về kiểm tra đại số và M2 Natural/Reversal, nhưng chưa xác nhận giải pháp temporal hallucination hoàn chỉnh.** Không nên bỏ các kết quả dương; cũng không nên dùng chúng để kết luận mọi mắt xích đã được sửa triệt để.

## 1. Phạm vi và khả năng tái lập

Giữ nguyên 252 pseudo pairs: 202 train từ 16 origins, 50 held-out từ 4 origins đã xem ở các vòng trước. Ba seeds 20261015/16/17; M1 có 36 fits/server, 180 steps/fit; M2 có hai bridges, 120 steps/bridge, và 27 probes, 200 steps/probe. Hai replicas cho kết quả M1/M2 trùng nhau. Không tăng số nguồn held-out độc lập bằng cách cộng seeds hoặc số episodes.

M3 giữ đúng 8 mẫu micro cũ, 4/shard, rotation 0. Không chạy QA40 đầy đủ, A0–A7 hoặc reserve400. Source gốc và alias có cùng SHA; không đổi công thức, nhãn, seeds hoặc siêu tham số trong mainrun. Các thí nghiệm bổ sung được lưu riêng dưới tên controls/exploratory.

## 2. Những phần đã sửa đúng

Phân rã even/odd không còn lỗi nhân hai đại lượng cùng đổi dấu như v4. Chín toy cases qua checks nội dung giữ nguyên, kênh order đổi dấu, hai kênh trực giao. Uniform gate bằng 0 và effective scale bằng 0 đã sửa đúng sàn .35 của v4. Gold lookup đã dùng trường gold của manifest.

Tên grid-clock interpolation và việc thừa nhận chưa phải frame PTS oracle là chính xác hơn. Assert patches/grid và audio span cũng giúp phát hiện lỗi dữ liệu. Clock vẫn suy visual timestamps bằng chia đều duration; annotation thời điểm event thật cần trace sampled frames/PTS độc lập.

M2 có baseline B1b dùng cùng C5 retrieval để so với B8. Raw affinity thêm thông tin mà softmax shift-invariant không giữ lại; đây là thay đổi có lý do toán học. Báo cáo strict/lexical/string foil proxies trung thực hơn tuyên bố v4 rằng mọi foil đã được xác minh.

## 3. M1: chuẩn hóa đường chéo đơn giản vẫn mạnh hơn ZCA/hybrid

| Cache / condition | Global mean | Exact mean | Multipos mean | Nonnull mean |
|---|---:|---:|---:|---:|
| Native C1 | 22,00% | 32,00% | 42,00% | 32,30% |
| Native C3 | 26,67% | 38,00% | 50,00% | 35,50% |
| Native C5 | 29,33% | 33,33% | 43,33% | 27,93% |
| Native C8 shuffle | 26,00% | 36,00% | 40,00% | 38,70% |
| Pooled C1 | 32,00% | 38,00% | 46,00% | 45,20% |
| Pooled C3 | 46,00% | 46,00% | 52,00% | 51,60% |
| Pooled C5 | 42,00% | 42,00% | 48,00% | 48,40% |
| Pooled C8 shuffle | 46,00% | 43,33% | 49,33% | 50,53% |

C3 pooled có tín hiệu đáng giữ. ZCA/hybrid C5 chưa thắng C3 về localization; native Nonnull C5 còn dưới C1 và shuffle. C8 hiện là shuffle của txt_diag_v4, chưa là shuffle khớp C5 ZCA/hybrid. Muốn quy kết đóng góp temporal của C5 cần đối chứng cùng transform, input mixture, loss và budget.

C2/C3 đã khớp within-rec weight .20. Nhưng C3 v5 không phải exact replay C3 v4: init_shift nay tính media mean sau _pre_whiten_media; v4 dùng media normalize trước phép whitening. Đây là thay đổi thuật toán cần ghi rõ khi so sánh hai phiên bản.

Quan trọng hơn, nhánh mod_zca của _pre_whiten_media vẫn trả:

```python
normalize((x_in - txt_mean) * txt_inv_std + 0.35 * out_zca)
```

Vì vậy mã chưa bỏ hoàn toàn việc trừ text mean trên media như phần mô tả nói. Nó là một hybrid giữa nhánh diagonal cũ và ZCA riêng modality. Nếu đây là ý định, hãy mô tả đúng và có ablation cho hai nhánh; nếu không, sửa rồi đánh giá như một cấu hình mới. Không tự gọi toàn bộ phương pháp là pure modality-specific whitening.

## 4. M2: kết quả dương cần giữ, giới hạn cần nói rõ

| Suite | C5 mean-time B1b | Full C5 B8 | Control tốt nhất | Gap B8/control | During B8 |
|---|---:|---:|---:|---:|---:|
| Natural | 46,6% | 67,7% | 54,3% | +13,4 điểm | 63,8% |
| Global reversal | 35,9% | 59,9% | 49,3% | +10,6 điểm | 59,7% |
| Cross-modal two-hop roll | 24,4% | 34,6% | 32,7% | +1,9 điểm | 10,3% |

Natural/Reversal là cải thiện đáng kể của **bài phân loại proxy 4 lớp** trong lượt này. Chúng vượt cả coordinate-shuffle và mean-time ở 4-way. Roll vẫn chưa đạt: During chỉ 4/39, và gain so control tốt nhất chỉ 1,9 điểm. Không kết luận đã giải quyết two-hop joint temporal reasoning.

Phải đọc thêm relation-only 3-way. B8 đạt 60,9/55,8/39,3%, trong khi C1 mean-time đạt 64,4/60,0/34,2%. Nghĩa là ở hai suites đầu, B8 chưa thắng baseline đơn giản tốt nhất về ba quan hệ thời gian. C5 mean-time riêng yếu hơn C1, nên thay teacher không đồng nghĩa mọi khía cạnh tốt hơn.

B8 foil recall tới 98,3/97,2/87,2%. Coordinate-shuffle cũng nhận foil rất tốt, 96,6/99,3/87,2%, dù gán thời gian đã bị phá. Presence là năng lực có ích cho giảm hallucination, nhưng gain đó không tự chứng minh đúng quan hệ thời gian. Đối chứng bổ sung quan trọng của chúng tôi là **B1b + presence, bỏ riêng Phi**, giữ C5 retrieval và cùng classifier budget; xem kết quả ở phần exploratory trong report. Nó tách hiệu quả của thông tin mức affinity khỏi đóng góp của Phi.

Đối chứng đó đạt **67,2/57,3/34,0%**, so full B8 **67,7/59,9/34,6%**. Thêm Phi chỉ tăng 1/232 câu ở Natural, 15/576 ở Reversal và 1/156 ở Roll. Natural relation-only 3-way của nhánh bỏ Phi còn cao hơn full: 63,8% so 60,9%. Như vậy phần lớn gain 4-way không cần Phi; đây là một bài học chính của v5. Giữ presence baseline mạnh và chỉ đầu tư thêm Phi khi nó thắng matched controls trên nhãn event/time độc lập.

Chẩn đoán “During và Foil đều ở vector 0” cũng cần sửa: chỉ kênh signed center difference bằng 0 khi đồng thời. Với K slots bằng nhau và support tập trung hoàn hảo vào cùng interval, overlap channel trước gate là 1−1/K sau trừ uniform, vẫn dương. Vì vậy vector Phi đầy đủ của During không mặc nhiên bằng 0. Null/presence có thể giúp, nhưng chưa thể gọi origin collision là nguyên nhân duy nhất đã chứng minh.

Foil proxy audit: train có 5 origins strict, 10 lexical, 1 string fallback; held-out có 0 strict, 2 lexical, 2 string fallback. Đã báo cáo trung thực, nhưng nhãn absence vẫn chưa độc lập. A query khác recording có thể có raw affinity thấp do nguồn/nội dung/cách mô tả, ngay cả khi event không được xác minh là vắng mặt. Cần hard negatives trong cùng recording và cùng chủ đề, có annotation presence/absence thật.

M2 v5 đổi cả train episode selection, foil selection và classifier lr/weight decay so với v4. Natural/Reversal held-out có cùng cỡ; train counts khác. Chỉ dùng matched comparisons trong v5 để quy đóng góp. Ladder cũng chưa hoàn toàn cộng dồn: B6 là presence-only C5, B5 là mean-time+Phi C1. B8 và B7 đổi teacher đồng thời; không đọc một cột tăng như hiệu quả duy nhất của một module.

## 5. Bridge có học, nhưng bài toán inference còn khác bài toán huấn luyện

Bridge được huấn luyện trên media vector của đúng index anchor, va=zm[ia]. Inference dùng v_hat_1 là trung bình có trọng số của nhiều candidates truy hồi. Đây là teacher-forced anchor ở train và noisy retrieved anchor ở test. Độ đúng hop 1 chưa được đo riêng; nếu nó sai, phép khử thành phần có thể loại sai thông tin.

Tập train ghép mọi cặp video/audio nonnull trong recording, chưa lọc tham chiếu anchor nhập nhằng. Episode generator lọc exact duplicate target descriptions, nhưng không xử lý các câu paraphrase hoặc nhiều instance cùng event; target có ít mô tả lặp hơn không đồng nghĩa query phân biệt đúng instance.

Đề xuất: đo hop-1 accuracy, hop-2 conditional accuracy khi hop-1 đúng/sai, và oracle-anchor upper bound trên cùng labels. Nếu gap chủ yếu đến từ retrieved mixtures, train bridge trên mixtures thật từ teacher đóng băng và làm đối chứng teacher-forced/mixed; không thêm module mới trước khi xác định mắt xích hỏng. Đây là thí nghiệm đề nghị, chưa chạy hay được coi là giải pháp đã xác nhận.

## 6. M3: parity đúng chưa đồng nghĩa model sử dụng order đúng

Cosine khoảng .1611 đến từ tan(theta)=.85 và công thức hai kênh trực giao. Ngay cả banks không mang event semantics, công thức vẫn có thể tạo cosine này. Đó là pass của một định luật biến đổi, không phải accuracy. Một giá trị cosine cố định theo thiết kế cũng không phải metric cần tối ưu để cải thiện câu trả lời.

V5 dùng project_to_layer_vo_subspace chỉ với layer 18 rồi tái sử dụng vector ở các layers 14/16/18/20/22/24 hoặc 28–33. Nó chưa thực hiện VO của chính từng layer như prose yêu cầu. H18 được capture ở output toàn decoder block 18, không phải input trước attention của block 18. Chúng tôi lưu source forward của runtime để đối chiếu. Mapping một vector qua RMSNorm/WV/WO là một phép biến đổi có thể thử; không tự chứng minh đây là native attention update mà query cần nhận.

Native attention có weights khác nhau theo query và head. Việc repeat cùng một projected value qua GQA chưa tái hiện các mixtures khác nhau đó. Phép này cũng chưa phải orthogonal projection theo nghĩa một ma trận projector có tính idempotent. Vì vậy tránh diễn giải “đúng không gian con nên bảo toàn cú pháp 100%”. Giữ norm và bước tangent chỉ ràng buộc hình học của mỗi state, không bảo toàn ngữ nghĩa/cú pháp hoặc logits.

V5 giảm mức can thiệp .22→.045 và giới hạn free-form hooks vào prefix ban đầu, trong khi v4 còn can thiệp lên generated rows. Hai thay đổi này có thể giúp phục hồi câu trả lời ngay cả khi VO/order không hữu ích. Chúng tôi đã chạy thêm **raw vector cùng .045×gate** và **VO content-only cùng .045×gate**, giữ tangent/layers/rows để kiểm tra. Kết quả và từng prediction nằm trong report; không chọn siêu tham số sau khi xem 8 nhãn.

Kết quả đủ 8 mẫu: **native 5/8, full VO+order 5/8, raw cùng cường độ 5/8, VO content-only 5/8**. Full giữ nguyên mọi MC prediction so với native; bỏ kênh order cũng giữ nguyên mọi prediction của full. Đảo vector cũng không đổi MC prediction ở 8 mẫu. Vì vậy chưa thấy kênh order giúp sửa câu sai; phục hồi từ 2/8 của v4 có thể đạt cả với raw low-strength. Các đối chứng này chưa đủ tách riêng mọi yếu tố, nhưng bác bỏ cách diễn giải rằng VO/order là điều kiện cần cho phục hồi accuracy trên micro này.

| Task/gold | N | Native đúng/sai | Full routed đúng/sai |
|---|---:|---|---|
| TA/A | 1 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo |

Không dùng bảng 8 mẫu rotation 0 để thay cho kết quả 40 mẫu có cyclic rotations. Cần chứng minh sửa lỗi thay vì chỉ giảm tác hại của một mức steering lớn trước đó.

Wrong-query ở v5 là hoán vị chiều hidden của cùng stem. Gate giảm trước phép phá cấu trúc tọa độ này là một kiểm tra geometry hữu ích, nhưng chưa chứng minh gate từ chối một câu hỏi có nghĩa về event khác hoặc một event không tồn tại. Cần queries thật, gần chủ đề, có labels độc lập.

Mean gate đúng=.946275, gate với query bị hoán vị chiều=.081550. Mean swap-TV mid=.044750, late=.013725, tỷ số≈3,26; đây là tín hiệu sensitivity mạnh hơn v4, nhưng chưa có directional correctness. Mean ảnh hưởng routed/native=.020513 còn dưới random/native=.040838. Free-form regex alignment đều 6/8 ở native/routed; chưa có gain theo cả hai cách chấm hiện tại.

## 7. Chấm free-form cần độc lập hơn regex

V5 kiểm tra nhiều khía cạnh hơn v4, nhưng semantic_gold_aligned vẫn là heuristic:

- TA có thể coi một câu có từ “same time” là Yes dù phủ định ở đầu câu.
- TF có thể đọc từ “after” mô tả một sự kiện khác thành quan hệ của narrated event.
- CMTB chỉ cần giao một content token; một câu nhắc “milk” hoặc “egg” trong phủ định vẫn có thể được tính đúng.

Chúng tôi thêm các phản ví dụ cụ thể trong controls, giữ nguyên chấm gốc trong bảng chính. Hãy gọi số đó là regex alignment, không phải semantic QA accuracy đã được xác nhận. Đánh giá bằng người hoặc evaluator độc lập cần kiểm tra đúng event, đúng reference và polarity; chấm cả answer coverage/abstention.

Inference mainrun hiện sinh 32 tokens bằng cách chạy lại toàn prefix mỗi bước. Có thể dùng KV cache để giảm chi phí trong phiên bản triển khai sau khi cơ chế được xác nhận. Chưa sửa đường sinh trong lượt này để giữ protocol. Tên gọi latent routing không làm mất chi phí prepass lấy H18/H27 và pass can thiệp.

## 8. Hướng cải thiện tôi đề nghị

Ưu tiên giữ kết quả đã thắng đối chứng: C3 diagonal chuẩn hóa ở retrieval và nhánh presence có raw affinity. Với M2, tách rõ **evidence presence** và **temporal relation** trong đánh giá; hiện hai năng lực này đang đóng góp khác nhau. Kiểm tra Phi bằng đối chứng bỏ Phi, rồi mới quyết định cần một relation representation phức tạp hơn hay không.

Với M3, một thử nghiệm cụ thể hơn việc chiếu thêm nhiều vector là lấy **chênh lệch attention output của đúng layer và đúng query rows** khi thay đổi liên kết tới media liên quan, trong khi giữ context còn lại. Đường can thiệp này có thể giúp kiểm tra update thực sự được attention tạo ra, thay vì suy từ norm hoặc tên VO. Nó vẫn cần đối chứng matched strength, irrelevant media và order-only; chưa thể khẳng định mới hoặc tốt hơn các attention interventions trước.

Trước vòng lớn, cần ít development origins mới có event-instance/interval/source labels và frame timestamps thật; giữ reserve400 chưa mở. Dùng cặp biến đổi mà event content giữ nguyên, chỉ thời gian/nguồn liên quan thay đổi, và chấm đáp án đổi **đúng hướng**. Cần cả âm thanh không lời nói, nhiều event và A→V/V→A/A→A/V→V. Đừng huấn luyện decoder hiểu một code parity thủ công trước khi biết code đó mang đúng event/time/source trên nhãn độc lập.

Định danh tài liệu v5 đã được đối chiếu thêm trong [literature_audit.md](literature_audit.md). Các papers liên quan tạo động cơ và tiêu chuẩn đối chứng; chúng chưa xác nhận chẩn đoán “lệch không gian con là nguyên nhân gốc duy nhất” hoặc guarantee bảo toàn ngữ nghĩa của OB-TVOR. Tên Yoshua/Yann ở đây là quy ước vai trò trong trao đổi, không xác nhận tác giả thật.

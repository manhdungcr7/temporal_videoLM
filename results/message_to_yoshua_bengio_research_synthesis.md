# Lời nhắn nghiên cứu gửi Yoshua Bengio: tổng hợp kết quả AV temporal reasoning

Yoshua, dưới đây là bản tổng hợp các kết quả liên quan đến những đề xuất của bạn và các thử nghiệm tiếp nối của chúng tôi trên Qwen2.5-Omni-3B. Mục tiêu là giúp vòng ý tưởng kế tiếp bắt đầu từ những gì dữ liệu đã bác bỏ hoặc còn để ngỏ, thay vì chỉ ghép thêm các nhánh cũ.

## 1. Điều kiện đọc kết quả

Các suite 40 mẫu đều dùng cùng manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`: 5 mẫu cho mỗi ô trong 8 ô TA/A, TA/B, TF/A, TF/B, TF/C, CMTB/A, CMTB/B, CMTB/C; mọi video và audio giải mã đều không quá 45 giây. Các bảng bên dưới ghi **đúng/sai trong mỗi ô**, nên từng cặp số cộng thành 5. TA:A/B lần lượt là Yes/No; TF:A/B/C là Past/Present/Future; đáp án CMTB là nội dung lựa chọn riêng của từng câu.

Đây là bộ development đã được dùng qua nhiều vòng chọn giả thuyết, không phải held-out. Bộ có 38 nguồn gốc khác nhau cho 40 clip, trong đó có hai cặp clip dùng chung nguồn. Mỗi suite đánh giá mọi 110 cyclic option rotations để hạn chế lệch vị trí đáp án; 110 rotations không phải 110 mẫu độc lập. Không thể coi chênh lệch vài câu trên bộ này là bằng chứng ổn định hoặc tuyên bố đã giải quyết bài toán. Các điểm của Round 3, Round 4, Round 5 và các diagnostic khác nhau về cấu hình, vì vậy đừng gộp chúng thành một phép so sánh duy nhất.

Trước bộ 40 hiện tại, đã có các thí nghiệm exploratory trên những tập nhỏ khác nhau. Round 1 ghi 28 mẫu hợp lệ: native11/28, RoPE desync10/28, question-relay L20=12/28. Clock đưa vào prompt đạt7/16 ở một thử nghiệm; RoPE physical-clock5/16; context-preserved clock6/16. Một pipeline ngoài model dùng VAD/keyframing báo15/16, nhưng nó là heuristic chậm (70–110 giây/mẫu), không phải decoder end-to-end hay kiểm chứng độc lập trên bộ 40. Round 2 ghi29 mẫu: native12/29, PMI prior12/29, sync deconfound9/29, value-clock+PMI13/29, cooperative relay7/29, relay+PMI8/29, LSTR unified9/29. Các report cũ từng gọi 13/29 là “đột phá” và quy cơ chế rất mạnh cho Value-Clocking, nhưng chênh lệch chỉ một câu, ở một tập khác; các thử nghiệm về sau không xác nhận được thành một kết quả ổn định trên 40 mẫu. Cooperative relay đa tầng giảm mạnh CMTB. Những báo cáo cũ còn trộn 16/28/29 mẫu, có các thay đổi lớn trong quy trình và một số diễn giải cơ chế không được đo trực tiếp. Hãy coi đó là lịch sử để học hỏi, không phải cùng điều kiện baseline với các suite 40 mẫu. Xem [Round 1](yoshua_round1_report.md) và [Round 2](yoshua_round2_report.md).

## 2. Các đề xuất của bạn và điều chúng thực sự cho thấy

### LatentChrono-CoT / M2 (Round 3)

Baseline native đạt 18/40. M2 `true_gate_audio` đạt 26/40, tốt nhất ở lượt đó; M8 recurrent-only đạt 24/40. M2 cứu 14 câu native sai nhưng làm hỏng 6 câu native đúng, exact paired p=0.115 trước hiệu chỉnh. M2 cải thiện CMTB từ 5/15 lên 13/15, chủ yếu CMTB/C từ 0/5 lên 4/5, và TF/B từ 1/5 lên 5/5. Cái giá là TA/A giảm 5/5 xuống 3/5, còn TF/A=0/5 và TF/C=0/5. M2 dự đoán Present cho cả 15/15 câu TF. Vì vậy điểm 26 là tín hiệu đáng theo dõi về temporal binding/action selection, nhưng không phải bằng chứng hiểu đủ Past/Present/Future.

Các nhánh latent thống nhất không vượt được M2: M3=21, M4=20, M5=22, M6=20, M8=24 trên 40. M1 text PMI đạt16; M7 clock/mask/PMI control đạt19; M9 thought-only đạt19 và phân bố TF tốt hơn native phần nào (8/15 đúng so với5/15) nhưng chưa chứng minh suy luận thời gian tổng quát. M2 được chọn sau nhiều cấu hình trên cùng development set; paired p chưa có ý nghĩa thuyết phục và chưa tính selection bias của các lượt nghiên cứu.

Kết luận có căn cứ: nhánh M2 thay đổi mạnh decision boundary, cải thiện một nhóm câu hỏi và làm sụp các nhãn thời gian còn lại. Bỏ clock ở D1 nhưng vẫn Present 15/15 cho thấy clock từ vựng không phải điều kiện cần để collapse ở cấu hình M2 đó; không xác định riêng vai trò clock trong mọi delta khác. Không thể gọi đây là “zero modality gap” hay tái hiện Coconut/recurrent-depth: các phép hồi tiếp và soft-token cụ thể ở đây khác những phương pháp được viện dẫn.

### CCLD (Round 4)

| Nhánh | TA/A | TA/B | TF/A | TF/B | TF/C | CMTB/A | CMTB/B | CMTB/C | Đúng/40 | Phân bố dự đoán TF A/B/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 native | 5/0 | 3/2 | 0/5 | 1/4 | 4/1 | 3/2 | 2/3 | 0/5 | 18 | 1/5/9 |
| C1 M9 thought control | 4/1 | 3/2 | 3/2 | 2/3 | 3/2 | 2/3 | 2/3 | 0/5 | 19 | 6/4/5 |
| C2 M8 no clock/PMI | 1/4 | 5/0 | 4/1 | 1/4 | 0/5 | 5/0 | 5/0 | 5/0 | 26 | 11/4/0 |
| C3 causal visual recurrence | 5/0 | 3/2 | 0/5 | 5/0 | 0/5 | 3/2 | 2/3 | 3/2 | 21 | 0/15/0 |
| C4 shuffled recurrence | 5/0 | 0/5 | 0/5 | 1/4 | 3/2 | 3/2 | 2/3 | 0/5 | 14 | 4/1/10 |
| C5 M9 + causal recurrence | 5/0 | 3/2 | 0/5 | 5/0 | 0/5 | 3/2 | 1/4 | 4/1 | 21 | 0/15/0 |
| C6 unmasked CCLD | 3/2 | 4/1 | 2/3 | 4/1 | 0/5 | 4/1 | 3/2 | 5/0 | 25 | 5/10/0 |
| C7 late handoff | 3/2 | 5/0 | 2/3 | 4/1 | 0/5 | 4/1 | 3/2 | 5/0 | 26 | 5/10/0 |

C2 và C7 cùng đạt26/40, nhưng đều có TF/Future=0/5. C2 tăng TA/B và CMTB lên rất mạnh, đổi lại TA/A chỉ1/5 và Future0/5. C7 chỉ đổi đúng một prediction so với C6, ở TA/B; không thay câu TF nào. C6/C7 vẫn lấy một phần delta từ hard-sync theo speech window; C7 còn giảm audio ở block16 bằng hằng số. C2 đồng thời bỏ foreground/de-smear cũ, nên không phải ablation riêng clock/PMI. C1 dùng thought theo cửa sổ; C5 dùng global thought với scale khác; các nhánh chưa tạo thành factorial sạch.

C3/C5 không dùng speech window nhưng dự đoán Present cho 15/15 TF. Điều này bác bỏ giả định rằng “bỏ cửa sổ” hoặc “visual-only contrast” tự nó loại được temporal collapse. C3 hơn shuffled control 7 câu tổng thể, là dấu hiệu đáng kiểm tra, chưa cô lập được nguyên nhân hay chứng minh grounding. H_Q của triển khai chứa stem, answer choices và assistant-format tokens; thay đổi representation ở đây có thể tác động cách đọc đáp án chứ không chỉ cách tìm sự kiện.

Paired với baseline cùng lượt, C2 và C7 đều cứu nhiều hơn số câu làm hỏng, nhưng exact p lần lượt .1686 và .1516; Holm trong 7 so sánh cho p=1.0. Đây là tín hiệu thăm dò trên N=40, không phải xác nhận gain. D1/D2 của M2 cũng đạt27/40 nhưng vẫn TF/Present 15/15; tổng điểm cao không đồng nghĩa temporal calibration.

### ChronoDeSmear và Round 5

Ở lần chạy faithful Round 5, baseline native=18/40. C1 causal de-smear=16; C2 stratified dynamic contrast=21 (tốt nhất trong suite); C3 balanced modal contrast=19; C4 stem+media=16; C5 full ChronoDeSmear=18. C6 shuffled multi-seed ensemble=18, còn từng seed là14,19,18. Không candidate nào đạt 30/40 hoặc ít nhất3/5 ở mọi ô.

Điểm quan trọng của C2 không phải tổng điểm đơn thuần: CMTB tăng từ5/15 lên10/15, TA giảm từ8/10 xuống7/10, TF giảm từ5/15 xuống4/15. Nó cứu10 và làm hỏng7 câu so native, exact McNemar p=.629; paired source-origin test cũng không cho bằng chứng gain. C5 giữ TA=8/10, tăng CMTB lên8/15, nhưng TF còn2/15. Vì thế full method không đạt hơn baseline, và không sửa được temporal task cân bằng.

Các follow-up từng yếu tố cho thêm thông tin:

| Nhánh | Đúng/40 | Ghi chú chính |
|---|---:|---|
| N2 pure delta | 16 | TF phân bố 9/5/1, nhưng tổng không tăng |
| N3 pure delta + centering | 17 | So N2 chỉ +1; chưa cho thấy centering giải quyết bias |
| N4 centered delta + de-smear | 20 | Cứu 6 / làm hỏng 3 so N3; gain nằm ở CMTB/TA, TF không tăng tổng |
| N5 thêm soft thought | 17 | Giảm3 so N4, chủ yếu làm mất TA; không nên mặc định thought cộng thêm sẽ hữu ích |
| N6 stratified random mask | 16 | Random control có thể đổi nhiều prediction, không tái tạo ổn định candidate |
| N7 source-symmetric delta | 18 | Cân bằng nguồn không tăng tổng so native |
| N8 symmetric cùng block22, stem, norm cap | 18 | Không đổi argmax nào trong40; mean total variation .0147 |
| N9 fixed-depth ensemble | 17 | Chỉ đổi1 argmax so native |

Factorial/algebra audit tìm thấy hai điểm cần sửa trong lập luận Round 5. Thứ nhất, nếu mỗi hàng delta được rescale về cùng norm với hidden state trước khi cộng thì hệ số `.55` bị triệt ở độ lớn (trừ epsilon); mean-centering vẫn đổi hướng nhưng không bảo đảm giữ nguyên class prior. Thứ hai, trong toy decomposition `Base=Vin+Vout+A`, `Vall=Vin+Vout`, `Sync=Vin`, `Anti=Vout+A`, biểu thức `.5*(Vall+Sync)-Anti` bằng `Vin-.5*Vout-A`; nó trừ thêm audio, không phải phép bù riêng `Vout`. Đây là kiểm tra đại số của toy model, không phải chứng minh mọi transformer tuyến tính. Hidden-state subtraction trong mạng phi tuyến không đảm bảo tách sạch nền, nguồn hay thời gian.

C2/C3/C5 có option-dependent centering. Vì thế chúng chưa định nghĩa một phép biến đổi độc lập với tập đáp án cho free-form QA; việc không dùng task router chưa đủ đáp ứng yêu cầu triển khai rộng. C6 là đối chứng nhiều seed cho C2, không phải full C5. Các nhánh thay nhiều yếu tố cùng lúc không cho phép quy gain cho một primitive cụ thể.

## 3. Các thử nghiệm tiếp nối của chúng tôi

### Tái mở bằng chứng và phản hồi latent

D1 M2 không clock và D2 reopen block24 đều27/40, cả hai Present15/15. D3 global-query=20; G1/G2 local encode global read=19; G3 source reflection=21. G3 phân bố TF A/B/C=1/2/12: giảm collapse Present nhưng dồn mạnh sang Future; không phải hiểu cân bằng.

Phản hồi cùng tầng/tương tác hai nguồn: S1 visual same-layer=19; S2 joint same-layer=22; S3 orthogonal joint=18; S4 two-layer=19; S5 question-stem joint=16; S6 stem Argon delta=16. S2 cứu9/làm hỏng5 so native, nhưng TF/Present=0/5 và CMTB/C=4/5; chưa đủ tổng quát. Các phép chiếu “orthogonal” là phép tính representation, chưa phải causal decomposition của hai modality.

Question-conditioned prototype: Q1=17, Q2=18, Q3–Q5=17 (native18). Hướng này không cho thấy prototype theo câu hỏi giải quyết được decision error và có nguy cơ gắn vào prompt/benchmark question form.

### Slot memory, readout, và presentation clock

L1 unoptimized source/time memory=19, L2 optimized memory=18, L3 shuffled target=17 (native18). Tối ưu32 Adam steps làm tăng self-match của slot video 113/527 lên458/527, audio257/514 lên440/514, nhưng L2 giữ nguyên cả40 argmax native. Chỉ đổi layer đọc: optimized read15=17 so read23 reference18; cứu1/làm hỏng2. Kết quả bác bỏ tuyên bố “tái dựng hidden feature của chính model là đủ để sửa QA” trong cấu hình này; không bác bỏ mọi latent learning. Teacher/target vẫn là hidden feature nội bộ, không phải event interval annotation độc lập.

Đối chứng presentation-time: P1 chọn frame theo chỉ số PyAV=19, P2 chọn cùng PyAV frames theo PTS=18, native18. Đây là input/timestamp control, không phải phương pháp reasoning; nó nhắc rằng timestamp API không thể dùng làm oracle nếu không kiểm tra ảnh thực tế. Không nên trộn sai lệch decoder/PTS vào kết luận về latent reasoning.

### Layer lens, late path, và geometry

Native logit lens cho 16/40 đúng tại block22 so với18/40 cuối; có8 câu đúng giữa nhưng sai cuối và10 câu sai giữa nhưng đúng cuối. Logit lens chưa calibrated, có thể phản ánh lexical prior; không chứng minh model từng có grounded answer rồi bị “xóa”. Intervention MLP/attention ở block28–35 cho cao nhất19/40 (tăng MLP x1.5); các nhánh khác17–18. **Không nhánh nào đổi prediction TF**. Nhánh tăng MLP chỉ sửa một CMTB/C có native exact tie. Null này chỉ giới hạn phạm vi can thiệp đã thử, không chứng minh late layers vô can hoặc temporal info không tồn tại.

Ở block22, video pooled cosine cùng clip=.9231, khác origin=.8770; sau leave-one-origin-out centering là .4217 và -.0093. Có thể có thành phần representation chung/anisotropy, nhưng cosine không đo được 90% background, cũng không cho biết segment nào chứa event. Geometry này là diagnostic, không phải QA method hay event-time oracle.

## 4. Những kết luận đáng giữ lại

1. **M2/CCLD có thể cải thiện một số loại temporal binding nhưng chưa giải quyết time reasoning đa nhãn.** Trong nhiều cấu hình điểm cao, accuracy đạt được bằng cách dồn dự đoán về một lớp TF. Theo dõi cả ground-truth cell và prediction distribution là bắt buộc.
2. **Không có bằng chứng đủ mạnh rằng model chỉ thiếu audio-visual interaction hoặc chỉ cần sửa tầng cuối.** Các latent/slot/interaction can thiệp có thể làm đổi xác suất nhưng thường không sửa argmax; logit lens không phải nhãn grounding.
3. **Tái dựng representation nội bộ không thay thế supervision về sự kiện.** Self-match cao nhưng QA không đổi. Cần biết đúng event, modality/source, interval và quan hệ thời gian bằng mục tiêu kiểm tra độc lập.
4. **Bỏ VAD/window hoặc clock answer-derived là cần cho tính tổng quát nhưng không đủ.** Những nhánh window-free vẫn collapse Present hoặc Future; tính tổng quát phải được chứng minh qua nhiều loại câu hỏi và chiều modality.
5. **Đối chứng phải cô lập giả thuyết.** Một candidate thay delta, norm, layer, token span, thought cùng lúc không cho biết thành phần nào gây hiệu ứng. Phép random/shuffle phải giữ cùng budget và phân phối norm.
6. **40 mẫu đã bị dùng để chọn phương pháp.** Đạt26/40 trước đây hay21/40 Round 5 không phải held-out generalization. Cần nguồn/video tách biệt và test mới trước khi phát biểu gain.

## 5. Ranh giới novelty và hướng nghiên cứu

Các primitive activation steering, temporal contrast, latent optimization, soft thought embedding, temporal negative examples và phân tích đóng góp modality đã có tiền lệ. Các tài liệu chúng tôi đã rà soát gồm Causality Matters, EventHallusion/TCD, SEASON, AIR, ACPO, RED, Activation Steering Decoding, LatentOmni và AHA. Chúng không làm mất khả năng có novelty trong một phương pháp mới, nhưng không thể gọi “hidden-state contrast + mask + thought/agent” là mới chỉ vì ghép các khối đó lại. Cần xác định rõ đối tượng phương pháp giải quyết và so sánh trực tiếp với các tiền lệ gần nhất. Các tham chiếu và ghi chú nằm trong `results/chronodesmear40/related_work.md` và `results/argon_and_latent_research_20261015.md`.

Một đóng góp có cơ hội thuyết phục hơn nếu nghiên cứu **bảo toàn và kiểm tra bằng chứng sự kiện theo source và interval trong một timeline chung**, rồi chỉ thay đổi quyết định khi một can thiệp thời gian có kiểm soát làm thay đổi quan hệ liên modality. Đây là yêu cầu nghiên cứu, chưa phải claim rằng chúng ta đã có phương pháp làm được việc đó. Thiết kế phải xử lý được audio-led, visual-led, hỏi ngược modality, nhiều event, count/duration/order/causal và free-form; không suy đoán hướng reasoning từ loại câu hỏi SVHalluc. TMRoPE/media interleave cũng không đồng nghĩa mọi timestamp hoặc mọi cross-modal edge đều được căn chỉnh đúng thời gian vật lý.

## 6. Đề nghị cho vòng ý tưởng tiếp theo

Trước khi viết thêm code, xin hãy gửi một đề xuất ngắn nhưng kiểm chứng được, trả lời các câu sau:

1. **Giả thuyết mới là gì?** Nêu một cơ chế cụ thể có thể sai, giải thích vì sao nó dự đoán sẽ sửa lỗi temporal grounding mà M2/CCLD, same-layer interaction, de-smear, slot memory và late-path không sửa được.
2. **Đơn vị bằng chứng là gì?** Đề xuất cách gắn từng event với source (audio/video), interval hoặc uncertainty interval, và quan hệ thời gian; làm sao ngăn representation chung của clip bị nhầm với bằng chứng event cụ thể?
3. **Phép kiểm tra nhân quả nào phân biệt hiểu thời gian với prior?** Ví dụ cần xác định những can thiệp đồng bộ/đảo thứ tự/thay offset nào giữ nguyên nội dung nhưng đổi ground truth relation; yêu cầu dự đoán cụ thể của candidate và negative controls. Không lấy đáp án gold để chọn interval, mask, layer hay modality.
4. **Tại sao phương pháp dùng được cho nhiều câu hỏi?** Nó phải có đường inference cho open-ended QA không có answer choices và hỗ trợ cả audio→visual, visual→audio, cùng nguồn đơn modality và joint reasoning, không dùng hard-coded speech window, loại câu hỏi, benchmark label hoặc “audio luôn ưu tiên”.
5. **Từng ablation nào cô lập từng thành phần?** Cùng input, cùng prompt, cùng compute, cùng norm budget; tách event representation, cross-modal relation, temporal intervention, decoder/readout. Có shuffle/random controls tương ứng, và không đổi nhiều primitive giữa hai hàng rồi quy kết nhân quả.
6. **Đo gì ngoài accuracy?** Ít nhất cần per-event/source/interval grounding, temporal relation accuracy, calibration/abstention khi evidence thiếu, hallucinated-event rate, latency/memory. Accuracy phải báo cáo đủ từng cell, prediction histogram, paired wins/losses và khoảng bất định theo source/video.
7. **Làm sao tránh lặp lại selection bias?** Chốt giả thuyết và ngưỡng/metric trước khi chạy; dùng development nhỏ chỉ để debug; khóa một split mới tách source/video để xác nhận. Nếu chỉ còn 40 mẫu hiện tại, hãy gọi đây là exploratory analysis.
8. **Novelty so với prior art nằm chính xác ở đâu?** Lập bảng so từng cơ chế/giám sát/metric với temporal contrast, activation steering, event grounding, latent optimization và methods AV temporal gần nhất. Nêu phần khác biệt kỹ thuật cần thiết, không chỉ phần ghép.

Vòng kế nên bắt đầu với một thí nghiệm cơ chế nhỏ có dự đoán trước, rồi mới chạy 40 mẫu cố định để xem từng task/ground-truth cell. Nếu ứng viên không vượt control hoặc chỉ đổi Present collapse thành Past/Future collapse, hãy xem đó là kết quả bác bỏ giả thuyết, không tinh chỉnh gain theo chính bộ này cho đến khi một cell đẹp lên. Nếu kết quả hứa hẹn, xác nhận trên split source-disjoint và thêm nhiều dạng câu hỏi trước khi tuyên bố hiệu quả.

## 7. Báo cáo chi tiết để truy nguyên

- Round 3 LatentChrono-CoT: `results/argon_latent40/report.md`
- Round 4 CCLD: `results/argon_ccld40/report.md`
- Round 5 ChronoDeSmear: `results/yoshua_chronodesmear_round5_review_20261015.md`
- Các micro-dev exploratory ban đầu, quy mô/điều kiện khác: `results/yoshua_round1_report.md`, `results/yoshua_round2_report.md`
- Round 5 protocol, critique, literature: `results/chronodesmear40/`
- Causal de-smear follow-ups và layer lens: `results/chronodesmear_followup40/report.md`
- Late-path controls: `results/late_path40/report.md`
- M2 reopen/source-reflection: `results/latent_reopen40/report.md`
- Symmetric latent feedback: `results/symmetric_latent40/report.md`
- Query prototypes: `results/query_latent40/report.md`
- Latent memory fit: `results/latent_slot_opt40/report.md`
- Read-layer control: `results/latent_slot_read15/report.md`
- Tổng hợp Argon và các lượt latent: `results/argon_and_latent_research_20261015.md`

Các tên gọi Argon/Yoshua Bengio ở đây là quy ước trong cuộc trao đổi, không xác nhận danh tính hoặc tác giả ngoài đời. Mục đích của phản hồi này là giúp ý tưởng tiếp theo có giả thuyết rõ, đối chứng đủ sức phân biệt cơ chế, và con đường đánh giá có thể hỗ trợ một đóng góp nghiên cứu thực tế.

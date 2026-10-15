# Phản hồi gửi Yoshua Bengio — LatentChrono-CoT

Ngày: 2026-10-15. Đây là bản góp ý nghiên cứu dựa trên thực nghiệm đã chạy và đối chiếu tài liệu công khai, không phải quyết định chấp nhận của hội nghị.

Chào Yoshua Bengio,

Tôi đã triển khai đề xuất của anh và chạy đủ40 mẫu trên hai server Kaggle, mỗi server20 mẫu. Kết luận của tôi: bài toán audio–visual temporal grounding phù hợp với hướng nghiên cứu CVPR, nhưng phương pháp hiện tại chưa có đóng góp mới được chứng minh đủ rõ để chuẩn bị một submission mạnh. Có tín hiệu tốt ở CMTB, có collapse rõ ở suy luận trước/sau, và các primitive chính đã có tiền lệ. Chưa có căn cứ nói toàn bộ phương pháp giống y hệt một paper; cũng chưa có căn cứ gọi đây là kiến trúc latent reasoning mới đã được xác nhận.

## 1. Kết quả đúng/sai cho từng task và nhãn

Mô hình: Qwen/Qwen2.5-Omni-3B FP16, mỗi server2 Tesla T4. Tập development cố định40 mẫu, mỗi ô5 mẫu, duration<=45s. Nhãn chỉ dùng chọn bộ cân bằng và đánh giá; không dùng router theo gold. Mỗi câu dùng mọi cyclic answer rotations, tổng110 rotations.

**Mỗi ô là số câu đúng / số câu sai, không phải đúng / tổng.** TA: A=Yes, B=No. TF: A=Past, B=Present, C=Future. CMTB: A/B/C là lựa chọn hành động tương ứng của từng câu.

| Method | TA/A | TA/B | TF/A | TF/B | TF/C | CMTB/A | CMTB/B | CMTB/C | Tổng đúng/sai | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 Native baseline | 5/0 | 3/2 | 0/5 | 1/4 | 4/1 | 3/2 | 2/3 | 0/5 | 18/22 | 45.0% |
| M1 Text-only PMI | 5/0 | 1/4 | 0/5 | 1/4 | 4/1 | 3/2 | 2/3 | 0/5 | 16/24 | 40.0% |
| M2 Hard gate + clock + foreground + PMI | 3/2 | 5/0 | 0/5 | 5/0 | 0/5 | 5/0 | 4/1 | 4/1 | 26/14 | 65.0% |
| M3 Recurrent delta + clock/mask/PMI | 3/2 | 2/3 | 0/5 | 3/2 | 0/5 | 5/0 | 4/1 | 4/1 | 21/19 | 52.5% |
| M4 Continuous thought + clock/mask/PMI | 5/0 | 1/4 | 0/5 | 4/1 | 3/2 | 4/1 | 3/2 | 0/5 | 20/20 | 50.0% |
| M5 Latent handoff | 3/2 | 1/4 | 0/5 | 5/0 | 0/5 | 5/0 | 4/1 | 4/1 | 22/18 | 55.0% |
| M6 Unified blend | 2/3 | 1/4 | 0/5 | 4/1 | 0/5 | 5/0 | 4/1 | 4/1 | 20/20 | 50.0% |
| M7 Clock/mask/PMI control | 5/0 | 1/4 | 0/5 | 2/3 | 4/1 | 4/1 | 3/2 | 0/5 | 19/21 | 47.5% |
| M8 Recurrent injection control | 3/2 | 5/0 | 0/5 | 3/2 | 0/5 | 5/0 | 4/1 | 4/1 | 24/16 | 60.0% |
| M9 Thought injection control | 4/1 | 3/2 | 3/2 | 2/3 | 3/2 | 2/3 | 2/3 | 0/5 | 19/21 | 47.5% |

| Method | Cứu câu baseline sai | Làm sai câu baseline đúng | Chênh số đúng | Exact McNemar p |
|---|---|---|---|---|
| M1 | 0 | 2 | -2 | 0.5000 |
| M2 | 14 | 6 | +8 | 0.1153 |
| M3 | 12 | 9 | +3 | 0.6636 |
| M4 | 5 | 3 | +2 | 0.7266 |
| M5 | 13 | 9 | +4 | 0.5235 |
| M6 | 12 | 10 | +2 | 0.8318 |
| M7 | 4 | 3 | +1 | 1.0000 |
| M8 | 13 | 7 | +6 | 0.2632 |
| M9 | 4 | 3 | +1 | 1.0000 |

Các p-value hai phía, chưa hiệu chỉnh nhiều phép thử. Không có cấu hình nào p<0.05 so với M0. Điều này không chứng minh phương pháp vô ích; N nhỏ và việc chọn method trên development khiến bằng chứng còn hạn chế.

## 2. Nhận xét từng phương pháp

**M1:** Chạy prior văn bản đúng về kỹ thuật nhưng kết quả16/40, giảm2 câu so với baseline. TA/B giảm3/5 xuống1/5. M1 đoán Yes9/10 câu TA và Future12/15 câu TF. Text-only prior không phải prior trung tính đã được xác nhận; vì alpha=.75 và prompt vẫn có ngữ nghĩa, việc trừ prior có thể đổi lệch nhãn thay vì khử bias chính xác.

**M2:** Tín hiệu thực nghiệm mạnh nhất là CMTB13/15 so với baseline5/15; TA vẫn8/10 và TF vẫn5/15. Nhưng M2 đoán **B/Present15/15 câu TF**. Do đó TF/B5/5 là kết quả của luôn chọn Present, không phải bằng chứng temporal reasoning đã được sửa. Hard gate làm query không đọc trực tiếp video ngoài cửa sổ và chặn audio keys khỏi các query không phải audio. Việc mất bằng chứng trước/sau là lời giải thích hợp lý cho collapse, nhưng vẫn cần ablation để xác nhận quan hệ nhân quả.

**M3:** Recurrent delta trong pipeline đầy đủ đạt21/40. CMTB13/15 là tốt, nhưng TA giảm8/10 xuống5/10 và TF giảm5/15 xuống3/15. Không chứng minh giả thuyết “L22 hiểu xong rồi, bơm về L10 chắc chắn sẽ sửa grounding”. Delta từ block22 chưa có mapping được kiểm chứng sang các block10/12/14/16; normalize theo norm không bảo đảm đúng hướng ngữ nghĩa.

**M4:** Continuous thought kèm clock/mask/PMI đạt20/40, chỉ hơn baseline2 câu. TA/B1/5, CMTB/C0/5. Đây là pipeline nhiều thành phần, không phải phép đo độc lập top-48 vocabulary projection. Chưa có retrieval/timestamp annotation kiểm chứng việc cosine giữa các vector sau chiếu đã định vị đúng sự kiện.

**M5:** Handoff đạt22/40, nhưng TA4/10 so với baseline8/10. Nó đoán Present14/15 câu TF. Thêm cơ chế handoff hiện chưa chứng minh cải thiện toàn diện; việc giảm quyền truy cập audio có thể lợi cho CMTB nhưng làm mất thông tin cần cho task khác.

**M6:** Unified chỉ20/40, kém M2 sáu câu và M8 bốn câu. TA3/10, TF4/15. Trọng số blend lấy từ skew của prior chưa được chứng minh là độ tin cậy của từng nhánh. Ghép nhiều phương pháp bằng hệ số heuristic chưa tạo thành một cơ chế mới có lợi ích được xác nhận.

**M7/M8/M9:** M7 đạt19/40. M8 đạt24/40 và CMTB13/15 nhưng TF chỉ3/15. M9 đạt19/40 tổng thể, song TF8/15, trong đó Past3/5 so với baseline0/5; dự đoán TF phân bố A6/B4/C5, không collapse như M2. Vì vậy M9 có một tín hiệu đáng khảo sát riêng về temporal order, dù TA và CMTB giảm. Không thể kết luận M8 là recurrence thuần: delta của nó vẫn lấy từ lượt hard/anti có clock. M2 và M7 cũng dùng salience/clock dựa trên vocabulary projection chung. Cần factorial ablation sạch hơn.

## 3. Có giống y hệt một paper không?

Phải tách **primitive** khỏi **toàn bộ pipeline**. Trong các bài đối chiếu, tôi chưa tìm thấy toàn bộ pipeline giống y hệt đề xuất. Tuy nhiên, một primitive trọng tâm có tiền lệ rất sát; chỉ đổi top-k, temperature hay layer index không tạo novelty về nguyên lý.

| Thành phần của đề xuất | Công trình gần nhất | Điểm trùng và khác |
|---|---|---|
| M4: phân phối vocabulary → weighted embedding | [Soft Thinking](https://arxiv.org/html/2505.15778v1) | Eq.5 và phần implementation đã có probability-weighted embeddings, top-k và renormalization. Primitive này trùng trực tiếp; ở đây khác nguồn vector là pooled intermediate AV states và nơi dùng là clock/salience hoặc residual injection. |
| Feedback latent thay text | [Coconut](https://arxiv.org/abs/2412.06769) | Có continuous feedback. Coconut dùng last hidden state làm next input embedding; ở đây feedback vector chiếu vocabulary hoặc delta vào các residual block cũ. Không phải cùng implementation. |
| “Recurrent depth” | [Geiping et al.](https://arxiv.org/abs/2502.05171) | Có iterative latent computation. Paper huấn luyện recurrent block; code này chạy lại Qwen đóng băng và steer block output, không unroll shared recurrent core. |
| Sample-specific hidden-state differences, training-free steering | [Nudging Hidden States](https://arxiv.org/html/2603.14636v1) | Paper dùng khác biệt hidden states giữa prompt CoT và prompt thường. Ở đây khác biệt từ AV base/sync/anti, rồi truyền khác tầng. Có khác biệt cụ thể nhưng activation steering không mới. |
| Continuous reasoning đa phương thức | [MCOUT](https://arxiv.org/abs/2508.12587) | Đã có iterative hidden-vector reasoning và latent attention cho VLM; áp dụng đa phương thức tự nó không mới. |
| Latent reasoning audio–visual và đồng bộ thời gian | [LatentOmni](https://arxiv.org/abs/2605.22012) | Đã có latent AV reasoning, feature supervision và temporal synchronization. Phương án ở đây training-free và residual steering nên khác, nhưng không thể claim first AV latent reasoning. |
| “Unsilencing” latent để cải thiện answer | [Visual Latents Know More Than They Say](https://arxiv.org/abs/2605.02735) | Paper tối ưu latent bằng contrastive alignment và confidence progression; code hiện không triển khai objective này. Cùng động cơ chưa phải tái hiện kết quả paper. |
| M1: prior-adjusted scoring | [Surface Form Competition](https://aclanthology.org/2021.emnlp-main.564/) | PMI và prior correction đã có tiền lệ; ở đây dùng text-only question/options prior và alpha khác. Không nên claim PMI là đóng góp mới. |

Đối chiếu này không phải tìm kiếm exhaustive hay chứng nhận quyền ưu tiên. Nó đủ để bác bỏ claim novelty rộng cho weighted soft embeddings hoặc AV latent reasoning, nhưng chưa đủ để khẳng định không có bất kỳ combination novelty nào.

## 4. Novelty hiện có và novelty còn cần xây

**Chưa thể claim mới:** continuous thought; top-k expected embedding; prior subtraction; additive attention masking; activation steering; hoặc việc dùng latent trong AV nói chung. Fix dấu attention suppression, last_media, lm_head memory, sdpa compatibility và chạy được T4 là công việc kỹ thuật quan trọng, nhưng tự nó chưa phải contribution nghiên cứu CVPR.

**Ứng viên novelty phương pháp:** từ cùng một mẫu, tạo delta giữa AV factual/sync/anti ở block22 rồi dùng nó hướng dẫn các block sớm hơn, đồng thời giữ dấu nguồn và thời gian. Đây là cách kết hợp cụ thể có thể khác các bài đã kiểm tra. Nhưng recurrence contribution đang confounded và TF suy giảm, nên hiện mới là ứng viên, chưa là đóng góp đã chứng minh. Tương tự, cross-modal temporal matching từ pooled AV logit-lens features là một adaptation cụ thể cần baseline so với raw hidden cosine, projected cosine và random clocks.

**Ứng viên novelty phân tích:** đồng bộ cục bộ giúp CMTB nhưng có thể đánh đổi với temporal order; hard gate có thể biến cải thiện nhãn Present thành collapse. Nếu lặp lại ở nhiều backbone và dùng causal intervention tách nguyên nhân, đây có thể thành kết quả phân tích có giá trị. Trên40 mẫu, gọi là quan sát phát triển ban đầu là chính xác hơn.

Không nên dùng “zero modality gap”: cùng không gian tọa độ không đảm bảo semantic alignment. Top-k projection cũng mất thông tin; nó không chứng minh giữ nguyên mọi “superposition” của trạng thái gốc. Probe tốt ở block trung gian không đồng nghĩa lm_head chưa huấn luyện cho vị trí đó đọc được event concepts; “single-pass depth paradox” vẫn cần kiểm chứng nhân quả bằng patching có đối chứng.

## 5. Mức độ phù hợp với CVPR

**Phù hợp về chủ đề; chưa sẵn sàng về bằng chứng và đóng góp.** Phương pháp có raw-AV grounding, temporal reasoning và phân tích failure mode của MLLM, đều có nội dung thị giác rõ. Điểm yếu hiện tại là novelty chưa tách khỏi prior art, nhánh unified không thắng thành phần đơn, temporal collapse và thiếu kiểm chứng ngoài development.

[Hướng dẫn CVPR2026](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines) cân nhắc technical soundness, contribution, novelty và impact; không quy định phải đạt một accuracy cụ thể hay luôn hơn SOTA. Vì thế65% không tự nó làm bài đủ hay thiếu chuẩn. Mốc30/40 và3/5 mỗi cell là mục tiêu development của repo, không phải luật CVPR. Với bằng chứng hiện nay tôi khuyên tiếp tục nghiên cứu trước khi chuẩn bị submission main conference; không dự đoán xác suất nhận bài hoặc thay cho quyết định reviewer.

## 6. Góp ý cho vòng tiếp theo

1. Chọn đóng góp hẹp: bảo toàn evidence trước/trong/sau khi tăng synchrony grounding, hoặc một feedback map được chứng minh dùng được giữa tầng. Bỏ các claim first/zero-gap/superposition và diễn đạt đúng residual intervention thay “Value-Clock” nếu chưa sửa v_proj.
2. Ablation riêng hard gate, audio blocking, clock, vocabulary projection, foreground, PMI và feedback. Đo nguyên nhân collapse trước khi tăng số hyperparameter. Dùng random/shuffled vector, cùng norm và cùng số forward passes làm controls.
3. Thử bản giữ cả ba miền thời gian, có provenance riêng cho audio/video thay hard-delete ngoài cửa sổ; đây là hướng đề xuất mới cho vòng tới, chưa có kết quả trong lần chạy này. Kiểm tra music/noise và nhiều speech spans vì energy window hiện không phải VAD xác nhận speech.
4. Khảo sát riêng M9: nó cải thiện TF và hồi phục Past3/5, nhưng phải tái lập trên held-out; không chọn method theo nhãn hoặc cherry-pick mẫu.
5. Đóng băng hyperparameter trên dev; dùng video/nguồn độc lập cho test, ưu tiên đủ mẫu cho mọi cell và model khác. So prior art trực tiếp và controls ngang compute; báo macro/per-cell accuracy, confusion, rescue/regression, uncertainty, runtime và VRAM từng method.70s/sample hiện là cả harness10 method, không phải runtime riêng của M2.
6. Nếu claim semantic alignment, đo event-window retrieval với timestamp annotations và audio shift/reversal interventions, không chỉ answer accuracy. Nếu claim layer feedback, kiểm tra layer index, map/projection và độ nhạy beta; không lấy linear probe accuracy làm bằng chứng thay thế.

Tôi khuyên ưu tiên giải thích và sửa collapse trước khi ghép thêm thành phần. Kết quả26/40 tạo lý do tiếp tục nghiên cứu CMTB, nhưng chưa chứng minh LatentChrono-CoT là lời giải tổng quát cho temporal AV reasoning.

## 7. Artifact và phạm vi tái lập

Manifest SHA256: `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.
Raw results: [server1](results_s1.jsonl), [server2](results_s2.jsonl).
Counts: [summary](summary.json). Predictions: [CSV](per_sample.csv).
Nguồn đề xuất: [nguyên văn](proposal_original.txt), [Python](proposal_original.py).
Code đã chạy: [runner](../../server_jobs/argon_latent40.py).

Code đã sửa last_media, dùng manifest40 thay random32, raw AV thay cut/reencode, prompt letter-only chung, decoder replay và bounded-memory SDPA. Không có sample skipping/replacement. Cả110 rotations native/decoder replay logit error0.0; không OOM. Max sequence7113, fp16 dense mask96.5MiB/device. Baseline hiện18/40; baseline lịch sử19/40 do1 mẫu đổi dự đoán. Kết luận ghép cặp dùng M0 của cùng phiên này.

Nới causal V->A gần đồng bộ của code gốc được giữ ở các mask có desmear: đây là thay đổi connectivity của pretrained decoder, phải xem như một thành phần cần ablation. Không diễn giải mọi chênh lệch điểm là tác dụng độc lập của “latent reasoning”.

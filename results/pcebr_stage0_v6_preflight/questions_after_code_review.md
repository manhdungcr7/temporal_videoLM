# Gửi Yoshua Bengio — đối chiếu source v6 với proposal, cần chốt trước khi chạy

Yoshua, đã nhận source v6 đầy đủ. Tôi đã đọc và đối chiếu tĩnh với proposal trước đó. **Có các sai khác ảnh hưởng trực tiếp đến thuật toán và controls, nên theo yêu cầu người dùng, tôi dừng trước thực nghiệm và hỏi lại cậu. Chưa import/chạy source, chưa chạy M0/M1/M2/M3, chưa gửi job lên hai server và chưa tự sửa phương pháp.**

Source nhận nguyên byte được lưu ở [stage0_pcebr_v6_original.py](stage0_pcebr_v6_original.py), SHA256 `d4899261862df5e0bf6ba890d722b212bcd26f8ebc2090454fe6dcf191871c74`. Proposal đối chiếu là [proposal_original.txt](proposal_original.txt), SHA256 `16a414ff3cff51119b1f0669391d346a0ed8973b359dc97ab01406e28ea76b27`. Đây là bản hỏi bổ sung sau khi đã nhận code; vấn đề thiếu source ở lượt trước đã được giải quyết.

Các phần đã hiện thực rõ hơn và đúng hướng trong source: C4/C5 không còn trừ text mean trên media; bridge dùng retrieved mixtures và correlation loss; có B7 no-Phi matched trong từng suite; CAOR nhận native Q/K/V của từng layer; shared logsumexp normalization xác định được cách chuyển mass audio/video mà giữ tổng media unnormalized mass. Các điểm dưới đây cần được quyết định trước khi đánh giá.

## 1. CAOR đặt eta trong logits hay trộn hai attention outputs?

**Proposal:** `p_tilde = softmax(log p + eta * [kappa_sink*(a-mean(a)) + beta_ev])`, với `a = log p(r,media) - .65*log p(r0,media)`. Sau đó cập nhật bằng exact attention-output difference.

**Code:** [profile builder](stage0_pcebr_v6_original.py:2047) đặt gate vào token biases/sink weight; [kernel](stage0_pcebr_v6_original.py:2459) dùng `lv_cf = lv_nat - sink_w*sink_v + tok_bias_v`, audio tương tự. Sau softmax và `P_cf@V`, [dòng2479](stage0_pcebr_v6_original.py:2479) mới trộn `(1-eff_eta)*orig_slice + eff_eta*updated`, trong đó `eff_eta=.70*gamma_gate`.

Đây là hai thuật toán khác nhau vì softmax phi tuyến. Không thể coi trộn outputs sau softmax là cùng phép đặt eta trong logits. Code cũng chưa có thành phần tăng cường log-p của chính query row trong `kappa_sink*(a-mean(a))`; nó chỉ trừ centered delimiter logits và thêm profile. Hệ số sink trong kernel là `.60*gamma`, còn `.65` của proposal xuất hiện ở phép khử hướng query trong bước salience, không phải cùng vị trí toán học.

**Câu hỏi:** Cậu chọn công thức proposal hay công thức code làm phương pháp chính? Nếu chọn code, nhờ cập nhật rõ hiệu dụng `P_eff=(1-eff_eta)P_native+eff_eta P_cf`, các lần nhân gate và toàn bộ coefficients. Nếu chọn proposal, nhờ gửi patch để eta và query-dependent de-sinking thực sự nằm đúng chỗ. Cả hai có thể có local mass-preservation, nhưng kết quả và cách diễn giải sẽ khác nhau.

Ngoài ra, code thêm `late_aud_damp` ở layers>=20, trong khi phần công thức chính không nêu quy tắc này. Profile dùng linear tilt trên toàn clip `slot_ord_v=ord_tilt*t_tilde`, `slot_ord_a=-.85*ord_tilt*t_tilde`, thay vì một cửa sổ quanh anchor có dịch thời gian như prose mô tả. Nhờ chốt đây là lựa chọn chủ đích hay phần còn thiếu; không tự thêm cửa sổ mới khi chạy.

## 2. Xác nhận chi phí inference và định nghĩa r0

[M3 dòng2247–2276](stage0_pcebr_v6_original.py:2247) chạy native forward lấy states ở input self-attention18/27, sau đó mới tạo evidence/profile và chạy routed forward từ block14. Như vậy một nhánh trả lời cần native prepass + routed pass. Wrong-query semantic là forward diagnostic bổ sung và cần được báo cáo riêng. Việc lấy delimiter logits trong routed pass không cần null-prompt pass là một lợi ích có thật, nhưng chưa làm toàn pipeline thành một forward.

**Câu hỏi:** Cậu xác nhận đổi mô tả thành “de-sinking không thêm null-prompt pass; inference chính có prepass” chứ? Nếu định làm one-pass toàn pipeline, cần source tạo profile online từ tensors đã có tại thời điểm block14, không phụ thuộc q27 chưa tính.

Code dùng `payload['last_media']`, nhưng không kiểm tra `input_ids[last_media]` có phải `<|vision_eos|>` hay không. Nhờ chốt muốn dùng last media embedding hay delimiter đóng khối; hai loại token không thể coi mặc nhiên như nhau. Question-independent cũng chưa chứng minh pure structural sink; cần coi phần đó là giả thuyết và có control. Kernel chỉ lấy media columns của logits_r0, nên tôi không kết luận nó đã đọc future question chỉ vì chưa áp causal mask đầy đủ cho r0.

## 3. Phi_EMD: adjacent-band hay Gaussian; CDF theo index hay physical time?

**Sai khác band:** Proposal định nghĩa `sync_banded` bằng tổng đúng các cặp `|i-j|=1` trừ uniform. [Code dòng200](stage0_pcebr_v6_original.py:200) dùng Gaussian `exp(-.5*(abs_d/1.15)^2)`, có cả diagonal và các khoảng cách xa hơn một bin. Đây là feature khác, không phải cùng một cách viết.

**CDF:** Khi K1=K2, [dòng209–217](stage0_pcebr_v6_original.py:209) tính cumsum trực tiếp theo thứ tự phần tử và lấy mean; không sử dụng c1/c2 hay khoảng cách support cho hai kênh W. Khi K1!=K2, [dòng220](stage0_pcebr_v6_original.py:220) interpolate theo index rồi normalize, vẫn không dùng physical support. Trong khi đó exact/band/order kernels lại dùng centers thực. Chúng đang trộn hai cách định nghĩa thời gian.

Phản ví dụ suy ra trực tiếp từ công thức, **chưa chạy như một test**: p1=p2=[1,0], bank1 có intervals[0,1],[1,2], bank2 có[10,11],[11,12]. Hai CDF theo index giống nhau nên code cho W_plus=W_minus=0, dù hai point masses ở centers.5 và10.5 cách nhau10giây. Code W đúng cho một shared regular rank grid thích hợp; nó không tự là physical 1D-Wasserstein của mọi `iv_1/iv_2` mà hàm nhận. Coordinate shuffle thay iv_2 cũng không đổi hai kênh CDF nếu giữ nguyên p2/order mảng.

**Câu hỏi:** Cậu muốn giữ rank-based feature và đổi claim, hay thực sự tính Wasserstein trên common physical support? Nếu chọn physical time, nhờ cung cấp quy tắc hợp/sort support, resampling hay tích phân CDF có trọng số khoảng cách, cùng normalization và uniform reference. Đồng thời chọn adjacent-only hay Gaussian và cập nhật code/proposal cho khớp.

Tên “full-rank6D” và claim During không bao giờ nhầm vẫn chưa được xác nhận: M0 chỉ lưu `phi_samples` ở K6 cho3 seeds, không tính rank; và uniform joint gate0 vẫn làm mọi Phi thành0. Cần hạ các khẳng định tuyệt đối thành giả thuyết có miền áp dụng.

## 4. C8b có được phép shuffle cả global branch không?

**Proposal:** C8b chỉ hoán vị innovation x_rec, giữ nguyên global branch để tách contribution innovation.

**Code:** [train_bilevel_innovation_aligner dòng1037](stage0_pcebr_v6_original.py:1037) truyền `shuffle_within_rec=shuffle_within_rec` cho cả `m_glob` và `m_innov`. C8b được gọi vớiTrue ở [dòng1128](stage0_pcebr_v6_original.py:1128), nên cả x_ctx–text matching của global branch cũng bị tráo. Đây không còn là control chỉ phá innovation.

Docstring còn nói innovation “trained exclusively with within-recording loss”, nhưng [dòng1044](stage0_pcebr_v6_original.py:1044) đặt weight.75; [loss dòng1006](stage0_pcebr_v6_original.py:1006) vẫn có.25 global loss.

**Câu hỏi:** Cậu muốn giữ global branch giống C6 và chỉ shuffle innovation như prose, hay chấp nhận control shuffle cả hai? Nếu chỉ innovation, nhờ gửi patch rõ seed/training budget và loss weight; nếu muốn innovation vẫn có.25 global loss, sửa mô tả “exclusively”.

## 5. C6 có hai định nghĩa similarity không tương đương

[BilevelGlobalInnovationAligner dòng883–898](stage0_pcebr_v6_original.py:883):

```text
encode_text:   zt_global
encode_media:  normalize(zm_global + beta*zm_innov)
score_matrix:  (score_global + beta*score_innov)/sqrt(1+beta^2)
```

`score_innov` dùng text encoder của innovation branch, khác text encoder global. Vì vậy nói chung `score_matrix(text,media)` không bằng `encode_text(text) @ encode_media(media).T`.

M1 và Natural/Reversal dùng score_matrix. Bridge training và TwoHop Roll dùng encoded vectors/dot product ở [dòng1263](stage0_pcebr_v6_original.py:1263) và [dòng1495](stage0_pcebr_v6_original.py:1495). Cùng object C6 nhưng không cùng score definition. B7/B8 trong từng suite vẫn được xây từ cùng episodes, nên tôi không gọi riêng cặp no-Phi/full đó là unmatched; vấn đề là claim chuyển đúng teacher C6 xuyên M1 và toàn bộ M2 cần được giới hạn hoặc sửa.

**Câu hỏi:** Cậu chủ ý dùng late score fusion cho direct retrieval và một embedding fusion khác cho multi-hop, hay muốn một scorer duy nhất? Nếu scorer duy nhất, nhờ chốt encoding/fusion sao cho hai interfaces nhất quán; nếu chủ ý khác nhau, cần tên/đối chứng riêng thay vì mặc định cùng teacher/readout.

## 6. Chốt các hyperparameters và filter đang khác prose

- Proposal nêu ZCA shrink.35; [dòng810/813](stage0_pcebr_v6_original.py:810) dùng.22.
- Proposal C6 ghi `x_ctx+.25*x_rec`; [dòng1047](stage0_pcebr_v6_original.py:1047) tạo beta_innov=.20, ngoài khác biệt giữa late score fusion và embedding fusion ở câu5.
- Proposal bridge training lọc `classify_interval_relation>=0`; [vòng tạo train pairs dòng1300](stage0_pcebr_v6_original.py:1300) ghép tất cả cặp strings unique/non-null mà không gọi relation filter. Relation filter chỉ có trong episode builder đánh giá ở dòng1560.
- “Unambiguous” hiện là exact normalized-string uniqueness trong từng modality, chưa xử lý paraphrase hoặc event-instance identity độc lập; cần giữ đúng tên proxy.

**Câu hỏi:** Cậu chọn giá trị/filter nào? Nhờ cập nhật source và prose tương ứng để không có sự thay đổi phương pháp ngầm trong wrapper.

## 7. Controls và các phép kiểm tra được hứa chưa đủ/không có trong đường chạy

- Proposal nêu `legacy_v5_vo_bivector`. Trong source không có nhánh chạy legacy-v5 thực sự; tạo `vec_fwd` bằng hàm đã đổi sang EMD và salience mới không phải replay nguyên v5. Nhờ xác nhận thêm exact legacy replay hay dùng kết quả archive có ghi rõ provenance.
- Random/irrelevant chuẩn hóa theo một norm của visual slot vector, dùng chung norm đó cho audio và đặt floor.5 ở [dòng2118/2129](stage0_pcebr_v6_original.py:2118). Full còn có token innovation/sink/audio damping mà random không có. Vì vậy chưa phải matched norm của toàn logit perturbation theo token/head/row. Nhờ định nghĩa muốn match ở tensor nào và bỏ claim “matched100%” nếu chỉ cùng eta.
- `reverse_order` chỉ đổi dấu rel_scalar; không đảo lại toàn bộ evidence/innovation profile. Nhờ chốt tên đúng là order-sign control hay physical/full-profile reversal.
- [M0 dòng573](stage0_pcebr_v6_original.py:573) chưa có kiểm tra bảo toàn media/non-media mass như prose tuyên bố. [Zero-scale dòng2406](stage0_pcebr_v6_original.py:2406) đi tắt trực tiếp native nên không kiểm chứng no-op parity của kernel khi recompute logits/P@V. Cần control đi qua chính wrapper với perturbation0 nhưng eta khác0, cùng mask/scaling/dtype/backend, rồi mới dùng claim exact attention-output differential trên runtime.

**Câu hỏi:** Nhờ cậu chốt/bổ sung các controls này và các tolerance M0 float64 so runtime float32/FP16. Bảo toàn local matrix trước/sau perturbation với cùng Q/K/V không bảo đảm mọi attention non-media ở các tầng sau vẫn giống native full-model.

## 8. Bộ chấm và denominators chẩn đoán cần ghi đúng phạm vi

TF evaluator mới vẫn có fallback tìm từ after mà không buộc đúng event/reference. Phản ví dụ gốc “After the speech, the camera changes angle.” với gold Future vẫn có đường qua fallback đó theo đọc source; M0 đang dùng một câu khác về subject inversion. Nhờ giữ nguyên cả ba counterexamples đã archive, thêm case sai reference ID1821, và không thay test cũ bằng test mới dễ hơn. Đây là nhận xét từ phân tích nhánh regex, chưa chạy evaluator để báo số pass/fail.

Hop counters được cộng trong `make_ep` [dòng1507](stage0_pcebr_v6_original.py:1507), trước khi cắt buckets cân bằng ở [dòng1605](stage0_pcebr_v6_original.py:1605). Vì vậy denominator có thể chứa các episodes không vào tập eval cuối. Nhờ ghi rõ pre-balancing population, hoặc tính diagnostics trên đúng retained episodes; không gộp hai tập mẫu số.

Nhờ cậu trả lời/sửa các điểm1–7 trước khi chạy, đặc biệt CAOR eta, Wasserstein physical support và C8b. Khi đã có một source/spec khớp, chúng tôi sẽ thực hiện đúng cấu hình được chốt. **Lượt này chỉ review tĩnh, chưa có kết quả thực nghiệm v6.** Tên Yoshua/Yann là quy ước trao đổi, không xác nhận tác giả thật của code.

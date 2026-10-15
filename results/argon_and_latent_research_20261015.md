# Đánh giá Argon CCLD và các lượt tiếp nối trên development40

2026-10-15. Toàn bộ các suite dưới đây hoàn tất đúng40/40 ID, mỗi ô task/ground-truth có5 mẫu; raw video và decoded audio đều<=45s. Mọi suite dùng tất cả110 cyclic option rotations. Không thay mẫu khi lỗi.

Manifest SHA256: `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.

**Kết quả chính: C2 và C7 của Argon đạt26 đúng/14 sai, ngang tổng M2 cũ. C3/C5 vẫn dự đoán Present15/15. Chưa tìm được phương pháp giải quyết tốt đồng thời các ô temporal và đủ bằng chứng triển khai đa ngữ cảnh.**

40 mẫu này đã được dùng nhiều lần để chọn và phân tích phương pháp. Chúng là development, không phải held-out. Không chọn method theo task/nhãn hoặc ghép đáp án bằng oracle. Độ đa dạng câu hỏi thực tế chưa được xác nhận.

## Đề xuất Argon: kết quả đúng/sai từng ô

TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future; CMTB:A/B/C là nội dung lựa chọn riêng từng câu. Trong các ô, `đúng/sai` cộng lại bằng5.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|R4_C0_baseline|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R4_C1_m9_thought_ctrl|4/1|3/2|3/2|2/3|3/2|2/3|2/3|0/5|19/21|6/4/5|
|R4_C2_m8_noclk_nopmi|1/4|5/0|4/1|1/4|0/5|5/0|5/0|5/0|26/14|11/4/0|
|R4_C3_causal_vis_rec|5/0|3/2|0/5|5/0|0/5|3/2|2/3|3/2|21/19|0/15/0|
|R4_C4_shuffled_rec_ctrl|5/0|0/5|0/5|1/4|3/2|3/2|2/3|0/5|14/26|4/1/10|
|R4_C5_m9_plus_causal_rec|5/0|3/2|0/5|5/0|0/5|3/2|1/4|4/1|21/19|0/15/0|
|R4_C6_ccld_unmasked|3/2|4/1|2/3|4/1|0/5|4/1|3/2|5/0|25/15|5/10/0|
|R4_C7_ccld_late_handoff|3/2|5/0|2/3|4/1|0/5|4/1|3/2|5/0|26/14|5/10/0|

![Recall từng ô và phân phối nhãn TF](argon_ccld40/figures/ccld_development40.png)

## Đánh giá, nhận xét và góp ý với Argon

C2 đạt CMTB15/15 nhưng TA/Yes1/5 và TF/Future0/5; TF dự đoán Past11,Present4,Future0. Đây là thay đổi đánh đổi, không phải giải quyết joint temporal reasoning.

C3/C5 không dùng speech-window trong đường suy luận tương ứng, nhưng vẫn collapse Present15/15. Vì vậy phép trừ visual-blind không bảo đảm giữ đúng temporal evidence hay loại bỏ thiên lệch. C3 hơn shuffled C4 thêm7 câu; một hướng có ý nghĩa hơn nhiễu chưa đủ chứng minh cơ chế nhân quả hoặc độ chính xác cao.

C6/C7 vẫn lấy45% delta từ hard_sync theo cửa sổ; unmasked pass2 không làm toàn bộ thuật toán window-free. C7 còn giảm audio từ block16 bằng hằng số, chưa phù hợp mọi hướng reasoning hoặc câu hỏi cần nghe chi tiết. C6/C7 đều TF/Future0/5. C7 chỉ khác C6 trên một dự đoán TA/B idx2054 (sai thành đúng); không đổi câu TF nào.

Phân biệt giả thuyết với chứng minh: hiệu hidden state của một mạng phi tuyến không triệt sạch prior và không bảo toàn100% tín hiệu. H_Q trong triển khai Argon gồm stem, choices và format; tác dụng lên readout lựa chọn có thể lẫn với grounding. C2 cũng bỏ foreground/desmear; C1 thought-window/.22 khác C5 global/.20, nên các đối chứng chưa hoàn toàn trực giao.

Lượt D1 giữ gate/foreground/PMI nhưng bỏ clock đạt27/40 và vẫn Present15/15. Nó chứng minh lexical clock không cần thiết cho collapse M2 trong cấu hình này; không xác định riêng tác dụng clock ở delta M8. M9 tái lập19/40, TF8/15, chưa chứng minh TMRoPE tự suy luận đúng mọi quan hệ thời gian.

So native cùng lượt, C2 cứu17/làm sai9 và C7 cứu16/làm sai8. Exact paired p lần lượt.1686/.1516; toàn bộ7 phép so sánh C1–C7 có Holm-adjusted p=1.0000. Không coi đây là chứng cứ gain ổn định; hiệu chỉnh nội bộ này cũng chưa xử lý toàn bộ quá trình chọn nhiều lượt.

## Đáp ứng lưu ý về tổng quát và CVPR

C3/C5 đáp ứng hình thức tốt hơn: fullAV ở pass cuối, không router theo task, không VAD window hoặc từ đáp án. Nhưng cơ chế ưu tiên phần đóng góp visual và kết quả chưa chứng minh hiệu quả cho audio-led, visual-led, reverse lookup, nhiều sự kiện, count/duration, causal hoặc free-form QA.

C1/C2/C6/C7 còn dựa cửa sổ. C7 có chính sách giảm một nguồn cố định. Không nên trình bày các cấu hình này như phương pháp đáp ứng đầy đủ yêu cầu nhiều ngữ cảnh của người dùng.

Chưa có cấu hình nào được xác nhận đủ mới, đủ mạnh hoặc sẽ được nhận CVPR. Các primitive counterfactual, activation steering, weighted vocabulary embedding và latent optimization đã có tiền lệ. Đóng góp cần được xác định cụ thể và kiểm chứng trên held-out, đa benchmark và can thiệp thời gian có kiểm soát.

## Các hướng riêng đã hiện thực và chạy sau Argon

|Suite|Cấu hình nổi bật|Kết quả|Điểm thất bại chính|
|---|---|---:|---|
|D/G|D1 bỏ clock; D2 mở lại từ24|27/40 mỗi cấu hình|TF vẫn Present15/15|
|S|S2 tương tác cùng tầng|22/40|TF/Present0/5; S1/S3/S4/S5/S6 lần lượt19/18/19/16/16|
|Q|Q2 prototype theo stem|18/40|Ngang native; Q1/Q3/Q4/Q5 đều17|
|L|L2 input slots được tối ưu|18/40|L1 chưa fit19, L3 đảo target17; feature self-match không thành QA gain|
|E|E2 cùng slots, đọc từ15|17/40|E1 chưa fit16, E3 đảo target17; E4 đọc từ23=18|
|P|P2 presentation-time|18/40|P1 frame-index=19/40; input correctness control|

Tối ưu latent đã chạy32 Adam steps/clip/variant, source-only teachers ở block22, fixed2s slots, cosine+within-source contrastive loss, backbone đóng băng. Tất cả parameters được giữ nguyên khi so E2/E4. Video slot self-match từ113/527 lên458/527; audio từ257/514 lên440/514, nhưng L2 vẫn18/40 và giữ nguyên từng argmax trên40 câu. Mean TV của phân phối lựa chọn=.0463; xác suất có đổi, không suy ra mọi layer bỏ qua memory. Đó là bằng chứng mục tiêu tái dựng nội bộ chưa đủ cho downstream temporal QA ở cấu hình này, không phải bằng chứng mọi latent optimization đều thất bại.

E2 so với E4 chỉ cứu1 câu, làm sai2; parity dựng lại read23 bằng0. Đổi riêng giai đoạn đọc không sửa được thất bại của các slot đã fit. Cần mục tiêu grounding/decision có supervision độc lập thay vì chỉ lặp lại feature của chính model.

## Research đã dùng và ranh giới với hướng tương lai

[LatentOmni](https://arxiv.org/html/2605.22012v1) kết hợp latentAV với feature supervision và đồng bộ vị trí. [Unsilencing Visual Latents](https://arxiv.org/html/2605.02735v1) đã có inference-time latent optimization trong backbone đóng băng; không claim primitive đó mới.

[RED](https://arxiv.org/html/2610.02976v1) nghiên cứu đóng góp audio/video/interaction ở decoding; [Activation Steering Decoding](https://aclanthology.org/2025.acl-long.634/) là tiền lệ steering hidden states. Các nguồn này không chứng minh công thức CCLD hoạt động trên40 mẫu của chúng ta.

[AHA](https://arxiv.org/html/2512.24052v2) dùng preference hard negatives cho temporal/audio hallucination trên Qwen2.5-Omni-7B. Đó là hướng supervision liên quan, nhưng audio-only và thiết kế dữ liệu của paper không tự giải quyết jointAV tổng quát ở model3B. Không chạy checkpoint7B thay model mục tiêu hoặc tuyên bố đã tái lập DPO.

[PyAV time documentation](https://pyav.org/docs/stable/api/time.html) giải thích pts*time_base. Đối chứng P giữ audio và budget nguyên, tách PyAV index/PyAV PTS. Nó kiểm tra clock đầu vào, không làm oracle xác định sự kiện.

Hướng tiếp theo có căn cứ hơn: huấn luyện riêng adapter/latent reasoner trên train pool tách video/source khỏi40 mẫu, với segment grounding có kiểm chứng và temporal counterfactual hard negatives giữ nguyên nội dung. Bao phủ cả hai hướng modality, nhiều sự kiện và quan hệ count/duration/order; không đặt luật suy luận theo câu chữ của SVHalluc. Phải đánh giá đúng sự kiện và khoảng thời gian cùng câu trả lời. Đây là đề xuất chưa được chạy; không đưa điểm dự kiến hoặc gọi là phương pháp thành công.

[Thiết kế chi tiết và các điều kiện chống supervision sai](latent_research_next_grounded_training.md).

## Tái lập và artifacts

Baseline Argon và M9 giống lịch sử trên từng ID (0/40 thay đổi). Native replay CCLD max0.0; shuffled row-norm error5.72e-6. Suite S v1 gặp prefix replay.03125>.03; giữ log riêng implementation_v1, v2 dùng full fallback và không nới ngưỡng. Các sửa kỹ thuật không thay mẫu hoặc scale.

C/S/Q/L/E đều có JSONL từng sample, mọi110 rotation JSON và audit tính lại điểm. L có40 checkpoint input slots lưu local trong latent_slot_opt40/fits, tổng25,741,416bytes, SHA từng file ở fit_archive_audit.json. Chỉ input slots được fit; không sửa backbone.

Server1 mất tunnel ở cuối đối chứng P, khi19/20 sample đã lưu local. Đúng ID còn thiếu1898 được chuyển sang server2, cùng code toán học và C0 tham chiếu đã archive; không thay mẫu. P có đủ40 aggregate records/110 rotation audit entries, nhưng raw per-rotation files của19 mẫu server1 chưa tải được trước khi tunnel offline. Không gọi các aggregate records là đã archive đủ raw110 rotation scores; xem recovery_protocol.json và raw archive coverage riêng. Điều này không ảnh hưởng raw110 của Argon hoặc các suite C/S/Q/L/E đã tải xong.

Code trong server_jobs; mỗi folder có protocol trước chạy, report, summary, per_sample.csv, job statuses và rotations. proposal_original.txt/.py là bản Argon nguyên gốc; không thực thi phần sampling/xóa folder của code đó. Thay bằng đúng manifest40<=45s và native raw input theo yêu cầu người dùng. Những persona/tên tác giả trong đề xuất không phải xác nhận thật của nhà nghiên cứu hoặc hội nghị.

## Bảng đầy đủ từng suite

### D/G: bỏ clock, mở lại query và đọc evidence cục bộ

[Báo cáo](latent_reopen40/report.md) · [Protocol](latent_reopen40/protocol.md) · [Số liệu](latent_reopen40/summary.json)

Xem report.md cho bảng đúng/sai.

### C0–C7: đề xuất CCLD của Argon

[Báo cáo](argon_ccld40/report.md) · [Protocol](argon_ccld40/protocol.md) · [Số liệu](argon_ccld40/summary.json)

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|R4_C0_baseline|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R4_C1_m9_thought_ctrl|4/1|3/2|3/2|2/3|3/2|2/3|2/3|0/5|19/21|6/4/5|
|R4_C2_m8_noclk_nopmi|1/4|5/0|4/1|1/4|0/5|5/0|5/0|5/0|26/14|11/4/0|
|R4_C3_causal_vis_rec|5/0|3/2|0/5|5/0|0/5|3/2|2/3|3/2|21/19|0/15/0|
|R4_C4_shuffled_rec_ctrl|5/0|0/5|0/5|1/4|3/2|3/2|2/3|0/5|14/26|4/1/10|
|R4_C5_m9_plus_causal_rec|5/0|3/2|0/5|5/0|0/5|3/2|1/4|4/1|21/19|0/15/0|
|R4_C6_ccld_unmasked|3/2|4/1|2/3|4/1|0/5|4/1|3/2|5/0|25/15|5/10/0|
|R4_C7_ccld_late_handoff|3/2|5/0|2/3|4/1|0/5|4/1|3/2|5/0|26/14|5/10/0|

### S0–S6: đúng tầng, norm giới hạn và tương tác đối xứng

[Báo cáo](symmetric_latent40/report.md) · [Protocol](symmetric_latent40/protocol.md) · [Số liệu](symmetric_latent40/summary.json)

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|S0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|S1_visual_same_layer|0/5|5/0|2/3|0/5|4/1|3/2|2/3|3/2|19/21|4/1/10|
|S2_joint_same_layer|1/4|4/1|3/2|0/5|4/1|4/1|2/3|4/1|22/18|5/0/10|
|S3_orthogonal_joint|2/3|3/2|1/4|1/4|4/1|3/2|2/3|2/3|18/22|1/3/11|
|S4_two_layer_orthogonal_joint|3/2|3/2|1/4|1/4|4/1|3/2|2/3|2/3|19/21|1/3/11|
|S5_question_stem_joint|3/2|3/2|0/5|1/4|4/1|3/2|2/3|0/5|16/24|1/5/9|
|S6_question_stem_argon_delta|0/5|5/0|0/5|2/3|3/2|3/2|3/2|0/5|16/24|0/7/8|

### Q0–Q5: thought theo câu hỏi và đối chứng scaffold

[Báo cáo](query_latent40/report.md) · [Protocol](query_latent40/protocol.md) · [Số liệu](query_latent40/summary.json)

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|Q0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|Q1_global_prototype|5/0|2/3|0/5|2/3|3/2|3/2|2/3|0/5|17/23|1/7/7|
|Q2_question_prototype|3/2|3/2|1/4|3/2|3/2|3/2|2/3|0/5|18/22|2/8/5|
|Q3_question_evidence_contrast|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|
|Q4_contrast_plus_joint|3/2|3/2|1/4|1/4|4/1|3/2|2/3|0/5|17/23|2/4/9|
|Q5_shuffled_thought|2/3|3/2|1/4|2/3|3/2|3/2|2/3|1/4|17/23|1/6/8|

### L0–L3: thật sự tối ưu input latent, backbone đóng băng

[Báo cáo](latent_slot_opt40/report.md) · [Protocol](latent_slot_opt40/protocol.md) · [Số liệu](latent_slot_opt40/summary.json)

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|L0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|L1_unoptimized_memory|4/1|3/2|0/5|2/3|4/1|3/2|2/3|1/4|19/21|0/6/9|
|L2_optimized_memory|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|L3_time_shuffled_target|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|0/4/11|

### E0–E5: giữ nguyên slots, đọc sớm hơn

[Báo cáo](latent_slot_read15/report.md) · [Protocol](latent_slot_read15/protocol.md) · [Số liệu](latent_slot_read15/summary.json)

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|E0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|E1_unoptimized_read15|4/1|3/2|0/5|2/3|2/3|3/2|2/3|0/5|16/24|1/7/7|
|E2_optimized_read15|4/1|3/2|0/5|1/4|4/1|2/3|2/3|1/4|17/23|2/3/10|
|E3_shuffled_read15|5/0|1/4|1/4|1/4|4/1|2/3|2/3|1/4|17/23|2/3/10|
|E4_optimized_read23_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|E5_unoptimized_read23_reference|4/1|3/2|0/5|2/3|4/1|3/2|2/3|1/4|19/21|0/6/9|

### P0–P2: kiểm tra đồng hồ frame đầu vào

[Báo cáo](presentation_resample40/report.md) · [Protocol](presentation_resample40/protocol.md) · [Số liệu](presentation_resample40/summary.json)

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|P0_original_native_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|P1_pyav_frame_index|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/4/10|
|P2_pyav_presentation_time|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/4/10|

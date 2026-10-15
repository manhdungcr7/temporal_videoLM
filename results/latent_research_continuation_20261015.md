# Tiếp tục nghiên cứu latent reasoning,15/10/2026

Mục tiêu gốc: giảm jointAV temporal hallucination trên Qwen2.5-Omni-3B, theo
hướng tổng quát, không router/fix theo câu hỏi SVHalluc. Dùng frozen40<=45s,
năm mẫu mỗi ô; không đổi sample khi lỗi. Manifest0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459.

Yêu cầu mới: ưu tiên tám cấu hình CCLD của Argon trong attachment; lưu nguyên
đề xuất, kết quả đúng/sai từng ô và đánh giá tính tổng quát/CVPR. Trong lúc đợi,
chuẩn bị và xếp phương pháp riêng ngay sau Argon để GPU không nhàn rỗi.

Các phase theo thứ tự FIFO trên mỗi server:
1. latent_reopen40: hoàn tất40/40. D1 no-clock27/40, D2 reopen24 27/40 nhưng
   cả hai vẫnPresent15/15 TF. D3 20/40; G1/G2 19/40; G3 21/40. Baseline18,
   M2 control26. Xem results/latent_reopen40/report.md.
2. argon_ccld40: original proposal/code và protocol ở results/argon_ccld40.
   Jobs S1a364decccc, S273d43fe37e. Generator report_argon_ccld40.py yêu cầu
   đủ đúng40 và110 rotations. Lưu table/summary/per_sample và review Việt.
3. symmetric_latent40: same-layer, norm-bounded interaction, projected delta,
   stem-only controls. V1 lỗi prefix replay.03125>.03, hai record chẩn đoán
   được lưu implementation_v1, không dùng bảng chính. V2 remote folder
   symmetric_latent40_v2; full fallback thay vì tăng tolerance. Jobs mới
   S15b8581e888, S284e81506bd. Candidate parameters không đổi.
4. query_latent40: thought theo câu hỏi trước choices, contrast với neither
   content; so global/native/shuffled và thêm joint delta. JobsS17b1ce5a068,
   S220626b05dd. Không speech window, clock từ nhãn hay audio handoff.

Theo dõi: monitor_latent_research.py cho ba phase đầu, và cùng script với
argument query_latent40 cho phase cuối. Chúng download JSONL/job status mỗi45s,
không chạy inference. Status local: latent_research_live_status.json và
query_latent40_live_status.json. Nếu monitor dừng vì lỗi, sửa an toàn rồi
khởi động lại monitor; GPU jobs là worker độc lập, không bị dừng theo monitor.

Lệnh tổng hợp:
- .venv/Scripts/python.exe server_jobs/argon_ccld_manage.py checkpoint
- .venv/Scripts/python.exe server_jobs/report_argon_ccld40.py
- .venv/Scripts/python.exe server_jobs/symmetric_latent_manage.py checkpoint
- .venv/Scripts/python.exe server_jobs/query_latent_manage.py checkpoint
- .venv/Scripts/python.exe server_jobs/report_latent_followups40.py symmetric_latent40
- .venv/Scripts/python.exe server_jobs/report_latent_followups40.py query_latent40
- .venv/Scripts/python.exe server_jobs/archive_latent_rotations.py argon_ccld40 symmetric_latent40 query_latent40

Network và Python executable cần sandbox escalation ở máy này; gọi bằng
.venv/Scripts/python.exe. Config auth nằm trong remote_config*.json bị gitignore,
không chép token vào reports. Server2 token cũ đã được user sửa trong hội thoại.

Điểm phê bình đã xác định trước kết quả cuối:
- C6/C7 vẫn phụ thuộc hard_sync/speech window, dù pass2 không mask.
- C3/C5 window independent nhưng vẫn visual-asymmetric; không bảo đảm mọi
  câu hỏi audio-led, reverse, count/duration hay free-form.
- C2 cũng bỏ foreground/desmear, không phải clock-only ablation củaM8.
- C1 windowed thought/.22 khác C5 global thought/.20; suite không fully factorial.
- CCLD “H_Q” gồm cả choices/format, chưa tách tác dụng đọc đáp án khỏi grounding.
- Hidden-state subtraction không chứng minh triệt tiêu sạch prior hoặc giữ
  nguyên100% bằng chứng thời gian. Không dùng persona trong đề xuất như xác
  nhận của nhà nghiên cứu thật hoặc hội nghị.
- RED, ASD, SoftThinking/Coconut/LatentOmni đã có các primitive liên quan;
  không claim novelty rộng. Đây là development đã dùng nhiều phương pháp.

Audit CPU khác: PyAV/FFmpeg so Decord và pixel matching trên1449/1528/2054.
TimestampAPI Decord âm không có nghĩa pixel đầu thật sự là preroll; phát biểu
ban đầu này đã được sửa với user. Best-pixel clocks cho thấy sai lệch sampler
đáng kiểm tra, nhưng chưa sửa đầu vào CCLD. Xem presentation_clock_audit/report.md.
Không chạy lại research40_pts_clock.py cũ nguyên xi: assertmonotone có thể fail
và không được coi DecordAPI làm oracle. Cần đối chứng display clock riêng.

## Bổ sung sau các suite phản hồi hướng vector

5. latent_slot_opt40: đã đủ40/40,110 rotations. Jobs S1 `1a6209e492`,
   S2 `5aa502070b`; source server_jobs/latent_slot_opt40.py. Native18,
   unoptimized memory19, optimized memory18, shuffled-target memory17.
   Thật sự optimize input tensors32 Adam steps, lr.05, target source-only
   block22; cosine + within-source contrastive. Backbone frozen và restored
   requires_grad flags. Đúng slot nội bộ video113/527->458/527,
   audio257/514->440/514, nhưng QA không tăng. Không coi feature self-match
   là event-grounding hoặc một phương pháp thành công.
   Native replay/zero-memory identity max.03125 dưới ngưỡng.06 đã định trước.
   Probe ed7078719e chỉ kiểm tra gradient, không dùng QA chọn hyperparameters.
   Fits đã lưu local đủ40 trong latent_slot_opt40/fits,25,741,416bytes;
   hash từng file ở fit_archive_audit.json. Raw110 rotations reaggregate0.

6. latent_slot_read15: đã đủ40/40,110 rotations. Jobs S1 `e6b94fa2c0`,
   S2 `8149560a83`; giữ nguyên từng float slot từ phase5, chỉ đổi read23->15.
   Native18; unoptimized read15=16; optimized read15=17; shuffled read15=17;
   optimized read23 reference18; unoptimized read23 reference19.
   Parity dựng lại read23 max0.0. Optimized early cứu1/làm sai2 so late.
   Kết quả không ủng hộ sửa riêng giai đoạn đọc. Raw110 rotations reaggregate0.

7. presentation_resample40: đang chạy, KHÔNG lấy số partial làm kết quả cuối.
   Jobs full S1 `80bac8beb6`, S2 `511a86a99b`; smoke `20b0a948dd`/
   `b2479aa095` thành công, cùng code/math được giữ nguyên.
   Source presentation_resample40.py, manager presentation_resample_manage.py.
   P0 original reference; P1 PyAV uniformly rounded frame indices;
   P2 same PyAV pictures selected by latest PTS<=grid time i/native_fps.
   Giữ nguyên original audio waveform, nframes, FPS, geometry, spatial budget.
   Stable-sort PTS cùng picture index, duplicate PTS chọn last; trước frame
   đầu black/sau cuối hold, lưu cả hai số. Không dùng API Decord làm oracle.
   Đây là input-correctness control, chưa phải đóng góp reasoning mới.
   P1/P0 không decoder-only vì decoded frame counts có thể khác nhau;
   P2/P1 chỉ đổi rule chọn ảnh. Không reencode MP4 hoặc dùng sample/task router.
   CPU3 tests qua: unequal intervals không lấy ảnh tương lai, reordered/duplicate
   timestamps đúng index, black trước frame đầu. Smoke encoder replay qua.
   Monitor session51062 còn độc lập với GPU; interval45s.

Các báo cáo C/S/Q/L/E đã có report.md, table.md, summary.json, per_sample.csv
và raw rotation JSON. Script report_latent_research_overview.py chỉ chạy khi
cả7 suite đủ40; nó tạo results/argon_and_latent_research_20261015.md.
Sau P full: checkpoint, report_latent_followups40.py presentation_resample40,
archive_latent_rotations.py presentation_resample40, rồi overview. Chưa claim
đạt mục tiêu kết quả cao hoặc được accept CVPR. Không ensemble oracle theo ô.

Research bổ sung: AHA https://arxiv.org/html/2512.24052v2 dùng temporal hard
negatives cho DPO trên7B/audio; không phải proof jointAV/3B và không tái lập
training đó trong lượt này. Hướng cần kiểm chứng tiếp là independent segment
grounding + decision supervision trên train pool tách nguồn/video khỏi40,
hai hướng modality, count/duration/order/free-form và temporal counterfactuals.

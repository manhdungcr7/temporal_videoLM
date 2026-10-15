PC-EBR Stage-0 v2 — kết quả và audit, 2026-10-15

**Chưa đủ bằng chứng thông qua Stage-0 để chạy A0–A7. M1 two-stream không hơn matched controls một cách thuyết phục; M2 có shortcut mạnh; M3 xác nhận sensitivity giữa các tầng, chưa xác nhận temporal correctness.**

M1/M2 chạy trên cả hai server, mọi kết quả trùng nhau. Ba seeds cố định, 42 lượt huấn luyện/server, cùng 202 train/50 held-out pairs từ 16/4 origins. M3 dùng 4 ID đầu mỗi shard, rotation 0, tổng 8 ID đúng manifest. Bản gốc M3 lỗi mask GPU 0/hidden GPU 1. Devicefix chỉ chuyển visual mask/chunk indices về device của hidden. CUDA context server1 sau lỗi bị illegal access; 4 ID shard 1 được mã hóa/chạy tiếp trên server 2, không thay mẫu, M1/M2 không chạy lại. Native decoder replay max error=0.0.

**M1 — mọi condition và baseline**

Global/exact/multipos có 50 queries; nonnull có 32 queries theo substring filter trong code, không phải annotation tồn tại event. Với condition học: cột Global/Exact là số đúng của từng seed 20261015/16/17; Multipos/Nonnull là mean %. Các 50 queries đến từ 4 origins; 3 seeds hoặc 2 replicas không làm tăng số held-out độc lập.

**Native post-media readout**

| Condition | Global đúng/50 | Exact đúng/50 | Multipos % | Nonnull % |
|---|---|---|---:|---:|
| 0_unadapted_raw | 0 | 9 | 20.0 | 18.8 |
| 0_unadapted_media_rec_only | 1 | 10 | 24.0 | 18.8 |
| 0_step0_mean_shift_two_stream | 4 | 11 | 32.0 | 18.8 |
| F1_uncentered_no_hneg | [6, 6, 6] | [12, 12, 12] | 28.0 | 21.9 |
| F2_uncentered_with_hneg | [6, 6, 6] | [12, 12, 12] | 28.0 | 21.9 |
| F3_media_sub_no_hneg | [4, 4, 4] | [15, 15, 15] | 40.0 | 31.2 |
| F4_media_sub_with_hneg | [4, 4, 4] | [15, 15, 15] | 40.0 | 31.2 |
| F5_two_stream_no_hneg | [6, 6, 6] | [12, 12, 12] | 28.0 | 28.1 |
| F6_two_stream_with_hneg | [6, 6, 6] | [12, 12, 12] | 28.0 | 28.1 |
| F7_two_stream_shuffled_ctrl | [4, 6, 6] | [11, 9, 9] | 24.0 | 18.8 |

**Pooled media**

| Condition | Global đúng/50 | Exact đúng/50 | Multipos % | Nonnull % |
|---|---|---|---:|---:|
| 0_unadapted_raw | 2 | 12 | 26.0 | 31.2 |
| 0_unadapted_media_rec_only | 3 | 12 | 24.0 | 34.4 |
| 0_step0_mean_shift_two_stream | 3 | 11 | 28.0 | 28.1 |
| F1_uncentered_no_hneg | [14, 14, 14] | [15, 15, 15] | 36.0 | 37.5 |
| F2_uncentered_with_hneg | [14, 14, 14] | [15, 15, 15] | 36.0 | 37.5 |
| F3_media_sub_no_hneg | [5, 5, 5] | [18, 18, 18] | 42.0 | 43.8 |
| F4_media_sub_with_hneg | [5, 5, 5] | [18, 18, 18] | 42.0 | 43.8 |
| F5_two_stream_no_hneg | [8, 8, 8] | [12, 12, 12] | 30.0 | 31.2 |
| F6_two_stream_with_hneg | [8, 8, 8] | [12, 12, 12] | 30.0 | 31.2 |
| F7_two_stream_shuffled_ctrl | [5, 9, 9] | [12, 12, 13] | 31.3 | 31.2 |

Matched F2→F6: native giữ global 6/50, exact 12/50, multipos 14/50; chỉ đổi phân bố câu đúng nonnull từ 7/32 lên 9/32. Pooled F2 tốt hơn F6: global 14→8/50, exact 15→12/50, multipos 18→15/50, nonnull 12→10/32. Pooled F6 không hơn shuffle về exact (24.0% so 24.7%), multipos (30.0% so 31.3%) hoặc nonnull (31.2% bằng nhau). F3/F4 cải thiện localization nhưng đổi lại global thấp; đây vẫn là trade-off. F1/F2, F3/F4, F5/F6 có cùng các số đếm, chưa có bằng chứng hard-negative boost cải thiện trong budget này.

Hướng mới giữ query 2048-D và bỏ text-batch centering là cải tiến về protocol. Query-batch score check max error=2.384185791015625e-07. Tổng 77,827 parameters khớp code. Nhưng `einsum(nd,ndr,nrd->nd)` chỉ tạo `x_d*sum_r down[d,r]*up[r,d]`, không phải full low-rank mixing `x@down@up`. Không sửa công thức đó trong mainrun vì sẽ đổi phương pháp.

Zero-sum advantage chỉ bảo toàn arithmetic mean score trong từng group, không bảo toàn global argmax hay softmax mass. Counterexample: ctx[.9,.8,.7,.6], adv[0,0,.4,-.4], adv có tổng 0 mỗi nhóm 2 slots nhưng global winner đổi 0→2. Vì vậy claim “không thể giảm global retrieval” không được bảo đảm bằng đại số.

**M2 — affine và synthetic retrieved-coordinate probes**

Part A: affine 2t+12.5 error 0; anchor coverage .2, target coverage 1; center antisymmetry error 0. Đây là unit check được sửa đúng ở transform thử. Không suy affine-invariance toàn LLM hoặc mọi phép scale/float32 có duration clamp.

Part B có 3,848 train và 1,072 held-out synthetic episodes, từ 962/268 ordered event pairs, mỗi pair 4 states. Mỗi state held-out có 268 episodes. Các episodes phụ thuộc cùng 4 held-out origins, không phải 1,072 AV samples độc lập.

| Probe | Đúng/1072 | Accuracy % | Before đúng/268 | During đúng/268 | After đúng/268 | Foil đúng/268 |
|---|---:|---:|---:|---:|---:|---:|
| B1_matched_std_attn_baseline | 501 | 46.7 | 158 | 22 | 171 | 150 |
| B2_pcebr_time_coord_shuffled | 550 | 51.3 | 165 | 37 | 179 | 169 |
| B3_pcebr_phi_only_no_null | 583 | 54.4 | 166 | 48 | 173 | 196 |
| B4_pcebr_null_only_no_phi | 229 | 21.4 | 53 | 21 | 0 | 155 |
| B5_pcebr_full_phi_plus_null | 572 | 53.4 | 170 | 46 | 178 | 178 |

| Probe | A→V % | V→A % | V→V % | A→A % |
|---|---:|---:|---:|---:|
| B1_matched_std_attn_baseline | 46.2 | 51.0 | 45.5 | 47.5 |
| B2_pcebr_time_coord_shuffled | 50.5 | 58.2 | 49.0 | 55.0 |
| B3_pcebr_phi_only_no_null | 55.3 | 62.0 | 51.3 | 57.5 |
| B4_pcebr_null_only_no_phi | 21.2 | 21.2 | 21.6 | 20.0 |
| B5_pcebr_full_phi_plus_null | 51.9 | 56.2 | 52.6 | 57.5 |

Full 53.4% chỉ hơn coordinate-shuffle 51.3% là 2.1 điểm phần trăm, thua Phi-only 54.4%; During 17.2% rất thấp. Natural-unshifted evaluation được ghi docstring nhưng không có episode offset0 được label bằng order tự nhiên; vòng thực thi chỉ tạo 3 offsets từ true event centers và foils. Hai retrievals lấy từ hai pseudo descriptions riêng, không phải learned hop 2 conditioned on hop 1. Không dịch waveform/frames hoặc kiểm tra generated QA. Same-source “dịch target bank” vẫn giữ anchor bank cùng source không dịch, là pair-coordinate intervention chứ chưa thể coi một single-stream physical shift.

**Đối chứng không có semantic retrieval**

Dùng nguyên episode generator và linear probe training nhưng đặt mọi similarity=0: cả p1, p2 uniform, không có thông tin event matching.

| Uniform-retrieval control | Accuracy % | Before % | During % | After % | Foil % |
|---|---:|---:|---:|---:|---:|
| B1_std | 50.7 | 62.7 | 16.0 | 64.9 | 59.0 |
| B3_phi | 60.5 | 61.9 | 15.3 | 64.9 | 100.0 |
| B5_phi_null | 60.5 | 61.9 | 15.3 | 64.9 | 100.0 |

**Uniform Phi/full 60.5%, cao hơn learned full 53.4%, với foil 100%.** Điểm này bác bỏ cách diễn giải rằng đạt điểm cao trên episode generator này đã chứng minh learned event retrieval/null evidence. Riêng ba relation states (bỏ foil), learned full đạt 49.0% so uniform 47.4%; phần lớn lợi thế overall của uniform đến từ foil. Nó chỉ ra shortcut trong construction/coordinate distributions; không chứng minh mọi mô hình quan hệ temporal vô dụng. Foils lấy từ origin khác chưa được annotation rằng event không tồn tại, và clock ranges/offsets có thể phân biệt label.

**M3 — kết quả đầy đủ micro-probe**

| Shard | ID | Task | TV late/native | TV mid/native | TV swap late | TV swap mid | TV mid shuffled/native |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 1449 | TA | 0.0452 | 0.0933 | 0.0363 | 0.1085 | 0.1473 |
| 1 | 2150 | TF | 0.0582 | 0.1964 | 0.0846 | 0.6315 | 0.0454 |
| 1 | 1929 | TF | 0.0249 | 0.3873 | 0.0903 | 0.1290 | 0.4336 |
| 1 | 2054 | TA | 0.1640 | 0.0519 | 0.1792 | 0.4475 | 0.0384 |
| 2 | 1956 | CMTB | 0.0273 | 0.4131 | 0.0847 | 0.3905 | 0.0239 |
| 2 | 1513 | TF | 0.1762 | 0.0504 | 0.2234 | 0.4621 | 0.2565 |
| 2 | 2264 | CMTB | 0.0277 | 0.4710 | 0.0454 | 0.4995 | 0.0357 |
| 2 | 1821 | TF | 0.1104 | 0.4893 | 0.1517 | 0.2970 | 0.3542 |

Tổng 8 ID: mean TV swap late=0.111950, mid=0.370700, ratio=3.311×. Shard 1=3.37×, shard 2=3.26×, đạt dự đoán sensitivity ≥3× trên micro này. Đây không phải directional accuracy hoặc QA improvement.

Hai vector early/late sau trừ toàn visual mean gần như đối hướng: cosine≈−1 ở mọi 8 samples. Với 2 nhóm, các centered prototypes có dạng ±c*(mean early−mean late), nên swap thực tế là đổi dấu residual. Không có physical time shift. M3 không dùng trained M1 projector hoặc M2 Phi; nó mean-pool visual block 18 rồi cộng 14% norm vào mọi tokens sau media (gồm stem, choices, format). Vì vậy chưa kiểm chứng PC-EBR option-free path. `freeform_hook_ok` chỉ lặp lại MC forward và isfinite ở sample đầu, các samples khác default True; không gọi generate hoặc bỏ options.

**Kết luận gửi Yoshua**

V2 cải thiện protocol query independence, multipos reporting, affine unit check và thêm executable M3. Tuy nhiên phải sửa low-rank implementation nếu đó là ý định, kiểm tra global rank thay vì suy từ zero-sum, và thay M2 episode generator để uniform/coordinate-only/null controls không thắng bằng shortcut. M3 sensitivity hữu ích để chọn thí nghiệm tiếp, nhưng chưa xác nhận đúng event/time/source. Không chạy A0–A7 hoặc reserve400 trong lượt này.

**Artifacts và verification**

Có 97 raw PT files/server: 42 projector checkpoints, 48 retrieval score matrices, 2 episode tensors, 5 classifier probes. 2,400 retrieval query rows/server được recompute đúng; hai archives SHA/từng entry được check local, không extract. M1/M2 replica results identical. M3 raw choice probabilities/native parity lưu JSON; CUDA errors và migration job logs được giữ. Không coi pseudo annotations là event ground truth hoặc 50 pairs/four origins là held-out final mới.

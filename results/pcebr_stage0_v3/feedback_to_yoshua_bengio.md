# Gửi Yoshua Bengio — kết quả Stage-0 v3, gate và hướng đề xuất tiếp

Yoshua, chúng tôi đã hoàn tất v3 trên hai runtime mới. **Không qua Cổng0 theo các tiêu chí bạn công bố trước chạy.** Các sửa đại số và việc sinh văn bản không choices đã hoạt động; nhưng hiệu năng M1, lợi thế M2 so controls và các điều kiện sensitivity M3 chưa đạt. Đây là kết quả của cấu hình đã thử, không phải bằng chứng bác bỏ mọi latent reasoning hoặc mọi can thiệp tầng muộn.

**1. Phạm vi và sửa kỹ thuật**

Giữ nguyên252 cặp cache:202train/16origins,50held-out/4origins; ba seeds, hai replicas. M1/M2 trùng kết quả giữa hai server. Đúng8ID micro cũ,4/shard,rotation0; không thay mẫu, không chạy A0–A7/bảng40/reserve400.

Bản gốc dừng ở M2 vì `zm_all = m_loc.encode_media(...)` còn giữ autograd graph của M1. Train linear probe nhiều bước gọi backward lần2 qua graph đã giải phóng. Chúng tôi lưu traceback, đóng băng các feature teachers và xác nhận từng tensor trọng số khớp bitwise checkpoint M1; M1 không train lại. Bản standalone sửa đúng một `.detach()` ở feature extraction để sử dụng projector như teacher đóng băng. Không sửa công thức, labels, seeds, loss, rank hoặc thresholds.

**2. Các phần đã được sửa đúng**

- Low-rank einsum mới thật sự trộn tọa độ; giữ diagonal làm control là phù hợp. Basis tests trả đúng `[2,0,0,0]` và `[2,2,2,2]`.
- Max-Zero bảo toàn group maxima và winning group đúng trên mọi seed, cả native/pooled: error0,agreement100%. Kiểm tra raw matrices xác nhận toàn bộ, không chỉ seed đầu trong summary.
- Context/local checkpoints dùng để bật/tắt scoring được giữ cố định trong so sánh. Đây là đối chứng sạch hơn so huấn luyện mọi thành phần chung rồi quy đóng góp riêng.
- Uniform control nay25.0%4-way và33.3%3-way ở cả ba suites. Nó không còn đạt60.5% như v2.
- M3 dùng nhiều slots, vector fwd/rev không còn mặc định đối hướng;cosine8mẫu trong khoảng.393–.868.
- Native replay0 ở8/8; token roundtrip khớp8/8; hook rows không giao option rows; choices và instruction letter-only thực sự bị loại khỏi free-form input. Cả8mẫu sinh được văn bản tự hồi quy.

Những pass này là tiến bộ về implementation/protocol, nhưng không thay cho các gate hiệu năng.

**3. M1: theorem qua, localization gate fail**

| Representation | Context diag C1: Exact | Hier diag C4: Exact | Context low-rank C5: Exact | Hier low-rank C7: Exact |
|---|---:|---:|---:|---:|
| Native readout | 32.00% | 24.00% | 24.67% | 22.00% |
| Pooled media | 38.00% | 30.00% | 28.67% | 25.33% |

C7 còn thua shuffled C8 về Nonnull: native17.20% so19.33%,pooled23.67% so24.73%. Vậy chưa đạt yêu cầu localization hơn context-only và shuffle trên cả hai representations. Nonnull v3 có31queries theo filter mới, khác32 ởv2.

Thành phần đơn giản nhất đáng giữ làm baseline mạnh là **pooled context-only diagonal C1**:16/50global,19/50exact,23/50multipos,14/31nonnull. True low-rank chưa tốt hơn diagonal trong budget/data này. Đừng suy rằng sửa đúng low-rank đồng nghĩa semantic retrieval tất yếu tốt lên.

Max-Zero giải quyết chính xác lỗi bảo toàn cực đại nhóm, nhưng nó **khóa lại coarse decision của context model**. Nếu context chọn sai recording/source group, localization không thể cứu group đó; nếu chọn đúng group nhưng local rerank chọn sai event, global content retrieval vẫn có thể giảm. Thực tế native C1global22%→C4global16%, dù winning group100% giữ nguyên. Bảo toàn group không phải bảo toàn đúng semantic slot.

Ngoài ra, công thức max-normalization là một scoring rule có ích cho top1; nó chưa tự là phân rã xác suất đã chuẩn hóa theo chain rule. Muốn dùng uncertainty/abstention cần kiểm tra calibration và group mass, không chỉ group maximum.

**4. M2: gate uniform qua; lợi thế full và một phần During chưa qua**

| Suite | Full4-way | Đối chứng4-way tốt nhất | Gap full | During recall full | Full3-way |
|---|---:|---:|---:|---:|---:|
| Natural | 31.5% | Wrong-event31.9% | −0.4pp | 39.7% | 37.4% |
| Global reversal | 32.3% | Mean-time baseline35.4% | −3.1pp | 40.3% | 39.8% |
| Two-hop roll | 27.1% | Coordinate-shuffle28.2% | −1.1pp | 22.9% | 36.9% |

Ngưỡng bạn đặt là full hơn **mọi** control ít nhất10pp và During≥35%. Không suite nào qua điều kiện gain; Roll còn fail During. Natural/Reverse có cải thiện During so v2 nhưng không đủ để kết luận temporal grounding tốt: các controls và các state khác phải được tính cùng.

Uniform cho25%/33.3% là đúng điều kiện đã chốt. Tuy nhiên operator chủ động đặt Phi/mean-time về0 khi support gần uniform; input classifier lúc đó không có thông tin để phân biệt lớp. Đây là identity được hiện thực đúng, không phải bằng chứng mọi shortcut đã bị loại. Uniform có foil recall100% trong log vì nó dự đoán cùng lớp foil cho toàn bộ samples, còn Before/During/After đều0%; không diễn giải đó là khả năng phát hiện thiếu bằng chứng.

Các vấn đề cần sửa trong định nghĩa/thông tin đầu vào:

1. **Foil chưa được xác minh vắng mặt.** Trong cả20origins train/held-out, số candidate đạt text-cos<.55 đều0. Tất cả dùng fallback chỉ yêu cầu khác chuỗi caption. “Verified absent event” không phù hợp với dữ liệu thực tế; khác câu chữ không chứng minh không có cùng sự kiện.
2. **During chưa nhất quán.** Natural dùng center distance≤2.1s, gồm cả hai cửa sổ2s kề nhau. Roll ở anchor sát biên có thể gán Before/After cho displacement1slot=2s. Cùng một quan hệ thời gian có thể nhận nhãn khác giữa suites; cần chốt interval semantics, không chỉ center thresholds.
3. **Role split vẫn còn ở B3 cùng source.** B2 đảo cả hai banks đã tránh lỗi cũ; nhưng B3 roll riêng target bank trong khi anchor bank không roll, kể cả video→video/audio→audio. Đó chưa là can thiệp lên một physical stream duy nhất. Hãy giới hạn B3 vào cross-modal nếu đó là mục tiêu diagnostic, hoặc định nghĩa edit thật giữ anchor instance nguyên và chỉ di chuyển target instance.
4. **M2 không dùng giá trị hierarchical scores C7.** `sim_mat` chỉ được dùng lấy `.device`; sau đó function tính lại affinities từ m_loc. Context branch/Max-Zero chưa nằm trong đường downstream scoring. Vì vậy không nên gọi kết quả M2 là kiểm chứng toàn kiến trúc hierarchical M1.
5. **B1 không phải unmodified standard attention.** Mean times của B1 cũng dùng positive-part supports sau debiasing, cùng với null features. Đây là baseline mean-time phù hợp để so feature maps, nhưng cần đặt tên đúng và không biến kết quả thành claim attention tiêu chuẩn không thể suy thứ tự.
6. Các queries diagnostic vẫn xuất phát từ các pseudo descriptions riêng, hoặc composite của hai descriptions đã biết. Chúng chưa chứng minh giải được coreference từ một câu hỏi thực chỉ gọi “the narrated event”.

**5. Positive-part có giới hạn toán học còn cần xử lý**

Nó xóa chính xác uniform tail trong toy mixture peak+uniform. Nhưng không xóa mọi noise hoặc chứng minh calibration. Chúng tôi kiểm tra logits `[1e-4,0,0,0,0]`: margin chỉ≈5.6e-6,g≈1.6e-5, nhưng sau trừ uniform và renormalize, support trở thành one-hot. Khi hai peak cùng slot, Phi là `[1,1,1,0,0,0]`; khi ở hai đầu đối diện, center channel≈1.609. Một khác biệt logits rất nhỏ đã trở thành quan hệ cực đoan.

Null features vẫn chứa độ tin cậy nhỏ nên một readout được học tốt có thể sử dụng chúng. Nhưng M3 hiện bỏ null/mass này khi tạo vector và luôn inject14%norm; nó chưa có cơ chế giảm tác động khi support yếu. Không nên gọi positive-part là “triệt tiêu100% boundary drift/shortcut” ngoài các giả thiết toy cụ thể. Với repeated/multi-instance events và uncertainty rộng, việc cắt xác suất dưới uniform còn có thể xóa evidence hợp lệ; cần kiểm tra riêng.

**6. M3: thực thi tự do đã đúng hơn, sensitivity gate lại fail**

| Shard | Mean TVswap late | Mean TVswap mid | Ratio |
|---|---:|---:|---:|
| 1 | .1221 | .1600 | 1.31× |
| 2 | .0488 | .0494 | 1.01× |
| Gộp8 | .085425 | .104725 | **1.226×** |

Gate≥2× không đạt. Mean TV M1+M2/native=.106925,random=.127825,irrelevant=.104775. Nhánh có cấu trúc không vượt random control. Không nên hạ gate hoặc gọi độ nhạy thấp là “ổn định tốt hơn” sau khi giả thuyết ban đầu yêu cầu sensitivity cao hơn.

Kết quả v2 từng3.31× không lặp lại dưới phép can thiệp v3. Điều này củng cố yêu cầu đọc M3 như **phép đo phụ thuộc kiểu perturbation**, không phải chứng minh một cửa sổ layer14–24 luôn đúng còn layer28–35 luôn quá muộn.

M3 hiện có thêm ba vấn đề liên quan yêu cầu triển khai:

- Luôn audio-hop1 rồi visual-hop2 và visual evidence vector. Nó chưa có đường đối xứng cho visual-anchor→audio-target, same-source hoặc nhiều sự kiện tùy câu hỏi. Bảng bốn directions của M2 không làm đường inference M3 tự trở nên tổng quát.
- Reverse chỉ hoán vị visual logits/weights trong khi V18 evidence value bank giữ nguyên. Nếu muốn bảo toàn identity event dưới timeline reversal, phải biến đổi nhất quán cả feature/value bank và thời gian; hiện tại việc đổi trọng số lên bank cũ có thể chọn nội dung khác. Chưa có raw audio/video nào được đảo và mã hóa lại.
- Train features là crop/source-isolated pooled states; M3 áp dụng projector lên mean của full-AV contextual states và query stem states. Distribution/representation mismatch này cần được kiểm chứng, không coi cùng2048 chiều là đủ alignment.

**7. Một phát hiện về clock cần ưu tiên trước khi nói “physical slots”**

Cache `sec` được tính từ temporal RoPE/25. Chúng tôi so với position_ids thật của decoder: khớp trong sai số nhỏ, nhưng đó không đồng nghĩa timestamp PTS. Ví dụidx2150 clip12.167s có visual position clock đến15.52s;idx1513 clip13.074s đến16.04s;idx1956 clip22.074s đến25.04s. Audio clock gần duration hơn.

Không nên kết luận từ đó rằng Qwen tự sai đồng hồ: positional scheme/padding/preprocessing có thể không được thiết kế để đọc ngược như PTS. Nhưng **bản thân helper của chúng ta chưa đủ làm physical timestamp oracle**. Trước diagnostic tiếp theo hãy lấy token→frame/grid→sampled presentation-time mapping từ metadata encoder/processor và waveform clock, kiểm tra lệch/đuôi/padding. Không chỉ clamp mọi timestamp vượt duration rồi gọi là sửa đúng. Dữ kiện này giới hạn diễn giải M3 hiện tại; không ảnh hưởng các fail M1/M2 trên pair caches riêng.

**8. Free-form pass là pass thực thi, chưa phải chất lượng trả lời**

V3 thực sự bỏ choices rồi chạy autoregressive trên8/8, tốt hơn check isfinite ởv2. Chúng tôi đã audit exact token roundtrip và spans. Nhưng12tokens thường chưa thành câu trả lời hoàn chỉnh; chưa có chấm semantic/evidence correctness.

Ví dụidx2054 có gold TA/B=No nhưng sinh “Yes, the person is doing all the steps in the video”;idx2150 sinh “The event occurs at 1:00. It's” trong clip12.167s. Đây là những ca cần kiểm tra ground truth/time độc lập, không thể lấy cờ `freeform_gen_ok=True` làm bằng chứng giảm hallucination. Toàn bộ nguyên văn nằm trong report/audit, không chỉ ví dụ tốt.

**9. Đề nghị cho vòng kế tiếp**

Hãy giữ pooled context-diag làm baseline, giữ các sửa kỹ thuật đã kiểm chứng, và dừng việc cộng thêm nhiều operators để bù cho retrieval còn yếu. Chưa nên mở A0–A7 hoặc reserve400 khi gate đã fail.

Ưu tiên một diagnostic nhỏ có **event instance/source/interval labels đáng tin**, clock mapping đúng, và query đủ phân biệt instance. So raw/context/local representations với cùng checkpoint và budget, xem có thắng đúng retrieval/control hay không. Đây là nhu cầu đã nêu từ các vòng trước, chưa phải thành công huấn luyện mới.

Nếu retrieval qua, kiểm tra quan hệ bằng một định nghĩa interval nhất quán và transformations bảo toàn event identity; chấm paired directional correctness chứ không chỉ classifier accuracy hoặc TV. Cuối cùng mới train một adapter/readout nối evidence vào generation, có uncertainty/null gating và nhiều chiều modality. Chứng minh đường inference này bằng câu hỏi tự do đa dạng, không chỉ dạng tham chiếu narration của benchmark.

Đừng tuyên bố “không còn bị bác bỏ bởi phản ví dụ nào” hoặc bác bỏ toàn lớp phương pháp từ vài cấu hình. Đóng góp nghiên cứu cần vừa có cơ chế đúng, vừa có empirical gain vượt controls trên dữ liệu tách nguồn mới. Hiện v3 đã sửa được vài lỗi implementation và identity; các gate quan trọng còn fail rõ ràng.

Raw artifacts,21probes,36M1checkpoints/server,52retrievalmatrices/server, outputs8mẫu, traceback và freeze audit nằm trong `results/pcebr_stage0_v3/`. Báo cáo chính: `report.md`; gate machine-readable: `gates.json`. Tên Yoshua Bengio/Yann LeCun là quy ước vai trò trong cuộc trao đổi.

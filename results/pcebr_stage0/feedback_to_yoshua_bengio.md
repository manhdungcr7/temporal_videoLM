# Gửi Yoshua Bengio — kết quả Stage-0 và phản biện PC-EBR

Yoshua, chúng tôi đã chạy kịch bản Stage-0 bạn gửi trên cả hai server Kaggle. **Cổng 0 chưa đạt điều kiện chuyển sang A0–A7: H1 không đáp ứng ngưỡng bạn đặt, M2 chỉ qua kiểm tra đại số, còn H3/M3 chưa có implementation.** Kết luận này áp dụng cho cấu hình và dữ liệu vừa kiểm tra; chưa đủ để bác bỏ mọi phương pháp event-binding hoặc PC-EBR có huấn luyện về sau.

**1. Chúng tôi đã chạy gì?**

Bốn cache pair local R14/R18 được upload lên hai runtime mới; SHA khớp cả hai server. Mỗi server chạy cùng toàn bộ script để kiểm tra tái lập. Bản gốc lỗi ở `torch.randperm(len(m_idx), generator=g_shuf)` vì generator ở CUDA nhưng randperm mặc định CPU. Chúng tôi chỉ thêm `device=device`, giữ nguyên các phép tính, seeds, architecture,384 training steps và split. Bản gốc, bản sửa và cả hai traceback đều được giữ lại. Sau sửa, mỗi replica chạy khoảng13–14 giây; mọi số đếm M1/M2 trùng nhau giữa hai server.

Đúng252 pairs:202 train từ16 recording origins,50 held-out từ4 origins. Không trùng origin train/held-out, không trùng38 origins của research40; dữ liệu cũng không trùng danh sách182 origins loại trừ đã lưu trong metadata. Chúng tôi không mở hoặc chạy reserve400. Đừng gọi50 held-out này là test mới nguyên vẹn: chính báo cáo R14–R18 đã nói nó được xem qua nhiều lựa chọn representation trước đây.

**2. Kết quả M1 của chính script bạn gửi**

Mọi cặp dưới là số đúng/tổng, không phải đúng/sai.

| Representation | Phương pháp | Retrieval toàn cục | Định vị nội recording/source |
|---|---|---:|---:|
| Native post-media readout | R14 checkpoint cũ, replay | 7/50 | 13/50 |
| Native post-media readout | Raw chưa học | 0/50 | 9/50 |
| Native post-media readout | Chỉ deconfound, chưa học | 1/50 | 9/50 |
| Native post-media readout | R24 uncentered | 13/50 | 13/50 |
| Native post-media readout | PC-EBR deconfounded R24 | **6/50** | **16/50** |
| Native post-media readout | PC-EBR shuffle | 3/50 | 9/50 |
| Pooled media | R18 checkpoint cũ, replay | 1/50 | 11/50 |
| Pooled media | Raw chưa học | 2/50 | 12/50 |
| Pooled media | Chỉ deconfound, chưa học | 4/50 | 14/50 |
| Pooled media | R24 uncentered | 2/50 | 8/50 |
| Pooled media | PC-EBR deconfounded R24 | **5/50** | **11/50** |
| Pooled media | PC-EBR shuffle | **10/50** | **11/50** |

Điểm tốt: trên native readout, candidate centered có within-rec16/50, hơn uncentered13/50 và shuffle9/50. Đây là tín hiệu cho khả năng phân biệt cục bộ cần nghiên cứu tiếp. Projector uncentered mới còn có global13/50, hơn R14 cũ7/50; thay kiến trúc/loss/budget có thể có ích.

Điểm chưa đạt: **khử mean không giúp đồng thời cả hai metric**. Native global tụt13→6 khi thêm deconfounding; within-rec tăng13→16 chỉ+6 điểm phần trăm, không phải+15. So native raw9→16 cũng chỉ+14 điểm phần trăm. Pooled candidate global5/50 chỉ hơn R18+8 điểm phần trăm; within-rec11/50 bằng R18 và shuffle, còn global thua shuffle10/50. Vì vậy không đáp ứng lời dự đoán tăng ít nhất15 điểm phần trăm ở cả retrieval và localization, đồng thời hơn shuffle. Không nên chuyển sang ablation QA vì một metric tăng ba câu.

Không so `within_rec16/50` với historical `R14 global7/50` rồi ghi tăng18 điểm phần trăm: hai candidate sets/metrics khác nhau. Chúng tôi đã replay checkpoint cũ để có baseline within-rec cùng định nghĩa, thay vì suy từ số global.

**3. Vì sao M1 vẫn chưa đo đúng giả thuyết mạnh bạn mô tả?**

Thứ nhất, đây không phải cache raw-media tầng22 cho cả hai nhánh. `avbind_native_pairs` đã lấy last assistant/readout-token state sau media tại `model.layers[27]`, tức block index27/tầng28. `avbind_pooled_pairs` mới là mean các media tokens cùng tầng. Cả hai dùng crop2s cô lập, đã được trích trước đây; script không hiện thực một ngân hàng probe mới tầng22 chạy song song trên một full-clip forward. Vì vậy không thể nói Stage-0 đã kiểm chứng toàn bộ representation mới trong prose.

Thứ hai, nhãn `content` là pseudo description do chính model sinh từ crop, không phải annotation độc lập rằng một event thật xảy ra đúng interval đó. Trong50 held-out có nhiều audio descriptions lặp nguyên văn ở các start khác nhau, chẳng hạn “there is no speech in this audio clip.” Một query như vậy không chỉ rõ instance thời gian, nhưng metric lại đòi exact start. Thất bại exact-start có thể do mô tả không phân biệt instance; thành công cũng có thể là self-consistency với pseudo teacher, chưa phải event grounding đáng tin.

Thứ ba, code center **cả media và text held-out**, theo recording ID/source của toàn bộ tập query held-out. Đây là giao thức transductive dùng các query cùng group để tính mean; không phải query-independent inference đã được xác lập cho một câu hỏi mới. Nó không đưa held-out vào optimizer, nhưng representation của một query phụ thuộc các query held-out khác. Cần ghi rõ điểm này và kiểm tra bản chỉ center media, text query độc lập. Đừng gọi đây là benchmark QA label leakage nếu chưa có bằng chứng đó; vấn đề chính là giao thức có thêm thông tin/batch query context chưa có trong deploy.

Thứ tư, `shuffled_time_ctrl` không hề dùng numeric start/end trong training objective. Nó tráo media rows trong cùng recording/source, làm hỏng semantic pair alignment. Nó kiểm tra phụ thuộc vào matching đúng, chưa cô lập được việc model học vật lý thời gian thay vì semantic caption matching. Muốn kiểm tra dùng clock, hãy giữ feature/text identity nguyên, chỉ tráo interval coordinates trong một model thực sự đọc coordinates, rồi đo relation/QA đã có nhãn độc lập.

Thứ năm, đây không phải factorial2×2×2 đầy đủ hay tái lập R14 trong cùng một họ architecture. Các nhánh học r24 cùng có hard-negative boost; không có đầy đủ crossed factors representation/centering/hard-negative. R14 gốc là residual mapper397,440 params, không phải r24 bilinear projector. Projector gửi lần này thực tế **294,985 params**, không phải~98K; docstring còn nói rank16 trong khi class/default/run dùng24. Chúng tôi theo code rank24, không chọn lại rank theo kết quả.

Đối chứng bổ sung ba seeds và bản chỉ center media được lưu riêng trong `report.md`, cùng1200 query rows và similarity matrices. Đây là kiểm tra độ nhạy exploratory, không đổi ngưỡng gate sau khi thấy kết quả và không biến50 queries thành150 mẫu độc lập.

Qua ba seeds, native centered đạt mean12.0% global/27.3% within so uncentered18.7%/22.0% và shuffle6.7%/16.7%. Có tín hiệu localization cục bộ hơn shuffle nhưng vẫn trade-off global; chỉ center media đạt15.3%/24.7%. Pooled centered mean13.3%/22.7%, shuffle12.7%/23.3%: không có lợi thế localization so control. Các số này không giải cứu dự đoán+15pp. Bảng seed audit chính giữ nguyên AST training gốc, và seed20261015 tái lập đúng mọi số của mainrun; một bản audit viết lại loss tương đương toán học nhưng làm đổi trajectory đã được lưu riêng, không trộn vào bảng này.

**4. M2 qua unit check nhưng chưa kiểm chứng một model temporal reasoning**

Kết quả chính xác: common shift+12.5s cho max difference0; directed-order difference4.3944; tổng interval giao hoán cho difference0. Các vector Before/During/After khác nhau đúng theo hàm bạn viết. Điểm tốt là interval function có chứa hướng và không phụ thuộc gốc đồng hồ trong toy translation check.

Nhưng script không có audio/video nào được dịch, không có learned hop1/hop2, không có event retrieval hoặc language-model prediction. Nó đưa interval đúng vào trực tiếp rồi kiểm tra hàm số tính từ chúng. Vì thế chưa kiểm tra được model tự tìm đúng event/interval, không chứng minh100% directional equivariance của PC-EBR, và không cung cấp tỷ lệ đảo đáp án đúng trên counterfactual media.

Lập luận “attention value pooling giao hoán nên không thể hiểu thứ tự” cần sửa. Hoán vị cách liệt kê **nguyên cặp** `(content,time)` không đồng nghĩa tráo timestamp giữa hai content. Attention weights phụ thuộc query/content/position, có thể khác nhau giữa hai event. Chỉ với weights[.9,.1], centers[3,11], weighted time đã là3.8; tráo time của hai event cho10.2. Test `(I1+I2)-(I2+I1)=0` không đại diện đủ cho attention tiêu chuẩn, positional encoding, nhiều heads hay nhiều layers. Không được dùng nó làm chứng minh bất khả thi hoặc novelty nền tảng.

Cần phân biệt thêm:

- Translation và positive scaling là hai phép khác. Code hiện chỉ bảo đảm toy translation; scale×2 làm full Phi đổi khoảng0.646627 vì signed log gaps và epsilon0.1 trong duration ratio. Các nhãn order có thể giữ nguyên khi scale dương; không suy toàn bộ vector Phi, duration answers hoặc latent readout bất biến.
- Full Phi không antisymmetric; center-offset channel có tính đó. Gap-after/gap-before đổi vai trò khi swap, không nhất thiết đơn giản đổi dấu từng tọa độ với cùng magnitude.
- `inter/min(da,db)` là overlap coefficient. Nếu định nghĩa channel1 là tỷ lệ anchor được target phủ thì cần `inter/da`: anchor[0,10],target[2,4] hiện trả1, trong khi anchor coverage là0.2. Phải chốt nghĩa trước khi train relation adapter.

**5. H3/M3 chưa chạy được vì chưa có trong code**

Docstring có micro-limit4, mid-vs-late và điều kiện TV_swap gấp3, nhưng không có hàm thực thi chúng. Không có occurrence bank hai hop được train, relation-to-LLM adapter, middle/late routing hay foil/null evaluation. Chúng tôi không tự thay phần còn thiếu bằng heuristic rồi gọi đó là PC-EBR của bạn.

Cũng cần sửa claim lịch sử: đọc lại `layer_predictions` trên40 records cho thấy block30 đồng ý **37/40**, không phải39/40, với final; khác ở idx2109,1513,2281. Block34 đồng ý40/40. Quan sát agreement không phải chứng minh quyết định “khóa cứng”; late-path trước đây đã có thay đổi quyết định ở một số câu, dù không sửa TF và chưa có gain ổn định. Các kết quả đó chỉ làm giảm độ hứa hẹn của những can thiệp cụ thể đã thử, không bác bỏ tuyệt đối mọi can thiệp từ tầng27 trở đi.

Ngay cả nếu M3 tương lai có TV_swap cao gấp3, TV cao chỉ nói output nhạy với can thiệp, chưa nói đổi đúng hướng. Cần directionally correct rate và matched random/irrelevant-time controls. Khi late TV gần0, tỷ lệ3× có thể đạt với tác động tuyệt đối rất nhỏ; cần một ngưỡng effect size tuyệt đối chốt trước.

**6. Góp ý cho kiến trúc PC-EBR và A0–A7**

Hướng dùng source provenance, event intervals, retrieval không phụ thuộc options và có unsupported state đáp ứng yêu cầu triển khai tốt hơn những nhánh ép audio-first hoặc đọc choices. Nhưng chúng mới là thiết kế cần kiểm chứng. Những điểm sau cần được giải quyết trước khi đóng băng:

1. `u-alpha*mean` là subtract-mean, chưa phải orthogonalization hay tách đúng background. Nó có thể xóa sự kiện kéo dài toàn clip. Giữ context/dynamic hai nhánh là hợp lý, nhưng phải đo bằng interval labels/event identity thật, không suy từ cosine cao.
2. Innovation norm/signal energy không tự xác định có sự kiện. Cảnh tĩnh có thể chứa bằng chứng đúng, im lặng có thể là trạng thái có nghĩa. Gate có thể biểu diễn uncertainty nhưng chưa được gọi là event-existence probability nếu chưa calibrated.
3. Hai hop không tự bảo đảm mọi câu hỏi. Nhiều event/repeated instances cần phân phối hoặc tập instance rõ; count bằng tổng activation của sliding windows có thể đếm cùng event nhiều lần, duration có thể bị cộng chồng. Phải định nghĩa instance grouping/union và kiểm chứng khi thay stride/window overlap.
4. “Không có sự kiện được nhắc” khác “hai sự kiện có tồn tại nhưng không đồng thời”. `unsupported/unknown` không nên tự động ánh xạ sang đáp án TA `No`; null cần annotation và calibration riêng. Ngưỡng0.7 hiện là giả định, chưa có checkpoint hoặc validation hỗ trợ.
5. A5 thay cả directed interval features **và null gate** so A4, nên không cô lập riêng Phi. Hãy có Phi-only, null-only, và full, hoặc giữ null giống nhau ở cả hai. So mid14–24 với late28–35 cũng cần kiểm soát số lần tác động, norm budget, training/inference compute, không chỉ đặt cùng gain.
6. `MODEL_.generate(answer_options=None)` trong prose chưa có một wrapper option-free được hiện thực/kiểm tra ở runtime. Muốn claim đồ thị100% giống nhau, cần API/token-span/hook cụ thể và thực nghiệm free-form; MC wrapper có thể khác phần scoring nhưng core không nên cần choice information.
7. Một model tự giới hạn pair relations có thể hỗ trợ nhiều hướng modality, nhưng tính đối xứng cấu trúc không chứng minh hiệu quả đối xứng. Cần đánh giá audio→video, video→audio, audio→audio, video→video, gồm event lệch pha và mô tả không xuất hiện trong nguồn kia.

**7. Đề nghị bước kế — sửa gate và dữ liệu trước khi mở bảng40**

Đề nghị giữ mô hình event/interval như giả thuyết, nhưng bỏ các kết luận “bác bỏ dứt điểm4 nhánh” hoặc “bắt buộc chỉ có kiến trúc này”. Thử nghiệm hiện tại không cho phép định lý một vector không thể biểu diễn quan hệ2-hop, không chứng minh late layers vô dụng, không chứng minh mean-centering là nguyên nhân giải quyết static dominance.

Vòng kế nên gửi một implementation nhỏ của **event retrieval có interval/source supervision đáng tin**, kiểm tra trên source-disjoint validation với query phân biệt được instance. Trước tiên hãy xem center media đơn thuần có tăng interval retrieval so cùng projector chưa center, trên đủ nguồn và có uncertainty; tách query centering khỏi candidate centering. Nếu không tăng ổn định, sửa representation/objective trước, không dùng thêm relation algebra để che một hop retrieval sai.

Sau đó mới kiểm tra directed relation từ **intervals được truy xuất**, không phải intervals gold đưa sẵn, với can thiệp thời gian có nhãn đúng và matched standard-attention baseline. Cuối cùng mới train relation-to-LLM adapter rồi so mid/late bằng cùng bank, weights, budget và directionally correct QA changes. Mỗi tầng kiểm chứng cần phân biệt lỗi extraction, retrieval, relation và language readout.

Đừng lấy một supervised loss/classifier chuyên Past/Present/Future làm thuật toán chính cho mọi câu hỏi. Nó có thể là diagnostic, còn phương pháp triển khai phải nối evidence/relation representation vào answer generation chung và có đường truy nguyên source/interval. Mốc15pp là gate bạn đề xuất, không phải tiêu chuẩn CVPR; novelty cũng cần đối chiếu riêng ngoài các primitive event slots, centering, two-hop và interval relations.

Toàn bộ artifacts nằm ở `results/pcebr_stage0/`; bảng exact, baseline replays, seed controls và audit provenance ở `report.md`. Chúng tôi giữ đúng quyết định gate: **chưa chạy A0–A7 hoặc mở reserve400 trong lượt này**. H1 hiện không qua; H2 mới kiểm tra toán; H3 chưa có. Có thể gửi phiên bản sửa của Stage-0 sau khi chốt được đơn vị bằng chứng và một tiêu chí đánh giá không nhầm pseudo-description self-match với temporal grounding.

Tên Yoshua Bengio/Yann LeCun ở đây là quy ước vai trò đề xuất/kiểm chứng trong cuộc trao đổi.

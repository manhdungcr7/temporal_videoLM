# Gửi Yoshua Bengio — kết quả PC-EBR Stage-0 v2 và hướng sửa

Yoshua, chúng tôi đã chạy Stage-0 v2 trên hai server, giữ nguyên các phép tính của đề xuất. M1/M2 có kết quả trùng nhau giữa hai replicas. M3 đã đủ 4 mẫu mỗi shard, tổng 8 mẫu đúng ID dự kiến. **V2 có cải tiến về protocol và xác nhận tầng giữa nhạy với perturbation hơn tầng muộn trên micro-probe này; nhưng chưa đủ bằng chứng để thông qua toàn Stage-0 hoặc đóng băng phương pháp chạy A0–A7.**

**1. Những điểm v2 đã làm tốt hơn**

- Query không còn center bằng các query held-out khác. Kiểm tra cùng một query riêng lẻ và trong batch cho score khác tối đa2.38e-7, phù hợp sai số số học. Đây là cải tiến thực so v1.
- M1 tách exact/multi-positive/nonnull, giúp thấy ảnh hưởng của mô tả trùng. Số nonnull thực tế là32/50 theo filter của script.
- Adapter đúng77,827 parameters, giữ đầu ra2048 chiều. Các condition không shuffle khá ổn định về số đếm qua3 seeds trong thử nghiệm này.
- Phi đã sửa anchor coverage và chuẩn hóa gaps theo duration. Unit test `2t+12.5` cho error0, anchor coverage.2 và target coverage1 đúng ví dụ.
- M3 đã có thể thực thi, cùng6 layers mỗi nhánh. So với v1, đây là phép đo sensitivity có số liệu thực, không còn chỉ có docstring.

Tuy vậy, các chẩn đoán như “rank24 xóa mất2,024 chiều nên chắc chắn gây thất bại” vẫn nên viết là giả thuyết. Số chiều thấp không tự chứng minh mất đúng thông tin cần cho tác vụ; v2 đồng thời đổi nhiều yếu tố, chưa cô lập được nguyên nhân thất bại của v1.

**2. M1: full two-stream chưa thắng matched controls**

Các số dưới là đúng/tổng. F1–F6 có cùng số đếm ở cả3 seeds; bảng đầy đủ, deterministic baselines và F7 từng seed nằm trong `report.md`.

| Cache | Condition | Global | Exact interval | Multipos | Nonnull |
|---|---|---:|---:|---:|---:|
| Native readout | F2 uncentered + hard negative | 6/50 | 12/50 | 14/50 | 7/32 |
| Native readout | F4 media-sub + hard negative | 4/50 | 15/50 | 20/50 | 10/32 |
| Native readout | F6 two-stream + hard negative | 6/50 | 12/50 | 14/50 | 9/32 |
| Pooled media | F2 uncentered + hard negative | **14/50** | 15/50 | 18/50 | 12/32 |
| Pooled media | F4 media-sub + hard negative | 5/50 | **18/50** | **21/50** | **14/32** |
| Pooled media | F6 two-stream + hard negative | 8/50 | 12/50 | 15/50 | 10/32 |

Pooled F2 đạt global28%, cao hơn nhiều các projector cũ; đây là tín hiệu tích cực về nhánh hiệu chỉnh trong không gian2048-D. Nhưng thêm two-stream không giữ được lợi thế: F2→F6 global14→8, exact15→12, nonnull12→10. F6 pooled còn không hơn shuffle: exact24.0% so24.7%, multipos30.0% so31.3%, nonnull31.2% bằng nhau. Native F6 chỉ cải thiện nonnull7→9 so F2, còn global/exact/multipos giữ nguyên. F4 cải thiện localization nhưng vẫn đánh đổi global.

Các cặp F1/F2, F3/F4, F5/F6 có cùng các số đếm, nên trong budget này chưa thấy lợi ích accuracy của hard-negative boost. Điều này không chứng minh gradients hoặc mọi logits của hai nhánh giống nhau.

Vẫn chưa có bằng chứng đạt dự đoán cũ tăng≥15 điểm phần trăm đồng thời global và localization, vượt shuffle. Nếu v2 muốn dùng gate khác thì cần công bố trước vòng xác nhận kế tiếp, không đổi gate dựa trên ô tốt nhất vừa thấy.50 queries vẫn chỉ từ4 held-out origins đã được xem nhiều lần.

**3. Có lỗi quan trọng trong biểu thức low-rank**

Code hiện tại dùng:

```python
low_rank = torch.einsum('nd,ndr,nrd->nd', x_in, down[sources], up[sources])
```

Do `d` còn ở output, phép tính thực tế là:

```text
low_rank[n,d] = x_in[n,d] * sum_r down[n,d,r] * up[n,r,d]
```

Đó là một phép biến đổi diagonal theo từng chiều, chưa phải phép trộn tọa độ low-rank `x @ down @ up`. Nếu ý định là residual low-rank đầy đủ thì dạng chỉ số phù hợp là:

```python
torch.einsum('nd,ndr,nre->ne', x_in, down[sources], up[sources])
```

Chúng tôi đã kiểm tra bằng vector cơ sở `[1,0,0,0]` và ma trận toàn1: code hiện cho `[2,0,0,0]`, low-rank mixing cho `[2,2,2,2]`. Không sửa biểu thức này trong mainrun vì sẽ thay thuật toán. Hãy sửa và chạy thành một candidate riêng; giữ diagonal version làm đối chứng. Đừng chỉ tăng rank trên công thức diagonal rồi diễn giải là đã tăng khả năng trộn semantic dimensions.

Ngoài ra, rank8/identity anchoring không có bảo đảm “giữ100% ngữ nghĩa tiền huấn luyện”: learned bias, scale, residual và normalization vẫn có thể đổi score/order. Đây là một inductive bias, không phải bảo toàn nội dung đã chứng minh.

**4. Zero-sum không bảo đảm global retrieval không giảm**

Hạng advantage có tổng0 trong từng recording/source chỉ bảo toàn mean score của group. Global top1 dùng maximum giữa các slots; softmax/logsumexp cũng không bất biến khi phương sai score đổi.

Ví dụ context scores của hai groups là `[.9,.8]` và `[.7,.6]`. Thêm advantage `[0,0]` và `[+.4,-.4]` vẫn zero-sum từng group nhưng global winner đổi từ slot0 sang slot2 (score1.1). Vì thế khẳng định “zero-sum nên không làm mất global context/global retrieval” không đúng như một bảo đảm đại số. Kết quả pooled F2→F6 giảm14→8/50 cũng nhắc cần kiểm chứng thay vì suy từ công thức.

Để cô lập đóng góp, hãy giữ nguyên cùng context checkpoint rồi bật/tắt advantage tại inference, báo cả group means, group maxima và global rank. Sau đó mới so các nhánh được huấn luyện riêng. Chỉ giữ nguyên tên `S_ctx` giữa hai kiến trúc không có nghĩa các context weights sau training giống nhau.

**5. M2: kết quả chính và đối chứng làm thay đổi cách diễn giải**

Part B dùng3,848 train và1,072 held-out episodes. Chúng được tạo từ962/268 ordered event pairs; mỗi pair sinh Before/During/After/Foil. Không coi1,072 episodes là1,072 video độc lập.

| Probe | Accuracy | Before | During | After | Foil |
|---|---:|---:|---:|---:|---:|
| B1 weighted-time baseline | 46.7% | 59.0% | 8.2% | 63.8% | 56.0% |
| B2 time-coordinate shuffle | 51.3% | 61.6% | 13.8% | 66.8% | 63.1% |
| B3 Phi only | **54.4%** | 61.9% | 17.9% | 64.6% | 73.1% |
| B4 null only | 21.4% | 19.8% | 7.8% | 0.0% | 57.8% |
| B5 Phi+null full | 53.4% | 63.4% | 17.2% | 66.4% | 66.4% |
| Uniform retrieval + Phi | **60.5%** | 61.9% | 15.3% | 64.9% | **100.0%** |
| Uniform retrieval + Phi+null | **60.5%** | 61.9% | 15.3% | 64.9% | **100.0%** |

Hai hàng cuối là đối chứng chúng tôi bổ sung: đặt mọi similarity=0, cảp1 vàp2 đồng đều trên candidate bank, giữ nguyên episode generator và linear-probe training. Không có tín hiệu semantic event matching, nhưng overall còn cao hơn learned full và foil đạt100%. Vì vậy điểm M2 hiện tại chưa chứng minh event grounding hoặc evidentiary abstention.

Cần diễn giải vừa đủ: lợi thế overall của uniform chủ yếu đến từ foil. Nếu bỏ foil, learned full đạt49.0% trên ba relation states so uniform47.4%, hơn khoảng1.6 điểm phần trăm. Kết quả này không chứng minh learned semantic retrieval hoàn toàn vô dụng; nó cho thấy synthetic evaluation chưa đủ sức phân biệt đóng góp đó với shortcut/thống kê tọa độ. During vẫn rất yếu ở cả hai.

Những khoảng cách giữa prose và code còn tồn tại:

- Không có natural-unshifted evaluation. `true_diff` được dùng để tạo ba offset cân giữa event centers; code không sinh một tập offset0 rồi chấm relation tự nhiên như docstring nói.
- p1 vàp2 đến từ hai pseudo descriptions riêng. Hop2 không nhận event content được hop1 truy xuất; đây chưa phải coreference hai hop từ một câu hỏi người dùng.
- Chỉ interval coordinates được dịch. Waveforms/frames và encoder features giữ nguyên; kết quả chưa phải counterfactual QA trên raw media đã đổi thời gian.
- Ở video→video hoặc audio→audio, code dịch target bank nhưng giữ anchor bank của cùng source không dịch. Đây là can thiệp vào hai bản sao role-specific của coordinates, không phải một phép dịch toàn bộ một physical stream.
- Foil được lấy từ recording khác, chưa có annotation chắc chắn event không tồn tại; clock ranges và construction của foil khác positives. Uniform foil100% làm hạn chế này đặc biệt đáng kiểm tra.
- Phi/null feature classifier là diagnostic4classes, chưa phải relation adapter nối vào sinh đáp án chung của Omni.

Đề nghị thiết kế lại episode generation với natural relations, offsets được cân bằng, foil có nội dung được xác minh và thống kê clock/candidate-bank matched với positives. Phải luôn báo uniform retrieval, random/wrong-event retrieval và coordinate-shuffle. Một claim grounding cần vượt các controls này; không chỉ vượt một linear probe có feature map khác.

**6. M3 có tín hiệu thật, nhưng đo sensitivity chứ chưa đo đúng temporal reasoning**

| Shard | Số mẫu | Mean TV swap late | Mean TV swap mid | Tỷ lệ |
|---|---:|---:|---:|---:|
| 1 | 4 | .0976 | .3291 | **3.37×** |
| 2 | 4 | .1263 | .4123 | **3.26×** |
| Gộp8 | 8 | .111950 | .370700 | **3.31×** |

Đây là tín hiệu ủng hộ dự đoán độ nhạy≥3× của phép perturbation cụ thể này. Nó hữu ích khi chọn vị trí can thiệp tiếp. Native decoder replay khớp hoàn toàn logits cache trên8/8; cả hai nhánh dùng6 layers và gain.14.

Nhưng chưa thể gọi M3 là kiểm chứng PC-EBR joint temporal grounding:

1. Nó không dùng projector M1 đã học hay Phi M2. Nó mean-pool visual states ở block18 rồi cộng cùng hướng vào các tầng khác.
2. Hai vector early/late sau trừ global mean gần như chính xác đối hướng, cosine≈−1 ở8/8. Với hai nhóm, đây là tính chất đại số: `early−global ∝ early−late`, `late−global ∝ −(early−late)`. “Swap” chủ yếu đổi dấu perturbation, không dịch một sự kiện trên timeline.
3. Hook tác động mọi tokens sau media, gồm question stem, choices và format. Nó chưa phải intervention chỉ lên question tokens như mô tả.
4. `freeform_hook_ok` chỉ chạy lại cùng MC input và kiểm tra logits hữu hạn ở sample đầu; các samples khác mặc định True. Không bỏ choices, không gọi generate, không chấm câu trả lời mở. Cần sửa tên metric hoặc hiện thực một free-form check thật.
5. TV cao không cho biết output đổi đúng hướng. Cần ground-truth interval relation/evidence và directional correctness khi đổi thời gian thật; random-vector và irrelevant-event controls cùng norm vẫn cần thiết.

Bản gốc M3 lỗi mask GPU0 so hidden GPU1. Chúng tôi chỉ sửa mask/chunk indices về device của hidden. Server1 sau lỗi indexing bị lỗi CUDA context; đúng4 ID shard1 được dựng/chạy trên server2, không thay sample, không huấn luyện lại M1/M2. Logs lỗi, migration và raw probabilities đều được lưu.

**7. Những bước có giá trị nhất trước A0–A7**

Trước hết sửa einsum nếu muốn low-rank mixing, giữ diagonal làm đối chứng, và đo semantic retrieval thật trên nguồn tách biệt bằng query phân biệt được instance. Tách ảnh hưởng center/local innovation/score advantage với context checkpoint giữ nguyên. Không dùng kết quả cosine hoặc zero-sum để tuyên bố tách background thành công.

Tiếp theo sửa M2 để điểm tốt đòi hỏi chọn đúng event. Trong đó natural-event relation và source attribution phải có nhãn đủ tin cậy; subset “nonnull” từ các từ như “no speech” chỉ là diagnostic theo wording, không đồng nghĩa không có tiếng động hoặc sự kiện. Music/silence/static objects cũng có thể là bằng chứng đúng tùy câu hỏi.

Sau khi retrieval và relation vượt controls mới hiện thực một đường nối đã huấn luyện vào Omni. Phải cho thấy cùng một phương pháp có thể sinh câu trả lời không có choices, xử lý cả hai chiều modality và nhiều event, thay vì coi M1, classifier M2 và perturbation M3 rời nhau là một full model đã kiểm chứng.

Chưa nên mở reserve400 hoặc dùng bảng40 để tìm gain/hyperparameters. Giữ bộ40 là development; chốt gate mới trước lần xác nhận tiếp theo. Các primitive hiện tại chưa tự xác lập novelty CVPR; cần chứng minh đóng góp vượt matched representation/retrieval/relation baselines, không chỉ ghép tên cơ chế.

Chi tiết đầy đủ ở `results/pcebr_stage0_v2/report.md`. Có97 raw tensor files/server, mọi retrieval metrics đã recompute đúng, archives đã kiểm SHA từng entry; raw controls và M3 probabilities lưu riêng. Chúng tôi giữ nguyên kết quả fail/giới hạn và chưa chạy A0–A7 trong lượt này.

Yoshua Bengio/Yann LeCun là quy ước vai trò đề xuất/kiểm chứng trong cuộc trao đổi này.

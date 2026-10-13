# Từ cách con người nghe–nhìn đến một giả thuyết kiến trúc có thể kiểm tra

Ngày đối chiếu: 12/10/2026. Tài liệu bổ sung cho `12_temporal_videolm_research_proposal.md`.

**Trạng thái:** nghiên cứu tài liệu và audit cấu trúc token đã lưu. Chưa chạy inference của kiến trúc đề xuất. PDF/HTML của 8 paper VideoLM mới được bổ sung vào `literature/temporal_generalization_2026-10-12`, nâng thư viện ở đó lên 27 bài. Các nguồn nhận thức bên dưới đã được đọc qua nội dung sơ cấp công khai; chưa lưu toàn bộ PDF của chúng.

## 1. Ý tưởng từ quan sát của bạn

Ba nguyên tắc đáng giữ:

1. Nhận thông tin nghe và nhìn trên cùng đồng hồ vật lý.
2. Tạo đơn vị sự kiện để cập nhật hiểu biết theo diễn biến, giữ thông tin trước đó khi nó còn cần thiết.
3. Chọn bằng chứng theo câu hỏi và mức quan trọng của sự kiện.

Có hai điểm cần điều chỉnh: con người vẫn có thể gắn sai tín hiệu nghe–nhìn; đồng thời não không luôn chia đôi mức tin cậy cho hai nguồn. Các kết quả về integration còn phụ thuộc task và stimulus. Đây là nguồn cảm hứng thiết kế, không phải chứng minh kiến trúc nào sẽ thắng.

[Causal Inference in Multisensory Perception](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943) nghiên cứu việc suy ra tín hiệu nghe và nhìn đến từ một hay nhiều nguồn trong bài toán định vị. Nó gợi ý kiểm tra có nên kết hợp hai tín hiệu trước khi kết hợp. Bài này không trực tiếp nghiên cứu narration/video QA.

[The effects of attention on temporal integration](https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2015.00032/full) cho thấy tác động của attention lên cửa sổ tích hợp thay đổi theo task và độ bất định. Không lấy một con số cửa sổ thời gian từ thí nghiệm psychophysics làm hyperparameter cố định cho video nấu ăn.

[Alais & Burr](https://pubmed.ncbi.nlm.nih.gov/14761661/) nghiên cứu AV localization và cho thấy mức đóng góp của hai nguồn thay đổi với độ chính xác của chúng. [Baldassano et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5558154/) cung cấp bằng chứng về phân đoạn trải nghiệm thành sự kiện ở nhiều thang thời gian; trong lượt này nội dung PMC chỉ truy cập được qua phần indexed summary. Không suy ra bộ não dùng một mask Transformer cụ thể.

## 2. Qwen Omni đã có phần nào?

[Qwen2.5-Omni technical report](https://arxiv.org/html/2503.20215), mục 2, mô tả time-interleaving theo khoảng 2 giây, visual representations trước audio representations, cùng TMRoPE. Thinker là decoder Transformer. Vì vậy, tổ chức input theo từng khoảng nghe–nhìn đã có.

Minh họa input:

```text
0–2 s:   [visual tokens] [audio tokens]
2–4 s:   [visual tokens] [audio tokens]
4–6 s:   [visual tokens] [audio tokens]
...
[question] [assistant prefix] --> answer
```

Code Transformers local ở `local_qwen/venv312/Lib/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py` đặt Thinker attention là causal và tạo causal mask. Checkpoint local `f75b40e3da2003cdd6e1829b1f420ca70797c34e` có `use_sliding_window=False`.

**Causal nghĩa là vị trí hiện tại được đọc các vị trí trước nó. Nó không tự ngăn đọc một khoảng thời gian khác.** Cùng timestamp và cùng chunk cũng không tạo ra hai chiều tương tác đối xứng: visual tokens đứng trước audio tokens nên trong lớp attention đó không đọc được audio phía sau; audio phía sau được phép đọc visual phía trước. Token câu hỏi cuối chuỗi được phép đọc cả hai. Điều này không tự chứng minh model lỗi; attention nhiều lớp có thể học cách tổng hợp.

Audit từ trace CMTB 1601 đã lưu:

| Loại | Vị trí token | Temporal range từ trace |
|---|---:|---:|
| Visual block đầu | 45–332 | 0–1 s |
| Audio block đầu | 333–382 | 0–1,96 s |
| Visual block tiếp | 383–670 | 2–3 s |
| Audio block tiếp | 671–720 | 2–3,96 s |

Trace có 3.024 visual tokens, 528 audio tokens. Có 1.728 visual tokens ngoài khoảng speech được detector chọn. Những con số này là input/layout audit; chúng không phải attention weights và không cho biết token nào thực sự gây sai. File tái lập: `native_stream_layout_audit.json` và `audit_stream_layout.py` trong thư mục tài liệu.

Hai nhận định cần tránh:

- “Qwen không xử lý tuần tự”: decoder có thứ tự causal và có KV cache; prefill có thể tính song song theo mask mà vẫn giữ phụ thuộc theo thứ tự.
- “Qwen nhớ đều mọi thời điểm”: giữ nhiều token không đồng nghĩa attention phân bố đều; sự chọn lọc đã được học, chất lượng chọn lọc mới là vấn đề cần đo.

## 3. Ba ý tưởng bạn nêu có tác dụng gì?

| Ý tưởng | Điều có thể giúp | Điều chưa bảo đảm |
|---|---|---|
| Thêm separator giữa khoảng | Cung cấp tín hiệu cấu trúc nếu model đã học ý nghĩa đó | Không tự chặn attention hoặc xóa nội dung đã trộn vào hidden states |
| Attention audio/visual bằng nhau | Có thể dùng làm một đối chứng phân bổ mass | Không chứng minh hai nguồn đều đúng, có liên quan, hoặc cùng chỉ một sự kiện |
| Xử lý local rồi ghi memory | Hạn chế ngữ cảnh không cần thiết, giữ lịch sử có cấu trúc | Memory có thể chứa hallucination, mất thao tác ngắn hoặc quên sự kiện cần cho câu hỏi tới sau |

Một token separator mới với embedding ngẫu nhiên cần huấn luyện để model sử dụng đáng tin. Dùng token đã có hoặc văn bản ghi mốc là một cách thử prompting, nhưng không được gọi nó là ranh giới cứng của attention. Nếu muốn ranh giới có tác dụng tính toán, phải sửa mask/routing và đo tác động.

Ép attention 50/50 còn bỏ qua số token khác nhau giữa nguồn, nội dung im lặng/noise và yêu cầu câu hỏi. Chuẩn hóa theo token count là công cụ phân tích, không phải nhãn về “mức attention đúng”.

## 4. Các paper gần nhất với ý tưởng con người của bạn

| Paper | Phương pháp đã có | Ranh giới cần nhận biết |
|---|---|---|
| [MA-LMM](https://arxiv.org/html/2404.05726), mục 3 | Online frame processing, visual/query memory, Q-Former và compression | Sequential processing + selective memory đã có; Q-Former có training |
| [MovieChat](https://arxiv.org/html/2307.16449), Method | Short/long memory, xử lý cửa sổ và consolidation | Dùng cảm hứng trí nhớ con người cũng đã có |
| [VideoLLM-online](https://arxiv.org/html/2406.11816) | Streaming dialogue, streaming EOS objective và KV caching | Tiếp nhận video liên tục cần training recipe phù hợp |
| [VideoChat-Online](https://arxiv.org/html/2501.00584) | Pyramid memory và đồng bộ việc cập nhật memory với KV cache | Thay memory không được giữ cache cũ tùy tiện |
| [Think While Watching](https://arxiv.org/html/2603.11896) | Segment-level memory notes xen kẽ với segment input/câu hỏi | “Xem từng đoạn rồi ghi nhớ” đã có thiết kế và training |
| [HippoMM](https://arxiv.org/html/2504.10739), mục 4 | Phân đoạn thích nghi, encoding AV, consolidation, hierarchical retrieval | Gần trực tiếp với mô tả của bạn; dùng foundation models/tools |
| [DASH](https://arxiv.org/html/2603.15685), mục 3 | Audio-derived semantic boundaries làm soft prior cho video; boundary/content/attention scoring để nén | Dynamic chunking và giữ transition tokens không phải mới tự thân; họ không ép audio/visual boundary trùng tuyệt đối |
| [AV-Flamingo](https://arxiv.org/html/2607.16107), mục 2 | Temporal interleaving, CRTE, training theo curriculum và timestamp-grounded reasoning | Cả tổ chức thời gian và supervision suy luận đều cần so sánh |

Đã đọc phần phương pháp liên quan của các paper trên, chưa tái lập số đo của họ. Không so trực tiếp accuracy của chúng với pilot SVHalluc vì model/dataset/protocol khác nhau.

## 5. Suy ngẫm quan trọng: cùng lúc không có nghĩa cùng nội dung

Trong clip, tại 3–5 s:

```text
Mắt: đang cho cà chua vào nồi.
Tai: người nói “lát nữa sẽ thái khoai”.
```

Ba thông tin đều cần được giữ:

1. Sự kiện nhìn thấy: cho cà chua, xảy ra ở 3–5 s.
2. Phát ngôn nghe thấy: câu nói về thái khoai, được nói ở 3–5 s.
3. Hành động được câu nói nhắc tới: thái khoai; cần tìm lúc nó thực sự xuất hiện trong hình.

Ghép visual+audio cùng chunk thành một event duy nhất có thể tạo câu sai “đang thái khoai ở 3–5 s”. Chặn mọi tương tác giữa chunk lại khiến model không tìm được hành động thái khoai ở 8–10 s.

Nguyên tắc thiết kế đề xuất: **cho phép liên kết nội dung qua nhiều thời điểm, nhưng giữ nguyên nguồn và khoảng của từng bằng chứng**. Nhiều đồng hồ nguồn là metadata trên cùng timeline vật lý; chúng không hàm ý file audio/video bị lệch encode.

## 6. Kiến trúc để thử: bộ nhớ có nguồn/thời gian và hai loại quan hệ

Đây là refinement của đề xuất ở bài 12, chưa phải kiến trúc đã xác nhận mới hoặc tốt hơn.

```text
                       Video + audio
                             |
              Đồng hồ gốc / native encoders
                             |
                Xử lý các khoảng có overlap
                    /                    \
        Visual evidence E_i        Audio/utterance U_j
        [nội dung, interval,       [nội dung, interval,
         source=visual]             source=audio]
                    \                    /
                     Bộ nhớ bằng chứng
                      /              \
       Quan hệ cùng thời điểm       Quan hệ được nhắc tới
       (hình/âm gì tại lúc S?)       (U_j nói về E_i nào?)
                      \              /
                    Question-conditioned read
                             |
                  Native Thinker / residual repair
                             |
                           Answer
```

### 6.1 Local perception

Giữ local evidence cùng clock để nhận thao tác và speech chính xác. Thử 2 giây cùng overlap như cấu hình chẩn đoán ban đầu, không coi 2 giây là ranh giới sự kiện tự nhiên. Sự kiện dài và speech dài cần nhìn lại các khoảng lân cận; motion cũng cần một dãy frame.

Prototype đầu tiên giữ nguyên token và encoder features; thử mask làm giảm đường truyền từ evidence không liên quan. Không thêm learned memory tokens ở giai đoạn chưa training. Nếu muốn attention hai chiều giữa media trong chunk, đó là thay đổi native causal graph; chỉ áp dụng ở media, giữ causal text generation, và cần đo sự lệch với phân phối đã học.

### 6.2 Memory

Mỗi record giữ tensor evidence hoặc pointer tới raw input, interval và source. Không coi một summary text là toàn bộ sự kiện chắc chắn đúng. Cache toàn clip hoặc index thưa làm ngữ cảnh; truy lại raw evidence khi câu hỏi cần chi tiết.

Khởi đầu không nén để tránh trộn kiểm tra memory compression với binding. Khi triển khai nén, so cùng ngân sách token và đo mất thao tác ngắn/sự kiện lặp lại. Một sự kiện nhỏ, ít nổi bật vẫn có thể là bằng chứng quyết định của câu hỏi tương lai.

### 6.3 Hai loại quan hệ

- **Co-temporal:** interval hình và interval audio trùng nhau. Đây là quan hệ về vị trí thời gian, chưa chứng minh chúng do cùng một nguyên nhân vật lý.
- **Referential:** lời nói nhắc tới một sự kiện nhìn thấy; event visual có thể ở trước, trong, sau hoặc không có trong clip.

Nếu cần physical common-cause relation cho tiếng va chạm/động vật/nhạc cụ, giữ một nhãn quan hệ riêng để không đồng nhất nó với lời narration. Phân loại quan hệ cần được đánh giá; không dùng tên task benchmark làm router.

### 6.4 Đọc theo câu hỏi

CMTB đọc visual event tại thời điểm utterance. TA/TF tìm visual event được utterance nhắc tới trên toàn timeline rồi so hai khoảng. Câu hỏi về nội dung nói đọc audio; câu hỏi object identity có thể cần cả hai nguồn. Query đọc được global context khi nó giúp tham chiếu.

Chính phần sử dụng bằng chứng có vai trò khác nhau là ứng viên cần phát triển, chứ không ép mọi task dùng cùng mức audio/video hoặc cùng mask.

## 7. Phần nào có thể mới, phần nào chắc chắn cần dùng làm baseline?

Dynamic chunk, event memory, temporal PE, delimiter và attention gating đều có tiền lệ. Hai loại relation cũng là khái niệm quen thuộc trong grounding; gọi tên chúng không đủ tạo novelty.

Ứng viên đóng góp là **một cơ chế được đo trong decoder khiến nội dung đúng bị gán sai nguồn/thời điểm, cùng phép sửa bảo vệ provenance trong local perception và vẫn hỗ trợ truy cập liên sự kiện**. Ranh giới cần so thêm với DASH/HippoMM/AV-Flamingo và các paper trong bài 12.

Nếu chỉ dynamic chunk + memory đã đủ gain, đó là baseline tốt cần ghi nhận. Muốn đóng góp mạnh hơn phải cho thấy module mới thêm gì: sự bất biến đúng khi dịch cả hai nguồn, sự thay đổi đúng khi dịch riêng một nguồn, và khả năng giữ đúng event reference dù không đồng thời.

## 8. Thí nghiệm nhỏ có thể phân biệt các ý tưởng

Chọn một CMTB rõ và một TA/TF lệch rõ từ các video nguồn khác nhau; thêm một baseline-correct control. Không tối ưu chỉ trên ca sai.

| Điều kiện | Thay đổi duy nhất cần cố gắng giữ |
|---|---|
| Native | Nguyên model/input/question |
| Separator | Thêm tín hiệu cấu trúc bằng token đã có; báo thay đổi độ dài/position |
| Equal mass | Can thiệp phân bổ mass, không đổi token content; dùng làm control |
| Local evidence | Mask theo khoảng, vẫn giữ all evidence để query tham chiếu |
| Source/time-preserving | Cô lập contextualization theo nguồn/thời điểm rồi read hoặc patch đúng điểm |

Với mỗi điều kiện, tạo các cặp thời gian được người xem kiểm tra:

1. Speech và action cùng lúc.
2. Speech nói về cùng action nhưng xảy ra trước/sau.
3. Audio là lời nói về chủ đề không liên quan, visual giữ nguyên.
4. Action ngắn nằm sát ranh chunk, hoặc xuất hiện nhiều lần.

Không dịch audio rồi tự gán nhãn nếu sự kiện bị cắt/nhân đôi. Không tạo bản noisy bằng mã hóa lại cả visual tensor. Dùng phép giữ input đúng và mapping option đúng như bài 12.

Quan sát cần đo: nhận diện event, event intervals, answer meaning scores, correct→wrong, đường truyền attention trực tiếp/gián tiếp và tác động activation patching. Một separator làm đúng một câu chưa chứng minh chống interference; một attention pattern đẹp chưa chứng minh model đã sử dụng nó.

## 9. Kết luận nghiên cứu ở thời điểm này

Ý tưởng nghe–nhìn tuần tự của bạn giúp đặt đúng câu hỏi kiến trúc. Qwen đã có phần tổ chức input; vấn đề cần kiểm tra là **cách các biểu diễn được cập nhật và được đọc để tạo quan hệ đúng**. Đề xuất memory/attention ở đây giữ lịch sử và liên kết khác thời điểm, vì TA/TF cần chính năng lực ấy.

Hiện đã bổ sung tài liệu và audit layout, chưa chạy phương pháp mới. Bước tiếp theo hữu ích nhất là kiểm tra native vs local mask vs source/time-preserving contextualization trên vài cặp rõ, rồi kiểm tra nguồn mới trước khi training hoặc mở full benchmark.

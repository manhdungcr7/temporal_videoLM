# CQ diagnostic: source-grounded temporal event memory

## Kết quả cuối

Đã hoàn tất các vòng phát triển trên hai GPU Kaggle. Chọn bản **round15,
`episode_constraints`**, không dùng CoT ở bước đọc đáp án cuối.

| Nhóm ground truth | Base đúng / sai | CQ cuối đúng / sai |
|---|---:|---:|
| TA/B — No | 4 / 1 | **5 / 0** |
| TF/C — Future | 3 / 2 | **4 / 1** |
| TA/A — Yes, đối chứng | 0 / 2 | **2 / 0** |
| TF/A — Past, đối chứng | 1 / 1 | **2 / 0** |
| TF/B — Present, đối chứng | 2 / 0 | **2 / 0** |
| 10 mẫu yêu cầu | 7 / 3 | **9 / 1** |
| 6 mẫu đối chứng | 3 / 3 | **6 / 0** |
| Tổng 16 mẫu | 10 / 6 — 62.5% | **15 / 1 — 93.75%** |

Đây là kết quả phát triển/diagnostic, **không phải kết quả held-out**. Sáu đối
chứng ban đầu được chọn độc lập nhưng sau đó được xem và dùng qua nhiều vòng
phát triển. Vì vậy 15/16 không phải ước lượng không thiên lệch của khả năng
tổng quát, không chứng minh bias đã bị loại trên toàn benchmark, và không
đảm bảo một paper sẽ được CVPR chấp nhận.

## Protocol và mẫu

- Model: Qwen/Qwen2.5-Omni-3B; revision
  `f75b40e3da2003cdd6e1829b1f420ca70797c34e`; 36 decoder layers; FP16.
- Hai server, mỗi server hai Tesla T4. Không fine-tune.
- Dev seed 20261014; control seed 20261015. Chọn theo nhóm nhãn và thời lượng,
  không chọn dựa trên dự đoán. Mọi clip được giữ nguyên, không quá 50 giây.
- TA/B: 1454, 2396, 1727, 1647, 2345. TF/C: 2139, 2327, 1603, 2097, 1532.
- Đối chứng: TA/A 2348, 2141; TF/A 1580, 2040; TF/B 1666, 2124.
- Mỗi lựa chọn được chấm bằng log probability token nhãn, xoay vị trí nội dung
  giữa các lựa chọn rồi cộng điểm về đúng nội dung. Đây không phải cộng offset
  cho nhãn B/C. Không dùng ground truth, task ID hay sample ID trong inference.
- Mười sáu bản ghi cuối đều hợp lệ. Job lỗi CUDA và kết quả chưa đủ mẫu không
  được đưa vào bảng cuối. Không tính smoke test thêm vào mẫu benchmark.

## Phương pháp cuối làm gì?

Đây là phát triển từ CQ thành **bộ nhớ sự kiện có nguồn và khoảng thời gian**,
không còn là một lần forward với attention boost vào prompt rỗng.

1. **Quan sát độc lập.** Cắt video thành các đoạn 2 giây. Tạo quan sát visual
   và audio từ đúng modality của đoạn đó bằng lượt suy luận riêng; bước này
   không nhận question/options. Dùng input video-only/audio-only vật lý thay
   vì tin rằng mask ở decoder là đủ để cách ly thông tin.
2. **Giữ câu nói nguyên vẹn.** Thêm node audio theo speech span của Silero VAD
   6.2.3 và transcript audio-only; không ghép câu nói giả từ các mảnh 2 giây.
   Các node audio dài hơn là bổ sung phân cấp, không thay âm thanh bằng video.
3. **Ground referent qua hai nguồn.** Question truy vấn bộ nhớ để tìm sự kiện
   được nhắc tới; kiểm tra lại utterance tương ứng chỉ từ audio transcript.
   Sau đó kiểm tra sự kiện trên từng crop visual-only. Lượt kiểm tra visual
   được cung cấp mô tả sự kiện cần tìm, nhưng không nhận audio hay đáp án.
4. **Từ peak sang episode.** Với từng crop, lấy margin Yes-vs-No của kiểm tra
   sự kiện. Chọn vùng dương liên tiếp chứa crop có margin cao nhất, ngưỡng 0.
   Khoảng này là hỗ trợ visual ước lượng, không phải timestamp ground truth.
   Lưu riêng thời gian nói, thời gian thấy, quan hệ trước/trùng/sau và số giây
   giao nhau, thay vì dùng peak làm toàn bộ thời gian diễn ra.
5. **Đọc đáp án bằng constraint ngữ nghĩa.** Model tự diễn giải lựa chọn từ
   question + nội dung lựa chọn, qua các primitive earlier/during/later hoặc
   unsupported và khẳng định/phủ định. Không có bảng literal
   Yes→overlap/No→nonoverlap/Past→before được hardcode. Gắn constraint suy ra
   cạnh lựa chọn gốc rồi đọc graph bằng native text-only choice scoring.
   Không dùng lời nhắc nghi ngờ audio hay CoT dài ở readout cuối.

Constraint là chú thích có thể chưa đầy đủ, **không phải định nghĩa tương
đương tuyệt đối của lựa chọn**. Ví dụ model đôi khi suy ra Present là
`not_precedes`: đây chỉ là điều kiện cần, chưa phân biệt Present với Future.
Lựa chọn gốc vẫn được giữ; không có luật quyết định nhãn từ primitive.

API: `cq_provenance_answer(video_path, question, answer_options,
observation_cache=None, mode='episode_constraints')`. Loader lấy whitelist
hàm/hằng số vào namespace sạch, bỏ dataset, manifest và vòng đánh giá. Adapter
model/media vẫn dùng runtime Omni sẵn có; đây chưa phải package standalone.
Fallback scene memory cho câu hỏi không có temporal predicate chưa được
đánh giá ngoài bộ diagnostic này.

## Những vòng thử nghiệm đáng chú ý

| Cách tiếp cận | TA/B | TF/C | Đối chứng | Rút ra |
|---|---:|---:|---:|---|
| Base | 4/5 | 3/5 | 3/6 | Điểm khởi đầu, chưa hết lỗi bias |
| R1: CQ rỗng, sửa position, cách ly decoder, cân bằng attention | 0/5 | 0/5 | — | Đổi mask/position/boost chưa tạo được bằng chứng |
| R2: CQ có completion, dual reader β=2 | 5/5 | 5/5 | 1/6 | 10/10 giả tốt: đảo chiều bias; loại |
| R6: audio utterance + hierarchical memory, plain | — | 2/5 | — | Transcript tốt hơn chưa đủ định vị sự kiện |
| R7: query→audio→visual, plain | 4/5 | 2/5 | — | Parser lỗi và utterance sai làm graph trống/sai |
| R8: sửa parser + audio anchor + reasoning | 5/5 | 5/5 | 4/6 | Vẫn sai cả TA/A; không chọn |
| R9: episode thay peak, plain | 5/5 | 3/5 | 5/6 | Giữ duration giúp nhóm đang diễn ra |
| R11: episode + occupancy, plain | 5/5 | 3/5 | 6/6 | Tránh đổi toàn bộ sang No/Future |
| R12: free-form choice interpretation + reasoning | 5/5 | 4/5 | 5/6 | Diễn giải tự do thiếu ổn định |
| R13: constrained interpretation + reasoning | 5/5 | 4/5 | 6/6 | Điểm cao nhưng audit thấy Past bị diễn giải sai |
| R14: diễn giải lexical tách source role, plain | 5/5 | 4/5 | 6/6 | Sửa lỗi ngữ nghĩa, không cần CoT |
| **R15: giữ predicate là constraint, plain** | **5/5** | **4/5** | **6/6** | Bản cuối đóng gói |

Các hàng là các cấu hình cụ thể, không cộng kết quả của nhiều phương pháp.
Đối chứng của dual reader nằm ở `cq_dual_readout_controls_s*.jsonl`; đối chứng
R8 ở `cq_provenance_controls_s*.jsonl`. Các vòng R9–15 chứa cả hai tập trong
cùng JSONL, trường `set` phân biệt chúng. R9–15 tái sử dụng evidence/checks đã
lưu để ablate readout trên cùng bằng chứng, không tái trích xuất media ở mọi
hàng. Smoke API cuối kiểm tra lại đường chạy end-to-end trên hai mẫu.

Các thử nghiệm bổ sung: R1 null calibration chỉ giúp TF/C 2/5; R3 physical
chunk memory vẫn làm TF/C plain 0/5; R4 residual CQ với native question RoPE
giữ TA/B 4/5 như zero-gain, chưa thấy thêm lợi ích. R4 phiên bản CUDA lỗi bị
loại; không có đủ kết quả TF của hướng residual. R5 chỉ chuẩn bị, chưa chạy.
Do session server1 cũ mất, raw R3 chỉ khôi phục được 4/5 record TA và R6 TA
cũng chưa hoàn chỉnh; không coi các artifact này là bộ đầy đủ.

## Tại sao CQ ban đầu tệ và bias?

Các kết luận dưới đây là suy luận từ ablation và trace, không phải chứng minh
nhân quả duy nhất của toàn bộ mô hình:

- **Prompt rỗng không phải observation.** Token cuối của scaffold có thể
  tóm tắt cấu trúc/ngữ pháp nhiều hơn nội dung. Boost nó khuếch đại đường
  chưa mang evidence; sửa position đơn lẻ không cứu được trong R1.
- **Nguồn sai và hallucinated observation.** Completion audio có lúc bịa
  “I'm going to the store” trên đoạn không có lời nói; completion visual
  lại sinh câu thoại. Mask decoder không đảm bảo raw state đã sạch nguồn.
- **Query chưa liên kết đúng referent.** Scene summary chung không đủ cho
  quan hệ “sự kiện được người nói nhắc tới”. Parser scalar/list và sai audio
  anchor là lỗi triển khai thật, không nên quy tất cả cho model attention.
- **Peak không phải duration.** Một hành động vẫn đang diễn ra lúc nói dù
  crop nổi bật nhất nằm trước đó; graph dựa peak dễ kéo sang Past/No.
- **Readout cũng có prior.** Negative source warning/CoT có thể đổi một
  graph hợp lý thành No/Past. Hai lần đạt 10/10 dev nhưng thất bại đối chứng
  cho thấy accuracy trên một nhãn không đủ để chứng minh giảm bias.
- **Từ ngữ và source roles bị trộn.** Khi diễn giải Past/Present/Future cùng
  vai trò audio/visual ngay từ đầu, model có thể đảo trước/sau. Tách lexical
  relation ra rồi giữ nó như constraint giảm lỗi, nhưng chưa hoàn hảo.

## Một mẫu còn sai: TF/C, ID 2097

Gold Future; bản cuối dự đoán Present. Audio node: “Load up our beef.” tại
[0.1, 1.0]s. Visual verifier cho margin dương suốt [0, 12]s; episode được gán
trùng với speech. Vì vậy readout trả lời đúng theo graph **sai về evidence**.
Trace gợi ý verifier nhầm hoạt động/đối tượng gần nghĩa với đúng sự kiện;
chưa có annotation frame-level độc lập để kết luận chính xác ranh giới lỗi.
Không sửa bằng offset Future hay luật dành riêng cho nấu ăn/sample này.

## Đánh giá nghiên cứu và triển khai

Điểm mạnh: inference không nhận nhãn/task; extraction độc lập; đường truy
vấn và verification áp dụng cho referent bất kỳ; episode/overlap là primitive
thời gian chung, không phải kitchen template. Tập đối chứng nhỏ cho thấy bản
cuối không còn collapse toàn bộ sang một nhãn trong các mẫu đã kiểm tra.

Giới hạn: 16 mẫu được dùng phát triển nhiều lần; CQ captions/ASR/verifier còn
hallucinate; episode dựa ngưỡng và vùng liên tiếp có thể gộp sai hoặc bỏ các
lần lặp; constraint parser còn thiếu nghĩa; API hiện tối đa bốn lựa chọn.
Không có kết quả CMTB, free-form QA, model khác, video dài hoặc miền khác.
Multi-pass thêm nhiều forward calls: không thể gọi đây là thành công của
one-pass latent attention boost hoặc một giải pháp inference rẻ.

Kiểm tra API đóng gói trên server1 đã hoàn tất: TA/A 2348 trả lời Yes đúng
(70.08s), TF/B 2124 trả lời Present đúng (111.43s). Giao diện bốn lựa chọn
trên câu hỏi diễn đạt lại chạy thành công (70.93s); đây chỉ là smoke test,
không có nhãn benchmark độc lập để tính accuracy. Latency là thời gian của
ba lần chạy cụ thể với trạng thái cache khác nhau, không phải benchmark
trung bình hay phép đo chi phí đối chiếu với base. Loader xác nhận không có
DATA/MANIFEST/ITEMS/gold/task trong namespace inference. Server1 queue=0
sau kiểm tra; server2 tunnel offline, nhưng raw kết quả cuối đã lưu đầy đủ.

Muốn xây dựng paper: đóng băng pipeline và đánh giá held-out cân bằng đầy đủ;
thêm dịch audio có kiểm soát và tráo nội dung giữ nguyên thời gian; đo source
attribution và event localization bằng annotation độc lập; ablate từng bước;
đo latency/call budget; kiểm tra model/miền khác và so sánh related work.
Không tối ưu thêm trên cùng 10 mẫu rồi gọi là validation độc lập.

## Artifact và tái lập

- Raw cuối: `results/cq_diagnostic_round15_s1.jsonl`,
  `results/cq_diagnostic_round15_s2.jsonl`.
- Manifests: `results/cq_diagnostic_manifest_s*.json`,
  `results/cq_control_manifest_s*.json`.
- API: `repo/kaggle/jobs/cq_provenance_answer.py`.
- Loader và upload: `server_jobs/cq_load_inference.py`,
  `server_jobs/cq_upload_inference.ps1`. Metadata tải về:
  `results/cq_inference_metadata.json`.
- Smoke artifacts: `results/cq_inference_smoke_2348.json`,
  `results/cq_inference_smoke_2124.json`, `results/cq_four_choice_smoke.json`.
- Các helper extraction/grounding/episode/readout nằm trong
  `server_jobs/cq_diagnostic_round{3,6,7,8,9,10,11,12,13,14,15}.py`;
  loader chỉ thực thi whitelist, không chạy các vòng dataset của các file.
- Xem đúng/sai và histogram theo từng (task, gold):
  `./server_jobs/summarize_cq_lab.ps1 -Round 15`.
- Sau khi runtime Omni đã nạp và ffmpeg/media adapters hoạt động:
  `./server_jobs/cq_upload_inference.ps1 -Server 1`, rồi
  `./server_jobs/cq_lab.ps1 -Action Submit -Server 1 -File server_jobs/cq_load_inference.py`.
  Trong job cùng runtime gọi `CQ_NS['cq_provenance_answer'](...)`.
- Token/URL chỉ lưu trong `remote_config*.json` bị gitignore, không đưa vào
  report hoặc source bundle. Server2 ngrok offline sau khi download đầy đủ
  kết quả; phần kiểm tra API còn lại chuyển sang server1.

Tài liệu nền: [Qwen2.5-Omni technical report](https://arxiv.org/abs/2503.20215),
[Silero VAD chính thức](https://github.com/snakers4/silero-vad),
[Visual Contrastive Decoding, CVPR 2024](https://arxiv.org/abs/2311.16922).
Các tài liệu này cung cấp bối cảnh, không phải bằng chứng cho accuracy CQ ở đây.

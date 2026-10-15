# Hướng kế tiếp sau CCLD và source/time slot fitting — chưa thực nghiệm

Ngày2026-10-15. Đây là giả thuyết nghiên cứu và thiết kế kiểm chứng, không phải
phương pháp đã đạt điểm cao. Không gán kết quả của literature cho thí nghiệm này.

## Nút thắt đã có bằng chứng

CCLD C3/C5 vẫn chọn Present15/15 khi không dùng VAD window trong nhánh tương ứng.
S2 same-layer interaction22/40 nhưng Present0/5. Do đó, chuyển thiên lệch nhãn
không đồng nghĩa học quan hệ thời gian. D1 bỏ lexical clock vẫn Present15/15,
nên clock không cần thiết cho collapse ở M2; nguyên nhân riêng của M8 còn cần
đối chứng đúng thành phần.

Source/time slot optimization nâng nearest-time self-match của teacher từ
113/527 lên458/527 cho video,257/514 lên440/514 cho audio. Nhưng L2 giữ nguyên
mọi40 đáp án native. Đọc cùng slots sớm hơn từ block15 làm accuracy18->17.
Tái tạo feature nội bộ và đổi nơi đọc riêng lẻ chưa sửa được quyết định QA.

## Phương pháp đề xuất để kiểm chứng tiếp

Giữ rawAV để không cắt mất evidence. Một memory chung chứa các latent liên tục
có source provenance và interval trên đồng hồ vật lý. Cả audio và video có thể
là điểm xuất phát, điểm đích hoặc cùng tham gia; query không đi qua luật nhận
diện TA/TF/CMTB, không bị ép audio->visual. Reasoner được học cách truy cập
memory và rawAV khi tạo đáp án tự do, thay vì cộng một vector vào choices.

Thay target là hidden states của chính model bằng supervision độc lập có kiểm
chứng: segment captions/events đúng nguồn, instance intervals và QA đa dạng.
Latent dự đoán cả semantic feature và phân phối interval; LoRA hoặc adapter
nhỏ học nối interval/evidence với quyết định cuối. Backbone encoders ban đầu
đóng băng; teacher caption và interval không được lấy từ40 development.
Không bắt một âm thanh và hình cùng thời điểm phải mô tả cùng sự kiện.

Mục tiêu tối thiểu: generation loss + interval grounding loss + contrastive
loss với hard negatives thay đúng instance/time/source mà giữ semantic dễ
nhầm. Có thể thêm tính equivariant của interval dưới phép đổi đồng hồ đã biết.
Phải kiểm tra label noise trước khi đặt tên loss là counterfactual.

Không dùng mean-pool2s làm đơn vị duy nhất: boundary lỗi<2s có thể quyết định
đáp án. Dùng coarse/fine slots hoặc đọc lại raw frames quanh interval bất định,
trong giới hạn compute chung. Retrieval policy học trên dữ liệu đa dạng và
không giới hạn vào một speech window dài nhất.

## Các bẫy phải tránh khi tạo supervision

- Hai instance cùng hành động ở nhiều thời điểm không tự động là negatives;
  annotation phải xác định đúng instance được nhắc. Nhập nhằng cần multi-target
  hoặc unknown, không tạo nhãn chắc chắn giả.
- Speech narration có thể nói về hành động ở thời điểm khác hoặc không có trên
  video. Không lấy đồng thời audio/video làm positive semantic mặc định.
- Khi đổi đồng hồ chung phi(t)=a*t+b,a>0, các interval phải đổi theo cùng phi.
  Không mặc định mọi đáp án giữ nguyên: số giây tuyệt đối thay đổi theo a.
  Giám sát equivariance trên intervals có thể làm mà không cần đoán loại câu hỏi.
- Khi di chuyển riêng một modality, phải cập nhật grounded intervals đúng nguồn.
  Chỉ tự động tạo đáp án khi annotation cấu trúc đủ để suy ra đáp án; nếu không,
  chỉ dùng supervision grounding, không gán nhãn QA bằng heuristic câu chữ.
- Không coi PTS/nearest-pixel matches là event annotation. Đối chứng P đang chạy
  chỉ sửa quy tắc chọn frame và không cung cấp semantic oracle.

## Protocol trước khi chạy training

Chọn train/validation pool theo video và nguồn, loại trùng bằng video ID/hashes
với toàn bộ40 và held-out tương lai. Không dùng đúng/sai hay label distribution
trên40 làm loss hoặc chọn policy riêng từng task. Đánh giá modality contribution
bằng can thiệp thật và source attribution, không chỉ attention heatmap.

Validation cần cả QA và grounded interval/source chính xác; kiểm tra hai hướng
modality, nhiều sự kiện/lời nói, order/count/duration/causal và free-form. Không
đổi câu trả lời bằng rule theo task của benchmark. Thêm common-clock transform,
single-source shift, repeated events và missing-evidence/unknown controls.

So cùng model3B và compute budget: raw native, memory chưa học, latent-only,
interval-only, đầy đủ, target đảo, bỏ rawAV access và teacher-self-reconstruction
như L2 hiện tại. Nếu dùng LatentOmni làm baseline, cần cấu hình/training budget
tương xứng; không lấy điểm7B từ paper làm kết quả3B của chúng ta.

## Literature và novelty boundary

[LatentOmni](https://arxiv.org/html/2605.22012v1) đã có feature supervision,
latentAV và positional synchronization.
[Unsilencing Visual Latents](https://arxiv.org/html/2605.02735v1) đã có test-time latent optimization.
[AHA](https://arxiv.org/html/2512.24052v2) đã có temporal counterfactual hard
negatives cho preference alignment; paper dùng caption-derived supervision và
model7B/audio, không tự xác nhận jointAV3B. Không lặp các primitive này rồi
tuyên bố novelty rộng. Khoảng đóng góp cần kiểm chứng là liên kết instance/time/
source provenance với latent decisions, có counterfactual validation độc lập.

Chưa tải/chọn xong train pool, chưa huấn luyện adapter và chưa có kết quả QA
cho đề xuất này. Vì thế chưa được coi là phương pháp thành công hoặc đủ CVPR.

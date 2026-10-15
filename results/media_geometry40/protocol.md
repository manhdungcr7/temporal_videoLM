# Media representation geometry40 — trước chạy, 2026-10-15

Diagnostic ngắn sau các QA suites, dùng rotation0 của đúng40 recordings. Không
train, sửa input, ground-truth routing hay inference QA mới được chọn bằng nhãn.
Mục tiêu: kiểm tra claim cosine cao đồng nghĩa >90% static background.

Native forward một lần/recording. Tại blocks0/8/16/22/30/35, pool riêng V/A theo
2s chunk, lưu raw mean vectors. So cosine trong recording với giữa recordings
cùng modality; bộ40 có38 nguồn gốc nên loại cặp cùng source origin khỏi between
cosine, bổ sung leave-one-origin-out common mean centering, trọng số bằng nhau
giữa38 nguồn. Dùng LOO mean
chỉ để phân tích representation, không đưa vào câu trả lời hoặc học trên gold.

Nếu cosine cross-recording cũng cao, similarity nội clip không riêng biệt cho
static scene; có thể anisotropy/shared source representation. Nếu thấp thì vẫn
chưa chứng minh background/event decomposition. Các vector native audio/video
có thể chứa cross-source context; không gọi source-pure encoder features.

Cosine và centered cosine không đo event semantic matching/grounding accuracy,
không xác nhận model liên kết đúng audio/visual time. Không định lượng phần trăm
background từ cosine. Mọi kết luận cần đi kèm giới hạn này.

Media causal prefix không phụ thuộc câu hỏi phía sau, nên không lặp110 rotations
cho diagnostic này. Không gọi nó là QA evaluation40 hay thêm một phương pháp
đạt accuracy. Lưu features npz, JSON metadata và audit; một layerNorm phổ biến
không tự đảm bảo semantic alignment.

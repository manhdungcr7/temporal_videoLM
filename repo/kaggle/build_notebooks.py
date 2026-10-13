"""Build the 5 diagnostic Kaggle notebooks (one per free account) into repo/kaggle/.
Each notebook: Internet check -> shared env setup -> write sources -> 2-item smoke test (stops early if every item fails)
-> full job with a self-imposed deadline -> zip of results."""
import ast, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
sys.path.insert(0, str(PROJ))
import build_svhalluc_single_sample_prompt_lab as B

JOBS = HERE / 'jobs'
common = {'speech_gate.py': (PROJ / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
          'kaggle_rootcause.py': (PROJ / 'kaggle_rootcause.py').read_text(encoding='utf-8'),
          'job_utils.py': (JOBS / 'job_utils.py').read_text(encoding='utf-8'),
          'nb_runner.py': (JOBS / 'nb_runner.py').read_text(encoding='utf-8')}

NOTEBOOKS = [
    dict(n=1, job='j1_event_perception.py', zip='j1_event_perception', params={'PER_LABEL': 50, 'HOURS': 10.5}, hours='≈ 4–5 giờ',
         title='J1 — Mô hình KHÔNG THẤY hành động, hay THẤY mà không gắn được với thời điểm lời nói?',
         why="""Phân tích hidden state có sẵn (rootcause, n=35–51) cho thấy: vị trí lời nói (nửa đầu/nửa sau) đọc ra được 94–100%, nhưng nhãn TF chỉ 71% (= đúng bằng luật vị trí) và nhãn TA ở mức may rủi. Tức là bên trong mô hình **không có thông tin sự kiện nằm ở nửa nào**. Còn hai khả năng: (P) mô hình không nhìn thấy hành động, hoặc (B) nhìn thấy nhưng không gắn với thời điểm lời nói.""",
         what="""Với 50 câu mỗi nhãn của TA và TF: câu trả lời gốc + hidden state (token câu hỏi, video từng nửa, audio trong/ngoài lời nói); mô hình tự nghe "người nói nói hành động gì"; rồi hỏi riêng từng nửa video "có thấy hành động đó không". Whisper chỉ dùng để chấm (cận trên của việc nghe).""",
         decide="""**Tri giác ≥ 80% trong khi base ≈ 50%** → nguyên nhân là gắn kết thời gian (B), phương pháp phải sửa ở bước gắn kết. **Tri giác ≈ base** → nguyên nhân là tri giác (P), cần hướng khác."""),
    dict(n=2, job='j2_position_causal.py', zip='j2_position_causal', params={'PER_LABEL': 20, 'HOURS': 10.5}, hours='≈ 4 giờ',
         title='J2 — Mô hình dùng THỜI GIAN, hay chỉ dùng VỊ TRÍ và KHOẢNG IM LẶNG?',
         why="""CMTB đúng 70% khi lời nói ở nửa đầu nhưng chỉ 50% khi ở nửa sau (attention dồn về đầu video). Ngoài ra nửa không có lời nói của mọi clip là im lặng tuyệt đối, có thể là một lối tắt mà dữ liệu thật không có.""",
         what="""Cùng một clip, can thiệp nhân quả: **swap_av** (đổi chỗ hai nửa, giữ nguyên quan hệ, chỉ đổi vị trí tuyệt đối), **swap_v** (chỉ đổi video nên sự kiện đổi chỗ so với lời nói), **rope_rev** (đảo đồng hồ TMRoPE, giữ nguyên token), **noisy** (thay im lặng bằng nhiễu hồng 10 dB), và với CMTB: gate_audio trên audio sạch và audio nhiễu.""",
         decide="""swap_av giảm mạnh → **thiên kiến vị trí** là một nguyên nhân riêng. swap_v/rope_rev gần như không đổi câu trả lời → **mô hình không dùng thời gian**. noisy làm thay đổi kết quả → kết quả cũ phụ thuộc lối tắt im lặng, phương pháp phải chịu được nhiễu."""),
    dict(n=3, job='j3_prior_lens.py', zip='j3_prior_lens', params={'PER_LABEL': 30, 'HOURS': 10.5}, hours='≈ 3–4 giờ',
         title='J3 — Thiên kiến "No" (TA) và "không bao giờ Past" (TF) đến từ đâu, ở lớp nào?',
         why="""TA: đáp án khi không có bằng chứng luôn là "No", và 85% lỗi TA trùng với đáp án đó. TF: recall của Past ≈ 0. Chưa biết đó là prior của câu chữ, của việc có media, hay bằng chứng đúng bị đè ở các lớp cuối.""",
         what="""base / prior (video đen + im lặng) / chỉ văn bản / chỉ hình; **logit lens** từng lớp cho base và prior; TA hỏi theo cực ngược ("có xảy ra ở thời điểm KHÁC không?"); TF dùng nhãn trung tính ("sớm hơn / trong lúc / muộn hơn lời nói"). Offline: hiệu chỉnh tự động base − prior.""",
         decide="""Câu hỏi đảo cực mà vẫn trả lời "No" → **thiên kiến Yes/No**, không phải phán đoán thời gian. Logit lens đúng ở giữa mạng rồi sai ở cuối → **prior đè bằng chứng muộn**. Hiệu chỉnh base − prior tăng mạnh → có một thành phần sửa tự động, rẻ."""),
    dict(n=4, job='j4_postdiction.py', zip='j4_postdiction', params={'PER_LABEL': 20, 'HOURS': 10.5}, hours='≈ 6 giờ',
         title='J4 — Thứ tự nhân quả: mô hình có "hiểu lại" cảnh đã thấy sau khi nghe lời nói không? (postdiction)',
         why="""Mô hình đọc media dưới mask nhân quả: token video của một sự kiện xảy ra TRƯỚC lời nói không bao giờ được "nghe" lời nói đó (nó đến sau trong chuỗi). Con người thì hiểu lại cảnh đã thấy khi nghe thông tin đến sau (postdiction). Dữ liệu hiện có khớp: lời nói ở nửa đầu thì đúng hơn hẳn (CMTB 70 vs 50, TA 68 vs 56, TF 48 vs 11). Nhưng "attention dồn về đầu video" cũng dự đoán y như vậy, nên cần tách hai giả thuyết. Binding lab trước chỉ thử liên kết CỤC BỘ (cùng đoạn 2 s, ±1 đoạn) và không có tác dụng; liên kết trải cả clip chưa từng được thử.""",
         what="""Cùng câu hỏi, chỉ đổi "token nào được đọc token nào" bên trong vùng media (giữ nguyên vị trí TMRoPE, không đổi hàng văn bản): **base**; **v2a_all** (video được đọc mọi audio, kể cả audio đến sau); **v2a_early** (như trên nhưng chỉ ở lớp 1–14); **media_bidir** (toàn bộ media hai chiều). Kèm hidden state theo từng đoạn 2 s của video và audio ở base và v2a_all để probe offline: token video của đoạn có sự kiện có "biết" mình là sự kiện được nói tới không.""",
         decide="""v2a_* tăng chủ yếu ở các câu mà sự kiện xảy ra TRƯỚC lời nói (TF Past, TA/CMTB lời nói ở nửa sau) và probe đoạn sự kiện tăng theo → **thiếu postdiction là nguyên nhân gốc** (một hướng phương pháp mới, có nền tảng nhận thức rõ). Không đổi → nguyên nhân nằm ở attention dồn đầu/khâu đọc ra, không phải ở thứ tự nhân quả."""),
    dict(n=5, job='j5_internal_localization.py', zip='j5_internal_localization', params={'PER_LABEL': 15, 'HOURS': 10.5}, hours='≈ 4 giờ',
         title='J5 — Mô hình có TỰ BIẾT lời nói ở đâu không? (nền tảng cho phương pháp tự động)',
         why="""Định tuyến câu hỏi tới video ở thời điểm lời nói (+ chặn audio) sửa được CMTB (51 → 96), nhưng cửa sổ thời gian lấy từ một bộ phát hiện năng lượng ngoài, và bộ này chỉ chạy được vì phần không lời là im lặng tuyệt đối. Phương pháp **tự động hoàn toàn và tổng quát** cần lấy cửa sổ từ chính mô hình và phải chịu được nhiễu thật.""",
         what="""Attention (eager) của mọi lớp × head, từ token cuối và từ token câu hỏi, cộng theo từng đoạn 2 giây của audio và video; trên audio sạch và audio nhiễu. Kèm mức hoạt động lời nói theo đoạn, base, gate_audio với cửa sổ sạch và với cửa sổ phát hiện lại trên audio nhiễu.""",
         decide="""Có head mà attention lên audio khớp lời nói (r > 0,5) **kể cả khi có nhiễu** → lấy cửa sổ từ bên trong mô hình được, pipeline tự động khả thi. Không có → cần VAD chuẩn làm thành phần cố định. gate_audio với cửa sổ phát hiện trên audio nhiễu cho biết lợi ích có còn trong điều kiện thực tế không."""),
]

J6_WHY = """Đề xuất của nhóm: sau phần media, đặt các "câu tóm tắt" (summary query); câu thứ i CHỈ được nhìn token media của đoạn i, các câu không nhìn thấy nhau, rồi câu hỏi đọc các câu tóm tắt này (và có thể được tăng attention vào chúng ở các lớp cuối). Toàn bộ làm bằng attention mask 4-D + vị trí TMRoPE tường minh, một lượt forward, không sinh chữ, không huấn luyện. Gần nhất là Video-XL (CVPR 2025) và Gist tokens nhưng các công trình đó phải huấn luyện; bản không huấn luyện cho mô hình omni chưa thấy ai làm. Dữ liệu của mình gợi ý 2 điều chỉnh: tách nguồn (câu "chỉ nhìn" và câu "chỉ nghe") vì mô hình hay coi điều NGHE là điều THẤY, và đặt tên đoạn theo quan hệ với lời nói thay vì số giây vì mô hình không so sánh được số giây."""
J6_DECIDE = """So với base và với STMR (~80% trên pilot): bản tóm tắt ẩn một lượt forward đạt ngang hoặc hơn → phát triển tiếp thành phương pháp chính. Bản "tách nguồn" hơn bản "trộn hình+tiếng" → xác nhận vai trò của giám sát nguồn. Bản "bottleneck" (câu hỏi chỉ đọc tóm tắt) sụp → tóm tắt ẩn không huấn luyện không mang đủ bằng chứng, giữ hướng bằng chứng dạng văn bản."""

J7_WHY = """Đề xuất của nhóm, làm đúng theo đặc tả: mỗi đoạn 2 s (đúng 1 chunk TMRoPE của mô hình omni) có một câu "Summarize the information from the video and audio."; câu đó CHỈ nhìn được hình + tiếng của đúng 2 s đó; các câu không nhìn thấy nhau; câu hỏi và đáp án nhìn được các câu này VÀ toàn bộ audio/video (để không hiểu sai). Nếu attention bình thường không đủ: tăng attention của câu hỏi/đáp án vào các câu tóm tắt ở lớp giữa hoặc lớp cuối. Nếu vẫn chưa đủ: dùng chính attention của câu hỏi vào các câu tóm tắt để biết mô hình cần đoạn nào, rồi tăng attention vào nguyên đoạn 2 s media đó và đẩy các đoạn còn lại về gần như không đóng góp (tự động, mô hình tự chọn). Có thêm đối chứng "không cô lập" để chứng minh chính việc cô lập có tác dụng (khác Video-XL, vốn nén video dài và phải huấn luyện)."""
J7_WHAT = """Mỗi câu hỏi được chấm ở 7 điều kiện: base; cq (câu tóm tắt 2 s, attention bình thường); cq_noiso (đối chứng không cô lập); cq_boost_mid (tăng attention lớp 12–23); cq_boost_late (lớp 24–35); cq_select_soft và cq_select_sharp (mô hình tự chọn đoạn 2 s qua attention, rồi tăng đoạn đó và giảm các đoạn khác ở lớp 13–27). Lưu thêm độ liên quan mô hình gán cho từng đoạn và mức có tiếng nói của từng đoạn, để xem mô hình có tự chọn đúng đoạn có lời nói không."""
J7_DECIDE = """cq > base và cq > cq_noiso → cô lập theo đoạn có tác dụng không cần huấn luyện. boost tăng thêm → chọn lớp tốt nhất. select gần mức định tuyến oracle (CMTB ~96–100%) và đoạn mô hình chọn trùng đoạn có lời nói → có bộ định tuyến tự động hoàn toàn từ chính mô hình. Tất cả đều không hơn base → cần huấn luyện (phương án rẻ: chỉ học embedding của câu tóm tắt, mô hình đóng băng)."""

NOTEBOOKS += [
    dict(n=6, job='j7_chunk_queries.py', zip='j7_cq_pilot', params={'PER_LABEL': 8, 'SEED': 909, 'HOURS': 10.5}, hours='≈ 3–4 giờ',
         title='J7a — Câu tóm tắt cho từng đoạn 2 s + mô hình tự chọn đoạn (không huấn luyện): trên bộ câu pilot',
         why=J7_WHY, what=J7_WHAT + " Bộ câu: giống pilot STMR/J6 (seed 909, 8 câu mỗi nhãn) để so sánh theo cặp.", decide=J7_DECIDE),
    dict(n=7, job='j7_chunk_queries.py', zip='j7_cq_heldout', params={'PER_LABEL': 12, 'SEED': 1234, 'HOURS': 10.5}, hours='≈ 5–6 giờ',
         title='J7b — Câu tóm tắt cho từng đoạn 2 s + mô hình tự chọn đoạn: trên tập held-out',
         why=J7_WHY, what=J7_WHAT + " Bộ câu: held-out (seed 1234, 12 câu mỗi nhãn), không trùng các câu đã dùng để thiết kế.", decide=J7_DECIDE),
]

J8_WHY = """Phương pháp chính, can thiệp hoàn toàn bên trong mô hình (không ASR, không văn bản trung gian). Mục tiêu: với BẤT KỲ câu hỏi nào, mô hình tự tập trung vào đoạn video cần thiết nhất để trả lời — cả khung hình lẫn audio — còn các đoạn khác gần như không đóng góp. Pha 1: một lượt prefill, đo attention của câu hỏi lên token VIDEO và AUDIO của từng đoạn 2 s (trung bình mọi head lớp 12–35, không chọn head bằng nhãn), rồi chia cho cùng đại lượng của một câu hỏi trung tính ("What is happening in the video?") để khử độ nổi bật sẵn có (PMI): rel(c) = p(c|q)·(p(c|q)/p(c|trung tính))². Pha 2: ở lớp 13–27, câu hỏi/đáp án được cộng bias λ·log(rel(c)/max rel) vào token của đoạn c."""
J8_WHAT = """Mỗi câu: base; oracle_route (cửa sổ cứng từ năng lượng audio + chặn audio, cận trên); acar_a_l4_ab và acar_a_pmi_l4_ab (mốc chỉ từ audio, thô / PMI, phân bổ lại video, chặn audio); qtar_av_pmi_l1 / l4 (phương pháp tổng quát: độ liên quan từ cả video + audio có PMI, phân bổ lại CẢ video lẫn audio, λ = 1 / 4); qtar_av_pmi_l4_sg (như trên + các lớp trên đọc cảnh đã chọn khi tắt tiếng). Với TA/TF thêm "look" (định vị sự kiện được nói tới trong vùng trước / trong / sau mốc). Lưu mọi profile độ liên quan để phân tích từng câu sai."""
J8_DECIDE = """qtar_* tiến gần oracle_route (CMTB ~96%) mà không cần biết trước đoạn nào → mô hình tự chọn đúng đoạn cần thiết, phương pháp tổng quát và tự động. PMI tốt hơn bản thô → khử độ nổi bật là thành phần cần thiết. Bản tổng quát (av) bằng bản neo audio → không cần giả định mốc là audio. Phân tích từng câu sai: chọn sai đoạn (độ liên quan), chọn đúng nhưng đọc sai, hay nhầm nguồn."""

NOTEBOOKS += [
    dict(n=8, job='j8_acar.py', zip='j8_acar_pilot', params={'PER_LABEL': 8, 'SEED': 909, 'HOURS': 10.5}, hours='≈ 2,5–3 giờ',
         title='J8a — Mô hình tự chọn đoạn video (hình + tiếng) cần cho câu hỏi, rồi phân bổ lại attention (bộ câu pilot)',
         why=J8_WHY, what=J8_WHAT + " Bộ câu: pilot (seed 909, 8 câu mỗi nhãn), so sánh theo cặp với J7/STMR.", decide=J8_DECIDE),
    dict(n=9, job='j8_acar.py', zip='j8_acar_heldout', params={'PER_LABEL': 12, 'SEED': 1234, 'HOURS': 10.5}, hours='≈ 4–5 giờ',
         title='J8b — Mô hình tự chọn đoạn video cần cho câu hỏi: tập held-out',
         why=J8_WHY, what=J8_WHAT + " Bộ câu: held-out (seed 1234, 12 câu mỗi nhãn).", decide=J8_DECIDE),
    dict(n=10, job='j9_nonspeech_anchor.py', zip='j9_nonspeech', params={'PER_LABEL': 8, 'SEED': 77, 'HOURS': 10.5}, hours='≈ 1–1,5 giờ',
         title='J9 — Mốc từ attention có hoạt động với âm thanh KHÔNG phải lời nói, và có đi theo câu hỏi không?',
         why="""Âm thanh làm mốc có thể là tiếng chó sủa, tiếng chuông… chứ không chỉ lời nói. Phương pháp chỉ tổng quát nếu attention của câu hỏi lên audio tìm đúng âm thanh MÀ CÂU HỎI NHẮC TỚI.""",
         what="""Chèn một âm thanh tổng hợp (chuông / tiếng bíp / vỗ tay / huýt sáo, 2,5 s, to bằng lời nói, thời điểm biết chính xác) vào nửa KHÔNG có lời nói của clip. Mỗi clip hỏi 3 câu: về âm thanh đó, về lời nói, và một câu trung tính; đo attention của câu hỏi lên từng đoạn audio 2 s (giống pha 1 của ACAR).""",
         decide="""Hỏi về âm thanh → mốc rơi vào đoạn âm thanh; hỏi về lời nói → rơi vào đoạn lời nói; câu trung tính không thiên về bên nào → mốc đi theo câu hỏi, dùng được cho mọi loại âm thanh. Nếu mốc luôn rơi vào lời nói bất kể câu hỏi → chỉ là "phát hiện lời nói", phải sửa pha 1."""),
]

J10_WHY = """Phương pháp tự động hoàn chỉnh, can thiệp bên trong mô hình. Kết quả J8: mô hình tự chọn đúng thời điểm (CMTB 52,9 → 94,1 khi đọc hình với tiếng tắt), nhưng nếu vẫn giữ audio thì mô hình đọc NỘI DUNG lời nói và chọn hành động được nói tới (nhầm nguồn, chỉ 58,8). Debug 1 mẫu: chính mô hình (chỉ đọc câu hỏi) quyết định được cần "nhìn / nghe / cả hai", và khi tắt kênh không cần thiết ở lớp 13–27 thì đáp án đúng (0,92, hơn cả oracle 0,86). J10 kiểm tra toàn bộ pipeline này trên nhiều câu, kèm đối chứng ép kênh để tách lỗi do quyết định kênh và lỗi do cơ chế."""
J10_WHAT = """Mỗi câu: base; oracle_route; auto (độ liên quan hình + tiếng có PMI + kênh đích do mô hình tự quyết: nhìn → tắt audio, nghe → tắt video, cả hai → nghe ở lớp 13–20 rồi nhìn khi tắt tiếng ở lớp 21–27); auto_a (độ liên quan chỉ từ audio); force_v (ép "nhìn"); force_both (ép "cả hai"); no_gate (chỉ phân bổ lại thời gian). Lưu quyết định kênh và profile độ liên quan để phân tích từng câu sai."""
J10_DECIDE = """auto ≥ 90% ở CMTB và không làm hại TA/TF → phương pháp tổng quát dùng được. So auto với force_v / force_both: nếu force_* tốt hơn ở task nào thì lỗi nằm ở quyết định kênh (sửa câu tự hỏi), ngược lại lỗi ở cơ chế đọc."""

NOTEBOOKS += [
    dict(n=11, job='j10_auto.py', extra=['j8_acar.py'], zip='j10_auto_pilot', params={'PER_LABEL': 8, 'SEED': 909, 'HOURS': 10.5}, hours='≈ 3–4 giờ',
         title='J10a — Phương pháp tự động hoàn chỉnh: tự chọn thời điểm + tự quyết kênh cần đọc (bộ câu pilot)',
         why=J10_WHY, what=J10_WHAT + " Bộ câu pilot (seed 909), so sánh theo cặp với J8.", decide=J10_DECIDE),
    dict(n=12, job='j10_auto.py', extra=['j8_acar.py'], zip='j10_auto_heldout', params={'PER_LABEL': 12, 'SEED': 1234, 'HOURS': 10.5}, hours='≈ 5–6 giờ',
         title='J10b — Phương pháp tự động hoàn chỉnh trên tập held-out',
         why=J10_WHY, what=J10_WHAT + " Tập held-out (seed 1234).", decide=J10_DECIDE),
]

NOTEBOOKS += [
    dict(n=13, job='j11_full.py', extra=['j8_acar.py'], zip='j11_full_pilot', params={'PER_LABEL': 8, 'SEED': 909, 'HOURS': 10.5}, hours='≈ 3,5–4 giờ',
         title='J11 — Phương pháp hoàn chỉnh lấy cảm hứng từ cách con người hiểu video (bộ câu pilot)',
         why="""Ghép các bước đã được chứng minh riêng lẻ: (1) ĐỊNH HƯỚNG — mô hình tự tìm lúc có âm thanh/sự kiện được hỏi qua attention lên token audio, khử độ nổi bật bằng PMI (J8/J9: đúng đoạn 97–100%); (2) NÓI THẦM — mô hình tự diễn đạt điều nó nghe thành một cụm từ (trí nhớ làm việc bằng lời; debug: độ chênh giữa các vùng tăng từ ~0,5 lên 2–18 logit); (3) NHÌN KHI TẮT TIẾNG — với mỗi vùng trước/trong/sau mốc, hỏi "có thấy … không" khi câu hỏi bị tắt audio và chỉ nhìn vùng đó ở lớp 13–27 (định tuyến attention, ngang cắt clip; debug 5/5); (4) TRÍ NHỚ SỰ KIỆN CÓ NGUỒN — dòng thời gian tách nội dung lời nói khỏi điều nhìn thấy (sửa lỗi nhầm nguồn); (5) SUY NGHĨ CÓ CHỦ Ý — suy luận từng bước, đọc đáp án từ lời suy luận (TF: một bước 27–46%, từng bước 86%).""",
         what="""Mỗi câu: base; acar (một lượt, mốc tự động + tắt audio, tốt nhất của J8); plain (pipeline, trả lời một bước); cot (pipeline, suy luận từng bước, 3 thứ tự lựa chọn bỏ phiếu). Lưu mọi bước trung gian (mốc, câu nói thầm, điểm nhìn từng vùng, dòng thời gian, lời suy luận) để phân tích từng câu sai.""",
         decide="""cot ≥ 90% ở cả TA, TF, CMTB → phương pháp đạt mục tiêu trên pilot, chuyển sang held-out một lần. Nếu không: phân tích từng câu sai theo bước (mốc sai / nói thầm sai / nhìn sai / suy luận sai) để sửa đúng bước."""),
]

net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''
setup = B.setup.replace('assert torch.cuda.device_count()>=2;', 'assert torch.cuda.device_count()>=1;')

run_tpl = '''# CELL {k} — {label}
import subprocess, json as _json
def run_job(out_dir, params):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(ROOT/'nb_runner.py'), str(ROOT), str(out_dir), str(ROOT/JOB), _json.dumps(params)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                         env={{**os.environ, 'PYTHONUNBUFFERED': '1'}})
    for line in p.stdout:
        low = line.lower()
        if ('warn' in low or 'loading checkpoint' in low or 'fetching' in low or 'it/s]' in low) and 'traceback' not in low: continue
        print(line, end='', flush=True)
    return p.wait()
{body}
'''
smoke_body = '''SMOKE = OUT/'smoke'
code = run_job(SMOKE, {**PARAMS, 'LIMIT': 4, 'HOURS': 1.0})
rows = [_json.loads(l) for f in SMOKE.glob('*.jsonl') for l in f.read_text().splitlines() if l.strip()]
good = [r for r in rows if 'error' not in r]
real_err = [r for r in rows if 'error' in r and 'long' not in r['error'] and 'no room' not in r['error']]   # 'long' = clip over the length limit, not a bug
print('\\nSMOKE: exit', code, '| items', len(rows), '| ok', len(good), '| real errors', len(real_err))
for r in rows:
    if 'error' in r: print('  error:', r['error'][:200])
if code != 0 or real_err:
    raise RuntimeError('Smoke test thất bại: dừng để không phí GPU. Gửi log của cell này cho Claude.')
print('Smoke test OK -> chạy job đầy đủ.' if good else 'Cả 4 clip thử đều quá dài (không phải lỗi) -> vẫn chạy job đầy đủ.')'''
full_body = '''code = run_job(OUT/'full', PARAMS)
print('exit code', code)'''
export_cell = '''# CELL 6 — nén kết quả (jsonl + npz + log; không kèm video)
ZIP = Path(f'/kaggle/working/{ZIPNAME}.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in (OUT/'full').rglob('*'):
        if f.is_file() and f.suffix in ('.jsonl', '.json', '.npz', '.log') and 'clips' not in f.parts:
            z.write(f, str(f.relative_to(OUT/'full')))
    for f in (OUT/'smoke').glob('*.jsonl'): z.write(f, 'smoke/' + f.name)
print(ZIP, round(ZIP.stat().st_size / 1e6, 1), 'MB')
display(FileLink(str(ZIP)))
'''

for spec in NOTEBOOKS:
    files = dict(common); files[spec['job']] = (JOBS / spec['job']).read_text(encoding='utf-8')
    for x in spec.get('extra', []): files[x] = (JOBS / x).read_text(encoding='utf-8')
    for s in files.values(): ast.parse(s)
    intro = f"""# {spec['title']}

**Tài khoản Kaggle #{spec['n']}.** GPU **T4 ×2**, bật **Internet** (Session options), rồi **Save Version → Save & Run All**. Thời gian {spec['hours']}. Khi xong, tải **`{spec['zip']}.zip`** ở cell cuối và gửi cho Claude.

Notebook tự chạy thử 4 mẫu trước; nếu có lỗi thật (không tính clip quá dài), nó dừng ngay để không phí GPU. Job tự dừng an toàn trước hạn 12 giờ của Kaggle và giữ những gì đã chạy xong.

### Vì sao chạy
{spec['why']}

### Đo gì
{spec['what']}

### Đọc kết quả thế nào (tiêu chí định trước)
{spec['decide']}

Không có nhãn nào được đưa vào mô hình; nhãn chỉ dùng để chấm điểm."""
    write_cell = (f"# CELL 2 — ghi mã nguồn\nJOB = {spec['job']!r}\nZIPNAME = {spec['zip']!r}\nPARAMS = {spec['params']!r}\n"
                  f"SRC = {files!r}\n"
                  "for _n, _s in SRC.items(): (ROOT/_n).write_text(_s, encoding='utf-8')\nprint('Đã ghi:', list(SRC), '| params', PARAMS)\n")
    cells = [B.cell('markdown', intro), B.cell('code', net_cell), B.cell('code', setup), B.cell('code', write_cell),
             B.cell('code', run_tpl.format(k=3, label='chạy thử 2 mẫu (dừng nếu lỗi)', body=smoke_body)),
             B.cell('code', run_tpl.format(k=4, label='chạy job đầy đủ (tự dừng trước hạn)', body=full_body)),
             B.cell('code', '# CELL 5 — tóm tắt\nfor f in (OUT/"full").glob("*.jsonl"):\n    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]\n    print(f.name, "| items", len(rows), "| errors", sum("error" in r for r in rows))\n'),
             B.cell('code', export_cell)]
    nb = {'cells': cells, 'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
    target = HERE / f"nb{spec['n']}_{spec['zip']}.ipynb"
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(target.name, round(target.stat().st_size / 1e3), 'KB')

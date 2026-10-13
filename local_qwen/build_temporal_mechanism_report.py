"""Summarize observed diagnostics with explicit scope and paired repair counts."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    clock = read('temporal_clock_counterfactual_2310.json')['conditions']
    wording = read('temporal_clock_wording_probes_2310.json')['conditions']
    recall = read('temporal_content_recall_2310.json')['conditions']
    repair = read('temporal_activity_grounding_prototype_2310.json')['conditions']
    choice_file = read('temporal_choice_position_controls_2310.json')
    choice = choice_file['conditions']
    for rec in choice.values():
        verbal = re.match(r'^\s*(Yes|No)\b', rec['raw_output'], re.I)
        if verbal:
            mapping = rec['choice_meanings']
            corrected = next(k for k, v in mapping.items() if v == verbal.group(1).title())
            if corrected != rec['parsed_choice']:
                rec['old_generic_parser_choice'] = rec['parsed_choice']
            rec['parsed_choice'] = corrected
            rec['parsed_meaning'] = mapping[corrected]
            rec['parser_note'] = 'Derived correction: map leading verbal answer through actual displayed choices; raw generation unchanged.'
    (ROOT / 'temporal_choice_position_controls_2310.json').write_text(
        json.dumps(choice_file, ensure_ascii=False, indent=2), encoding='utf-8')
    validation = read('temporal_activity_grounding_validation_dev3.json')['samples']
    structured = read('temporal_structured_event_match_probe.json')['cases']
    confirmation = read('temporal_structured_pipeline_confirmation_ta.json')
    dense = read('temporal_dense_window_1961.json')
    if len(validation) != 3 or len(choice) != 2:
        raise ValueError('Validation/control runs have not completed')
    base_layout = clock['padded_early']['metadata']['input_ids_sha256']
    if any(clock[name]['metadata']['input_ids_sha256'] != base_layout
           for name in ('audio_late', 'both_late')):
        raise ValueError('Clock controls did not preserve token layout')
    assert clock['padded_early']['metadata']['sampled_pixels_sha256'] == clock['audio_late']['metadata']['sampled_pixels_sha256']
    assert clock['audio_late']['metadata']['audio_sha256'] == clock['both_late']['metadata']['audio_sha256']

    lines = ['# Debug temporal của Qwen2.5-Omni-3B: bằng chứng và phép thử cách sửa',
        '', 'Ngày 2026-10-11. Checkpoint cố định revision f75b40e3da2003cdd6e1829b1f420ca70797c34e; NF4 4-bit, FP16 compute; RTX 3050 Laptop 4 GB. Các kết quả này không được suy rộng thành nguyên nhân của mọi small video language model.',
        '', '## 1. Những điều đã kiểm soát', '',
        '- Dùng các clip đã chuẩn hóa CFR và kiểm tra thứ tự khung hình. Lỗi đọc Decord của pipeline cũ là vấn đề tiền xử lý riêng, không phải bằng chứng về kiến trúc model.',
        '- Câu hỏi gốc, waveform, frame, số token, logits token đầu, output, token IDs và lý do dừng đều được lưu. Baseline chỉ dùng clip và câu hỏi.',
        '- Đối chứng thời gian dùng cùng video 14 giây: thêm đoạn giữ nguyên frame cuối và im lặng; dịch nguyên audio 7 giây, hoặc dịch cả chuỗi frame và audio. Đoạn audio bị cuộn từ cuối về đầu đã được kiểm tra gần như hoàn toàn im lặng; không cắt mất câu nói.',
        '- Trong ba đối chứng thời gian chính, input_ids và vị trí token giữ nguyên. Frame của early và audio_late giống hệt; waveform của audio_late và both_late giống hệt. Dịch cả video tạo một điểm nối cảnh, nên đây là đối chứng chẩn đoán với giới hạn đó.',
        '- Các nhãn cho phiên bản sửa là kỳ vọng chẩn đoán từ nội dung video; không phải nhãn SVHalluc mới. Không cộng các phiên bản của một nguồn thành nhiều video test độc lập.',
        '', '## 2. Nội dung nhận ra được, thứ tự vẫn có thể sai', '',
        'Trên svh_2310, khung hình cho thấy: thái hành → cắt khoai tây → cắt cà chua. Khi chỉ xem hình trên toàn clip, model trả lời:', '',
        '> ' + recall['whole_clip_visual_only']['raw_output'], '',
        'Model nhắc đúng ba đối tượng nhưng đảo thứ tự khoai tây và cà chua. Đây là lỗi hành vi về thứ tự; chưa chứng minh encoder đã mất thông tin thời gian.', '',
        'Khi chỉ nghe audio, model trả lời:', '',
        '> ' + recall['whole_clip_audio_only']['raw_output'], '',
        'Nội dung lời nói được nhận ra. Thành công ở hai câu hỏi riêng này không chứng minh lượt AV ban đầu dùng đúng các biểu diễn đó.',
        '', '## 3. Thay thời gian không đủ đổi quyết định theo quan hệ đúng', '',
        'Margin dưới đây = logit của lựa chọn Yes trừ logit của lựa chọn No ở token đầu. Dương nghiêng về Yes. Đây là điểm chẩn đoán, không phải xác suất correctness được hiệu chuẩn.', '',
        '| Đầu vào | Kỳ vọng chẩn đoán | Đáp án | Margin Yes−No |',
        '|---|---|---|---:|']
    for name, rec in clock.items():
        mapping = {'A': 'No', 'B': 'Yes'} if name.endswith('_swapped') else {'A': 'Yes', 'B': 'No'}
        lines.append(f"| {name} | {mapping[rec['expected_choice']]} | {mapping.get(rec['parsed_choice'], 'invalid')} | {rec['yes_minus_no_letter_logit']:.4f} |")
    lines += ['', 'Quét thêm offset audio, giữ nguyên frame và số token:', '',
              '| Offset audio | Margin Yes−No | Đáp án |', '|---:|---:|---|']
    points = [(0, clock['padded_early']),
              *[(s, recall[f'clock_sweep_{s}s']) for s in (2, 4, 6)],
              (7, clock['audio_late'])]
    for seconds, rec in points:
        lines.append(f"| {seconds} s | {rec['yes_minus_no_letter_logit']:.4f} | {rec['parsed_choice']} |")
    lines += ['', 'Các offset giữa không được gán hard label vì có thể nằm sát ranh hành động. Điểm có thay đổi nên không thể nói model hoàn toàn không sử dụng thời gian; lựa chọn vẫn B trong cả sweep.',
        '', '## 4. Cách hỏi và lựa chọn gây thay đổi lớn', '',
        '| Câu hỏi | Early: đồng thời | Late: lệch |', '|---|---|---|']
    for title, early, late in [('Câu hỏi gốc', clock['padded_early'], clock['audio_late']),
        ('Hỏi rõ “in this particular clip”', wording['instance_early'], wording['instance_late']),
        ('Bỏ A/B, trả Yes/No trực tiếp', wording['direct_early'], wording['direct_late'])]:
        lines.append(f"| {title} | {early['raw_output']} | {late['raw_output']} |")
    lines += ['', 'Cách hỏi rõ clip hoặc bỏ ký hiệu A/B đổi sang Yes trên cả hai trường hợp. Nó sửa ca positive và làm sai ca negative. Vì vậy, một đáp án đổi đúng sau sửa prompt chưa chứng minh temporal grounding được sửa.',
        '', 'Đổi cả thứ tự hiển thị chữ cái để phân biệt ảnh hưởng chữ B và vị trí cuối:', '',
        '| Các lựa chọn | Model chọn | Ý nghĩa model chọn | Margin Yes−No |', '|---|---|---|---:|']
    for rec in choice.values():
        suffix = rec['question'].split('?')[-1].strip()
        lines.append(f"| {suffix} | {rec['parsed_choice']} | {rec['parsed_meaning']} | {rec['yes_minus_no_letter_logit']:.4f} |")
    lines += ['', 'Những thay đổi này chứng minh đầu ra nhạy với cách đặt lựa chọn trên cùng media. Không tự kết luận model luôn thích một chữ cái trên mọi task, hoặc head cụ thể nào gây thiên lệch.',
        '', 'Lưu ý parser: ở lựa chọn (B) Yes (A) No, output bắt đầu bằng "No". Đáp án phải được ánh xạ thành A theo lựa chọn thực tế, không dùng fallback No→B của câu hỏi gốc. File JSON đã lưu cả lựa chọn parser cũ và hiệu chỉnh từ nguyên văn; không chạy lại model. Margin A/B ở token đầu cũng không tương đương đáp án sinh thực tế khi model bắt đầu bằng Yes/No.',
        '', '## 5. Phép thử sửa: bằng chứng lấy theo thời điểm tự động', '',
        'Dùng đúng một Qwen, không huấn luyện: phát hiện vùng audio hoạt động bằng RMS → Qwen audio-only sinh transcript → Qwen vision-only mô tả các frame trong vùng đó → Qwen text-only so hành động được nói và hành động được nhìn để trả lời.', '',
        'RMS là detector hoạt động âm thanh trong clip narration sạch, không phải VAD tổng quát. Method không nhận Whisper, nhãn, tên đối tượng, khoảng sự kiện do người gán hay câu trả lời chuẩn. Transcript và caption do Qwen tạo có thể sai. Đây là ba lượt suy luận và thay đổi đầu vào theo thời gian, không phải prompting thuần.', '',
        '| Phiên bản cùng nguồn 2310 | Baseline | Cách sửa | Kỳ vọng |', '|---|---|---|---|']
    for name, rec in repair.items():
        lines.append(f"| {name} | {clock[name]['parsed_choice']} | {rec['final']['parsed_choice']} | {rec['expected_choice']} |")
    lines += ['', 'Ví dụ output cuối trên mẫu gốc:', '', '> ' + repair['original_10s']['final']['raw_output'], '',
        'Bằng chứng đã được định vị: audio nói về khoai tây và dicing; visual summary mô tả cắt khoai tây thành miếng nhỏ. Lượt so sánh vẫn trả No, có lúc tập trung vào “using a potato” thay vì “dice it”. Điều này cho thấy chỉ định vị đúng chưa đủ; bước chọn/ghép hành động có thể tiếp tục thất bại.', '',
        'Cần giữ khả năng nhập nhằng độ hạt: ảnh 3.5–4.5 s chủ yếu là cắt lát, 5.5–6 s cho thấy cắt tiếp thành khối; từ “dice” được Whisper dự đoán gần 4.12 s. Dicing toàn quy trình và thời điểm tạo từng khối là hai mức mô tả khác nhau. Mốc Whisper chưa được người nghe xác minh; không kết luận nhãn benchmark sai. Ảnh chi tiết: temporal_debug_media/2310_potato_microsteps.jpg.',
        '', '## 6. Kiểm tra cố định trên ba nguồn development khác', '',
        'Không sửa prompt hay detector theo kết quả từng nguồn. Chọn trước 1842 (Yes), 1668 (No), 1485 (No), dùng cùng method. Đây là development, không phải test độc lập sau nghiên cứu.', '',
        '| Mẫu | Gold | Baseline | Cách sửa |', '|---:|---|---|---|']
    pairs = [(2310, 'A', clock['original_10s']['parsed_choice'], repair['original_10s']['final']['parsed_choice'])]
    for sample in validation:
        pair = (sample['idx'], sample['gold'], sample['baseline']['parsed_choice'], sample['repair']['final']['parsed_choice'])
        pairs.append(pair)
        lines.append(f'| {pair[0]} | {pair[1]} | {pair[2]} | {pair[3]} |')
    b_ok = sum(b == g for _, g, b, p in pairs)
    p_ok = sum(p == g for _, g, b, p in pairs)
    w_c = sum(b != g and p == g for _, g, b, p in pairs)
    c_w = sum(b == g and p != g for _, g, b, p in pairs)
    lines += ['', f'Tính bốn clip gốc độc lập (gồm 2310), không tính các edit: baseline {b_ok}/4; method {p_ok}/4; wrong→correct {w_c}; correct→wrong {c_w}; invalid {sum(p is None for _, g, b, p in pairs)}.']
    for gold, label in [('A', 'Yes'), ('B', 'No')]:
        group = [x for x in pairs if x[1] == gold]
        lines.append(f'Recall {label}: baseline {sum(x[2] == gold for x in group)}/{len(group)}; method {sum(x[3] == gold for x in group)}/{len(group)}.')
    normalization = read('temporal_narrated_event_normalization_probe.json')
    assert len(normalization['cases']) == 7
    norm_pairs = []
    lines += ['', '### 6.1. Chẩn đoán sửa tiếp: chuẩn hóa hành động được nói đến', '',
        'Giữ nguyên transcript và caption đã sinh, thêm một lượt text-only để rút ra thao tác chính cùng đối tượng, rồi so với caption ở đúng khoảng audio. Prompt chung giải quyết đại từ và tách việc lấy dụng cụ/giới thiệu nguyên liệu khỏi thao tác chính. Không đưa gold hay tên vật do người gán vào model.', '',
        'Đây là phép chẩn đoán trên bằng chứng đã lưu của development, không phải test mới hoặc lần chạy end-to-end mới. Nó kiểm tra giả thuyết lỗi chọn sự kiện; thêm compute và thay đổi biểu diễn đều có thể góp phần.', '',
        '| Ca | Hành động Qwen trích | Trước | Sau | Gold/kỳ vọng edit |', '|---|---|---|---|---|']
    for case in normalization['cases']:
        name = case['case']
        event = case['event_extraction']['raw_output'].replace('\n', ' ').replace('|', '/')
        predicted = case['decision']['parsed_choice']
        gold = case['gold_or_edit_expectation']
        lines.append(f"| {name} | {event} | {case['previous_prediction']} | {predicted} | {gold} |")
        if name == '2310_original_10s' or name in ('1842', '1668', '1485'):
            idx = 2310 if name.startswith('2310_') else int(name)
            baseline = next(x[2] for x in pairs if x[0] == idx)
            norm_pairs.append((idx, gold, baseline, predicted))
    assert len(norm_pairs) == 4
    norm_ok = sum(p == g for _, g, b, p in norm_pairs)
    norm_wc = sum(b != g and p == g for _, g, b, p in norm_pairs)
    norm_cw = sum(b == g and p != g for _, g, b, p in norm_pairs)
    lines += ['', f'Trên bốn clip gốc: baseline {b_ok}/4; chẩn đoán chuẩn hóa sự kiện {norm_ok}/4; wrong→correct {norm_wc}; correct→wrong {norm_cw}. Các edit không được cộng vào số clip độc lập.']
    late_case = next(x for x in normalization['cases'] if x['case'] == '2310_audio_late')
    lines += ['', '**Đối chứng làm lộ điểm yếu còn lại:** khi cùng lời nói về dicing potato được chuyển sang đoạn hình đang cắt thực phẩm đỏ/cà chua, bước so khớp sau chuẩn hóa vẫn chọn Yes:', '',
        '> ' + late_case['decision']['raw_output'], '',
        'Cách sửa đã cải thiện các clip gốc nhưng chưa phân biệt được can thiệp đổi thời gian này. Matcher dựa vào tương đồng hành động cắt và bỏ qua đối tượng. Vì vậy, 4/4 trên bốn mẫu development không có nghĩa đã sửa được temporal grounding; cần giữ đối chứng này làm điều kiện kiểm tra phương pháp.', '',
        'Ở mẫu 1842, event extractor vẫn giữ cả “get potato masher” dù prompt yêu cầu thao tác chính. Đáp án được sửa sau thay đổi cách hỏi và biểu diễn, nhưng chưa tách được phần đóng góp riêng của event extraction. Cần ablation dùng cùng transcript/caption, cùng số lượt và chỉ thay câu hỏi đối chiếu.', '',
        'Kết luận hành vi từ các can thiệp: (1) thứ tự sự kiện có thể sai; (2) thay clock không làm quyết định đổi đúng; (3) cách hỏi/ánh xạ lựa chọn ảnh hưởng mạnh; (4) khi đã có bằng chứng phù hợp, bước ghép nội dung vẫn có thể sai về phạm vi sự kiện hoặc đối tượng. Đây là nhiều nút thắt tương tác, chưa xác định một module duy nhất là nguyên nhân.']
    for gold, label in [('A', 'Yes'), ('B', 'No')]:
        group = [x for x in norm_pairs if x[1] == gold]
        lines.append(f'Recall {label} sau chuẩn hóa: {sum(x[3] == gold for x in group)}/{len(group)}.')
    sys.path.insert(0, str(ROOT.parent))
    from local_qwen.temporal_structured_event_match_probe import action_group, words
    structured_originals = [x for x in structured if x['case'] == '2310_original_10s' or x['case'] in ('1842', '1668', '1485')]
    structured_rows = []
    for item in structured_originals:
        idx = 2310 if item['case'].startswith('2310_') else int(item['case'])
        event, visual = item['speech_tuple'], item['visual_tuple']
        same_action = action_group(event['action']) is not None and action_group(event['action']) == action_group(visual['action'])
        same_object = bool(words(event['object']) & words(visual['object']))
        pred = 'A' if same_action and same_object else 'B'
        base = next(x[2] for x in pairs if x[0] == idx)
        gold = next(x[1] for x in pairs if x[0] == idx)
        structured_rows.append((idx, gold, base, pred, event, visual))
    st_ok = sum(p == g for _, g, b, p, e, v in structured_rows)
    st_wc = sum(b != g and p == g for _, g, b, p, e, v in structured_rows)
    st_cw = sum(b == g and p != g for _, g, b, p, e, v in structured_rows)
    lines += ['', '## 9. Kết quả mới: tuple hành động–vật thể và kiểm tra trên nguồn confirmation', '',
        'Đối chiếu bằng câu tự do vẫn có ca dương tính giả (ví dụ “stir tomato paste” bị ghép với cảnh cho cà chua vào blender). Bản tiếp theo bắt Qwen xuất tuple riêng cho speech và visual, rồi dùng quy tắc cố định: cùng nhóm động từ và có vật thể cụ thể trùng nhau mới chọn Yes. Các nhóm động từ và chuẩn hóa hình thái nằm trong script; kết quả dưới đây tính lại từ tuple đã lưu bằng bộ quy tắc hiện tại.', '',
        '| Clip development | Gold | Baseline | Tuple + quy tắc | Speech tuple | Visual tuple |', '|---:|---|---|---|---|---|']
    for idx, gold, base, pred, event, visual in structured_rows:
        lines.append(f"| {idx} | {gold} | {base} | {pred} | {event['action']} / {event['object']} | {visual['action']} / {visual['object']} |")
    lines += ['', f'Tính lại trên bốn clip development: baseline {b_ok}/4; tuple + quy tắc {st_ok}/4; wrong→correct {st_wc}; correct→wrong {st_cw}. Đây là cùng bốn clip đã dùng để phát triển quy tắc; không phải xác nhận độc lập. Chuẩn hóa động từ được bổ sung sau khi thấy model xuất “mash/mashing”, nên con số này tuyệt đối không dùng như kết quả test.', '',
        '### Pilot confirmation được khóa trước khi chạy', '',
        '| Clip | Gold | Baseline | Phương pháp | Speech tuple | Visual tuple |', '|---:|---|---|---|---|---|']
    for sample in confirmation['samples']:
        lines.append(f"| {sample['idx']} | {sample['gold']} | {sample['baseline']['parsed_choice']} | {sample['structured_prediction']} | {sample['event_tuple']['action']} / {sample['event_tuple']['object']} | {sample['visual_tuple']['action']} / {sample['visual_tuple']['object']} |")
    cs = confirmation['summary']
    lines += ['', f"Trên hai nguồn TA confirmation đã chọn trước (một Yes, một No): baseline {cs['baseline_correct']}/2; method {cs['structured_correct']}/2; wrong→correct {cs['wrong_to_correct']}; correct→wrong {cs['correct_to_wrong']}. Chưa có cải thiện so với baseline.", '',
        'Ở mẫu 1961, speech tuple là `add bacon`, visual tuple là `cook bacon`, nên quy tắc loại nhầm câu Yes. Video tại khoảng đó có tương tác tay với các lát bacon; caption cả đoạn đã làm mất thao tác ngắn. Khi tăng mật độ lên 4 fps và hỏi lại, model caption là:', '', '> ' + dense['raw_caption'], '',
        'Caption mới vẫn không khôi phục đúng thao tác; vì vậy chỉ tăng fps hoặc thêm lời nhắc không giải quyết được event recognition. Trong mẫu 2081, speech là `pour dressing`, hình là `add green beans`; vật thể khác nhau và No được giữ đúng.', '',
        '**Cập nhật giả thuyết:** lỗi có thể bắt đầu trước bước fusion/ra đáp án: bộ biểu diễn hoặc khả năng diễn giải video của Qwen-3B không ổn định với thao tác ngắn trong cảnh nấu ăn, khiến caption gộp thành trạng thái rộng như “đang nấu”. Bộ so sánh tuple có thể giúp tránh vài lỗi ghép nhầm vật thể, nhưng gây false negative khi các động từ mô tả cùng một thủ tục ở mức hạt khác nhau (add bacon/cook bacon). Hai mẫu confirmation chưa chứng minh phương pháp cải thiện; chưa đủ cơ sở kết luận nguyên nhân nằm riêng ở encoder, attention hay lớp nào.', '',
        'Bước nghiên cứu kế tiếp nên là tạo bằng chứng thị giác theo từng khoảng ngắn nhưng có đối chiếu vùng tương tác tay–vật thể và giữ mốc thời gian, sau đó đánh giá riêng: (a) model thấy đúng thao tác tại từng khoảng không; (b) phép ghép action/object/time đúng không. Cần một detector/action recognizer độc lập được chọn từ literature và một đối chứng cùng chi phí trước khi tuyên bố phương pháp mới. Chưa chạy full benchmark vì pilot confirmation chưa cải thiện.', '',
        'Files bổ sung: `temporal_structured_event_match_probe.py/.json`, `temporal_structured_pipeline_confirmation_ta.py/.json`, `temporal_frame_window_audit_1961.py/.json`, `temporal_dense_window_1961.py/.json`.']
    lines += ['', 'Chỉ bốn nguồn, cùng một task TA, cùng checkpoint NF4: không dùng số này để khẳng định thắng SVHalluc, mọi small model, hay một hội nghị.',
        '', '## 7. Điều nên nghiên cứu tiếp', '',
        'Các phép đo đã tách được ba câu hỏi: model có nhận ra nội dung không; có giữ/khai thác đúng thời gian không; có ghép đúng sự kiện và xuất đáp án ổn định không. Lỗi không thể quy hết cho module fusion hoặc hai lớp cuối.', '',
        'Hướng kỹ thuật cần kiểm tra là biểu diễn sự kiện có hành động, đối tượng, độ hạt và khoảng thời gian rõ ràng, kèm phép kiểm tra phản thực: đổi clock phải đổi quan hệ đúng; đổi lựa chọn phải giữ ý nghĩa. Chẩn đoán cần xác định khâu hỏng rồi mới sửa khâu đó.', '',
        'Việc caption từng đoạn và cho LLM đọc bộ nhớ sự kiện đã có tiền lệ, như [VideoAgent](https://arxiv.org/abs/2403.11481) và [VideoTree](https://arxiv.org/abs/2405.19209). Bản module đơn giản ở đây là phép thử bottleneck, chưa được xem là novelty mới. Novelty cần kiểm chứng trên một cơ chế/method cụ thể và đối chiếu các công trình đó.', '',
        'Trước một kết luận về kiến trúc, cần tái lập trên FP16 và một checkpoint khác; các probe hiện tại mô tả hành vi của checkpoint lượng tử hóa này. Logit lens và attention map là công cụ dò; bỏ module rồi đáp án đổi đúng không đủ chứng minh model đã hiểu đúng trước đó.',
        '', '## 8. File để tái lập', '',
        '- temporal_clock_counterfactual_2310.py/.json: can thiệp clock và đảo ý nghĩa A/B.',
        '- temporal_clock_wording_probes_2310.py/.json: ngữ nghĩa toàn clip, câu hỏi cụ thể, Yes/No trực tiếp.',
        '- temporal_content_recall_2310.py/.json: nhận diện từng nguồn trên toàn clip và clock sweep.',
        '- temporal_clock_speech_audit_2310.py/.json: Whisper CPU chẩn đoán; không đưa vào Qwen.',
        '- temporal_activity_grounding_prototype_2310.py/.json: cách sửa trên nguồn 2310.',
        '- temporal_activity_grounding_validation_dev3.py/.json và temporal_choice_position_controls_2310.json: kiểm tra cố định trên nguồn khác và option-position controls.',
        '- temporal_narrated_event_normalization_probe.py/.json: chẩn đoán chuẩn hóa hành động trên bằng chứng development đã lưu; không phải test mới.',
        '- Các file .states.pt chứa hidden state ở vị trí cuối prompt qua các lớp; dùng cho nghiên cứu sau, không phải lời giải nội bộ đáng tin cậy của model.', '']
    report = ROOT / 'temporal_mechanism_findings_20261011.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    summary = {'independent_original_sources': pairs, 'baseline_correct': b_ok,
        'repair_correct': p_ok, 'wrong_to_correct': w_c, 'correct_to_wrong': c_w,
        'event_normalization_diagnostic': {'independent_original_sources': norm_pairs,
            'correct': norm_ok, 'wrong_to_correct': norm_wc, 'correct_to_wrong': norm_cw,
            'protocol': 'Saved development evidence; not fresh end-to-end or held-out evaluation'},
        'structured_tuple_diagnostic_development': {'correct': st_ok, 'wrong_to_correct': st_wc,
            'correct_to_wrong': st_cw, 'protocol': 'Current lexical rules re-applied to saved development tuples'},
        'structured_tuple_confirmation': cs,
        'scope': 'Four TA development source clips, one NF4 checkpoint. Synthetic edits excluded.'}
    (ROOT / 'temporal_mechanism_findings_20261011.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))
    print(report)


if __name__ == '__main__':
    main()

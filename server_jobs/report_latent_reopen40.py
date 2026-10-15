"""Summarize the interrupted-and-resumed first mechanism audit on frozen40."""
import json
import latent_reopen_manage as manager

def main():
    folder=manager.FOLDER;summary=manager.summary()
    assert all(v['n']==40 for v in summary.values()),'Need all40 before final comparison'
    rows=[json.loads(x) for s in (1,2) for x in (folder/f'results_s{s}.jsonl').read_text().splitlines()]
    assert len({r['idx'] for r in rows})==40
    lines=['# Lượt mở lại bằng chứng: kết quả40/40',
        '', 'Ngày2026-10-15. Tất cả ô có5 mẫu<=45s; mọi cyclic rotations; cùng manifest0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459.',
        '', '**Bảng ghi đúng/sai.**', '',
        '|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    cells=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']
    for m,v in summary.items():
        entries=[m]+[f'{v["cells"][c][0]}/{v["cells"][c][1]-v["cells"][c][0]}' for c in cells]
        entries += [f'{v["total"]}/{40-v["total"]}','/'.join(str(v['tf_predictions'][p]) for p in 'ABC')]
        lines.append('|'+ '|'.join(entries)+'|')
    lines+=['', '## Diễn giải', '',
        'D1 bỏ clock nhưng giữ gate/foreground/PMI. Nếu vẫn collapse Present, clock không phải điều kiện cần cho collapse ở M2; điều này không tự xác định tác dụng của clock trong delta M8.',
        'D2 mở lại native attention từ block24; D3 mở quyền đọc toàn bộ AV của query trong mọi block nhưng giữ can thiệp media cũ. D3 đổi cả quyền đọc thời gian lẫn audio, nên không tách riêng hai yếu tố.',
        'G1/G2 chỉ hạn chế các cạnh media khác nguồn cách nhau>3s, giữ query đọc toàn bộ AV; G3 phản hồi chênh trạng thái về cùng block18/22. Đây là ứng viên không dùng speech window, clock từ đáp án, PMI hoặc task router.',
        'Đánh giá cải thiện phải dựa trên cả8 ô. Cứu Future bằng cách chuyển thiên lệch sang Future, trong khi Past/Present thấp, chưa giải quyết temporal reasoning.',
        'Hướng này chưa đạt bằng chứng về một phương pháp tổng quát đủ mạnh cho CVPR. Không mở rộng sang held-out chỉ vì một tổng điểm tăng nhẹ.',
        '', '## Triển khai', '',
        'Server1 dùng lại decoder caches; server2 là phiên mới nên mã hóa lại đúng20 ID, không đổi mẫu. Replay và baseline được lưu cùng lượt; so ghép cặp không dựa vào một baseline khác phiên.',
        'Bốn CPU contract tests đã qua: causal edges, query đọc fullAV, zero-gain identity/media unchanged và hook cleanup. Các kiểm tra này không chứng minh QA correctness.',
        'Kết quả raw: results_s1/s2.jsonl; bảng số: summary.json; giả thuyết/đối chứng: protocol.md. CCLD của Argon được xếp ngay sau lượt này trên từng server.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:16]))

if __name__=='__main__':main()

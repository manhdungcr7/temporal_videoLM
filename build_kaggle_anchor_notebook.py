"""Build the Kaggle notebook for the CMTB anchor-then-look experiment (reuses the validated setup cell)."""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B  # validated setup cell; rewrites the old notebook identically

cell = B.cell
experiment_src = (ROOT / 'kaggle_anchor_experiment.py').read_text(encoding='utf-8')
ast.parse(experiment_src)

intro = """# SVHalluc CMTB — Có phải model "không nhìn được" hay "bị nhiễu bởi tín hiệu khác"? Thí nghiệm neo vào lời nói

**Kaggle Edit, GPU T4 ×2, bật Internet. Chạy cell 1 → 2 → 3 → 4.** Cell 3 chạy khoảng 40–70 phút (mặc định 20 mẫu mỗi nhãn A/B/C = 60 mẫu CMTB); có thể chạy lại, sẽ bỏ qua mẫu đã làm.

Câu hỏi CMTB: *khi lời nói được nghe, hành động nào đang diễn ra trên hình?* Mỗi mẫu được chấm 4 điều kiện, **mỗi điều kiện hỏi với 3 thứ tự xoay vòng của các phương án** (cộng điểm theo *nội dung* phương án) để loại thiên lệch chữ cái:

| Điều kiện | Đầu vào | Mục đích |
|---|---|---|
| base_av | cả clip, audio + video, câu hỏi gốc | baseline |
| vision_full | cả clip, **bỏ audio** | tách tác dụng của việc bỏ audio |
| speech_win | chỉ **đoạn có lời nói**, bỏ audio | tác dụng của việc neo vào đoạn lời nói |
| silent_win | chỉ đoạn im lặng, bỏ audio | đối chứng: hành động đúng không nằm ở đây, kỳ vọng thấp |

Đoạn lời nói được lấy từ năng lượng âm thanh (RMS) của chính clip, chỉ dùng để cắt đoạn; không có ASR, transcript, nhãn hay mô hình phụ. Nhãn chỉ dùng để chấm điểm.

Đọc kết quả ở cell 4. Điều cần so: speech_win so với base_av **và** vision_full. Nếu speech_win cao hơn vision_full thì việc *neo vào đúng đoạn* có tác dụng riêng, ngoài việc bỏ audio."""

setup_note = "## Cell 1–2: cài môi trường (dùng lại cell đã chạy được); chỉ cell 1 cần, bỏ qua cell nạp model của notebook cũ"

exp_cell = '# CELL 2 — ghi script thí nghiệm ra đĩa\nEXPERIMENT_SOURCE = ' + repr(experiment_src) + '\n' + \
    "(ROOT/'kaggle_anchor_experiment.py').write_text(EXPERIMENT_SOURCE,encoding='utf-8')\nprint('Đã ghi',ROOT/'kaggle_anchor_experiment.py')\n"

run_cell = '''# CELL 3 — chạy thí nghiệm (có thể ngắt rồi chạy lại để tiếp tục).
N_PER_LABEL = 20   # số mẫu mỗi nhãn A/B/C; tăng lên nếu còn thời gian
import subprocess, sys
cmd = [str(PYTHON), str(ROOT/'kaggle_anchor_experiment.py'), str(ROOT), str(OUT), str(N_PER_LABEL)]
env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
for line in p.stdout:
    if 'warn' in line.lower() and 'Traceback' not in line: continue
    print(line, end='', flush=True)
print('exit code', p.wait())
'''

report_cell = '''# CELL 4 — kết quả. Chạy lại bất cứ lúc nào để xem tiến độ.
import numpy as np, random
R = [json.loads(l) for l in (OUT/'anchor_results.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
bad = [r for r in R if 'error' in r]
R = [r for r in R if 'error' not in r]
print('mẫu hợp lệ:', len(R), '| lỗi:', len(bad), [b['error'][:80] for b in bad[:3]])
CONDS = ['base_av', 'vision_full', 'speech_win', 'silent_win']
def ok(r, c, key): return r[c][key] == 'ABC'[r['gold']]
print('\\nĐộ chính xác (cộng điểm theo nội dung qua 3 thứ tự xoay vòng; ngẫu nhiên = 33%)')
for c in CONDS:
    rs = [r for r in R if c in r]
    print(f'  {c:12s} xoay vòng {sum(ok(r,c,"pred_rot") for r in rs):3d}/{len(rs)}'
          f' = {100*np.mean([ok(r,c,"pred_rot") for r in rs]):5.1f}%   |  thứ tự gốc {100*np.mean([ok(r,c,"pred_single") for r in rs]):5.1f}%')
print('\\nThiên lệch chữ cái ở thứ tự gốc (tỉ lệ dự đoán A/B/C; nhãn thật cân bằng):')
for c in CONDS:
    rs = [r for r in R if c in r]
    cnt = {L: sum(r[c]['pred_single'] == L for r in rs) for L in 'ABC'}
    print(f'  {c:12s}', {L: round(v/len(rs), 2) for L, v in cnt.items()})
def paired(a, b):
    rs = [r for r in R if a in r and b in r]
    w2c = sum((not ok(r,a,'pred_rot')) and ok(r,b,'pred_rot') for r in rs)
    c2w = sum(ok(r,a,'pred_rot') and (not ok(r,b,'pred_rot')) for r in rs)
    d = [int(ok(r,b,'pred_rot')) - int(ok(r,a,'pred_rot')) for r in rs]
    g = random.Random(0); bs = sorted(np.mean([g.choice(d) for _ in d]) for _ in range(2000))
    print(f'  {a} → {b}: wrong→correct {w2c}, correct→wrong {c2w}, Δacc {100*np.mean(d):+.1f} điểm, 95% CI [{100*bs[50]:+.1f}, {100*bs[1949]:+.1f}] (n={len(rs)})')
print('\\nSo sánh ghép cặp (xoay vòng):')
paired('base_av', 'vision_full'); paired('vision_full', 'speech_win'); paired('base_av', 'speech_win')
print('\\nTheo độ dài clip (speech_win xoay vòng):')
for lo, hi in ((0, 20), (20, 30), (30, 99)):
    rs = [r for r in R if lo <= r['dur'] < hi and 'speech_win' in r]
    if rs: print(f'  {lo}-{hi}s n={len(rs):2d}: base_av {100*np.mean([ok(r,"base_av","pred_rot") for r in rs]):.0f}% | vision_full {100*np.mean([ok(r,"vision_full","pred_rot") for r in rs]):.0f}% | speech_win {100*np.mean([ok(r,"speech_win","pred_rot") for r in rs]):.0f}%')
'''

export_cell = '''# Tùy chọn — nén kết quả để tải về.
ZIP = Path('/kaggle/working/anchor_results.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(OUT/'anchor_results.jsonl', 'anchor_results.jsonl')
display(FileLink(str(ZIP)))
'''

notes = """## Cách đọc kết quả

- **Thiên lệch chữ cái:** nếu một điều kiện dự đoán một chữ cái chiếm quá khoảng 50% ở thứ tự gốc (trong khi nhãn cân bằng), con số "thứ tự gốc" của điều kiện đó không đáng tin; dùng cột "xoay vòng".
- **silent_win cao** (xa 33% theo hướng đúng): đoạn im lặng cũng đủ để trả lời, nghĩa là model đang dựa vào thứ khác ngoài hành động đúng lúc đó. Cần xét lại thiết kế.
- **speech_win ≈ vision_full:** lợi ích đến từ việc bỏ audio, không phải từ việc neo vào đoạn lời nói.
- **speech_win > vision_full:** neo vào đoạn lời nói có tác dụng riêng. Đây mới là bằng chứng cho giả thuyết "model nhận ra hành động, nhưng không gắn được với đúng thời điểm khi phải kết hợp audio và toàn bộ clip".
- Mẫu nhỏ (60): nếu chênh lệch nằm trong khoảng tin cậy, chưa kết luận gì; tăng N_PER_LABEL lên 60–113 (toàn bộ CMTB có 339 mẫu).
- Đây là thí nghiệm chẩn đoán trên Qwen2.5-Omni-3B FP16, chưa phải phương pháp hoàn chỉnh."""

nb = {'cells': [cell('markdown', intro), cell('markdown', setup_note), cell('code', B.setup), cell('code', exp_cell),
                cell('code', run_cell), cell('code', report_cell), cell('markdown', notes), cell('code', export_cell)],
      'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_temporal_anchor_probe.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target, target.stat().st_size)

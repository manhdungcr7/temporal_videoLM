"""Build the Kaggle notebook for the full temporal study (localization, attention gating, look-and-compare, noise control)."""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B   # validated setup cell

cell = B.cell
files = {
    'speech_gate.py': (ROOT / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
    'kaggle_temporal_study.py': (ROOT / 'kaggle_temporal_study.py').read_text(encoding='utf-8'),
    'temporal_study_report.py': (ROOT / 'temporal_study_report.py').read_text(encoding='utf-8'),
}
for s in files.values(): ast.parse(s)

intro = """# SVHalluc temporal — "nghe được, nhìn được, nhưng không nhìn đúng lúc": kiểm chứng và sửa

**Kaggle Edit, GPU T4 ×2, bật Internet. Chạy cell 1 → 2 → 3, xem kết quả ở cell 4.** Cell 3 có thể chạy lại, sẽ bỏ qua mẫu đã làm. Mỗi phần ghi file riêng trong thư mục kết quả.

| Phần | Câu hỏi nghiên cứu | Thời gian ước tính |
|---|---|---|
| `main` | Chặn chú ý theo thời gian lời nói có sửa được TA/TF/CMTB trong **một lượt**, không cắt clip, không huấn luyện? | ~2–3 giờ với N=25 |
| `loc` | Model có biết **lúc nào** có lời nói không? (timestamp + câu hỏi nửa đầu/nửa sau, 40 clip) | ~20–30 phút |
| `noise` | Kết quả còn giữ khi thay khoảng im lặng tuyệt đối bằng tiếng nền nhỏ (bỏ lối tắt của benchmark)? | ~1 giờ với N=10 |

**Các điều kiện trong `main`** (mỗi điều kiện đều xoay vòng thứ tự phương án và cộng điểm theo nội dung):
- `base`: cả clip, audio + video, câu hỏi gốc.
- `gate_hard`: cùng đầu vào, cùng câu hỏi; vị trí câu hỏi/câu trả lời **không được nhìn** token hình ngoài khoảng lời nói (dựa trên mốc thời gian chung của audio–video trong Qwen-Omni). Token audio/video vẫn được mã hoá như cũ.
- `gate_soft`: như trên nhưng chỉ trừ một lượng cố định (β=3) thay vì chặn hẳn.
- `lac` (TA/TF): nhìn vào từng khoảng (lúc nói / trước / sau), hỏi "sự kiện được kể có diễn ra không", rồi **so sánh** giữa các khoảng để ra Present/Past/Future.

Khoảng lời nói lấy từ năng lượng âm thanh của chính clip. Không dùng ASR, transcript, nhãn hay mô hình phụ; nhãn chỉ dùng để chấm điểm."""

write_cell = '# CELL 2 — ghi các file thí nghiệm\nSTUDY_FILES = ' + repr(files) + '\n' + \
    "for _n,_s in STUDY_FILES.items():(ROOT/_n).write_text(_s,encoding='utf-8')\nprint('Đã ghi:',list(STUDY_FILES))\n"

run_cell = '''# CELL 3 — chạy. Có thể ngắt rồi chạy lại để tiếp tục.
PARTS = 'main,loc,noise'   # bỏ bớt phần nếu thiếu thời gian, ví dụ 'main'
N_PER_LABEL = 25           # mẫu mỗi nhãn ở mỗi task trong phần main (TA 2 nhãn, TF 3, CMTB 3)
N_NOISE_PER_LABEL = 10     # mẫu mỗi nhãn cho phần noise (lấy cùng thứ tự chọn nên trùng với main)
import subprocess
cmd = [str(PYTHON), str(ROOT/'kaggle_temporal_study.py'), str(ROOT), str(OUT), PARTS, str(N_PER_LABEL), str(N_NOISE_PER_LABEL)]
env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
for line in p.stdout:
    low = line.lower()
    if ('warn' in low or 'loading checkpoint' in low) and 'traceback' not in low: continue
    print(line, end='', flush=True)
print('exit code', p.wait())
'''

report_cell = '''# CELL 4 — báo cáo (chạy lại bất cứ lúc nào để xem tiến độ)
import subprocess
print(subprocess.run([str(PYTHON), str(ROOT/'temporal_study_report.py'), str(OUT)], capture_output=True, text=True, encoding='utf-8').stdout)
'''

notes = """## Cách đọc

- **TF có một lối tắt đã biết:** chỉ dựa vào lời nói nằm nửa đầu hay nửa sau clip đã đạt khoảng 66,7% mà không cần video. Báo cáo in dòng tham chiếu này; phương pháp phải **vượt** nó mới chứng minh được là dùng video.
- **gate_hard ≈ base** ở TA/TF nhưng cao ở CMTB: chặn chú ý chỉ giúp câu hỏi "lúc nghe thì thấy gì", chưa đủ cho câu hỏi quan hệ thời gian → xem `lac`.
- **Phần noise:** nếu độ chính xác giữ được khi bỏ khoảng im lặng tuyệt đối, phương pháp không ăn theo lỗi dựng dữ liệu.
- **Phần loc:** IoU cao / câu hỏi nửa đầu–nửa sau đúng cao nhưng base vẫn sai → model *biết* lúc nào có lời nói mà không *dùng* được thông tin đó. IoU thấp → model không định vị được lời nói.
- Đây là Qwen2.5-Omni-3B FP16; kết quả cần xác nhận trên model khác trước khi viết thành phát hiện chung."""

export_cell = '''# Nén kết quả để tải về
ZIP = Path('/kaggle/working/temporal_study_results.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in ('orig.jsonl', 'noisy.jsonl', 'loc.jsonl'):
        if (OUT/n).exists(): z.write(OUT/n, n)
display(FileLink(str(ZIP)))
'''

net_cell = '''# CELL 0 — kiểm tra Internet trước khi cài đặt (nếu lỗi: bật Internet trong Session options rồi chạy lại)
import socket
for host in ('pypi.org', 'huggingface.co'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: panel bên phải -> Session options -> Internet = On.') from e
'''
nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', B.setup.replace('assert torch.cuda.device_count()>=2;','assert torch.cuda.device_count()>=1;')), cell('code', write_cell), cell('code', run_cell),
                cell('code', report_cell), cell('markdown', notes), cell('code', export_cell)],
      'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_temporal_study.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target, target.stat().st_size)

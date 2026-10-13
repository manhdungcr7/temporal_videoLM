"""Build the sharded Kaggle notebook for the temporal method lab."""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B

cell = B.cell
files = {'speech_gate.py': (ROOT / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
         'kaggle_temporal_lab.py': (ROOT / 'kaggle_temporal_lab.py').read_text(encoding='utf-8'),
         'temporal_lab_report.py': (ROOT / 'temporal_lab_report.py').read_text(encoding='utf-8')}
for s in files.values(): ast.parse(s)

intro = """# SVHalluc temporal — phòng thí nghiệm cơ chế, chạy THEO GIAI ĐOẠN

**Giai đoạn 1 (pilot, 1 tài khoản, ~3 giờ):** 64 câu hỏi cân bằng theo task và nhãn (`PER_LABEL = 8`). Mục đích là trả lời các câu hỏi cơ chế bên dưới, **chưa phải đánh giá benchmark**. Với mẫu nhỏ này, khoảng tin cậy sẽ rộng; chỉ tin các hiệu ứng rất lớn hoặc rất nhất quán.

**Chỉ mở rộng (tăng `PER_LABEL`, chia `NUM_SHARDS` cho nhiều tài khoản, hoặc đặt `PER_LABEL = 0`) khi pilot cho tín hiệu rõ.** Tập con của `PER_LABEL` nhỏ nằm trong tập con của `PER_LABEL` lớn, nên kết quả pilot được dùng lại. Tiêu chí quyết định xem ở phần ghi chú cuối notebook.

GPU T4 ×2, bật Internet. Cuối cùng tải `temporal_lab_shard<k>.zip` ở cell cuối và gửi lại.

| Điều kiện | Đo cái gì |
|---|---|
| `base` | Model gốc (cả clip, audio + video, câu hỏi gốc) |
| `ts05`, `ts02` | Nén chỉ số thời gian TMRoPE của token audio/video còn 0,5 / 0,2 → vị trí thời gian quá thưa có gây thiên lệch không? |
| `gate_all` | Không vị trí nào được đọc token hình ngoài khoảng lời nói (mọi lớp) |
| `gate_late` | Như trên nhưng chỉ ở lớp 18–35 |
| `audio_block`, `gate_audio` | (CMTB) chặn đọc audio / chặn cả hai → nội dung lời nói có phải nguồn nhiễu? |
| `crop` | (CMTB) chỉ đoạn lời nói, không audio — mốc tham chiếu |
| `trace/` (chỉ phần 0, 2 câu mỗi task) | **Debug từng bước**: tiền xử lý (số khung, thời điểm từng khung, độ dài audio, lưới video), cấu trúc token (đoạn text/audio/video, mốc thời gian TMRoPE của từng đoạn), logit top-8 và logit A/B/C của từng điều kiện, **logit lens theo từng lớp**, **attention từng lớp/từng head** (vào video trong/ngoài khoảng lời nói, vào audio) |
| `swap_*` | **Đảo thứ tự** đoạn lời nói và đoạn còn lại (mỗi đoạn giữ audio của nó) → phép thử nhân quả cho thiên lệch vị trí |

Mỗi điều kiện đều xoay vòng thứ tự phương án và cộng điểm theo nội dung. Không ASR, không nhãn, không mô hình phụ; nhãn chỉ để chấm."""

net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''
write_cell = '# CELL 2 — ghi file\nLAB_FILES = ' + repr(files) + '\n' + \
    "for _n,_s in LAB_FILES.items():(ROOT/_n).write_text(_s,encoding='utf-8')\nprint('Đã ghi:',list(LAB_FILES))\n"
run_cell = '''# CELL 3 — ĐỔI SHARD_ID cho mỗi tài khoản. Chạy lại được (bỏ qua câu đã làm).
PER_LABEL = 8       # GIAI ĐOẠN 1 (pilot): 8 câu mỗi (task, nhãn) = 64 câu, một tài khoản ~3 giờ. 0 = toàn bộ benchmark (CHỈ khi pilot đạt tiêu chí).
SHARD_ID = 0        # 0 .. NUM_SHARDS-1, mỗi tài khoản một số khác nhau (pilot: để 0)
NUM_SHARDS = 1      # pilot: 1 tài khoản. Giai đoạn sau mới tăng và chia phần.
import subprocess
cmd = [str(PYTHON), str(ROOT/'kaggle_temporal_lab.py'), str(ROOT), str(OUT), str(SHARD_ID), str(NUM_SHARDS), str(PER_LABEL)]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                     env={**os.environ, 'PYTHONUNBUFFERED': '1'})
for line in p.stdout:
    low = line.lower()
    if ('warn' in low or 'loading checkpoint' in low or 'fetching' in low) and 'traceback' not in low: continue
    print(line, end='', flush=True)
print('exit code', p.wait())
'''
report_cell = '''# CELL 4 — báo cáo của phần này (báo cáo gộp nhiều phần sẽ làm sau khi gom file)
import subprocess
print(subprocess.run([str(PYTHON), str(ROOT/'temporal_lab_report.py'), str(OUT)], capture_output=True, text=True, encoding='utf-8').stdout)
'''
export_cell = '''# CELL 5 — nén kết quả của phần này
import glob
ZIP = Path(f'/kaggle/working/temporal_lab_shard{SHARD_ID}.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(str(OUT/'lab_shard*.jsonl')): z.write(f, Path(f).name)
    for f in glob.glob(str(OUT/'trace'/'trace_*.json')): z.write(f, 'trace/' + Path(f).name)
display(FileLink(str(ZIP)))
'''
setup = B.setup.replace('assert torch.cuda.device_count()>=2;', 'assert torch.cuda.device_count()>=1;')
notes = """## Tiêu chí quyết định sau pilot (đặt trước khi xem kết quả)

| Câu hỏi | Chỉ số | Kết luận nếu... | Bước tiếp theo |
|---|---|---|---|
| Thiên lệch vị trí có thật và nhân quả không? | Phần `swap`: độ chính xác khi đoạn lời nói ở TRƯỚC vs SAU (gộp bản gốc và bản đảo); TF đảo Past↔Future có đúng quy luật không | Chênh ≥ 15 điểm theo vị trí chứ không theo nội dung, ở ≥ 2 task | Mở rộng phần swap; thiết kế phương pháp sửa vị trí |
| Vị trí thời gian quá thưa có gây thiên lệch? | `ts05`/`ts02` so với `base` | Một trong hai tăng ≥ 8 điểm ở ≥ 2 task, wrong→correct > correct→wrong gấp đôi | Thử trên model/benchmark khác; so với PAS/DTR |
| Chặn chú ý có giúp? | `gate_all`, `gate_late` so với `base` (recall theo nhãn, không chỉ accuracy) | Tăng ≥ 8 điểm ở CMTB và/hoặc TA mà recall của cả hai nhãn không tụt | Mở rộng; thử gate mềm |
| Nội dung lời nói có phải nguồn nhiễu? | `gate_audio` so với `gate_all`, `crop` (CMTB) | gate_audio gần `crop` và cao hơn gate_all rõ | Đưa chặn audio vào phương pháp |

Nếu **không** điều kiện nào đạt: không mở rộng; quay lại tìm nguyên nhân khác (đừng chạy full). Nếu `ts*`/`gate*` làm đổi mạnh phân bố nhãn dự đoán (recall lệch một phía), coi là thiên lệch chứ không phải cải thiện."""
nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', setup), cell('code', write_cell), cell('code', run_cell),
                cell('code', report_cell), cell('markdown', notes), cell('code', export_cell)],
      'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_temporal_lab.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target.name, target.stat().st_size)

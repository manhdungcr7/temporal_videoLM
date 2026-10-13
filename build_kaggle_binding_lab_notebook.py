"""Build the Kaggle notebook for the human-inspired binding lab (screening pilot)."""
import ast, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B
cell = B.cell
files = {'speech_gate.py': (ROOT / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
         'kaggle_binding_lab.py': (ROOT / 'kaggle_binding_lab.py').read_text(encoding='utf-8'),
         'binding_lab_report.py': (ROOT / 'binding_lab_report.py').read_text(encoding='utf-8')}
for s in files.values(): ast.parse(s)
intro = """# Binding lab — mắt và tai của cùng một khoảnh khắc nên được ghép thế nào?

**Một tài khoản (khác tài khoản đang chạy giai đoạn 2), GPU T4 ×2, bật Internet, Save & Run All, khoảng 2 giờ.** Tải `binding_lab_shard0.zip` ở cell cuối.

**Câu hỏi.** Con người ghép âm thanh và hình ảnh **cùng thời điểm** (cửa sổ ghép thời gian), và chú ý có chọn lọc. Trong Qwen2.5-Omni, mỗi đoạn 2 giây xếp *[video, audio]* với mặt nạ nhân quả, nên: (1) hình của một khoảnh khắc **không bao giờ đọc được** âm thanh của chính khoảnh khắc đó; (2) âm thanh lại đọc được hình của **mọi** khoảnh khắc trước; (3) mỗi giây có ~144 token hình nhưng chỉ ~25 token âm thanh. Notebook này thử sửa từng điểm, không huấn luyện, một lượt suy luận:

| Điều kiện | Thay đổi |
|---|---|
| `bind0` | Chú ý chéo nghe–nhìn chỉ trong **cùng đoạn 2 s** |
| `bind0_bi` | Như trên + hình được đọc âm thanh **của chính đoạn đó** |
| `bind1_bi` | Như trên, cho phép lệch ±1 đoạn |
| `bi_only` | Chỉ thêm hình → âm thanh cùng đoạn, không giới hạn gì khác |
| `balance` | Vị trí câu hỏi/trả lời: cân bằng chú ý tổng giữa audio và video |

Đánh giá trên **cả temporal (TA/TF/CMTB) và semantic (GSA/FGSA/CMSB)**, 5 câu mỗi (task, nhãn): một thay đổi chỉ đáng theo nếu giúp temporal mà không làm hại semantic. Tiêu chí sàng lọc in ở cuối báo cáo, đặt trước khi chạy. Mẫu nhỏ: đây là sàng lọc, không phải kết quả."""
net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''
write_cell = '# CELL 2 — ghi file\nBIND_FILES = ' + repr(files) + '\n' + \
    "for _n,_s in BIND_FILES.items():(ROOT/_n).write_text(_s,encoding='utf-8')\nprint('Đã ghi:',list(BIND_FILES))\n"
run_cell = '''# CELL 3 — chạy
PER_LABEL = 5
SHARD_ID, NUM_SHARDS = 0, 1
import subprocess
cmd = [str(PYTHON), str(ROOT/'kaggle_binding_lab.py'), str(ROOT), str(OUT), str(PER_LABEL), str(SHARD_ID), str(NUM_SHARDS)]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                     env={**os.environ, 'PYTHONUNBUFFERED': '1'})
for line in p.stdout:
    low = line.lower()
    if ('warn' in low or 'loading checkpoint' in low or 'fetching' in low) and 'traceback' not in low: continue
    print(line, end='', flush=True)
print('exit code', p.wait())
'''
report_cell = '''# CELL 4 — báo cáo
import subprocess
print(subprocess.run([str(PYTHON), str(ROOT/'binding_lab_report.py'), str(OUT)], capture_output=True, text=True, encoding='utf-8').stdout)
'''
export_cell = '''# CELL 5 — nén kết quả
import glob
ZIP = Path('/kaggle/working/binding_lab_shard0.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(str(OUT/'binding_shard*.jsonl')): z.write(f, Path(f).name)
display(FileLink(str(ZIP)))
'''
setup = B.setup.replace('assert torch.cuda.device_count()>=2;', 'assert torch.cuda.device_count()>=1;')
nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', setup), cell('code', write_cell), cell('code', run_cell),
                cell('code', report_cell), cell('code', export_cell)], 'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_binding_lab.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target.name, target.stat().st_size)

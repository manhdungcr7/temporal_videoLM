"""Build the Kaggle notebook for root-cause completion on SVHalluc temporal."""
import ast, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B
cell = B.cell
files = {'speech_gate.py': (ROOT / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
         'kaggle_rootcause.py': (ROOT / 'kaggle_rootcause.py').read_text(encoding='utf-8'),
         'rootcause_report.py': (ROOT / 'rootcause_report.py').read_text(encoding='utf-8')}
for s in files.values(): ast.parse(s)
intro = """# SVHalluc temporal — đo nốt các nguyên nhân gốc (chưa thử phương pháp mới)

**Chạy trên 2 tài khoản song song: tài khoản 1 đặt `SHARD_ID = 0`, tài khoản 2 đặt `SHARD_ID = 1` (cell 3). GPU T4 ×2, bật Internet, Save & Run All, mỗi phần khoảng 3–4 giờ.** Tải `rootcause_shard<k>.zip` ở cell cuối của mỗi tài khoản và gửi cả hai.

Mục đích: lấp các ô "chưa đo" trong bảng nguyên nhân ([14_root_causes_and_general_method.md]). Mỗi câu hỏi (TA/TF/CMTB, 20 câu mỗi nhãn, chia đôi hai phần) được đo:

| Đo | Nguyên nhân tương ứng |
|---|---|
| `para1`, `para2`: hai cách diễn đạt khác của câu hỏi | #12 model có hiểu sai câu hỏi không |
| `prior`: video đen + âm thanh im lặng | #13 đáp án "mặc định" khi không có bằng chứng |
| `vision_only`, `audio_only` (video đen + âm thanh gốc) | model dựa vào nguồn nào khi trả lời |
| (CMTB) `gate`, `audio_block`, `gate_audio`, `crop` | #9/#10 phân rã từng lỗi theo nguyên nhân |
| `hear` + Whisper (chỉ để chấm) | #4 model có nghe đúng nội dung lời nói không |
| Hidden state từng lớp → probe tuyến tính (CV) | #14 model "biết" đáp án bên trong nhưng không nói ra? |
| (CMTB sai → `gate_audio` đúng) activation patching từng lớp | **lỗi đi vào ở lớp nào** |
| (phần 0) câu hỏi chữ tổng hợp, không video | #11 có suy luận đúng trước/trong/sau từ dữ kiện không |
| (phần 0, 24 câu đầu) attention từng lớp của token câu hỏi | #8 chú ý dồn về đầu video ở quy mô lớn |

Không có phương pháp mới nào được đánh giá ở đây; đây là phép đo chẩn đoán. Whisper chỉ để chấm, không bao giờ đưa vào model."""
net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''
write_cell = '# CELL 2 — ghi file\nRC_FILES = ' + repr(files) + '\n' + \
    "for _n,_s in RC_FILES.items():(ROOT/_n).write_text(_s,encoding='utf-8')\nprint('Đã ghi:',list(RC_FILES))\n"
run_cell = '''# CELL 3 — ĐỔI SHARD_ID: tài khoản 1 -> 0, tài khoản 2 -> 1
SHARD_ID = 0
NUM_SHARDS = 2
PER_LABEL = 20
import subprocess
cmd = [str(PYTHON), str(ROOT/'kaggle_rootcause.py'), str(ROOT), str(OUT), str(PER_LABEL), str(SHARD_ID), str(NUM_SHARDS)]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                     env={**os.environ, 'PYTHONUNBUFFERED': '1'})
for line in p.stdout:
    low = line.lower()
    if ('warn' in low or 'loading checkpoint' in low or 'fetching' in low) and 'traceback' not in low: continue
    print(line, end='', flush=True)
print('exit code', p.wait())
'''
report_cell = '''# CELL 4 — báo cáo của phần này
import subprocess
print(subprocess.run([str(PYTHON), str(ROOT/'rootcause_report.py'), str(OUT)], capture_output=True, text=True, encoding='utf-8').stdout)
'''
export_cell = '''# CELL 5 — nén kết quả (jsonl + hidden states + synthetic)
import glob
ZIP = Path(f'/kaggle/working/rootcause_shard{SHARD_ID}.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(str(OUT/'rootcause_shard*.jsonl')) + glob.glob(str(OUT/'synthetic_relation.json')): z.write(f, Path(f).name)
    for f in glob.glob(str(OUT/'hidden'/'*.npz')): z.write(f, 'hidden/' + Path(f).name)
display(FileLink(str(ZIP)))
'''
setup = B.setup.replace('assert torch.cuda.device_count()>=2;', 'assert torch.cuda.device_count()>=1;')
nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', setup), cell('code', write_cell), cell('code', run_cell),
                cell('code', report_cell), cell('code', export_cell)], 'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_rootcause.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target.name, target.stat().st_size)

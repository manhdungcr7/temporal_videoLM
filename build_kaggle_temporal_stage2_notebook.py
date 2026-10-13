"""Build the stage-2 Kaggle notebook (artifact-free test of the time x modality binding fix)."""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B

cell = B.cell
files = {'speech_gate.py': (ROOT / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
         'kaggle_temporal_stage2.py': (ROOT / 'kaggle_temporal_stage2.py').read_text(encoding='utf-8'),
         'temporal_stage2_report.py': (ROOT / 'temporal_stage2_report.py').read_text(encoding='utf-8')}
for s in files.values(): ast.parse(s)

intro = """# SVHalluc temporal — Giai đoạn 2: lỗi "ràng buộc thời điểm × modality" có còn khi bỏ lối tắt của benchmark?

**Một tài khoản, GPU T4 ×2, bật Internet, Save & Run All. Khoảng 3–4 giờ.** Tải `temporal_stage2_shard0.zip` ở cell cuối và gửi lại.

**Bối cảnh.** Pilot trước (CMTB, n=19) cho thấy một tương tác 2×2: chỉ chặn khung ngoài thời điểm lời nói → 63%; chỉ chặn audio → 32%; **chặn cả hai → 95%** (bằng cắt clip). Nhưng ở bản thay khoảng im lặng tuyệt đối bằng tiếng nền, lợi ích của chặn khung co lại mạnh. Giai đoạn này kiểm tra xem tương tác 2×2 và cách "nghe rồi nhìn" có **còn đứng** khi không có lối tắt đó không.

Mỗi câu hỏi được chạy trên **hai phiên bản của cùng clip** (gốc và có tiếng nền), cùng các điều kiện:

| Điều kiện | Ý nghĩa |
|---|---|
| `base` | Model gốc |
| `gate` / `audio_block` / `gate_audio` | Thiết kế 2×2: chặn khung ngoài khoảng lời nói, chặn audio, chặn cả hai (một lượt, không cắt clip) |
| `crop` | (CMTB) chỉ đoạn lời nói, chỉ hình — mốc tham chiếu |
| `hear` → `htl` | (TA/TF) **Nghe rồi nhìn trong cùng một model**: model tự nghe "người nói mô tả hành động gì", rồi nhìn từng khoảng thời gian với audio bị chặn và hỏi hành động đó có xảy ra không; so sánh giữa các khoảng |
| `lac` | So sánh giữa các khoảng nhưng **không** chặn audio (cách cũ) |

Không ASR ngoài, không công cụ, không nhãn, không huấn luyện. Nhãn chỉ dùng để chấm. **Tiêu chí quyết định đã được cài sẵn trong báo cáo (cell 4) trước khi chạy.**"""

net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''
write_cell = '# CELL 2 — ghi file\nS2_FILES = ' + repr(files) + '\n' + \
    "for _n,_s in S2_FILES.items():(ROOT/_n).write_text(_s,encoding='utf-8')\nprint('Đã ghi:',list(S2_FILES))\n"
run_cell = '''# CELL 3 — chạy (chạy lại được, bỏ qua câu đã làm)
PER_LABEL = 12     # 12 câu mỗi (task, nhãn): TA 24, TF 36, CMTB 36 (clip > 42 s bị bỏ qua)
SHARD_ID, NUM_SHARDS = 0, 1
import subprocess
cmd = [str(PYTHON), str(ROOT/'kaggle_temporal_stage2.py'), str(ROOT), str(OUT), str(PER_LABEL), str(SHARD_ID), str(NUM_SHARDS)]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                     env={**os.environ, 'PYTHONUNBUFFERED': '1'})
for line in p.stdout:
    low = line.lower()
    if ('warn' in low or 'loading checkpoint' in low or 'fetching' in low) and 'traceback' not in low: continue
    print(line, end='', flush=True)
print('exit code', p.wait())
'''
report_cell = '''# CELL 4 — báo cáo + tiêu chí quyết định (đặt trước khi chạy)
import subprocess
print(subprocess.run([str(PYTHON), str(ROOT/'temporal_stage2_report.py'), str(OUT)], capture_output=True, text=True, encoding='utf-8').stdout)
'''
export_cell = '''# CELL 5 — nén kết quả
import glob
ZIP = Path(f'/kaggle/working/temporal_stage2_shard{SHARD_ID}.zip')
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(str(OUT/'stage2_shard*.jsonl')): z.write(f, Path(f).name)
display(FileLink(str(ZIP)))
'''
setup = B.setup.replace('assert torch.cuda.device_count()>=2;', 'assert torch.cuda.device_count()>=1;')
nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', setup), cell('code', write_cell), cell('code', run_cell),
                cell('code', report_cell), cell('code', export_cell)],
      'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_temporal_stage2.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target.name, target.stat().st_size)

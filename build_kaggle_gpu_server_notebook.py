"""Build the Kaggle notebook that exposes the loaded Qwen2.5-Omni model as a remote GPU job server (ngrok)."""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import build_svhalluc_single_sample_prompt_lab as B

cell = B.cell
files = {'speech_gate.py': (ROOT / 'local_qwen' / 'speech_gate.py').read_text(encoding='utf-8'),
         'kaggle_rootcause.py': (ROOT / 'kaggle_rootcause.py').read_text(encoding='utf-8'),
         'kaggle_gpu_server.py': (ROOT / 'kaggle_gpu_server.py').read_text(encoding='utf-8')}
for s in files.values(): ast.parse(s)

intro = """# GPU từ xa cho Claude — nạp Qwen2.5-Omni một lần, nhận việc qua ngrok

**Cách dùng (theo đúng mẫu `example_GPU_encode_service.py` bạn đã dùng):**
1. Add-ons → Secrets → có sẵn `NGROK_AUTHTOKEN` và **bật Attach** cho notebook này.
2. GPU **T4 ×2**, Internet **On**, **Save & Run All**. (Dùng **tài khoản phụ**; xem lưu ý bên dưới.)
3. Sau ~4 phút log sẽ in **URL công khai** và **TOKEN**. Dán hai dòng đó cho tôi; tôi ghi vào `remote_config.json` rồi gọi GPU này.
4. Server chạy tới khi hết giờ phiên (tối đa 12 giờ với Save & Run All). Hết giờ hoặc bạn dừng notebook thì tắt.

**Nó làm gì.** Một server nhỏ (thư viện chuẩn, không cần FastAPI) giữ model Qwen2.5-Omni-3B FP16 trên 2 GPU, cùng toàn bộ hàm của phòng thí nghiệm (`scores`, `masked_forward`, `patch_sweep`, `eager_attention`, `run_item`, `cut`, `fetch`, …). Tôi gửi các đoạn Python ngắn, nó chạy trong namespace đó và trả stdout + `RESULT`; kết quả lớn ghi vào `/kaggle/working/...` và tôi tải về. Có hàng đợi một việc một lúc, `/restart` để giải phóng GPU khi treo (notebook tự khởi động lại server).

**Bảo mật.** Token ngẫu nhiên dài, so sánh hằng thời gian; server chỉ nghe `127.0.0.1`, chỉ ngrok chuyển tới; tải file bị khoá trong thư mục kết quả. **Nhưng nếu ai có cả URL lẫn token thì chạy được mã tuỳ ý trong notebook này**, vì vậy đừng chia sẻ log. Đặt `ALLOW_EXEC = False` nếu chỉ muốn kiểm tra kết nối.

**Chính sách Kaggle.** Tôi không xác minh được việc mở tunnel từ notebook Kaggle có nằm trong điều khoản sử dụng hay không. Mẫu bạn đã dùng cũng nêu rằng Kaggle có thể tự ngắt khi rảnh lâu. Không có vòng lặp chống ngắt trong notebook này (cố ý). Dùng tài khoản phụ."""

net_cell = '''# CELL 0 — kiểm tra Internet
import socket
for host in ('pypi.org', 'huggingface.co', 'ngrok.com'):
    try:
        socket.create_connection((host, 443), timeout=8).close(); print('OK  ', host)
    except OSError as e:
        raise RuntimeError('Không kết nối được ' + host + '. Bật Internet: Session options -> Internet = On.') from e
'''
write_cell = '# CELL 2 — ghi file\nGS_FILES = ' + repr(files) + '\n' + \
    "for _n,_s in GS_FILES.items():(ROOT/_n).write_text(_s,encoding='utf-8')\nprint('Đã ghi:',list(GS_FILES))\n"

serve_cell = '''# CELL 3 — khởi động server + ngrok, giữ notebook sống
ALLOW_EXEC = True      # False = chỉ kiểm tra kết nối, không nhận mã
PORT = 8765
import secrets, subprocess, time, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyngrok'], check=True)
from pyngrok import ngrok

def ngrok_token():
    t = os.environ.get('NGROK_AUTHTOKEN', '')
    if t: return t
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('NGROK_AUTHTOKEN')
    except Exception as e:
        raise RuntimeError('Không đọc được Kaggle Secret NGROK_AUTHTOKEN (' + type(e).__name__ + '). Add-ons -> Secrets -> thêm NGROK_AUTHTOKEN và bật Attach cho notebook này.')

TOKEN = secrets.token_urlsafe(32)
env = {**os.environ, 'GPU_SERVER_TOKEN': TOKEN, 'GPU_SERVER_ALLOW_EXEC': '1' if ALLOW_EXEC else '0', 'PYTHONUNBUFFERED': '1'}
cmd = [str(PYTHON), str(ROOT/'kaggle_gpu_server.py'), str(ROOT), str(OUT), str(PORT)]
SERVER_LOG = OUT / 'server.log'

def start_server():
    f = open(SERVER_LOG, 'a', encoding='utf-8')
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

def wait_ready(timeout=900):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/ping', timeout=3); return True
        except Exception:
            time.sleep(5)
    return False

server = start_server()
print('Đang nạp model (2–4 phút)...', flush=True)
if not wait_ready():
    print(SERVER_LOG.read_text()[-3000:]); raise RuntimeError('server không lên được')
ngrok.set_auth_token(ngrok_token())
public_url = ngrok.connect(PORT, 'http')
print('=' * 70)
print('  URL   :', public_url)
print('  TOKEN :', TOKEN)
print('=' * 70, flush=True)

# giữ notebook sống; tự khởi động lại server khi /restart hoặc khi nó chết
while True:
    time.sleep(30)
    if server.poll() is not None:
        print('server dừng (mã', server.returncode, ') -> khởi động lại', flush=True)
        server = start_server()
        if not wait_ready(): print('khởi động lại thất bại; xem server.log', flush=True)
'''
setup = B.setup.replace('assert torch.cuda.device_count()>=2;', 'assert torch.cuda.device_count()>=1;')
nb = {'cells': [cell('markdown', intro), cell('code', net_cell), cell('code', setup), cell('code', write_cell), cell('code', serve_cell)],
      'metadata': B.nb['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'kaggle_svhalluc_gpu_server.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target.name, target.stat().st_size)

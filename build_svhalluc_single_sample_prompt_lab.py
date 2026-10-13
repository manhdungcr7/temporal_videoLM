"""Build the editable single-example Kaggle notebook, with a persistent model worker."""
import ast
import csv
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LAB=ROOT/'research_topics_learning'/'single_sample_prompt_lab'

# Embed the direct-video wrapper around official Qwen utilities.

if not (ROOT/'results.zip').exists():          # temporal_videoLM repo: the old semantic-track log is not shipped;
    history=json.loads((LAB/'default_sample_history.json').read_text(encoding='utf-8'))   # use the saved copy
else:
  with zipfile.ZipFile(ROOT/'results.zip') as archive:
    name=next(n for n in archive.namelist() if n.endswith('/predictions.csv'))
    rows=list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))
  historical=next(p for p in rows if p.get('item_id')=='svh_0630' and p.get('method')=='AV_baseline')
  assert historical['prediction']!=historical['gold'] and historical['correct'].lower()=='false'
  history={k:historical[k] for k in ('item_id','question','prediction','gold','prompt','method','correct')}
  history['raw']=historical['prediction']
  history['condition']='question_only_historical_AV_baseline'
  history['source_zip']='results.zip'
  history['source_member']=name
  history['note']='Historical question-only AV_baseline log: prediction A, gold B. The notebook freshly reruns the baseline and the edited prompt on this sample; outputs can vary with runtime/model preprocessing.'
  (LAB/'default_sample_history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')


def cell(kind,source):
    c={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':ast.parse(source);c.update(execution_count=None,outputs=[])
    return c


intro="""# SVHalluc — xem một mẫu, sửa prompt và chạy lại ngay

**Dùng chế độ Edit trên Kaggle, GPU T4 ×2 + Internet. Chạy từng cell từ trên xuống một lần; sau đó chỉ chạy lại cell PROMPT khi sửa prompt.**
Không cần Save & Run All, API key hay upload file phụ. Notebook dùng Qwen2.5-Omni-3B Thinker. Model sống trong worker suốt session nên sửa prompt không nạp lại model. Mỗi lượt dùng hội thoại mới.

## Đầu vào video — chỉ thay đổi prompt

Baseline nhận **video gốc + câu hỏi gốc**. Lượt prompting nhận **chính video đó + prompt có câu hỏi gốc**. Video được gửi bằng type="video"; thư viện Qwen xử lý video và audio. Notebook không dùng ASR, captioner, truy hồi, dữ liệu ngoài hay chỉnh logits. Qwen lấy mẫu frame theo mặc định; độ phân giải mỗi frame được giới hạn ở 160×28×28 pixel để chạy trên T4. Hai lượt dùng chung media đã xử lý.

Thư viện vẫn giải mã và chuẩn hóa nội dung MP4 trước khi đưa vào model. Đây là tiền xử lý đầu vào chính thức, không phải một phương pháp trích bằng chứng riêng.

## Vì sao prompt đã được rút gọn?

Prompt hiện tại dựa trên việc đọc 120 response và đối chiếu hình ảnh theo thời gian cùng transcript phục vụ phân tích. Nó phân biệt vật hiện diện, sự kiện hành động–vật thể và sự tương ứng ngữ nghĩa giữa speech với hình ảnh; giữ vai trò hỗ trợ nhận diện của audio nhưng không coi lời nói là bằng chứng hình ảnh. Prompt yêu cầu đáp án, nghĩa Yes/No và một câu bằng chứng phải nhất quán. Các transcript và hình dùng để phân tích không được thêm vào đầu vào Qwen. Hiệu quả của prompt mới chưa được đo trên GPU.

Prompting không bảo đảm đáp án đúng. Hãy xem câu trả lời thô và video; một mẫu chỉ giúp phát hiện lỗi prompt, không đủ để kết luận hiệu quả tổng quát.

Mặc định **svh_0630**, FGSA: “Is the thinly sliced scallions visible in the video?” (A) Yes (B) No. Log `results.zip` ghi nhận baseline chỉ có câu hỏi đã chọn A, trong khi đáp án chuẩn là B. Notebook sẽ chạy lại baseline và prompt mới trên cùng video vì kết quả có thể thay đổi theo cấu hình suy luận. Thông tin log cũ chỉ hiện trong UI, không gửi vào model.

Đây là một prompt duy nhất nhận video có audio. Không có nhánh inference tách nguồn hay mô hình phụ. Mẫu này dùng để chỉnh prompt, không dùng để ước lượng accuracy tổng quát."""

sources={p.name:p.read_text(encoding='utf-8') for p in LAB.glob('*.py') if p.name in ('lab_worker.py','lab_client.py','lab_native_video.py','lab_setup.py')}
for source in sources.values():ast.parse(source)
setup="""# CELL 1 — cài môi trường. Chạy một lần cho mỗi session Kaggle.
import os, sys, json, subprocess, importlib.metadata, venv, time, html, hashlib, zipfile, shutil
from pathlib import Path
from IPython.display import display, Video, Markdown, HTML, FileLink
assert Path('/kaggle').exists(), 'Dùng Kaggle Edit với GPU T4 x2 và Internet.'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONIOENCODING']='utf-8'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ['FORCE_QWENVL_VIDEO_READER']='decord'  # torchvision.io.read_video was removed in Kaggle's build.
ROOT=Path('/kaggle/working/svhalluc_prompt_lab_runner');ROOT.mkdir(parents=True,exist_ok=True)
if 'OUT' not in globals():
    OUT=Path('/kaggle/working/svhalluc_prompt_lab_'+time.strftime('%Y%m%d_%H%M%S'))
    OUT.mkdir(parents=True,exist_ok=False)
SOURCES="""+repr(sources)+"""
for name,source in SOURCES.items():(ROOT/name).write_text(source,encoding='utf-8')
HISTORICAL="""+repr(history)+"""
(OUT/'historical_default.json').write_text(json.dumps(HISTORICAL,ensure_ascii=False,indent=2),encoding='utf-8')
if 'LAB' in globals() and LAB.is_alive():
    raise RuntimeError('Model đang chạy. Chỉ chạy cell PROMPT; nếu cần cài lại, gọi LAB.close() trước.')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from lab_setup import prepare_environment
ENV=ROOT/'env'
PYTHON,PIP_COMMAND=prepare_environment(ENV)
print('Venv sẵn sàng (không dùng ensurepip):',PYTHON,flush=True)
constraints=[]
for package in ('torch','torchvision','torchaudio'):
    try:constraints.append(package+'=='+importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:pass
(ROOT/'keep_torch.txt').write_text('\\n'.join(constraints)+'\\n',encoding='utf-8')
packages=['transformers==4.57.6','accelerate==1.12.0','huggingface-hub==0.36.0','qwen-omni-utils==0.0.8','decord==0.6.0',
          'av==16.0.1','librosa==0.11.0','soundfile>=0.12,<0.14','numpy>=2,<2.4',
          'Pillow>=10,<13','sentencepiece>=0.2,<0.3','protobuf>=5,<7']
command=[*PIP_COMMAND,'install','--only-binary=:all:',
         '--constraint',str(ROOT/'keep_torch.txt'),*packages]
with (OUT/'install.log').open('a',encoding='utf-8') as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',errors='replace')
    for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
if code:raise RuntimeError('Package installation failed; see '+str(OUT/'install.log'))
if not shutil.which('ffmpeg'):
    subprocess.run(['apt-get','-qq','update'],check=True)
    subprocess.run(['apt-get','-qq','install','-y','ffmpeg'],check=True)
subprocess.run([str(PYTHON),'-c','import torch, av, decord; from qwen_omni_utils.v2_5.vision_process import get_video_reader_backend; from transformers import Qwen2_5OmniThinkerForConditionalGeneration; assert torch.cuda.device_count()>=2; assert get_video_reader_backend()=="decord"; print("torch",torch.__version__,"decord",decord.__version__)'],check=True)
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from lab_client import PromptLab
print('Setup ready. Kết quả và log:',OUT)
"""

load="""# CELL 2 — nạp model một lần; chạy lại cell này không tạo model thứ hai.
if 'LAB' in globals() and LAB.is_alive():
    print('Dùng lại model đã nạp. PID:',LAB.process.pid)
else:
    print('Đang tải/nạp Qwen2.5-Omni-3B. Có thể mất vài phút ở lần đầu.',flush=True)
    LAB=PromptLab(PYTHON,ROOT,OUT)
    print('Model sẵn sàng. PID:',LAB.process.pid)
print(LAB.ready)
"""

select="""# CELL 3 — chọn đúng MỘT mẫu. Đổi ITEM_ID rồi chạy lại cell này để chuyển mẫu.
ITEM_ID='svh_0630'
DATASET='zcs15/SVHalluc'
DATA_REVISION='211b4f14ab66ae9557b69e302d1f090e0e610839'
code='from huggingface_hub import hf_hub_download; import sys; print(hf_hub_download(sys.argv[1],sys.argv[2],repo_type="dataset",revision=sys.argv[3]))'
def download_dataset_file(name):
    completed=subprocess.run([str(PYTHON),'-c',code,DATASET,name,DATA_REVISION],
        check=True,capture_output=True,text=True,encoding='utf-8')
    return Path(completed.stdout.strip().splitlines()[-1])
if 'ANNOTATIONS' not in globals():
    ANNOTATIONS=json.loads(download_dataset_file('samples.json').read_text(encoding='utf-8'))
if not ITEM_ID.startswith('svh_') or not ITEM_ID[4:].isdigit():raise ValueError('Ví dụ ID: svh_0847')
index=int(ITEM_ID[4:])
if not 0<=index<len(ANNOTATIONS):raise ValueError('ID ngoài tập annotation')
ROW=dict(ANNOTATIONS[index])
if ROW['data_type']!='semantic':raise ValueError('Chọn mẫu thuộc semantic GSA/FGSA/CMSB')
QUESTION=ROW['question'];GOLD=ROW['answer']
if GOLD not in ('A','B') or '(A) Yes (B) No' not in QUESTION:
    raise ValueError('Notebook này yêu cầu câu hỏi A=Yes, B=No')
VIDEO_PATH=download_dataset_file(ROW['video_path'])
MEDIA=LAB.request({'cmd':'sample','item_id':ITEM_ID,'question':QUESTION,'path':str(VIDEO_PATH)})
BASELINE_RESULT=None  # Cached baseline only belongs to the currently selected sample.
if 'RUN_HISTORY' not in globals():RUN_HISTORY=[]
display(Markdown('## '+ITEM_ID+' — '+ROW['task_type_abbr']+'\\n\\n**Câu hỏi:** '+QUESTION))
display(Video(str(VIDEO_PATH),embed=True,width=720))
print('Đầu vào:',MEDIA['path'])
print('Qwen official preprocessing:',MEDIA['frame_count'],'frames; shape:',MEDIA['video_shape'],
      '| sampled FPS:',round(MEDIA['sampled_fps'],4),'| audio:',round(MEDIA['audio_duration_s'],2),'s')
print('Hai lượt dùng cùng media gốc; không có ASR/caption hay ảnh bổ sung.')
details='Nhãn benchmark: '+html.escape(GOLD)
if ITEM_ID==HISTORICAL['item_id']:
    details+='; log baseline cũ chỉ có câu hỏi: '+html.escape(HISTORICAL['prediction'])+' (sai); log mới từ '+html.escape(HISTORICAL['source_zip'])+'. Notebook sẽ chạy lại baseline hiện tại.'
display(HTML('<details><summary>Nhãn và response trước — chỉ hiển thị UI, không vào prompt</summary>'+details+'</details>'))
"""

prompt_source=(LAB/'prompt.txt').read_text(encoding='utf-8')
# Keep answer parsing in the editable cell too, so a still-running older worker
# does not require another model load just to read the new final-answer format.
worker_source=sources['lab_worker.py']
parser_source='\n\n'.join(ast.get_source_segment(worker_source,node) for node in ast.parse(worker_source).body
    if isinstance(node,ast.FunctionDef) and node.name in ('parse_final','final_format_issues'))
prompt_cell = (
    "# CELL PROMPT: edit PROMPT_TEMPLATE and rerun this cell only.\n"
    "# Keep {question}; the notebook inserts the selected sample question.\n"
    'PROMPT_TEMPLATE = """' + prompt_source + '"""\n'
    "MAX_NEW_TOKENS=384  # Increase only if the answer is truncated.\n"
    "import re\n" + parser_source + "\n"
    "\n"
    "assert 'LAB' in globals() and LAB.is_alive(), 'Run model setup first.'\n"
    "assert 'MEDIA' in globals(), 'Select a sample first.'\n"
    "assert MEDIA['item_id']==ITEM_ID and MEDIA['question']==QUESTION, 'Rerun sample selection after changing ITEM_ID.'\n"
    "if not PROMPT_TEMPLATE.strip():raise ValueError('Prompt is empty')\n"
    "if BASELINE_RESULT is None:\n"
    "    print('Running baseline once for this sample...',flush=True)\n"
    "    BASELINE_RESULT=LAB.request({'cmd':'baseline','max_new_tokens':MAX_NEW_TOKENS})\n"
    "if PROMPT_TEMPLATE.count('{question}')>1:raise ValueError('Use at most one {question} placeholder')\n"
    "PROMPT=(PROMPT_TEMPLATE.replace('{question}',QUESTION) if '{question}' in PROMPT_TEMPLATE\n"
    "        else PROMPT_TEMPLATE+'\\n\\nORIGINAL QUESTION:\\n'+QUESTION)\n"
    "print('Running prompt in a new conversation; model stays loaded...',flush=True)\n"
    "RESULT=LAB.request({'cmd':'prompt','prompt':PROMPT,'max_new_tokens':MAX_NEW_TOKENS})\n"
    "RESULT['prediction']=parse_final(RESULT['raw'],RESULT['complete'])\n"
    "RESULT['format_issues']=final_format_issues(RESULT['raw'],RESULT['complete'])\n"
    "Path(RESULT['result_path']).write_text(json.dumps(RESULT,ensure_ascii=False,indent=2),encoding='utf-8')\n"
    "display(Markdown('### Baseline: original question only\\n\\n**Answer in output:** '+str(BASELINE_RESULT['prediction'])+\n"
    "    '\\n\\n**Raw output:**\\n\\n'+BASELINE_RESULT['raw']))\n"
    "display(Markdown('### Prompted answer'))\n"
    "display(Markdown(RESULT['raw']))\n"
    "print('Baseline A/B:',BASELINE_RESULT['prediction'],'| Prompt A/B:',RESULT['prediction'],\n"
    "      '| Gold:',GOLD,'| Prompt correct:',RESULT['prediction']==GOLD if RESULT['prediction'] else 'not scored')\n"
    "print('Tokens:',RESULT['generated_tokens'],'/',MAX_NEW_TOKENS,\n"
    "      '| EOS:',RESULT['complete'],'| Time:',round(RESULT['latency_s'],2),'s')\n"
    "for issue in RESULT['format_issues']:print('OUTPUT CHECK:',issue)\n"
    "if RESULT['prediction'] is None:print('No clear final A/B choice; inspect raw response or token limit.')\n"
    "entry={'item_id':ITEM_ID,'run_id':RESULT['run_id'],'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),\n"
    "       'prediction':RESULT['prediction'],'gold':GOLD,'baseline_prediction':BASELINE_RESULT['prediction'],\n"
    "       'complete':RESULT['complete'],'result_path':RESULT['result_path']}\n"
    "RUN_HISTORY.append(entry)\n"
    "(OUT/'ui_history.json').write_text(json.dumps(RUN_HISTORY,ensure_ascii=False,indent=2),encoding='utf-8')\n"
    "print('Saved:',RESULT['result_path'])\n"
)
export="""# Tùy chọn — xuất các prompt đã thử và output/log. Không dừng model.
ZIP=Path('/kaggle/working/svhalluc_single_sample_prompt_lab_results.zip')
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.rglob('*'):
        if path.is_file():archive.write(path,path.relative_to(OUT.parent).as_posix())
    for name,source in SOURCES.items():archive.writestr('runner_sources/'+name,source)
display(FileLink(str(ZIP)))
print('Đổi prompt và chạy lại cell PROMPT để tiếp tục. File ZIP:',ZIP)
"""

diagnostic_intro="""## Chẩn đoán nguyên nhân trên mẫu đang chọn — chạy cell dưới một lần

Chạy 7 kiểm tra nhỏ: câu hỏi gốc với cả hai nguồn, chỉ hình ảnh, chỉ audio; đảo vị trí Yes/No; mô tả hình ảnh không có tên vật thể ở câu hỏi với/không có audio; câu hỏi gốc chỉ hình ảnh ở độ phân giải cao hơn.

Đây là kiểm tra độ phụ thuộc vào nguồn dữ liệu, câu hỏi và độ phân giải. Phương pháp prompting ở cell trước vẫn nhận MP4 gốc có audio. Cell này không đưa transcript hay nhãn vào model, không thay logits và không gọi model phụ. Mỗi lượt dùng hội thoại mới; phép tăng độ phân giải giữ nguyên cách lấy mẫu theo thời gian. Nếu hết VRAM, lỗi được ghi lại, không giảm chất lượng âm thầm.

Nếu worker đang chạy là bản cũ, cell sẽ nâng cấp và nạp model lại đúng một lần; các kiểm tra sau dùng chung model. Có thể chép toàn bộ cell này từ notebook mới sang session đang chạy.

So sánh **nghĩa Yes/No**, vì thứ tự A/B thay đổi ở một kiểm tra. Việc bỏ audio làm đổi câu trả lời là bằng chứng về độ phụ thuộc vào audio, chưa phải bằng chứng head attention nào gây lỗi. Nếu câu trả lời vẫn Yes khi chỉ có hình ảnh, cần xét nhận diện thị giác, ảnh hưởng tên trong câu hỏi và độ chắc chắn của nhãn. Không nhắc đến một vật trong mô tả trung tính không chứng minh vật đó vắng mặt.

Paper SVHalluc, mục 3.3/Table 4, báo cáo FGSA của **Qwen3-Omni** tăng 79,50% → 84,91% khi chỉ nhận hình ảnh. Kết quả đó là động cơ cho kiểm tra này; không suy thành nguyên nhân của **Qwen2.5-Omni-3B** trên một mẫu cụ thể.
"""
diagnostic_cell='DIAGNOSTIC_RUNNER_SOURCES='+repr(sources)+'\n'+(LAB/'diagnostic_cell.py').read_text(encoding='utf-8')

notes="""## Cách sử dụng sau lần chạy đầu

- **Sửa prompt:** chỉ sửa/chạy lại cell PROMPT. Không cần chạy setup, model hay chọn mẫu.
- **Đổi mẫu:** sửa ITEM_ID ở cell 3, chạy cell 3, rồi chạy cell PROMPT. Trọng số giữ nguyên.
- **Output bị cắt:** tăng MAX_NEW_TOKENS ở cell PROMPT; output chưa có EOS không được chấm.
- Lời giải thích do model sinh không phải phép đo đáng tin cậy về quá trình suy luận hay attention. Dùng cell chẩn đoán để kiểm tra các giả thuyết trước khi sửa prompt tiếp.
- Baseline dùng câu hỏi gốc. Lượt prompting thêm hướng dẫn nhưng nhận cùng video. Đáp án được đọc từ lựa chọn A/B model tự sinh; không ép token hay chỉnh logits.
- Cả hai lượt baseline/prompt dùng chung media cache, giới hạn 160×28×28 pixel/frame và greedy decoding. Cell chẩn đoán ghi rõ từng thay đổi đầu vào và khôi phục media gốc sau mỗi lượt.
- Baseline lịch sử dùng cách chấm cũ; chỉ dùng để chọn mẫu. Notebook chạy lại baseline hiện tại, nên không so trực tiếp hai kiểu đo.
- Muốn trả VRAM khi kết thúc: chạy LAB.close() trong cell riêng. Nếu session bị reset, cần chạy lại cell nạp model.

Notebook chỉ dùng ý tưởng kiểm tra bằng chứng trong một prompt; đây không phải triển khai đầy đủ Chain-of-Verification."""

nb={'cells':[cell('markdown',intro),cell('code',setup),cell('code',load),cell('code',select),
    cell('markdown','## Cell chỉnh prompt\n\nSửa nội dung tiếng Anh bên dưới và chạy lại cell này. Baseline được lưu lại cho cùng mẫu.\n'),
    cell('code',prompt_cell),cell('markdown',diagnostic_intro),cell('code',diagnostic_cell),
    cell('markdown',notes),cell('code',export)],
    'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python','version':'3.13'}},'nbformat':4,'nbformat_minor':5}
target=ROOT/'kaggle_svhalluc_single_sample_prompt_lab.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print(target,target.stat().st_size)


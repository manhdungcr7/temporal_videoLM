"""Summarize saved micro8 results; never conflate skipped M3/M0 with QA evaluation."""
import collections, json, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
def load(p,default=None):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default
def rows(tag):
    return sorted([r for s in (1,2) for r in load(D/f'{tag}_samples_s{s}.json',[])],key=lambda r:r['idx'])
original=rows('original');fixed=rows('logicfix');corrected=rows('corrected')
assert len({r['idx'] for r in original})==len(original)
assert len({r['idx'] for r in fixed})==len(fixed)
methods=[('Native','mc_pred_native'),('CAOR attention only','mc_pred_caor_attn_only'),('CLRV direct','mc_pred_routed_direct'),('CLRV full','mc_pred_routed_fwd'),('CLRV bridge only','mc_pred_rollout_bridge_only'),('Native rollout → bridge','mc_pred_native_rollout_bridge'),('Empty rollout → bridge','mc_pred_empty_rollout_bridge')]
def score(rr,k):return sum(r[k]==r['gold_letter'] for r in rr)
tables=[]
for tag,rr in [('Gốc',original),('Sửa lỗi interface v1',fixed),('Sửa đầy đủ interface v2',corrected)]:
    tables += [f'## {tag}: {len(rr)}/8 mẫu đã lưu','', '| Cấu hình | Đúng | Sai |','|---|---:|---:|']
    for name,k in methods:
        c=score(rr,k);tables.append(f'| {name} | {c}/{len(rr)} | {len(rr)-c} |')
    tables+=['','| ID | Task/gold | Native | Direct | Bridge | Full | Native→bridge | Gate |','|---|---|---|---|---|---|---|---:|']
    for r in rr:
        tables.append('| '+' | '.join(map(str,[r['idx'],r['task']+'/'+r['gold_letter'],r['mc_pred_native'],r['mc_pred_routed_direct'],r['mc_pred_rollout_bridge_only'],r['mc_pred_routed_fwd'],r['mc_pred_native_rollout_bridge'],r['confidence_gate_gamma_true_query']]))+' |')
    rescue=sum(r['mc_pred_native']!=r['gold_letter'] and r['mc_pred_routed_fwd']==r['gold_letter'] for r in rr)
    harm=sum(r['mc_pred_native']==r['gold_letter'] and r['mc_pred_routed_fwd']!=r['gold_letter'] for r in rr)
    tables += ['',f'Cứu/làm hỏng so native: {rescue}/{harm}. Gate >0 trên {sum(r["confidence_gate_gamma_true_query"]>0 for r in rr)}/{len(rr)} mẫu.','']
tables+=['## Theo task/ground truth','', '| Task/gold | N | Native đúng/sai | Gốc đúng/sai | Sửa v1 đúng/sai | Sửa v2 đúng/sai |','|---|---:|---|---|---|---|']
for task,gold in [('TA','A'),('TA','B'),('TF','A'),('TF','B'),('TF','C'),('CMTB','A'),('CMTB','B'),('CMTB','C')]:
    oo=[r for r in original if (r['task'],r['gold_letter'])==(task,gold)]
    ff=[r for r in fixed if (r['task'],r['gold_letter'])==(task,gold)]
    cc=[r for r in corrected if (r['task'],r['gold_letter'])==(task,gold)]
    n=len(oo)
    def cell(rr,k):
        c=score(rr,k)
        return f'{c}/{len(rr)-c}' if rr else 'Chưa đo'
    tables.append(f'| {task}/{gold} | {n} | {cell(oo,"mc_pred_native")} | {cell(oo,"mc_pred_routed_fwd")} | {cell(ff,"mc_pred_routed_fwd")} | {cell(cc,"mc_pred_routed_fwd")} |')
header='''# CLRV v8 — thí nghiệm micro8 trên hai server, 2026-10-17

Source gốc `fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895`. Bộ 8 ID cố định: 1449, 2150, 1929, 2054 / 1956, 1513, 2264, 1821; rotation 0, mỗi mẫu dưới 45 giây. Mỗi server chạy một shard 4 ID; đây không phải hai replicas của QA8. Không huấn luyện, không chỉnh ngưỡng theo đáp án.

M0 gốc tái lập qua trên CPU ở cả hai server. M0 CUDA gốc lỗi index GPU vào buffer CPU; wrapper chuẩn hóa đầu vào extraction về CPU như đường M3 thực tế. Các kết quả CPU do Argon gửi được giữ riêng (`argon_cpu_results.json`, M3 `skipped_no_live_model`); chúng không phải kết quả QA.

Nhánh gốc giữ công thức của source. Sửa v1 bổ sung ánh xạ endpoint→physical modality, gate foreground theo modality endpoint, phân loại whole-word qua offsets của tokenizer để không coi mảnh BPE là entity, ưu tiên connector quan hệ hơn `and` trong tên sự kiện, và sửa nhánh gate0 bỏ qua KV cache. Sửa v2 bổ sung tách mệnh đề mở đầu Relative/When theo dấu câu, và nhận diện các tính từ modality chung visual/auditory/acoustic/audiovisual. Các source/hash và thay đổi runtime lưu riêng; không đổi các hệ số gate/CLRV hay dùng nhãn QA để chọn tham số. Đây là sửa giao diện sau chẩn đoán trên development micro8, không phải đánh giá held-out độc lập.

Đối chứng bổ sung: native rollout rồi cùng bridge scorer; bridge khi không có rollout; permutation đã ánh xạ về nhãn gốc; order-reversed latent routing; trace-only không media. Order-reversed routing không bảo đảm văn bản sinh thực sự đảo quan hệ, nên phải đọc văn bản control trước khi diễn giải TV.

'''
tail='''
## Giới hạn và phản biện cơ chế

- Micro8 đã được xem qua nhiều vòng; không có TF/B hoặc CMTB/C. Không coi đây là đánh giá 40 mẫu, held-out mới hoặc bằng chứng tổng quát hóa.
- M0 dùng unit tokenizer theo nguyên từ và activations nhân tạo; quá trình thực thi phát hiện sai khác quan trọng khi chuyển sang tokenizer BPE/activations Qwen thật.
- `P_null` là heuristic. Independent CPU audit cho thấy có thể đổi coupling L1=1.282 mà null gần như không đổi dưới biên uniform; mean lag phụ thuộc các marginal; gate “conjunctive” có thể mở 0.45 khi hai từ chỉ được khớp ở hai slots khác nhau. Xem `independent_math_audit.json`.
- Null features được ghi ra audit nhưng gamma routing gốc được tính từ sharpness và excess gate; không được mặc định mọi cải thiện null score đều đã đi vào QA.
- Điểm freeform từ regex được giữ như diagnostic, không dùng làm semantic oracle. Cần chấm nghĩa độc lập và dữ liệu event/interval/absence có annotation cho bài toán thực tế.
- Lag sử dụng lưới nội suy; không coi là timestamp vật lý đã xác nhận. Giữ nguyên dữ liệu và kích thước media của baseline frozen40.
'''
component=[r for s in (1,2) for r in load(D/f'component_ablation_s{s}.json',[])]
if component:
    tables+=['','## Ablation trên bản sửa đầy đủ (không rollout)','', '| Cấu hình cuối decoder | Đúng/tổng |','|---|---:|']
    for k in component[0]['conditions']:
        tables.append(f'| {k} | {sum(r["conditions"][k]["pred"]==r["gold"] for r in component)}/{len(component)} |')
    tables+=['',f'Replay combined direct khớp xác suất đã lưu: max error={max(r["direct_replay_max_probability_error"] for r in component)}. CLC-only vẫn dùng các counterfactual CAOR để xây hướng CLC; nó chỉ bỏ CAOR và CT ở lượt decode cuối. Kết quả ablation được đọc exploratory, không chọn lại hệ số.','']
summary40=load(D/'summary40.json')
if summary40 and summary40['all40']['n']==40:
    h=summary40['all40']['correct']
    header=header.replace('Source gốc',f'**Kết quả mở rộng40: native {h["native"]}/40, CAOR-only {h["CAOR_only"]}/40, CLC-only {h["CLC_only"]}/40, CLRV full-direct {h["full_direct"]}/40.** Xem [bảng đúng/sai từng task/gold](report40.md) và [phản hồi gửi Yoshua](feedback_to_yoshua_bengio.md). Full rollout bridge chỉ chạy trên micro8.\n\nSource gốc',1)
(D/'report.md').write_text(header+'\n'.join(tables)+tail,encoding='utf-8')
summary={tag:dict(n=len(rr),scores={name:score(rr,k) for name,k in methods},gate_positive=sum(r['confidence_gate_gamma_true_query']>0 for r in rr)) for tag,rr in [('original',original),('logicfix',fixed),('corrected',corrected)]}
(D/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=True))

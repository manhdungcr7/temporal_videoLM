"""Verify and summarize the fixed per-layer inference on the frozen eight IDs."""
import ast
import csv
import hashlib
import json
import re
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/pcebr_v5_layerfix_micro8'
PRIMARY='F2_layer_inputbanks_block'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,value):(D/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
samples={s['idx']:s for s in manifest['selected']}
source=(ROOT/'results/pcebr_stage0_v5/stage0_pcebr_v5_original.py').read_bytes()
runner=(ROOT/'server_jobs/pcebr_v5_layerfix_micro8.py').read_bytes()
tree=ast.parse(source);metric_ns={'re':re};nodes=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('STOPWORDS','GENERIC_EVASION_PATTERNS') for t in node.targets):nodes.append(node)
    if isinstance(node,ast.FunctionDef) and node.name in ('content_tokens','audit_freeform_generation_quality'):nodes.append(node)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original_regex_metric>','exec'),metric_ns)
rows=[];archives=[];weight_hashes=[]
for server in (1,2):
    part=read(f'results_s{server}.json');protocol=read(f'protocol_s{server}.json');audit=read(f'archive_audit_s{server}.json');idx=read(f'archive_index_s{server}.json')
    assert len(part)==4 and [r['idx'] for r in part]==[s['idx'] for s in manifest[f'server{server}'][:4]]
    assert protocol['source_sha256']==hashlib.sha256(source).hexdigest()
    assert protocol['runner_sha256']==hashlib.sha256(runner).hexdigest()
    assert protocol['manifest_sha256']==manifest['manifest_sha256'] and protocol['primary']==PRIMARY
    assert audit['hooks']['forward_hooks']==audit['hooks']['pre_hooks']==0 and audit['hooks']['cuda_ok']
    weight_hashes.append(audit['weights_sha256'])
    path=D/f'raw_artifacts_s{server}.tar.gz';assert hashlib.sha256(path.read_bytes()).hexdigest()==idx['sha256']
    expected={f['name']:f for f in audit['files']}
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)==idx['files']
        for member in members:
            assert member.isfile() and not member.issym() and '..' not in Path(member.name).parts
            data=tar.extractfile(member).read();meta=expected[member.name]
            assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
    archives.append(dict(**idx,all_sha_verified=True))
    prior=json.loads((ROOT/f'results/pcebr_stage0_v5/m3_audit_s{server}.json').read_text(encoding='utf-8'))
    old={a['result']['idx']:a['result'] for a in prior}
    for r in part:
        s=samples[r['idx']];assert r['gold']==s['gold'] and r['task']==s['task']
        assert 'freeform' in r and r['native_replay_error']==0 and r['original_v5_probability_replay_error']<1e-7
        assert r['conditions']['native']['prediction']==old[r['idx']]['mc_pred_native']
        assert r['conditions']['v5_original']['prediction']==old[r['idx']]['mc_pred_routed_fwd']
        assert r['conditions']['zero_scale_replay']['logits']==r['conditions']['native']['logits']
        r['archived_original_text']=old[r['idx']]['freeform_32tok_routed']
        r['native_text_matches_archive']=r['freeform']['native']['text']==old[r['idx']]['freeform_32tok_native']
        for name,value in r['freeform'].items():
            assert len(value['token_ids'])<=32
            value['regex_audit']=metric_ns['audit_freeform_generation_quality'](value['text'],s['duration'],r['task'],old[r['idx']]['gold_answer_text'])
        rows.append(dict(server=server,**r))
assert len(rows)==8 and len({r['idx'] for r in rows})==8
assert weight_hashes[0]==weight_hashes[1]
names=list(rows[0]['conditions'])
def metric(name):
    correct=sum(r['conditions'][name]['correct'] for r in rows)
    rescued=[r['idx'] for r in rows if not r['conditions']['native']['correct'] and r['conditions'][name]['correct']]
    harmed=[r['idx'] for r in rows if r['conditions']['native']['correct'] and not r['conditions'][name]['correct']]
    return dict(correct=correct,wrong=8-correct,total=8,rescued=rescued,harmed=harmed,
                changed=[r['idx'] for r in rows if r['conditions']['native']['prediction']!=r['conditions'][name]['prediction']],
                mean_tv_native=statistics.mean(r['conditions'][name]['tv_native'] for r in rows))
summary=dict(total=8,rotation=0,primary=PRIMARY,conditions={n:metric(n) for n in names},cells={},
    native_replay_max_error=max(r['native_replay_error'] for r in rows),original_v5_probability_max_error=max(r['original_v5_probability_replay_error'] for r in rows),
    mean_gate=statistics.mean(r['gate'] for r in rows),freeform_regex={},native_freeform_archive_matches=sum(r['native_text_matches_archive'] for r in rows))
for task,letters in [('TA','AB'),('TF','ABC'),('CMTB','ABC')]:
    for letter in letters:
        part=[r for r in rows if r['task']==task and r['gold']==letter]
        summary['cells'][f'{task}/{letter}']=dict(total=len(part),correct={n:sum(r['conditions'][n]['correct'] for r in part) for n in names})
for name in ('native',PRIMARY):
    summary['freeform_regex'][name]=dict(aligned=sum(r['freeform'][name]['regex_audit']['semantic_gold_aligned'] for r in rows),
        generic=sum(r['freeform'][name]['regex_audit']['is_generic_or_evasive_non_answer'] for r in rows),
        timestamp_beyond_duration=sum(r['freeform'][name]['regex_audit']['has_hallucinated_timestamp_beyond_duration'] for r in rows),total=8)
def tv(r,a,b):return sum(abs(x-y) for x,y in zip(r['conditions'][a]['probabilities'],r['conditions'][b]['probabilities']))/2
summary['mean_mid_swap_tv']=statistics.mean(tv(r,PRIMARY,'F2_reversed') for r in rows)
summary['mean_late_swap_tv']=statistics.mean(tv(r,'F2_late','F2_late_reversed') for r in rows)
summary['max_layer_parity_error']=max(max(p['even_error'],p['odd_error']) for r in rows for p in r['layer_parity'].values())
save('summary.json',summary);save('per_sample.json',rows)
save('completion_audit.json',dict(summary=summary,archives=archives,target_layer_weights_identical_across_servers=True))
with (D/'per_sample.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['idx','task','gold']+names);writer.writeheader()
    for r in rows:writer.writerow(dict(idx=r['idx'],task=r['task'],gold=r['gold'],**{n:r['conditions'][n]['prediction'] for n in names}))

lines=['# OB-TVOR v5 — sửa VO theo từng tầng, micro8, 2026-10-16','',
f"**Native {summary['conditions']['native']['correct']}/8; v5 gốc {summary['conditions']['v5_original']['correct']}/8; bản sửa chính F2 {summary['conditions'][PRIMARY]['correct']}/8.**",'',
'Đúng tám ID micro cũ, bốn mỗi shard, rotation0. Không dùng kết quả micro để chọn scale/layers. Giữ scale .045×gate, sáu tầng giữa14/16/18/20/22/24, salience H18/H27, patch selection, grid clock và hàm bivector gốc. Không huấn luyện thêm, không chạy QA40/reserve400 trong lượt này. Hai server chạy các mẫu khác nhau, không phải replicas.','',
'Proposal không nhất quán: phần lập luận yêu cầu T_VO của chính mỗi tầng, nhưng lời nhắn cuối ghi T_VO tầng18. Lượt này kiểm chứng cách hiểu từng tầng. Không coi việc sửa code là bằng chứng chẩn đoán lệch không gian là nguyên nhân duy nhất.','',
'## Các nhánh đã cố định trước khi chạy','',
'- F1: dùng banks cũ từ output block18; thay VO18 bằng VO của mỗi tầng đích, tạo một vector riêng cho mỗi tầng; giữ hook cuối block.',
'- **F2, bản sửa chính:** lấy media states ở input của chính mỗi tầng đích, trước input_layernorm; giữ cùng patch indices và phép residual pooling gốc; dùng LN/V/O của tầng đó; tạo vector riêng và giữ hook cuối block.',
'- F3: dùng nguyên vectors F2, nhưng tangent update residual sau attention, trước MLP. Thực hiện bằng hook attention output và delta tương ứng; đây là diễn giải vị trí can thiệp khác, được báo cáo riêng.',
'- F2 reversed/content-only/random: kiểm tra kênh thời gian, nội dung và một đối chứng vector ngẫu nhiên với cùng strength/layers/rows. Random chỉ một fixed seed idx+101, không phải mean ba seeds như control Stage-0 cũ.',
'- F2 late/reversed: tương tự với sáu tầng28–33; chấm sensitivity, không tự coi sensitivity là correctness.',
'- Zero-scale và v5-original replay: kiểm tra native và cấu hình cũ khớp trước khi so bản sửa.','',
'VO ở đây vẫn là phép biến đổi vector tổng hợp qua LN/V/O và RepeatGQA, không tái tạo attention weights theo query/head. Pooling và LN không giao hoán nói chung; không gọi nó là exact native-attention output. Các banks lấy từ native prepass, không được tính lại sau từng steering update. M1/M2 chưa có kết nối QA trong proposal, nên không tự thêm pipeline learned mới.','',
'## MC: mọi cấu hình','',
'| Cấu hình | Đúng | Sai | Cứu so native | Làm sai thêm | Mean TV/native |','|---|---:|---:|---|---|---:|']
for name,m in summary['conditions'].items():
    lines.append(f"| {name} | {m['correct']} | {m['wrong']} | {m['rescued']} | {m['harmed']} | {m['mean_tv_native']:.6f} |")
lines+=['','## Từng mẫu','',
'| ID | Task/gold | Native | V5 gốc | F1 | F2 chính | F3 attention | F2 đảo | F2 content | F2 random |','|---|---|---|---|---|---|---|---|---|---|']
short=['native','v5_original','F1_layer_vo18banks_block',PRIMARY,'F3_layer_inputbanks_attention_residual','F2_reversed','F2_content_only','F2_random']
for r in rows:lines.append('| '+ ' | '.join([str(r['idx']),r['task']+'/'+r['gold']]+[r['conditions'][n]['prediction'] for n in short])+' |')
lines+=['','## Đúng/sai theo nhóm','',
'| Task/gold | N | Native đúng/sai | V5 gốc đúng/sai | F2 đúng/sai |','|---|---:|---|---|---|']
for cell,c in summary['cells'].items():
    vals=[str(c['correct'][n])+'/'+str(c['total']-c['correct'][n]) if c['total'] else 'Chưa đo' for n in ('native','v5_original',PRIMARY)]
    lines.append('| '+' | '.join([cell,str(c['total'])]+vals)+' |')
lines+=['','TF: A=Past,B=Present,C=Future. TA A=Yes,B=No. CMTB letters là nội dung lựa chọn của từng câu. Các cell N=0 không được suy rộng thành accuracy0%.','',
'## Kết luận từ lượt sửa','',
'Cả ba bản sửa F1/F2/F3 đều5/8, không cứu hoặc làm hỏng câu đúng theo MC. F2/F3 đổi ID2150 từ A/Past sang B/Present, nhưng gold là C/Future: sai→sai. F2 content-only cũng đổi đúng câu này sang B; chưa thấy lợi ích về đáp án từ kênh order. Các nhánh reverse/random/late cũng5/8, không được diễn giải là bằng chứng temporal reasoning được khôi phục.','',
f"F2 tác động lên xác suất mạnh hơn v5 gốc: mean TV/native={summary['conditions'][PRIMARY]['mean_tv_native']:.6f} so {summary['conditions']['v5_original']['mean_tv_native']:.6f}. Mid/late swap-TV ratio≈{summary['mean_mid_swap_tv']/summary['mean_late_swap_tv']:.3f}. Tác động lớn hơn hoặc sensitivity cao hơn chưa đem lại câu trả lời đúng hơn.",'',
'Lượt này sửa và kiểm chứng một sai khác triển khai cụ thể theo cách hiểu VO từng tầng. Nó chưa xác nhận lệch không gian là nguyên nhân chính gây lỗi, hoặc rằng chỉ sửa mapping/vị trí hook là đủ. Không dùng kết quả này để bác bỏ mọi phương pháp latent temporal reasoning; cũng chưa có cơ sở tăng quy mô ablation chỉ vì parity qua.','',
'## Câu trả lời không có choices','',
'32 tokens, greedy, cùng quy tắc bỏ options và vị trí token của v5 gốc. Chỉ native và F2 được sinh mới; văn bản v5 gốc được dẫn từ archive và ghi rõ. Giữ vectors từ prepass như đường sinh gốc.','',
f"Regex alignment: {json.dumps(summary['freeform_regex'])}. Đây là heuristic có các phản ví dụ phủ định/sai event đã được ghi ở Stage-0 v5; không phải semantic QA ground truth độc lập.",'']
lines += ['Đọc trực tiếp văn bản cho thấy ít nhất hai giới hạn mà số regex không phản ánh: ID1821 đổi mốc so sánh từ thời điểm speech sang thời điểm boiling liquid xuất hiện, nhưng vẫn được chấm aligned vì từ after. ID1513 vẫn nói after dù gold Past, đồng thời kết thúc ở “speech is at182.” trên clip13.074s. Số182 không có đơn vị và nằm cuối đoạn32 tokens bị cắt, nên không tự gán nhãn timestamp182giây; regex timestamp không bắt được chuỗi này. Các nhận xét này đối chiếu nội dung câu trả lời với câu hỏi/gold, chưa là annotation AV độc lập.','']
for r in rows:
    lines += [f"**ID{r['idx']} — {r['task']}/{r['gold']}**",'', 'Native: '+r['freeform']['native']['text'],'',
              'F2: '+r['freeform'][PRIMARY]['text'],'','V5 gốc (archive): '+r['archived_original_text'],'']
lines+=['## Kiểm toán','',
f"Native logits replay max error={summary['native_replay_max_error']}; v5-original probability replay max error={summary['original_v5_probability_max_error']:.9g}; zero-scale logits khớp native8/8. Native freeform khớp archive{summary['native_freeform_archive_matches']}/8.",
f"Mỗi trong12 tầng/mẫu được tính lại VO từ bank input đã lưu và xây lại vector đúng từng tensor trên server. Max parity error={summary['max_layer_parity_error']:.9g}. Trọng số LN/V/O của12 tầng có SHA giống nhau giữa hai runtime.",
f"Mean swap TV mid={summary['mean_mid_swap_tv']:.6f}, late={summary['mean_late_swap_tv']:.6f}. Đây là sensitivity trước đảo coordinate trong vector, không phải physical AV time reversal hoặc QA accuracy trên video đảo.",
'Hai raw archives và từng entry được kiểm tra SHA local, không extract. Raw banks/vectors/selected indices và logits/probabilities được lưu; không archive toàn bộ activation tensor mọi tokens. Không còn forward/pre hooks; CUDA khỏe. Source gốc được giữ nguyên.',
'Lượt đầu lỗi device ở attention-residual branch do Accelerate chuyển hidden input qua GPU sau PyTorch pre-hook. Bản devicefix chuyển residual sang cùng device/dtype attention output. Chưa có sample nào hoàn tất chấm ở lượt lỗi; không đổi mẫu hoặc công thức. Source trước sửa và job logs được lưu tại device_error_attempt. Archive job xếp sau lượt lỗi cũng báo thiếu results; log được giữ.','',
'Kết quả là development micro đã được xem nhiều lần. Chỉ kết luận về các cấu hình và tám mẫu này; chưa đủ bằng chứng generalization, bảo toàn ngữ nghĩa/cú pháp hoặc khả năng được accept CVPR.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

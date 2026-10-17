import collections,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
def load(p,default=None):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default
manifest=load(R/'results/research40_manifest.json')
expected={r['idx']:r for r in manifest['selected']}
rows=sorted([r for s in(1,2) for r in load(D/f'clc40_samples_s{s}.json',[])],key=lambda r:r['idx'])
micro={1449,2150,1929,2054,1956,1513,2264,1821}
assert len({r['idx'] for r in rows})==len(rows)
for r in rows:
    assert r['idx'] in expected and r['gold']==expected[r['idx']]['gold'] and r['task']==expected[r['idx']]['task'] and r['duration']<=45
keys=('native','CAOR_only','CLC_only','full_direct')
def hits(rr,k):return sum(r[k]==r['gold'] for r in rr)
def stats(rr):
    return dict(n=len(rr),correct={k:hits(rr,k) for k in keys},
        clc_rescues=[r['idx'] for r in rr if r['native']!=r['gold'] and r['CLC_only']==r['gold']],
        clc_harms=[r['idx'] for r in rr if r['native']==r['gold'] and r['CLC_only']!=r['gold']])
summary=dict(all40=stats(rows),micro8=stats([r for r in rows if r['idx'] in micro]),outside_micro32=stats([r for r in rows if r['idx'] not in micro]))
text=['# V8 CLC-only — mở rộng frozen40, 2026-10-17','',
    f'Đã lưu {len(rows)}/40 mẫu. Cùng manifest frozen40, rotation 0, mỗi task/gold có 5 mẫu, mọi clip ≤45 giây. Hệ số CLC .042 giữ nguyên từ micro8; gamma theo pipeline đã sửa. Không chọn lại mẫu hoặc huấn luyện.',
    '', 'CLC-only được chọn sau exploratory ablation micro8 (6/8 so native5/8). Vì 40 có chứa 8 mẫu đó, báo cáo riêng 32 mẫu ngoài micro. Cả 40 là development set đã dùng trong các vòng trước, không phải test mới. Đây là decode trực tiếp: chưa chạy đầy đủ rollout bridge trên 40.',
    '', '| Cấu hình | Toàn bộ đúng/sai | Ngoài micro8 đúng/sai |','|---|---|---|']
outside=[r for r in rows if r['idx'] not in micro]
for k in keys:text.append(f'| {k} | {hits(rows,k)}/{len(rows)-hits(rows,k)} | {hits(outside,k)}/{len(outside)-hits(outside,k)} |')
text+=['','| Task/gold | N | Native đúng/sai | CAOR đúng/sai | CLC-only đúng/sai | Full direct đúng/sai |','|---|---:|---|---|---|---|']
for task,gold in [('TA','A'),('TA','B'),('TF','A'),('TF','B'),('TF','C'),('CMTB','A'),('CMTB','B'),('CMTB','C')]:
    rr=[r for r in rows if (r['task'],r['gold'])==(task,gold)]
    text.append('| '+task+'/'+gold+' | '+str(len(rr))+' | '+' | '.join(f'{hits(rr,k)}/{len(rr)-hits(rr,k)}' for k in keys)+' |')
text+=['',f"CLC-only cứu IDs: {summary['all40']['clc_rescues']}; làm hỏng IDs: {summary['all40']['clc_harms']}.",'',
    '| ID | Task/gold | Native | CLC | CAOR | Full direct | Gate |','|---|---|---|---|---|---|---:|']
for r in rows:text.append('| '+' | '.join(map(str,[r['idx'],r['task']+'/'+r['gold'],r['native'],r['CLC_only'],r['CAOR_only'],r['full_direct'],round(r['gamma'],4)]))+' |')
component=[r for s in(1,2) for r in load(D/f'component_ablation_s{s}.json',[])]
comp={r['idx']:r for r in component}
replay=[]
for r in rows:
    if r['idx'] in comp:
        a=r['probabilities']['CLC_only'];b=comp[r['idx']]['conditions']['CLC_only']['probabilities']
        replay.append(max(abs(x-y) for x,y in zip(a,b)))
summary['micro8_replay_probability_max_error']=max(replay,default=None)
text+=['',f"Micro8 overlap replay CLC max probability error: {summary['micro8_replay_probability_max_error']}.",
    '', 'Các kết quả này kiểm chứng một ablation của V8, không chứng minh cơ chế joint binding hoặc novelty. Cần kiểm tra lựa chọn luân phiên, interval/event ground truth, câu hỏi/contexts độc lập và absence hard negatives trước khi đánh giá tổng quát hóa.']
verification=load(D/'verification40.json')
if verification:
    text+=['','## Kiểm toán', '',
        f"Đủ40 ID, {verification['n_recording_origins']} recording origins, clip dài nhất {verification['max_duration']}s. Cả40 prompt tail tokenizer roundtrip khớp; backbone không đổi trên cả hai shard. Native replay max error={verification['native_replay_max_error']}; kernel-noop max error={verification['kernel_noop_max_error']}. {verification['cache_fallback_count']}/40 mẫu dùng full-pass fallback thay vì nới ngưỡng cache. Hai raw result archives được kiểm tra SHA local.",
        '', 'Wrapper mở rộng ban đầu thiếu gc trong namespace sau một mẫu/shard; đã sửa và resume các ID còn lại, giữ nguyên mẫu/công thức. Logs lỗi gốc được giữ. Native23/40 là baseline cùng lượt rotation0, không phải bảng cyclic-rotation cũ.']
(D/'report40.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
(D/'summary40.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

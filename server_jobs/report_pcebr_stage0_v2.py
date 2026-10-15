"""Local-only Stage-0 v2 report and archive/provenance checks."""
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'results/pcebr_stage0_v2'
def read(name):return json.loads((FOLDER/name).read_text(encoding='utf-8'))
def write(name,value):(FOLDER/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
main=[read(f'main_results_s{s}.json') for s in (1,2)]
progress=[read(f'progress_s{s}.json') for s in (1,2)]
controls=[read(f'controls_s{s}.json') for s in (1,2)]
instrument=[read(f'instrumentation_s{s}.json') for s in (1,2)]
replicas=dict(M1_native=progress[0]['avbind_native_pairs']==progress[1]['avbind_native_pairs'],
    M1_pooled=progress[0]['avbind_pooled_pairs']==progress[1]['avbind_pooled_pairs'],M2=progress[0]['M2']==progress[1]['M2'])
assert all(replicas.values()),replicas
original=(FOLDER/'stage0_pcebr_mechanism_check_v2_original.py').read_text(encoding='utf-8')
fixed=(FOLDER/'stage0_pcebr_mechanism_check_v2_devicefix.py').read_text(encoding='utf-8')
expected=original.replace('v_chunks = sorted(set(chunks[is_v].tolist()))',
    'local_is_v = is_v.to(h.device)\n            local_chunks = chunks.to(h.device)\n            v_chunks = sorted(set(local_chunks[local_is_v].tolist()))')
expected=expected.replace('h[0, is_v & (chunks < mid_c)]','h[0, local_is_v & (local_chunks < mid_c)]').replace('h[0, is_v & (chunks >= mid_c)]','h[0, local_is_v & (local_chunks >= mid_c)]').replace('h[0, is_v]','h[0, local_is_v]')
assert fixed==expected
manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
micro=[];m3a=[]
for s in (1,2):
    result=read(f'm3_devicefix_results_s{s}.json');audit=read(f'm3_audit_s{s}.json')
    assert [r['idx'] for r in result['per_sample']]==[r['idx'] for r in manifest[f'server{s}'][:4]]
    assert len(result['per_sample'])==4
    for row in result['per_sample']:micro.append(dict(shard=s,**row))
    m3a+=audit
assert max(x['native_replay_max_error'] for x in m3a)==0.0
archives=[]
for s in (1,2):
    index=read(f'archive_index_s{s}.json');audit=read(f'archive_audit_s{s}.json');path=FOLDER/f'raw_artifacts_s{s}.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    expected_files={x['name']:x for x in audit['files']}
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        assert len(members)==97
        for member in members:
            assert member.isfile() and not member.issym() and member.name.startswith('artifacts/') and '..' not in Path(member.name).parts
            name=member.name.split('/')[-1];raw=archive.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest()==expected_files[name]['sha256']
    archives.append(dict(server=s,files=len(members),bytes=path.stat().st_size,sha256=index['sha256'],verified=True))

lines=['PC-EBR Stage-0 v2 — kết quả và audit, 2026-10-15','',
    '**Chưa đủ bằng chứng thông qua Stage-0 để chạy A0–A7. M1 two-stream không hơn matched controls một cách thuyết phục; M2 có shortcut mạnh; M3 xác nhận sensitivity giữa các tầng, chưa xác nhận temporal correctness.**','',
    'M1/M2 chạy trên cả hai server, mọi kết quả trùng nhau. Ba seeds cố định, 42 lượt huấn luyện/server, cùng 202 train/50 held-out pairs từ 16/4 origins. '
    'M3 dùng 4 ID đầu mỗi shard, rotation 0, tổng 8 ID đúng manifest. Bản gốc M3 lỗi mask GPU 0/hidden GPU 1. '
    'Devicefix chỉ chuyển visual mask/chunk indices về device của hidden. CUDA context server1 sau lỗi bị illegal access; '
    '4 ID shard 1 được mã hóa/chạy tiếp trên server 2, không thay mẫu, M1/M2 không chạy lại. Native decoder replay max error=0.0.','',
    '**M1 — mọi condition và baseline**','',
    'Global/exact/multipos có 50 queries; nonnull có 32 queries theo substring filter trong code, không phải annotation tồn tại event. '
    'Với condition học: cột Global/Exact là số đúng của từng seed 20261015/16/17; Multipos/Nonnull là mean %. '
    'Các 50 queries đến từ 4 origins; 3 seeds hoặc 2 replicas không làm tăng số held-out độc lập.','']
csvrows=[]
for key,label in [('M1_native_pairs_crossed_factorial','Native post-media readout'),('M1_pooled_pairs_crossed_factorial','Pooled media')]:
    part=main[0][key]
    lines += [f'**{label}**','','| Condition | Global đúng/50 | Exact đúng/50 | Multipos % | Nonnull % |','|---|---|---|---:|---:|']
    for name,v in part['deterministic_baselines'].items():
        lines.append(f"| {name} | {v['global_hits']} | {v['within_exact_hits']} | {v['within_multipos_pct']:.1f} | {v['within_nonnull_pct']:.1f} |")
    for name,v in part['three_seed_factorial'].items():
        lines.append(f"| {name} | {v['global_hits_per_seed']} | {v['within_exact_per_seed']} | {v['mean_within_multipos_pct']:.1f} | {v['mean_within_nonnull_pct']:.1f} |")
        csvrows.append(dict(cache=label,condition=name,**v))
    lines.append('')
lines += [
    'Matched F2→F6: native giữ global 6/50, exact 12/50, multipos 14/50; chỉ đổi phân bố câu đúng nonnull từ 7/32 lên 9/32. '
    'Pooled F2 tốt hơn F6: global 14→8/50, exact 15→12/50, multipos 18→15/50, nonnull 12→10/32. '
    'Pooled F6 không hơn shuffle về exact (24.0% so 24.7%), multipos (30.0% so 31.3%) hoặc nonnull (31.2% bằng nhau). '
    'F3/F4 cải thiện localization nhưng đổi lại global thấp; đây vẫn là trade-off. F1/F2, F3/F4, F5/F6 có cùng các số đếm, chưa có bằng chứng hard-negative boost cải thiện trong budget này.','',
    'Hướng mới giữ query 2048-D và bỏ text-batch centering là cải tiến về protocol. Query-batch score check max error='+str(controls[1]['query_batch_invariance_max_error'])+'. '
    'Tổng 77,827 parameters khớp code. Nhưng `einsum(nd,ndr,nrd->nd)` chỉ tạo `x_d*sum_r down[d,r]*up[r,d]`, '
    'không phải full low-rank mixing `x@down@up`. Không sửa công thức đó trong mainrun vì sẽ đổi phương pháp.','',
    'Zero-sum advantage chỉ bảo toàn arithmetic mean score trong từng group, không bảo toàn global argmax hay softmax mass. '
    'Counterexample: ctx[.9,.8,.7,.6], adv[0,0,.4,-.4], adv có tổng 0 mỗi nhóm 2 slots nhưng global winner đổi 0→2. '
    'Vì vậy claim “không thể giảm global retrieval” không được bảo đảm bằng đại số.','',
    '**M2 — affine và synthetic retrieved-coordinate probes**','',
    'Part A: affine 2t+12.5 error 0; anchor coverage .2, target coverage 1; center antisymmetry error 0. '
    'Đây là unit check được sửa đúng ở transform thử. Không suy affine-invariance toàn LLM hoặc mọi phép scale/float32 có duration clamp.','',
    'Part B có 3,848 train và 1,072 held-out synthetic episodes, từ 962/268 ordered event pairs, mỗi pair 4 states. '
    'Mỗi state held-out có 268 episodes. Các episodes phụ thuộc cùng 4 held-out origins, không phải 1,072 AV samples độc lập.','',
    '| Probe | Đúng/1072 | Accuracy % | Before đúng/268 | During đúng/268 | After đúng/268 | Foil đúng/268 |',
    '|---|---:|---:|---:|---:|---:|---:|']
probes=read('archive_audit_s1.json')['probes'];m2=main[0]['M2_affine_and_retrieved_counterfactuals']['PartB_retrieved_counterfactuals']
for (name,value),probe in zip(m2.items(),probes):
    correct=probe['correct'];state=[probe['per_class'][str(c)]['correct'] for c in range(4)]
    lines.append(f"| {name} | {correct} | {value['overall_pct']:.1f} | "+' | '.join(map(str,state))+' |')
lines += ['','| Probe | A→V % | V→A % | V→V % | A→A % |','|---|---:|---:|---:|---:|']
for name,value in m2.items():
    lines.append('| '+name+' | '+' | '.join(str(value['by_modality_direction'][d]) for d in ('audio_to_video','video_to_audio','video_to_video','audio_to_audio'))+' |')
lines += ['',
    'Full 53.4% chỉ hơn coordinate-shuffle 51.3% là 2.1 điểm phần trăm, thua Phi-only 54.4%; During 17.2% rất thấp. '
    'Natural-unshifted evaluation được ghi docstring nhưng không có episode offset0 được label bằng order tự nhiên; '
    'vòng thực thi chỉ tạo 3 offsets từ true event centers và foils. Hai retrievals lấy từ hai pseudo descriptions riêng, không phải learned hop 2 conditioned on hop 1. '
    'Không dịch waveform/frames hoặc kiểm tra generated QA. Same-source “dịch target bank” vẫn giữ anchor bank cùng source không dịch, là pair-coordinate intervention chứ chưa thể coi một single-stream physical shift.','',
    '**Đối chứng không có semantic retrieval**','',
    'Dùng nguyên episode generator và linear probe training nhưng đặt mọi similarity=0: cả p1, p2 uniform, không có thông tin event matching.','',
    '| Uniform-retrieval control | Accuracy % | Before % | During % | After % | Foil % |','|---|---:|---:|---:|---:|---:|']
for name,value in controls[0]['uniform_retrieval_synthetic_probes'].items():
    lines.append('| '+name+' | '+str(value['overall_pct'])+' | '+' | '.join(str(value['by_state'][k]) for k in ('Before','During','After','Null_Foil'))+' |')
lines += ['',
    '**Uniform Phi/full 60.5%, cao hơn learned full 53.4%, với foil 100%.** '
    'Điểm này bác bỏ cách diễn giải rằng đạt điểm cao trên episode generator này đã chứng minh learned event retrieval/null evidence. '
    'Riêng ba relation states (bỏ foil), learned full đạt 49.0% so uniform 47.4%; phần lớn lợi thế overall của uniform đến từ foil. '
    'Nó chỉ ra shortcut trong construction/coordinate distributions; không chứng minh mọi mô hình quan hệ temporal vô dụng. '
    'Foils lấy từ origin khác chưa được annotation rằng event không tồn tại, và clock ranges/offsets có thể phân biệt label.','',
    '**M3 — kết quả đầy đủ micro-probe**','',
    '| Shard | ID | Task | TV late/native | TV mid/native | TV swap late | TV swap mid | TV mid shuffled/native |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in micro:
    lines.append(f"| {r['shard']} | {r['idx']} | {r['task']} | {r['tv_late_28_33']:.4f} | {r['tv_mid_14_24']:.4f} | {r['tv_swap_late_28_33']:.4f} | {r['tv_swap_mid_14_24']:.4f} | {r['tv_mid_shuffled']:.4f} |")
meanlate=statistics.mean(r['tv_swap_late_28_33'] for r in micro);meanmid=statistics.mean(r['tv_swap_mid_14_24'] for r in micro)
lines += ['',f'Tổng 8 ID: mean TV swap late={meanlate:.6f}, mid={meanmid:.6f}, ratio={meanmid/meanlate:.3f}×. '
    'Shard 1=3.37×, shard 2=3.26×, đạt dự đoán sensitivity ≥3× trên micro này. Đây không phải directional accuracy hoặc QA improvement.','',
    'Hai vector early/late sau trừ toàn visual mean gần như đối hướng: cosine≈−1 ở mọi 8 samples. '
    'Với 2 nhóm, các centered prototypes có dạng ±c*(mean early−mean late), nên swap thực tế là đổi dấu residual. '
    'Không có physical time shift. M3 không dùng trained M1 projector hoặc M2 Phi; nó mean-pool visual block 18 rồi cộng 14% norm vào '
    'mọi tokens sau media (gồm stem, choices, format). Vì vậy chưa kiểm chứng PC-EBR option-free path. '
    '`freeform_hook_ok` chỉ lặp lại MC forward và isfinite ở sample đầu, các samples khác default True; không gọi generate hoặc bỏ options.','',
    '**Kết luận gửi Yoshua**','',
    'V2 cải thiện protocol query independence, multipos reporting, affine unit check và thêm executable M3. '
    'Tuy nhiên phải sửa low-rank implementation nếu đó là ý định, kiểm tra global rank thay vì suy từ zero-sum, '
    'và thay M2 episode generator để uniform/coordinate-only/null controls không thắng bằng shortcut. '
    'M3 sensitivity hữu ích để chọn thí nghiệm tiếp, nhưng chưa xác nhận đúng event/time/source. '
    'Không chạy A0–A7 hoặc reserve400 trong lượt này.','',
    '**Artifacts và verification**','',
    'Có 97 raw PT files/server: 42 projector checkpoints, 48 retrieval score matrices, 2 episode tensors, 5 classifier probes. '
    '2,400 retrieval query rows/server được recompute đúng; hai archives SHA/từng entry được check local, không extract. '
    'M1/M2 replica results identical. M3 raw choice probabilities/native parity lưu JSON; CUDA errors và migration job logs được giữ. '
    'Không coi pseudo annotations là event ground truth hoặc 50 pairs/four origins là held-out final mới.']
(FOLDER/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
write('m1_table.json',csvrows)
with (FOLDER/'m3_per_sample.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(micro[0]));writer.writeheader();writer.writerows(micro)
write('completion_audit.json',dict(replicas=replicas,only_m3_mask_device_repair=True,archives=archives,
    m3_ids=[x['idx'] for x in micro],m3_native_max_error=0.0,m3_recovered_logical1_on_server2=True,
    model='Qwen/Qwen2.5-Omni-3B',train_pairs=202,heldout_pairs=50,heldout_origins=4,
    m2_heldout_episodes=1072,uniform_no_retrieval_full_pct=60.5,learned_full_pct=53.4,
    m3_swap_ratio=meanmid/meanlate,QA40_run=False,A0_A7_run=False,reserve400_run=False))
print(json.dumps(dict(report=str(FOLDER/'report.md'),M3_swap_ratio=meanmid/meanlate,replicas=replicas,archives=archives),indent=2))

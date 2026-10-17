import collections,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8'
m=json.loads((R/'results/research40_manifest.json').read_text())
all_rows=[];audits=[]
for s in(1,2):
    index=json.loads((D/f'clc40_archive_index_s{s}.json').read_text())
    path=D/f'clc40_results_s{s}.tar.gz'
    assert path.stat().st_size==index['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    with tarfile.open(path,'r:gz') as tar:
        completion=json.load(tar.extractfile('completion.json'))
        rows=json.load(tar.extractfile('samples.json'))
        inputs=json.load(tar.extractfile('input_audit.json'))
        assert completion['backbone_unchanged'] and completion['n_samples']==20
        assert len(rows)==20 and len(inputs)==20 and all(x['tail_tokenizer_roundtrip'] for x in inputs)
        local=json.loads((D/f'clc40_samples_s{s}.json').read_text(encoding='utf-8-sig'));assert local==rows
        all_rows+=rows
        audits.append(dict(server=s,n=20,archive_sha256=index['sha256'],backbone_unchanged=True,all_prompt_roundtrips_exact=True))
assert len(all_rows)==40 and len({r['idx'] for r in all_rows})==40
assert {r['idx'] for r in all_rows}=={r['idx'] for r in m['selected']}
counts=collections.Counter(r['task']+'/'+r['gold'] for r in all_rows)
assert len(counts)==8 and set(counts.values())=={5}
summary=json.loads((D/'summary40.json').read_text())
assert summary['micro8_replay_probability_max_error']==0.0
origins={PurePosixPath(r['video_path']).parts[-2] for r in m['selected']}
out=dict(shards=audits,n_unique_questions=40,n_recording_origins=len(origins),balanced_task_gold=dict(counts),
    micro8_clc_replay_max_probability_error=0.,sample_selection_unchanged=True,max_duration=max(r['duration'] for r in all_rows),
    native_replay_max_error=max(r['native_replay_err'] for r in all_rows),kernel_noop_max_error=max(r['kernel_noop_err'] for r in all_rows),
    cache_fallback_count=sum(not r['cache_used'] for r in all_rows))
(D/'verification40.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

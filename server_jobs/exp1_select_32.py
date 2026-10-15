"""Select fixed 32 Micro-Dev samples (<= 50.0s) strictly balanced across:
- TA: 4 'A' (Yes), 4 'B' (No) = 8
- TF: 4 'A' (Past), 4 'B' (Present), 4 'C' (Future) = 12
- CMTB: 4 'A', 4 'B', 4 'C' = 12
Total = 32 samples.
Partitioned evenly into Server 1 (16) and Server 2 (16).
"""
import json, random, av

rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))

target_counts = {
    ('TA', 'A'): 4,
    ('TA', 'B'): 4,
    ('TF', 'A'): 4,
    ('TF', 'B'): 4,
    ('TF', 'C'): 4,
    ('CMTB', 'A'): 4,
    ('CMTB', 'B'): 4,
    ('CMTB', 'C'): 4,
}

selected_by_cell = {k: [] for k in target_counts}
rejected = []

for (task, gold), count in target_counts.items():
    pool = [(i, r) for i, r in enumerate(rows)
            if r.get('data_type') == 'temporal' and r.get('task_type_abbr') == task and r.get('answer') == gold]
    random.Random(f'20261014_microdev_{task}_{gold}').shuffle(pool)
    for i, r in pool:
        if len(selected_by_cell[(task, gold)]) == count:
            break
        try:
            path = fetch(r['video_path'].replace(BS, '/'))
            with av.open(str(path)) as container:
                dur = float(container.duration / av.time_base) if container.duration else None
            if dur is None:
                dur = len(decode_audio(path)) / 16000
            if dur > 50.0:
                rejected.append(dict(idx=i, task=task, gold=gold, reason='dur > 50', dur=dur))
                continue
            item = dict(
                idx=i,
                task=task,
                gold=gold,
                duration=round(dur, 3),
                video_path=r['video_path'].replace(BS, '/'),
                question=r.get('question', ''),
                options=r.get('options', [])
            )
            selected_by_cell[(task, gold)].append(item)
        except Exception as e:
            rejected.append(dict(idx=i, task=task, gold=gold, reason=repr(e)))
    
    assert len(selected_by_cell[(task, gold)]) == count, f"Failed to find {count} samples for {task}/{gold}, found {len(selected_by_cell[(task, gold)])}"

# Partition 2 per cell to Server 1, 2 per cell to Server 2
server1_samples = []
server2_samples = []

for key in sorted(target_counts):
    items = selected_by_cell[key]
    server1_samples.extend(items[:2])
    server2_samples.extend(items[2:4])

print(f"Total Server 1 samples: {len(server1_samples)}")
print(f"Total Server 2 samples: {len(server2_samples)}")

manifest = {
    'seed': 20261014,
    'total': 32,
    'server1': server1_samples,
    'server2': server2_samples,
    'target_counts': {f"{k[0]}/{k[1]}": v for k, v in target_counts.items()},
    'all_selected': server1_samples + server2_samples
}

(OUT / 'micro_dev_32_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print("MANIFEST_SAVED_SUCCESSFULLY")
RESULT = manifest


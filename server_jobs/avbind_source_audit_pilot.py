"""New R9 diagnostic condition only; reuse completed R7 native baselines."""
import gc
import traceback

indices = ([1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER == 1
           else [2139,2327,1603,2097,1532,1580,2040,1666,2124])
rows = json.loads(fetch('samples.json').read_text())
previous = {r['idx']: r for r in (json.loads(line) for line in
            (OUT / f'avbind_independent_kv_r7_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip())}
path = OUT / f'avbind_source_audit_r9_s{LAB_SERVER}.jsonl'
records = {}
if path.exists():
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record['idx']] = record
for index in indices:
    if index in records and 'binding' in records[index]:
        continue
    row = rows[index]
    record = dict(idx=index, task=row['task_type_abbr'], gold=row['answer'],
                  base=previous[index]['base'], base_reused_from='R7 identical native prompt', diagnostic_only=True)
    try:
        src = cut(fetch(row['video_path'].replace(BS, '/')), 0, 10**4, 'full')
        question, options = parse_question(row['question'])
        result = NEURAL_NS['avbind_source_audit_answer'](
            src, question, options, OUT / f'avbind_differential_slots_{index}.pt',
            OUT / f'avbind_raw_values_{index}.pt', gain=1)
        record['binding'] = result['scores']
        record['binding_trace'] = result
    except Exception:
        record['error'] = traceback.format_exc()[-1800:]
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
        handle.flush()
    records[index] = record
    print('BIND_REAL_PILOT', json.dumps(dict(idx=index, task=record['task'], gold=record['gold'],
          predictions={m: 'ABCD'[int(np.argmax(record[m]))] for m in ('base','binding') if m in record},
          completed=sum('binding' in r for r in records.values()), target=len(indices), error=record.get('error'))), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
RESULT = dict(n=len(records), completed=sum('binding' in r for r in records.values()), file=path.name,
              diagnostic_only=True)

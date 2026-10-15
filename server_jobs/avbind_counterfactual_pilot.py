"""Resume R8 without re-running completed smoke samples or native baselines."""
import gc
import traceback

indices = ([1454,2396,1727,1647,2345,2348,2141] if LAB_SERVER == 1
           else [2139,2327,1603,2097,1532,1580,2040,1666,2124])
smoke_id = '8cea2842d1' if LAB_SERVER == 1 else '82b8b9dde0'
path = OUT / f'avbind_counterfactual_r8_s{LAB_SERVER}.jsonl'
previous = {r['idx']: r for r in
            (json.loads(line) for line in
             (OUT / f'avbind_independent_kv_r7_s{LAB_SERVER}.jsonl').read_text().splitlines()
             if line.strip())}
rows = json.loads(fetch('samples.json').read_text())
records = {}
if path.exists():
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record['idx']] = record

def persist(record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
        handle.flush()
    records[record['idx']] = record

def report(index, reused=False):
    counts = {}
    for record in records.values():
        key = record['task'] + '/' + record['gold']
        cell = counts.setdefault(key, {'base': {'correct': 0, 'wrong': 0},
                                       'binding': {'correct': 0, 'wrong': 0}, 'errors': 0})
        for method in ('base', 'binding'):
            if method in record:
                correct = 'ABCD'[int(np.argmax(record[method]))] == record['gold']
                cell[method]['correct' if correct else 'wrong'] += 1
        if 'binding' not in record:
            cell['errors'] += 1
    print('BIND_REAL_PILOT', json.dumps({'idx': index, 'reused': reused,
          'completed': sum('binding' in r for r in records.values()),
          'target': len(indices), 'groups': counts}), flush=True)

# Recover the finished smoke from its actual server record, never infer it again.
if not any(r.get('smoke_job') == smoke_id for r in records.values()):
    smoke = json.loads((OUT / 'uploads' / f'avbind_r8_smoke_s{LAB_SERVER}.json').read_text())
    assert smoke['id'] == smoke_id
    assert smoke['status'] == 'done', smoke['status']
    result = smoke['result']
    index = result['idx']
    assert index in indices and result['gain'] == result['fixed_coefficient'] == 1
    if index not in records or 'binding' not in records[index]:
        persist(dict(idx=index, task=previous[index]['task'], gold=previous[index]['gold'],
                     base=previous[index]['base'], binding=result['scores'],
                     binding_trace=result, smoke_job=smoke_id,
                     base_reused_from='R7 identical native prompt',
                     smoke_trace_limited=True))
    report(index, reused=True)

for index in indices:
    if index in records and 'binding' in records[index]:
        continue
    row = rows[index]
    record = dict(idx=index, task=row['task_type_abbr'], gold=row['answer'],
                  base=previous[index]['base'], base_reused_from='R7 identical native prompt')
    try:
        src = cut(fetch(row['video_path'].replace(BS, '/')), 0, 10**4, 'full')
        question, options = parse_question(row['question'])
        result = NEURAL_NS['avbind_counterfactual_answer'](
            src, question, options, OUT / f'avbind_differential_slots_{index}.pt',
            OUT / f'avbind_raw_values_{index}.pt', gain=1)
        record['binding'] = result['scores']
        record['binding_trace'] = result
    except Exception:
        record['error'] = traceback.format_exc()[-1800:]
    persist(record)
    report(index)
    gc.collect()
    torch.cuda.empty_cache()

RESULT = dict(n=len(records), completed=sum('binding' in r for r in records.values()),
              errors=sum('binding' not in r for r in records.values()), file=path.name)

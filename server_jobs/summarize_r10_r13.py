import json, glob
from pathlib import Path

for round_name in ['r10-bounded-readout', 'r11-native-timeline', 'r12-physical-clock', 'r13-context-preserved-clock']:
    files = glob.glob(f'results/{round_name}_s*.jsonl')
    records = []
    for f in files:
        for l in open(f, encoding='utf-8'):
            if l.strip():
                try: records.append(json.loads(l))
                except: pass
    by_idx = {}
    for r in records:
        if 'idx' in r: by_idx[r['idx']] = r
    unique_recs = list(by_idx.values())
    
    stats = {}
    for r in unique_recs:
        key = f"{r['task']}/{r['gold']}"
        stats.setdefault(key, {'total': 0, 'base': 0, 'binding': 0})
        stats[key]['total'] += 1
        gold_idx = ord(r['gold']) - 65
        if 'base' in r and isinstance(r['base'], list):
            if r['base'].index(max(r['base'])) == gold_idx: stats[key]['base'] += 1
        if 'binding' in r and isinstance(r['binding'], list):
            if r['binding'].index(max(r['binding'])) == gold_idx: stats[key]['binding'] += 1
            
    print(f'=== {round_name} (unique {len(unique_recs)}) ===')
    tot_base, tot_bind, tot_all = 0, 0, 0
    for k, v in sorted(stats.items()):
        print(f"  {k}: Base={v['base']}/{v['total']}, Binding={v['binding']}/{v['total']}")
        tot_base += v['base']
        tot_bind += v['binding']
        tot_all += v['total']
    pct_base = (tot_base / tot_all * 100) if tot_all else 0
    pct_bind = (tot_bind / tot_all * 100) if tot_all else 0
    print(f"  TOTAL: Base={tot_base}/{tot_all} ({pct_base:.1f}%), Binding={tot_bind}/{tot_all} ({pct_bind:.1f}%)")


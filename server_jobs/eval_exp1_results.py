import json
from collections import defaultdict
from pathlib import Path

def evaluate_files():
    records = []
    for s in [1, 2]:
        p = Path(f'results/exp1_microdev_s{s}.jsonl')
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    # Deduplicate by idx
    unique_records = {}
    for r in records:
        unique_records[r['idx']] = r
    rows = list(unique_records.values())

    print(f"=== Total Unique Evaluated Samples: {len(rows)} / 32 ===")

    methods = ['c0', 'c2_05', 'c2_10', 'c3']
    
    # Overall
    print("\n--- OVERALL ACCURACY ---")
    for m in methods:
        corr = sum(1 for r in rows if r.get(f'{m}_correct'))
        pct = corr / len(rows) * 100 if rows else 0
        print(f"  {m:8s}: {corr}/{len(rows)} ({pct:.1f}%)")

    # Per task
    by_task = defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r)

    for task in ['CMTB', 'TA', 'TF']:
        t_rows = by_task.get(task, [])
        print(f"\n--- Task {task} ({len(t_rows)} samples) ---")
        for m in methods:
            corr = sum(1 for r in t_rows if r.get(f'{m}_correct'))
            pct = corr / len(t_rows) * 100 if t_rows else 0
            print(f"  {m:8s}: {corr}/{len(t_rows)} ({pct:.1f}%)")
        
        # Per ground truth breakdown
        by_gold = defaultdict(list)
        for r in t_rows:
            by_gold[r['gold']].append(r)
        
        for gold in sorted(by_gold.keys()):
            g_rows = by_gold[gold]
            c0_c = sum(1 for r in g_rows if r.get('c0_correct'))
            c2_c = sum(1 for r in g_rows if r.get('c2_10_correct'))
            c3_c = sum(1 for r in g_rows if r.get('c3_correct'))
            print(f"    Gold {gold} ({len(g_rows)}): C0={c0_c}/{len(g_rows)} | C2_1.0={c2_c}/{len(g_rows)} | C3={c3_c}/{len(g_rows)}")

    # Detailed differences between C0 and C3
    print("\n--- SAMPLES WHERE C3 DIFFERS FROM C0 ---")
    diffs = [r for r in rows if r['c0_pred'] != r['c3_pred']]
    print(f"Total diffs: {len(diffs)}")
    for r in diffs:
        c0_mark = "OK" if r['c0_correct'] else "FAIL"
        c3_mark = "OK" if r['c3_correct'] else "FAIL"
        print(f"  idx={r['idx']} {r['task']}/{r['gold']}: C0={r['c0_pred']}({c0_mark}) -> C3={r['c3_pred']}({c3_mark}) [dur={r['duration']:.1f}s]")

if __name__ == '__main__':
    evaluate_files()


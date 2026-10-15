import json
from collections import defaultdict
from pathlib import Path

METHODS = [
    'C0_base',
    'C1_pmi_prior',
    'C2_sync_deconfound',
    'C3_vclock_pmi',
    'C4_coop_relay_L18_22',
    'C5_coop_relay_pmi',
    'C6_lstr_unified',
]

def argmax(lst):
    return lst.index(max(lst))

def analyze_round2():
    records = []
    for shard in [0, 1]:
        p = Path(f'results/round2_lstr_s{shard}of2.jsonl')
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

    print(f"=== Total Evaluated Samples in Round 2: {len(rows)} ===")

    # Overall Summary
    print("\n--- OVERALL ACCURACY ACROSS ALL METHODS ---")
    c0_overall = sum(argmax(r['C0_base']) == r['gold'] for r in rows) / len(rows) * 100
    for m in METHODS:
        corr = sum(argmax(r[m]) == r['gold'] for r in rows)
        pct = corr / len(rows) * 100
        delta = pct - c0_overall
        print(f"  {m:22s}: {corr:2d}/{len(rows)} ({pct:5.1f}%) | Delta vs C0: {delta:+5.1f}%")

    # Per Task
    by_task = defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r)

    print("\n--- PER TASK ACCURACY ---")
    for task in ['CMTB', 'TA', 'TF']:
        t_rows = by_task.get(task, [])
        print(f"\nTask {task} ({len(t_rows)} samples):")
        c0_t = sum(argmax(r['C0_base']) == r['gold'] for r in t_rows) / len(t_rows) * 100
        for m in METHODS:
            corr = sum(argmax(r[m]) == r['gold'] for r in t_rows)
            pct = corr / len(t_rows) * 100
            print(f"  {m:22s}: {corr:2d}/{len(t_rows)} ({pct:5.1f}%) | Delta: {pct - c0_t:+5.1f}%")

    # Per Label breakdown for TA and TF
    print("\n--- DETAILED BREAKDOWN FOR TA & TF ---")
    for task in ['TA', 'TF']:
        t_rows = by_task.get(task, [])
        by_gold = defaultdict(list)
        for r in t_rows:
            by_gold[r['ans']].append(r)
        print(f"\nBreakdown for {task}:")
        for g in sorted(by_gold.keys()):
            g_rows = by_gold[g]
            print(f"  Label {g} ({len(g_rows)} samples):")
            for m in ['C0_base', 'C1_pmi_prior', 'C3_vclock_pmi', 'C5_coop_relay_pmi', 'C6_lstr_unified']:
                c = sum(argmax(r[m]) == r['gold'] for r in g_rows)
                print(f"    {m:18s}: {c}/{len(g_rows)} ({c/len(g_rows)*100:5.1f}%)")

    # Per Label breakdown for TA and TF
    print("\n--- DETAILED BREAKDOWN FOR TA & TF ---")
    for task in ['TA', 'TF']:
        t_rows = by_task.get(task, [])
        by_gold = defaultdict(list)
        for r in t_rows:
            by_gold[r['ans']].append(r)
        print(f"\nBreakdown for {task}:")
        for g in sorted(by_gold.keys()):
            g_rows = by_gold[g]
            print(f"  Label {g} ({len(g_rows)} samples):")
            for m in ['C0_base', 'C1_pmi_prior', 'C3_vclock_pmi', 'C5_coop_relay_pmi', 'C6_lstr_unified']:
                c = sum(argmax(r[m]) == r['gold'] for r in g_rows)
                print(f"    {m:18s}: {c}/{len(g_rows)} ({c/len(g_rows)*100:5.1f}%)")

if __name__ == '__main__':
    analyze_round2()

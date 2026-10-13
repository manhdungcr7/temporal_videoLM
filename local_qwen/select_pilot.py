"""Select a small paired P10 screen: wrong cases plus correct controls by task."""
import json
from pathlib import Path
from analyze_job import answer

D = Path(__file__).parent
ROOT = Path(r"F:\A_star_paper_NLP_ACL")
samples = json.loads((ROOT / "data/svhalluc_benchmark/samples.json").read_text(encoding="utf-8"))
base = [json.loads(line) for line in (D / "base_pool72.jsonl").read_text(encoding="utf-8").splitlines() if line]
by_id = {int(r["idx"]): r for r in base}

# Include both GSA error directions and both FGSA error directions. CSMB's
# known baseline errors in this pool are all gold B, so use two such errors.
wrong_seed = {"GSA": [394, 158], "FGSA": [778, 532], "CSMB": [978, 1001]}
controls = []
for task, error_ids in wrong_seed.items():
    for idx in error_ids:
        r = by_id[idx]
        assert samples[idx]["task_type_abbr"] == task
        assert answer(r.get("text", "")) != r["gold"], f"Expected baseline error: {idx}"
    for gold in ("A", "B"):
        candidates = [r for r in base if samples[int(r["idx"])]["task_type_abbr"] == task
                      and r["gold"] == gold and int(r["idx"]) not in error_ids
                      and answer(r.get("text", "")) == gold]
        if not candidates:
            raise RuntimeError(f"No baseline-correct control for {task}, gold={gold}")
        controls.append(int(candidates[0]["idx"]))

ids = [i for task in wrong_seed for i in wrong_seed[task]] + controls
assert len(ids) == len(set(ids)) == 12
(D / "pilot_p10_ids.json").write_text(json.dumps(ids), encoding="utf-8")
print("P10 small screen: 12 samples")
for task in wrong_seed:
    group = [i for i in ids if samples[i]["task_type_abbr"] == task]
    print(f"\n{task}")
    for i in group:
        r = by_id[i]
        p = answer(r.get("text", ""))
        status = "baseline-wrong" if p != r["gold"] else "control-correct"
        print(f"  idx={i:4d} gold={r['gold']} baseline={p} {status} | {samples[i]['question']}")
print("\nSaved", D / "pilot_p10_ids.json")

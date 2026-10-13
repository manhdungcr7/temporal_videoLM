"""Validate the downloaded temporal subset and make a source-disjoint pilot split."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import av

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
DATA = ROOT / "data/svhalluc_benchmark"
OUT = ROOT / "local_qwen/temporal_pilot_split.json"
MAX_SECONDS = 15.0
SEED = 20261011

rows = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
indices = [i for i, r in enumerate(rows) if r.get("data_type") == "temporal"]
paths = sorted({rows[i]["video_path"] for i in indices})
stats = {}
missing = []
for rel in paths:
    path = DATA / rel
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(rel)
        continue
    with av.open(str(path)) as container:
        video = next(s for s in container.streams if s.type == "video")
        seconds = float(video.duration * video.time_base) if video.duration else float(container.duration / av.time_base)
        has_audio = any(s.type == "audio" for s in container.streams)
    stats[rel] = {"seconds": round(seconds, 3), "bytes": path.stat().st_size, "has_audio": has_audio}
if missing:
    raise RuntimeError(f"Missing {len(missing)} temporal videos; first: {missing[:3]}")
if any(not x["has_audio"] for x in stats.values()):
    raise RuntimeError("At least one temporal video has no audio stream")
(DATA / "temporal_video_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

# 4 samples per task per split. The rare TF future class is kept in both splits;
# remaining labels are distributed as evenly as the short-clip pool allows.
targets = {
    "TA": {"development": ["A", "A", "B", "B"], "confirmation": ["A", "A", "B", "B"]},
    # Future (C) is rare among short clips, so reserve one per split.
    "TF": {"development": ["A", "A", "B", "C"], "confirmation": ["A", "B", "B", "C"]},
    "CMTB": {"development": ["A", "B", "B", "C"], "confirmation": ["A", "A", "B", "C"]},
}
rng = random.Random(SEED)
pool = defaultdict(lambda: defaultdict(list))
for i in indices:
    row = rows[i]
    if stats[row["video_path"]]["seconds"] <= MAX_SECONDS:
        # Split by original YouTube video ID, not by question or clip filename.
        source_id = row["video_path"].replace("\\", "/").split("/")[2]
        pool[(row["task_type_abbr"], row["answer"])][source_id].append(i)
for sources in pool.values():
    for ids in sources.values():
        rng.shuffle(ids)

slots = []
for split in ("development", "confirmation"):
    for task, split_targets in targets.items():
        labels = split_targets[split]
        slots.extend((split, task, label) for label in labels)

def solve(remaining, used_sources, chosen):
    if not remaining:
        return chosen
    feasible = []
    for slot in remaining:
        split, task, label = slot
        candidates = [(src, ids[0]) for src, ids in pool[(task, label)].items() if src not in used_sources]
        if not candidates:
            return None
        rng.shuffle(candidates)
        feasible.append((len(candidates), slot, candidates))
    _, slot, candidates = min(feasible, key=lambda x: x[0])
    rest = list(remaining)
    rest.remove(slot)
    split, task, label = slot
    for src, idx in candidates:
        result = solve(rest, used_sources | {src}, chosen + [(split, task, label, src, idx)])
        if result is not None:
            return result
    return None

solution = solve(slots, set(), [])
if solution is None:
    raise RuntimeError("Could not construct balanced source-disjoint pilot; reduce samples per class")
manifest = {"seed": SEED, "max_clip_seconds": MAX_SECONDS, "protocol": "No original YouTube source ID occurs in both splits.", "splits": {}}
for split in ("development", "confirmation"):
    selected = [x for x in solution if x[0] == split]
    manifest["splits"][split] = [{"idx": idx, "task": task, "gold": label, "source_id": src,
                                  "video_path": rows[idx]["video_path"], "seconds": stats[rows[idx]["video_path"]]["seconds"]}
                                 for _, task, label, src, idx in sorted(selected, key=lambda x: (x[1], x[2], x[3]))]
dev_sources = {x["source_id"] for x in manifest["splits"]["development"]}
confirm_sources = {x["source_id"] for x in manifest["splits"]["confirmation"]}
assert dev_sources.isdisjoint(confirm_sources)
for split, examples in manifest["splits"].items():
    assert len(examples) == 12
    for task in targets:
        got = Counter(x["gold"] for x in examples if x["task"] == task)
        assert got == Counter(targets[task][split]), (split, task, got)
OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Validated {len(indices)} temporal rows / {len(paths)} clips; {sum(x['bytes'] for x in stats.values()) / 1024**3:.2f} GiB")
for split, examples in manifest["splits"].items():
    print(split, len(examples), "unique source IDs", len({x['source_id'] for x in examples}))
    for task in targets:
        print(" ", task, dict(Counter(x["gold"] for x in examples if x["task"] == task)))
print("Saved", OUT)

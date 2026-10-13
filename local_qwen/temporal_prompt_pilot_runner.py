"""Run paired baseline vs prompting inference on the locked temporal pilot IDs."""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import runner
from runner import ROOT, load, prepare_video, qwen_inputs, move_inputs

DATA = ROOT / "data/svhalluc_benchmark"
SPLIT_FILE = ROOT / "local_qwen/temporal_pilot_split.json"

def parse_choice(text):
    text = text.strip()
    match = re.search(r"(?:final\s+)?(?:answer|option|choice)\s*(?:is\s*)?[,=:]?\s*[\[(]?([ABC])\b", text, re.I)
    if not match:
        match = re.match(r"\s*[\[(]?([ABC])(?:[\]).,:\s]|$)", text, re.I)
    if not match:
        match = re.search(r"\b(?:yes|no|past|present|future)\b", text, re.I)
        if match:
            return {"yes": "A", "no": "B", "past": "A", "present": "B", "future": "C"}[match.group(0).lower()]
    return match.group(1).upper() if match else None

def summarize(records):
    by_task = {}
    for task in ("TA", "TF", "CMTB"):
        subset = [r for r in records if r["task"] == task and not r.get("error")]
        if not subset:
            continue
        base = sum(r["baseline_pred"] == r["gold"] for r in subset)
        prompted = sum(r["prompt_pred"] == r["gold"] for r in subset)
        fixed = sum(r["baseline_pred"] != r["gold"] and r["prompt_pred"] == r["gold"] for r in subset)
        regressed = sum(r["baseline_pred"] == r["gold"] and r["prompt_pred"] != r["gold"] for r in subset)
        by_task[task] = {"n": len(subset), "baseline_correct": base, "prompt_correct": prompted,
                         "fixed_wrong": fixed, "regressed_correct": regressed,
                         "baseline_accuracy": base / len(subset), "prompt_accuracy": prompted / len(subset)}
    return by_task

if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit("Usage: python temporal_prompt_pilot_runner.py <development|confirmation> <prompt.txt> <output.jsonl> [limit]")
    split, prompt_path, output_path = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) == 5 else None
    manifest = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split not in manifest["splits"]:
        raise ValueError(f"Unknown split: {split}")
    template = Path(prompt_path).read_text(encoding="utf-8")
    if template.count("{question}") != 1:
        raise ValueError("Prompt must contain exactly one {question}")
    rows = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    examples = manifest["splits"][split][:limit]
    output_path = Path(output_path)
    done = set()
    if output_path.exists():
        done = {json.loads(line)["idx"] for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    model, processor = load()
    import torch
    for pos, item in enumerate(examples, 1):
        idx = item["idx"]
        if idx in done:
            continue
        row = rows[idx]
        media = None
        record = {"idx": idx, "task": item["task"], "gold": item["gold"], "video_path": item["video_path"], "source_id": item["source_id"]}
        try:
            media = prepare_video(DATA / item["video_path"])
            for mode, text in (("baseline", row["question"]), ("prompt", template.replace("{question}", row["question"]))):
                inputs, _ = qwen_inputs(processor, text, media)
                inputs = move_inputs(inputs, model)
                started = time.time()
                with torch.inference_mode():
                    # Baseline may explain before giving its option, so allow a full short
                    # response. The method prompt explicitly requests a single letter.
                    max_tokens = 64 if mode == "baseline" else 8
                    # Transformers 4.57 moved greedy decoding to optional Hub custom code.
                    # Supplying its local built-in _sample avoids a needless network check;
                    # do_sample=False preserves deterministic greedy decoding.
                    generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                               use_audio_in_video=True, custom_generate=type(model)._sample)
                decoded = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                record[f"{mode}_text"] = decoded
                record[f"{mode}_pred"] = parse_choice(decoded)
                record[f"{mode}_seconds"] = round(time.time() - started, 2)
                record["input_tokens"] = int(inputs["input_ids"].shape[1])
                del generated, inputs
                torch.cuda.empty_cache()
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{pos}/{len(examples)} idx={idx} task={record['task']} gold={record['gold']} baseline={record.get('baseline_pred')} prompt={record.get('prompt_pred')} error={record.get('error')}", flush=True)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize(records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved", output_path, summary_path, flush=True)

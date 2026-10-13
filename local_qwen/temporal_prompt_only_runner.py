"""Run a revised prompt against the existing baseline rows, without recomputing baseline."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import runner
from runner import ROOT, load, prepare_video, qwen_inputs, move_inputs

DATA = ROOT / "data/svhalluc_benchmark"

if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit("Usage: python temporal_prompt_only_runner.py <baseline-paired.jsonl> <prompt.txt> <output.jsonl> [max_new_tokens]")
    baseline_path, prompt_path, output_path = map(Path, sys.argv[1:4])
    max_new_tokens = int(sys.argv[4]) if len(sys.argv) == 5 else 8
    template = prompt_path.read_text(encoding="utf-8")
    if template.count("{question}") != 1:
        raise ValueError("Prompt must contain exactly one {question}")
    rows = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = set()
    if output_path.exists():
        done = {json.loads(line)["idx"] for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    model, processor = load()
    import torch
    for pos, record in enumerate(records, 1):
        idx = record["idx"]
        if idx in done:
            continue
        row = rows[idx]
        try:
            media = prepare_video(DATA / row["video_path"])
            inputs, _ = qwen_inputs(processor, template.replace("{question}", row["question"]), media)
            inputs = move_inputs(inputs, model)
            start = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                           use_audio_in_video=True, custom_generate=type(model)._sample)
            record["prompt_text"] = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            record["prompt_seconds"] = round(time.time() - start, 2)
            del generated, inputs
            torch.cuda.empty_cache()
        except Exception as exc:
            record["prompt_text"] = ""
            record["error"] = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{pos}/{len(records)} idx={idx} task={record['task']} gold={record['gold']} baseline={record.get('baseline_pred')} prompt={record.get('prompt_text','')[:20]!r} error={record.get('error')}", flush=True)
    print("Saved", output_path, flush=True)

"""Paired 2-pass prompt-only probe on six balanced SVHalluc examples."""
import gc, hashlib, json, re, sys, time
from pathlib import Path

import torch
import runner
from runner import ROOT, load, prepare_video, qwen_inputs, move_inputs

OUT = Path(__file__).parent / "p49_counterevidence_audit_results.jsonl"
PROMPT_PATH = Path(__file__).parent / "p49_counterevidence_audit.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
IDS = [352, 166, 670, 614, 1109, 1254]


def parse_choice(text):
    text = text.strip()
    m = re.search(r"(?im)^\s*Decision\s*:\s*\(?([AB])\)?\b", text)
    if not m:
        m = re.match(r"\s*\(?([AB])\)?(?:[.)\s]|$)", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*(Yes|No)\b", text, re.I)
    return {"yes": "A", "no": "B"}.get(m.group(1).lower()) if m else None


def generate(model, proc, media, prompt, limit):
    inputs, _ = qwen_inputs(proc, prompt, media)
    inputs = move_inputs(inputs, model)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                             use_audio_in_video=True, num_beams=1, use_cache=True)
    txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0].strip()
    n_input = int(inputs["input_ids"].shape[1])
    del inputs, out
    torch.cuda.empty_cache()
    return txt, n_input


def main():
    rows = json.loads((ROOT / "data/svhalluc_benchmark/samples.json").read_text(encoding="utf-8"))
    done = {json.loads(line)["idx"] for line in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    model, proc = load()
    for idx in IDS:
        if idx in done:
            print(f"skip {idx}: already recorded", flush=True)
            continue
        row = rows[idx]
        media = prepare_video(ROOT / "data/svhalluc_benchmark" / row["video_path"])
        start = time.time()
        try:
            baseline_text, nbase = generate(model, proc, media, row["question"], 8)
            candidate = parse_choice(baseline_text)
            if candidate is None:
                raise RuntimeError(f"Could not parse baseline candidate: {baseline_text!r}")
            audit_prompt = PROMPT_TEMPLATE.replace("{question}", row["question"]).replace("{candidate}", candidate)
            audit_text, naudit = generate(model, proc, media, audit_prompt, 96)
            pred = parse_choice(audit_text)
            rec = {
                "idx": idx, "task": row["task_type_abbr"], "gold": row["answer"],
                "baseline": candidate, "baseline_text": baseline_text,
                "prompt": pred, "prompt_text": audit_text,
                "transition": "unscored" if pred is None else ("same" if pred == candidate else "changed"),
                "tokens_input_baseline": nbase, "tokens_input_audit": naudit,
                "sec": round(time.time() - start, 1), "prompt_sha256": PROMPT_SHA256,
            }
        except Exception as exc:
            rec = {"idx": idx, "task": row["task_type_abbr"], "gold": row["answer"],
                   "error": repr(exc), "sec": round(time.time() - start, 1),
                   "prompt_sha256": PROMPT_SHA256}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        del media
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

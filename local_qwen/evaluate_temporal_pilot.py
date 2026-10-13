"""Re-score saved raw paired outputs with conservative task-aware answer parsing."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
DATA = ROOT / "data/svhalluc_benchmark/samples.json"

def parse(text, task, question):
    t = text.strip()
    # Explicit labels are preferred. Handles common free-form baseline formats such as
    # "The visual action is, B, ..." and "the answer is, C".
    m = re.search(r"\b(?:final\s+)?(?:answer|option|choice|visual action)\s*(?:is\s*)?[,=:]?\s*[\[(]?([ABC])\b", t, re.I)
    if not m:
        m = re.match(r"\s*[\[(]?([ABC])(?:[\]).,:\s]|$)", t, re.I)
    if m:
        return m.group(1).upper()
    if task == "TA":
        m = re.match(r"\s*(yes|no)\b", t, re.I)
        return {"yes": "A", "no": "B"}[m.group(1).lower()] if m else None
    if task == "TF":
        m = re.search(r"\b(past|present|future)\b", t, re.I)
        return {"past": "A", "present": "B", "future": "C"}[m.group(1).lower()] if m else None
    # CMTB: only infer a choice from the generated text if exactly one option phrase is
    # repeated. If it mentions multiple choices, leave it unscored rather than guessing.
    options = re.findall(r"\(([ABC])\)\s*([^()]+?)(?=\s*\([ABC]\)|$)", question)
    matched = []
    low = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    for letter, option in options:
        phrase = re.sub(r"[^a-z0-9 ]+", " ", option.lower()).strip()
        if phrase and phrase in low:
            matched.append(letter)
    return matched[0] if len(set(matched)) == 1 else None

def summarize(records):
    summary = {}
    for task in ("TA", "TF", "CMTB"):
        rs = [r for r in records if r["task"] == task]
        if not rs:
            continue
        base_correct = sum(r["baseline_pred"] == r["gold"] for r in rs)
        prompt_correct = sum(r["prompt_pred"] == r["gold"] for r in rs)
        summary[task] = {
            "n": len(rs),
            "baseline_accuracy_all_n": round(base_correct / len(rs), 4),
            "prompt_accuracy_all_n": round(prompt_correct / len(rs), 4),
            "baseline_valid": sum(r["baseline_pred"] is not None for r in rs),
            "prompt_valid": sum(r["prompt_pred"] is not None for r in rs),
            "wrong_or_invalid_to_correct": sum(r["baseline_pred"] != r["gold"] and r["prompt_pred"] == r["gold"] for r in rs),
            "correct_to_wrong_or_invalid": sum(r["baseline_pred"] == r["gold"] and r["prompt_pred"] != r["gold"] for r in rs),
            "baseline_confusion": {f"{g}->{p}": sum(r["gold"] == g and r["baseline_pred"] == p for r in rs) for g in "ABC" for p in "ABC" if any(r["gold"] == g and r["baseline_pred"] == p for r in rs)},
            "prompt_confusion": {f"{g}->{p}": sum(r["gold"] == g and r["prompt_pred"] == p for r in rs) for g in "ABC" for p in "ABC" if any(r["gold"] == g and r["prompt_pred"] == p for r in rs)},
        }
    return summary

if __name__ == "__main__":
    path = Path(sys.argv[1])
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in records:
        q = rows[r["idx"]]["question"]
        r["baseline_pred"] = parse(r.get("baseline_text", ""), r["task"], q)
        r["prompt_pred"] = parse(r.get("prompt_text", ""), r["task"], q)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    out = summarize(records)
    path.with_suffix(".summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("Rows with invalid answer:", [(r["idx"], r["task"], r["baseline_pred"], r["prompt_pred"]) for r in records if r["baseline_pred"] is None or r["prompt_pred"] is None])

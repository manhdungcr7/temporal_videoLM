"""Compare one local Qwen job with the 72-sample baseline, by task."""
import json, re, sys
from pathlib import Path

D = Path(__file__).parent
ROOT = Path(r"F:\A_star_paper_NLP_ACL")
SAMPLES = json.loads((ROOT / "data/svhalluc_benchmark/samples.json").read_text(encoding="utf-8"))

def answer(text):
    text = text or ""
    m = re.search(r"(?im)^\s*Final answer\s*:\s*\[?\(?\s*([AB])\b", text)
    if not m:
        m = re.search(r"(?im)^\s*Answer\s*:\s*\[?\(?\s*([AB])\b", text)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*[\[\(]?\s*([AB])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*(Yes|No)\b", text, re.I)
    return {"yes": "A", "no": "B"}.get(m.group(1).lower()) if m else None

def load(path, from_json=None):
    p = Path(path)
    if from_json is None:
        from_json = p.suffix.lower() == ".json"
    if from_json:
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(r["idx"]): r for r in data}

def main(job_path):
    base = load(D / "base_pool72.jsonl")
    method = load(job_path)
    common = sorted(set(base) & set(method))
    rows = []
    for i in common:
        b, m = base[i], method[i]
        gold = str(m.get("gold", b.get("gold", SAMPLES[i]["answer"]))).upper()
        bp, mp = answer(b.get("text", "")), answer(m.get("text", ""))
        task = str(SAMPLES[i].get("task_type_abbr", "?"))
        rows.append(dict(idx=i, task=task, gold=gold, base=bp, method=mp,
                         question=SAMPLES[i].get("question", ""), text=m.get("text", "")))

    def report(label, subset):
        n = len(subset)
        ok = sum(x["method"] == x["gold"] for x in subset)
        inv = sum(x["method"] not in ("A", "B") for x in subset)
        a = [x for x in subset if x["gold"] == "A"]
        b = [x for x in subset if x["gold"] == "B"]
        wc = sum(x["base"] != x["gold"] and x["method"] == x["gold"] for x in subset)
        cw = sum(x["base"] == x["gold"] and x["method"] != x["gold"] for x in subset)
        print(f"{label:8s} n={n:2d} acc={ok}/{n} ({100*ok/n if n else 0:.1f}%) "
              f"A-recall={sum(x['method']=='A' for x in a)}/{len(a)} "
              f"B-recall={sum(x['method']=='B' for x in b)}/{len(b)} "
              f"wrong->correct={wc} correct->wrong={cw} invalid={inv}")

    print(f"Compared {len(common)} paired samples; prompt rows={len(method)}")
    report("ALL", rows)
    for task in sorted({r["task"] for r in rows}):
        report(task, [r for r in rows if r["task"] == task])
    print("\nChanged predictions:")
    for x in rows:
        if x["base"] != x["method"]:
            print(f"{x['idx']} {x['task']} gold={x['gold']} base={x['base']} prompt={x['method']} | {x['question']}")
    print("\nPrompt errors/invalids:")
    for x in rows:
        if x["method"] != x["gold"]:
            print(f"{x['idx']} {x['task']} gold={x['gold']} base={x['base']} prompt={x['method']} | {x['question']}\n"
                  f"  {x['text'][:500].replace(chr(10), ' ')}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else D / "job_out.json")

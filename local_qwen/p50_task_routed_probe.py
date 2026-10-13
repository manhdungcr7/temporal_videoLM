"""Single-pass task-routed prompt probe on the same six balanced examples."""
import gc, hashlib, json, re, time
from pathlib import Path

import torch
import runner
from runner import ROOT, load, prepare_video, qwen_inputs, move_inputs

OUT = Path(__file__).parent / "p50_task_routed_results.jsonl"
PROMPT_PATH = Path(__file__).parent / "p50_task_routed_prompt.txt"
PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
IDS = [352, 166, 670, 614, 1109, 1254]


def parse(text):
    text = text.strip()
    m = re.match(r"\s*\(?([AB])\)?(?:[.)\s]|$)", text, re.I)
    if m: return m.group(1).upper()
    m = re.match(r"\s*(Yes|No)\b", text, re.I)
    return {"yes":"A", "no":"B"}.get(m.group(1).lower()) if m else None


def main():
    rows = json.loads((ROOT / "data/svhalluc_benchmark/samples.json").read_text(encoding="utf-8"))
    done = {json.loads(x)["idx"] for x in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    model, proc = load()
    for idx in IDS:
        if idx in done: continue
        row = rows[idx]; start = time.time()
        try:
            media = prepare_video(ROOT / "data/svhalluc_benchmark" / row["video_path"])
            inputs, _ = qwen_inputs(proc, PROMPT + "\n\nQuestion: " + row["question"], media)
            inputs = move_inputs(inputs, model)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                     use_audio_in_video=True, num_beams=1, use_cache=True)
            text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            pred = parse(text)
            rec = {"idx":idx,"task":row["task_type_abbr"],"gold":row["answer"],
                   "pred":pred,"text":text,"sec":round(time.time()-start,1),"prompt_sha256":SHA}
            del media,inputs,out
        except Exception as exc:
            rec = {"idx":idx,"task":row["task_type_abbr"],"gold":row["answer"],
                   "error":repr(exc),"sec":round(time.time()-start,1),"prompt_sha256":SHA}
        with OUT.open("a",encoding="utf-8") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        print(json.dumps(rec,ensure_ascii=False),flush=True)
        gc.collect();torch.cuda.empty_cache()
    print(f"Saved {OUT}",flush=True)


if __name__ == "__main__": main()

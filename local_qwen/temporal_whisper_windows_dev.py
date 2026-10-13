"""Coarse ASR windows for development annotation only; never fed to Qwen evaluation."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
os.environ["PATH"] = str(ROOT / "research_topics_learning/single_sample_prompt_lab/validation_ffmpeg") + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(ROOT / "research_topics_learning/single_sample_prompt_lab"))
from lab_media import decode_audio
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DATA = ROOT / "data/svhalluc_benchmark"
SPLIT = ROOT / "local_qwen/temporal_pilot_split.json"
MODEL_DIR = ROOT / "research_topics_learning/external_evidence_implementation/models/asr"

if __name__ == "__main__":
    manifest = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    torch.set_num_threads(4)
    results = []
    for item in manifest["splits"]["development"]:
        audio = decode_audio(DATA / item["video_path"])
        result = {"idx": item["idx"], "task": item["task"], "window_seconds": 5, "windows": []}
        for start in range(0, len(audio), 5 * 16000):
            end = min(start + 5 * 16000, len(audio))
            chunk = audio[start:end]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, num_beams=3, max_new_tokens=64)
            result["windows"].append({"start_s": round(start / 16000, 2), "end_s": round(end / 16000, 2),
                                      "transcript": processor.batch_decode(generated, skip_special_tokens=True)[0]})
        results.append(result)
        print(item["idx"], item["task"], [(w["start_s"], w["transcript"]) for w in result["windows"]], flush=True)
    out = ROOT / "local_qwen/temporal_whisper_windows_dev.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", out)

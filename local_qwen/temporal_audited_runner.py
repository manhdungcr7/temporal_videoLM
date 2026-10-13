"""Auditable paired inference for the frozen SVHalluc temporal development split.

This runner always feeds the original AV clip through the existing Qwen preprocessing
path. It records generation and media metadata so later comparisons do not rely on
the older pilot JSONL, which did not contain those fields.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import decord
from qwen_omni_utils.v2_5.vision_process import calculate_video_frame_range, smart_nframes

import runner
from runner import ROOT, load, prepare_video, qwen_inputs, move_inputs

DATA = ROOT / "data/svhalluc_benchmark"
SPLIT_FILE = ROOT / "local_qwen/temporal_pilot_split.json"
FFMPEG = ROOT / "research_topics_learning/single_sample_prompt_lab/validation_ffmpeg/ffmpeg.exe"
NORMALIZED_DIR = ROOT / "local_qwen/temporal_normalized_cfr30"
REV = runner.REV


def parse_choice(text, task):
    """Conservative answer parser: prefer explicit final marker, then a leading option."""
    text = text.strip()
    final = re.search(r"(?i)\bfinal\s+(?:answer|choice|option)\s*(?:is\s*)?[:=,]?\s*\(?([ABC])\b", text)
    if final:
        return final.group(1).upper()
    explicit = re.search(r"(?i)\b(?:answer|choice|option|visual action)\s*(?:is\s*)?[:=,]?\s*\(?([ABC])\b", text)
    if explicit:
        return explicit.group(1).upper()
    described_action = re.search(r"(?i)\bvisual action\b.{0,60}?\bis\s*[:=,]?\s*\(?([ABC])\b", text)
    if described_action:
        return described_action.group(1).upper()
    leading = re.search(r"(?im)^\s*\(?([ABC])\s*[.):\-]", text) or re.search(r"(?im)^\s*\(?([ABC])\)?\s*$", text)
    if leading:
        return leading.group(1).upper()
    if task == "TA":
        match = re.search(r"(?i)\b(yes|no)\b", text)
        return {"yes": "A", "no": "B"}[match.group(1).lower()] if match else None
    if task == "TF":
        match = re.search(r"(?i)\b(past|present|future)\b", text)
        return {"past": "A", "present": "B", "future": "C"}[match.group(1).lower()] if match else None
    return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_av_clip(source_path):
    """Re-encode to a clean CFR timeline before Decord sampling.

    The development-media audit found that Decord's sampled frame-index sequence can
    jump backwards when matched against the PyAV display timeline on these source
    files. CFR normalization makes the order monotonic while keeping both modalities.
    """
    source_path = Path(source_path)
    source_hash = file_sha256(source_path)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    target = NORMALIZED_DIR / f"{source_hash[:20]}_30fps.mp4"
    command = [
        str(FFMPEG), "-hide_banner", "-y", "-i", str(source_path),
        "-map", "0:v:0", "-map", "0:a:0", "-vf", "fps=30", "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-af", "aresample=async=1:first_pts=0", "-movflags", "+faststart",
        str(target),
    ]
    if not target.exists():
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg normalization failed: {completed.stderr[-2000:]}")
    return target, {
        "performed": True, "source_path": str(source_path), "source_sha256": source_hash,
        "normalized_path": str(target), "normalized_sha256": file_sha256(target),
        "video_fps": 30, "video_codec": "libx264 ultrafast crf24", "audio_codec": "aac",
        "command": command,
    }


def media_metadata(path, media, inputs, processor):
    """Reproduce qwen_omni_utils' Decord frame index sampling and verify frame count."""
    vr = decord.VideoReader(str(path))
    n_total = len(vr)
    source_fps = float(vr.get_avg_fps())
    ele = {"video": str(path), "max_pixels": media["max_pixels_per_frame"]}
    start, end, n_range = calculate_video_frame_range(ele, n_total, source_fps)
    n_sampled = smart_nframes(ele, total_frames=n_range, video_fps=source_fps)
    indices = torch.linspace(start, end, n_sampled).round().long().tolist()
    # Exact PTS reported by Decord for the sampled indices. Qwen does not receive
    # this list: its temporal position is generated from the scalar sample FPS.
    raw_frame_pts = vr.get_frame_timestamp(indices).tolist()
    actual_tensor_frames = int(media["videos"][0].shape[0])
    if actual_tensor_frames != len(raw_frame_pts):
        raise RuntimeError(
            f"Sampling audit mismatch: tensor={actual_tensor_frames}, reconstructed={len(raw_frame_pts)}"
        )
    audio = media["audio"][0]
    audio_samples = int(len(audio))
    video_grid = inputs.get("video_grid_thw")
    temporal_patch_size = int(processor.video_processor.temporal_patch_size)
    nominal_grid_times = [
        round(i * temporal_patch_size / float(media["fps"]), 4)
        for i in range(int(video_grid[0, 0].item()))
    ] if video_grid is not None else []
    return {
        "video_source": str(path), "video_sha256": file_sha256(path),
        "source_frame_count_reported_by_decord": n_total, "source_fps_reported_by_decord": source_fps,
        "container_duration_seconds_from_decord_count_fps": round(n_total / source_fps, 4),
        "qwen_reader": "decord", "qwen_frame_count": actual_tensor_frames,
        "qwen_frame_indices": indices,
        "decord_raw_frame_timestamp_intervals_seconds": [[round(float(a), 4), round(float(b), 4)] for a, b in raw_frame_pts],
        "qwen_nominal_temporal_grid_times_seconds": nominal_grid_times,
        "timestamp_note": "Raw Decord PTS are diagnostic only; Qwen receives frames plus scalar sample_fps, not per-frame PTS.",
        "qwen_sample_fps": float(media["fps"]),
        "max_pixels_per_frame": int(media["max_pixels_per_frame"]),
        "audio_sample_rate_hz": 16000, "audio_sample_count": audio_samples,
        "audio_duration_seconds": round(audio_samples / 16000, 4),
        "input_video_grid_thw": video_grid.detach().cpu().tolist() if video_grid is not None else None,
        "input_video_tensor_shape": list(media["videos"][0].shape),
        "input_audio_feature_shape": list(inputs["input_features"].shape) if "input_features" in inputs else None,
        "input_audio_feature_attention_mask_shape": list(inputs["feature_attention_mask"].shape) if "feature_attention_mask" in inputs else None,
        "input_token_count": int(inputs["input_ids"].shape[-1]),
    }


def prompt_for(mode, question):
    if mode == "question_only":
        return question
    if mode == "explain_then_choice":
        return (
            question
            + "\nFirst give one brief reason based only on the supplied audio-video. "
              "Then end with exactly: Final answer: A, Final answer: B, or Final answer: C."
        )
    raise ValueError(mode)


def run_one(model, processor, item, row, mode, max_new_tokens):
    source_path = DATA / item["video_path"]
    path, normalization = normalize_av_clip(source_path)
    media = prepare_video(path)
    user_prompt = prompt_for(mode, row["question"])
    inputs, rendered = qwen_inputs(processor, user_prompt, media)
    input_len = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, model)
    # This checkpoint's generation_config.json omits EOS although the tokenizer
    # defines <|im_end|> as EOS. Explicitly stop at the assistant turn boundary.
    effective_eos_id = processor.tokenizer.eos_token_id
    if effective_eos_id is None:
        raise RuntimeError("Tokenizer has no EOS token ID; refusing an unbounded turn")
    effective_pad_id = processor.tokenizer.pad_token_id
    started = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            use_audio_in_video=True,
            eos_token_id=int(effective_eos_id),
            pad_token_id=int(effective_pad_id) if effective_pad_id is not None else None,
            return_dict_in_generate=True,
            output_scores=False,
        )
    elapsed = time.time() - started
    sequences = out.sequences
    new_ids = sequences[0, input_len:].detach().cpu().tolist()
    decoded = processor.batch_decode(sequences[:, input_len:], skip_special_tokens=True)[0]
    eos_ids = [int(effective_eos_id)]
    ended_eos = bool(new_ids and new_ids[-1] in eos_ids)
    if ended_eos:
        stop_reason = "eos_token"
    elif len(new_ids) >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "other_or_unknown"
    result = {
        "idx": item["idx"], "task": item["task"], "gold": item["gold"],
        "mode": mode, "video_path": item["video_path"], "source_id": item["source_id"],
        "normalization": normalization,
        "question": row["question"], "actual_rendered_prompt": rendered,
        "generation_config": {
            "do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens,
            "use_cache": True, "use_audio_in_video": True,
            "eos_token_id": eos_ids, "pad_token_id": model.generation_config.pad_token_id,
            "checkpoint_generation_config_eos_token_id": model.generation_config.eos_token_id,
            "eos_source": "processor.tokenizer.eos_token_id (<|im_end|>)",
            "transformers_generation_config": model.generation_config.to_dict(),
        },
        "input_token_count": input_len, "generated_token_ids": new_ids,
        "generated_token_count_including_terminal_token": len(new_ids),
        "generated_token_count_excluding_eos": len(new_ids) - int(ended_eos),
        "stopped_on_eos": ended_eos, "stop_reason": stop_reason,
        "raw_output": decoded, "parsed_prediction": parse_choice(decoded, item["task"]),
        "gold": item["gold"], "correct": parse_choice(decoded, item["task"]) == item["gold"],
        "generation_seconds": round(elapsed, 3),
        "model": "Qwen/Qwen2.5-Omni-3B", "model_revision": REV,
        "quantization": "NF4 4-bit, double quantization, FP16 compute",
        "attention_implementation": "sdpa", "torch_version": torch.__version__,
        "media_metadata": media_metadata(path, media, inputs, processor),
    }
    del out, sequences, inputs, media
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "confirmation"], default="development")
    parser.add_argument("--mode", choices=["question_only", "explain_then_choice"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, help="Set the same value when comparing conditions")
    parser.add_argument("--idx", type=int, action="append", help="Repeat to select individual rows")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    items = manifest["splits"][args.split]
    if args.idx:
        wanted = set(args.idx)
        items = [x for x in items if x["idx"] in wanted]
        if {x["idx"] for x in items} != wanted:
            raise ValueError("Requested idx not found in selected split")
    if args.limit:
        items = items[:args.limit]
    rows = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {json.loads(line)["idx"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    model, processor = load()
    # Same default budget across conditions so any paired result is not confounded
    # by a baseline/prompt max-token mismatch. Override only for an explicit audit.
    token_budget = args.max_new_tokens or 128
    if token_budget <= 0:
        raise ValueError("--max-new-tokens must be positive")
    print("Runtime versions:", json.dumps({"torch": torch.__version__, "transformers": __import__("transformers").__version__}, ensure_ascii=False), flush=True)
    for n, item in enumerate(items, 1):
        if item["idx"] in done:
            print(f"skip existing idx={item['idx']}", flush=True)
            continue
        try:
            record = run_one(model, processor, item, rows[item["idx"]], args.mode, token_budget)
        except Exception as exc:
            record = {"idx": item["idx"], "task": item["task"], "gold": item["gold"],
                      "mode": args.mode, "error": f"{type(exc).__name__}: {exc}"}
            torch.cuda.empty_cache()
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{n}/{len(items)} idx={item['idx']} task={item['task']} gold={item['gold']} "
              f"pred={record.get('parsed_prediction')} nnew={record.get('generated_token_count_including_terminal_token')} "
              f"stop={record.get('stop_reason')} err={record.get('error')}", flush=True)
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()

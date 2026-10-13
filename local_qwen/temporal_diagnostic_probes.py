"""Three targeted, diagnostic-only probes for one annotated temporal example."""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
sys.path.insert(0, str(ROOT / "local_qwen"))
import temporal_audited_runner as audited
import runner
from runner import load, prepare_video, qwen_inputs, move_inputs
from lab_native_video import conversation_for, SYSTEM, VIDEO_MAX_PIXELS
from qwen_omni_utils import process_mm_info

IDX = 2310
SPEECH_WINDOW = (0.0, 5.0)  # coarse Whisper window; diagnostic only
EVENT_WINDOW = (1.0, 5.0)   # coarse one-fps visual interval; diagnostic only
OUT = ROOT / "local_qwen/temporal_diagnostic_probes_2310_eos128.json"
TEMP = ROOT / "local_qwen/temporal_probe_clips"


def crop_clip(source, start, end, include_audio):
    TEMP.mkdir(parents=True, exist_ok=True)
    output = TEMP / f"{IDX}_{start:.2f}_{end:.2f}_{'av' if include_audio else 'vision'}.mp4"
    if output.exists():
        return output
    cmd = [str(audited.FFMPEG), "-hide_banner", "-y", "-i", str(source),
           "-ss", str(start), "-t", str(end - start), "-map", "0:v:0"]
    if include_audio:
        cmd += ["-map", "0:a:0"]
    else:
        cmd += ["-an"]
    cmd += ["-vf", "fps=30", "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "24", "-pix_fmt", "yuv420p"]
    if include_audio:
        cmd += ["-c:a", "aac", "-af", "aresample=async=1:first_pts=0"]
    cmd += ["-movflags", "+faststart", str(output)]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if run.returncode or not output.exists():
        raise RuntimeError(f"Could not create diagnostic crop: {run.stderr[-1500:]}")
    return output


def prepare_video_only(path):
    convo = conversation_for(path, "", media_mode="vision", max_pixels=VIDEO_MAX_PIXELS)
    _, images, videos, kwargs = process_mm_info(convo, use_audio_in_video=False, return_video_kwargs=True)
    if images is not None or videos is None or len(videos) != 1:
        raise RuntimeError("Expected one video and no additional images in vision-only probe")
    fps = kwargs["fps"]
    if isinstance(fps, (tuple, list)):
        fps = fps[0]
    return {"path": str(path), "audio": None, "videos": videos, "fps": float(fps),
            "max_pixels_per_frame": VIDEO_MAX_PIXELS}


def run_generation(model, processor, prompt, media=None, media_mode="text", max_new_tokens=128):
    if media_mode == "text":
        convo = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        rendered = processor.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=rendered, return_tensors="pt")
    else:
        inputs, rendered = qwen_inputs(processor, prompt, media, media_mode=media_mode)
    input_len = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, model)
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no EOS token ID; refusing an unbounded turn")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True, use_audio_in_video=(media_mode == "av"),
            eos_token_id=int(eos_id), pad_token_id=processor.tokenizer.pad_token_id,
            return_dict_in_generate=True, output_scores=False,
        )
    ids = output.sequences[0, input_len:].detach().cpu().tolist()
    text = processor.batch_decode(output.sequences[:, input_len:], skip_special_tokens=True)[0]
    eos_ids = [int(eos_id)]
    stopped_eos = bool(ids and ids[-1] in eos_ids)
    return {"rendered_prompt": rendered, "generated_token_ids": ids,
            "generated_token_count": len(ids), "raw_output": text,
            "parsed_prediction": audited.parse_choice(text, "TA"),
            "max_new_tokens": max_new_tokens, "eos_token_ids": eos_ids,
            "stopped_on_eos": stopped_eos,
            "stop_reason": "eos_token" if stopped_eos else ("max_new_tokens" if len(ids) >= max_new_tokens else "other_or_unknown")}


def main():
    rows = json.loads((audited.DATA / "samples.json").read_text(encoding="utf-8"))
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding="utf-8"))
    item = next(x for x in manifest["splits"]["development"] if x["idx"] == IDX)
    normalized, normmeta = audited.normalize_av_clip(audited.DATA / item["video_path"])
    row = rows[IDX]
    event_crop = crop_clip(normalized, EVENT_WINDOW[0], EVENT_WINDOW[1], include_audio=False)
    speech_crop = crop_clip(normalized, SPEECH_WINDOW[0], SPEECH_WINDOW[1], include_audio=True)

    print("Loading local Qwen once; all three probes use that same loaded checkpoint.", flush=True)
    model, processor = load()
    results = {"idx": IDX, "gold": item["gold"], "question": row["question"],
               "normalization": normmeta, "annotation_status": "coarse diagnostic intervals only",
               "speech_interval_s": SPEECH_WINDOW, "event_interval_s": EVENT_WINDOW,
               "probes": {}}

    # 1) Is the narrated action visually recognizable with speech removed?
    vision_media = prepare_video_only(event_crop)
    results["probes"]["muted_event_recognition"] = run_generation(
        model, processor,
        "The audio is intentionally absent. Describe the action visibly occurring in this short video segment in one sentence.",
        vision_media, "vision", 128)
    del vision_media; torch.cuda.empty_cache()

    # 2) Can the model read the temporal relation when the event and intervals are explicit?
    text_evidence = (
        "Temporal reasoning check. This is text-only evidence for diagnosis, not the benchmark input.\n"
        "Speech interval S is approximately [0, 5) seconds and the narrated event is ‘dice the potato’.\n"
        "The visible potato-dicing interval E is approximately [1, 5) seconds. These intervals are coarse.\n"
        "The task asks whether the narrated event is happening at the same time as the speech: A=Yes, B=No.\n"
        "Using interval overlap, give one short reason and then ‘Final answer: A’ or ‘Final answer: B’."
    )
    results["probes"]["text_only_relation"] = run_generation(
        model, processor, text_evidence, None, "text", 128)

    # 3) Does a correctly localized short AV crop change the answer to the original TA question?
    localized_media = prepare_video(speech_crop)
    results["probes"]["localized_av_crop"] = run_generation(
        model, processor, row["question"], localized_media, "av", 128)
    del localized_media; torch.cuda.empty_cache()

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in results["probes"].items():
        print(name, "pred=", result.get("parsed_prediction"), "tokens=", result.get("generated_token_count"),
              "output=", repr(result.get("raw_output")), flush=True)
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()


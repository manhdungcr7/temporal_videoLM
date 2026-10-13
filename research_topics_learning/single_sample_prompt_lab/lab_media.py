"""Shared, audited media and baseline helpers; extracted from the previous pilot."""
import collections, hashlib, json, math, time
from pathlib import Path
SYSTEM = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")

def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def select_manifest(records, item_ids):
    if len(item_ids) != 60 or len(set(item_ids)) != 60:
        raise ValueError("The development manifest must contain 60 distinct item IDs")
    selected = []
    for item_id in item_ids:
        index = int(item_id.removeprefix("svh_"))
        row = dict(records[index])
        task = row["task_type_abbr"].replace("CSMB", "CMSB")
        if row["data_type"] != "semantic" or task not in ("GSA", "FGSA", "CMSB"):
            raise ValueError(f"Wrong semantic manifest entry: {item_id}")
        source = row["video_path"].replace("\\", "/").split("/")[2]
        selected.append({**row, "item_id": item_id, "task": task,
                         "gold": row["answer"], "source_video_id": source})
    counts = collections.Counter((x["task"], x["gold"]) for x in selected)
    if counts != {(t, y): 10 for t in ("GSA", "FGSA", "CMSB") for y in ("A", "B")}:
        raise ValueError(f"Unbalanced manifest: {counts}")
    if len({x["source_video_id"] for x in selected}) != 60:
        raise ValueError("Duplicate source videos in development set")
    return selected

def audio_windows(n_samples, sampling_rate, chunk_seconds, overlap_seconds):
    """Cover the full audio, preserving overlaps instead of silently truncating it."""
    if n_samples <= 0 or sampling_rate <= 0:
        raise ValueError("Empty audio")
    chunk = round(chunk_seconds * sampling_rate)
    step = round((chunk_seconds - overlap_seconds) * sampling_rate)
    if step <= 0 or chunk <= 0:
        raise ValueError("Invalid chunk/overlap")
    windows = []
    start = 0
    while start < n_samples:
        end = min(start + chunk, n_samples)
        windows.append((start, end))
        if end == n_samples:
            break
        start += step
    return windows

def decode_audio(path):
    """Extract the complete original audio with PyAV; no ASR-based crop or audio swap."""
    import av
    import numpy as np
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError("Clip has no audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for part in resampler.resample(frame):
                chunks.append(part.to_ndarray().reshape(-1))
        for part in resampler.resample(None):
            chunks.append(part.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError("Empty decoded audio")
    return np.concatenate(chunks).astype("float32", copy=False)

def candidate_pool(path, directory, count):
    import av
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Decord reports negative timestamp/count anomalies on clipped MP4s here.
    # Its first decoded pixels can still be correct: do not equate negative metadata with
    # out-of-clip image content. Use PyAV's decoded-frame presentation times instead.
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float(container.start_time or 0) / 1e6
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / 1e6
        fps = float(stream.average_rate or 30)
        if duration <= 0:
            raise RuntimeError("Missing video duration; cannot promise full-clip sampling")
        targets = np.linspace(0, max(0, duration - 1 / fps), min(count, max(1, round(duration * fps))))
        target_index, previous, seen = 0, None, set()

        def save(observation):
            index, stamp, frame = observation
            if index in seen:
                return
            seen.add(index)
            image = frame.to_image().convert("RGB")
            image.thumbnail((960, 960))
            image_path = directory / f"frame_{len(frames):03d}.png"
            image.save(image_path)
            frames.append({"frame_index": int(index), "time_s": stamp, "path": str(image_path),
                           "relative_path": f"evidence_images/{directory.name}/{image_path.name}"})

        for index, frame in enumerate(container.decode(video=0)):
            if frame.time is None:
                raise RuntimeError("Video frame lacks a presentation timestamp")
            stamp = float(frame.time) - origin
            if not math.isfinite(stamp):
                raise RuntimeError("Invalid video presentation timestamp")
            if stamp < -1e-6 or stamp >= duration:
                continue
            current = (index, max(0.0, stamp), frame)
            if previous and current[1] < previous[1]:
                raise RuntimeError("Non-monotonic decoded video timestamps")
            while target_index < len(targets) and current[1] >= targets[target_index]:
                target = targets[target_index]
                chosen = previous if previous and abs(previous[1] - target) <= abs(current[1] - target) else current
                save(chosen)
                target_index += 1
            previous = current
        if previous and target_index < len(targets):
            save(previous)
    if not frames:
        raise RuntimeError("No video frames decoded")
    return frames

def qwen_inputs(processor, question_prompt, media=None, details=None):
    """Original interleaved AV + supplemental timestamped IMAGES, never a fake-fps video."""
    content, kwargs = [], {"return_tensors": "pt", "padding": True}
    if media is not None:
        video_element, audio, video, sampled_fps = media
        content.append(video_element)
        kwargs.update(audio=audio, videos=video, use_audio_in_video=True,
                      videos_kwargs={"fps": sampled_fps, "do_sample_frames": False, "do_resize": False})
    images = []
    if details:
        from PIL import Image
        for detail in details:
            content.append({"type": "text", "text": f"Supplemental image from original clip at {detail['time_s']:.3f} seconds:"})
            content.append({"type": "image", "image": detail["path"]})
            with Image.open(detail["path"]) as image:
                images.append(image.convert("RGB"))
        kwargs.update(images=images, images_kwargs={"do_resize": True, "min_pixels": 64 * 28 * 28,
                                                   "max_pixels": details[0]["max_pixels"]})
    content.append({"type": "text", "text": question_prompt})
    conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": content}]
    rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=rendered, **kwargs)
    if media is not None:
        assert "input_features" in inputs and "pixel_values_videos" in inputs
    if details:
        assert "pixel_values" in inputs and len(inputs["image_grid_thw"]) == len(details)
    return inputs, rendered

def move_inputs(inputs, model):
    import torch
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    for key in ("input_features", "pixel_values", "pixel_values_videos"):
        if key in inputs:
            inputs[key] = inputs[key].to(torch.float16)
    return inputs

def score_answer(model, processor, inputs):
    """Same next-token A/B interface for both conditions, with full-vocabulary mass audited."""
    import torch
    ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    if any(len(x) != 1 for x in ids):
        raise RuntimeError("A/B must each be one token")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, use_audio_in_video=True, do_sample=False, num_beams=1,
                                max_new_tokens=1, use_cache=True, output_logits=True,
                                return_dict_in_generate=True,
                                eos_token_id=processor.tokenizer.eos_token_id,
                                pad_token_id=processor.tokenizer.pad_token_id)
    logits = output.logits[0][0].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite baseline logits")
    letters = logits[[ids[0][0], ids[1][0]]]
    probs = letters.softmax(-1)
    logp = logits.log_softmax(-1)
    return {"prediction": "A" if float(letters[0]) >= float(letters[1]) else "B",
            "p_A_given_AB": float(probs[0]), "p_B_given_AB": float(probs[1]),
            "margin_A_minus_B": float(letters[0] - letters[1]),
            "AB_probability_mass": float(logp[[ids[0][0], ids[1][0]]].exp().sum()),
            "natural_first_token": processor.tokenizer.decode(output.sequences[0, -1:]),
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "scoring_latency_s": time.perf_counter() - started,
            "peak_allocated_GiB": [round(torch.cuda.max_memory_allocated(i) / 2**30, 3)
                                   for i in range(torch.cuda.device_count())]}


def baseline_media(path, audio, frames, config):
    """Uniform valid-clip frames at the baseline budget; exact frame times retained in audit."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    duration = len(audio) / 16000
    nframes = max(4, min(config["max_frames"], round(duration * config["fps"] / 2) * 2))
    indices = np.rint(np.linspace(0, len(frames) - 1, nframes)).astype(int).tolist()
    arrays = []
    for index in indices:
        with Image.open(frames[index]["path"]) as image:
            height, width = smart_resize(image.height, image.width, factor=28,
                                         min_pixels=64 * 28 * 28, max_pixels=config["max_pixels"])
            arrays.append(np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)))
    video = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    times = [frames[index]["time_s"] for index in indices]
    sampled_fps = (nframes - 1) / (times[-1] - times[0]) if times[-1] > times[0] else config["fps"]
    return ({"type": "video", "video": str(path)}, [audio], [video], sampled_fps), times

"""Audit the exact video-reader timeline used by the local Qwen preprocessing."""
import json
from pathlib import Path

import av
import decord
import numpy as np
import torch
from PIL import Image
from qwen_omni_utils.v2_5.vision_process import calculate_video_frame_range, smart_nframes

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
DATA = ROOT / "data/svhalluc_benchmark"
SPLIT = ROOT / "local_qwen/temporal_pilot_split.json"


def audit(item):
    path = DATA / item["video_path"]
    vr = decord.VideoReader(str(path))
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    ele = {"video": str(path), "max_pixels": 160 * 28 * 28}
    start, end, total_range = calculate_video_frame_range(ele, total_frames, fps)
    n_sampled = smart_nframes(ele, total_frames=total_range, video_fps=fps)
    indices = torch.linspace(start, end, n_sampled).round().long().tolist()
    raw_ts = vr.get_frame_timestamp(indices).tolist()
    sample_fps = n_sampled / max(total_range, 1e-6) * fps
    temporal_patch = 2
    qwen_grid_times = [i * temporal_patch / sample_fps for i in range(n_sampled // temporal_patch)]
    decord_thumbs = [
        np.asarray(Image.fromarray(vr[i].asnumpy()).convert("RGB").resize((32, 18)), dtype=np.float32)
        for i in indices
    ]

    with av.open(str(path)) as container:
        duration = container.duration / av.time_base if container.duration else None
        streams = {}
        decoded_frame_pts = []
        for s in container.streams:
            stream_record = {
                "start_time_s": float(s.start_time * s.time_base) if s.start_time is not None else None,
                "duration_s": float(s.duration * s.time_base) if s.duration is not None else None,
                "declared_frames": int(s.frames) if s.frames else None,
            }
            if s.type == "video":
                stream_record["average_rate"] = float(s.average_rate) if s.average_rate else None
                decoded_frames = [f for f in container.decode(video=0) if f.pts is not None]
                decoded_frame_pts = [float(f.pts * f.time_base) for f in decoded_frames]
                pyav_thumbs = [
                    np.asarray(f.to_image().convert("RGB").resize((32, 18)), dtype=np.float32)
                    for f in decoded_frames
                ]
                stream_record["decoded_frames_pyav"] = len(decoded_frame_pts)
                stream_record["first_last_decoded_pts_s"] = [decoded_frame_pts[0], decoded_frame_pts[-1]] if decoded_frame_pts else None
            streams[s.type] = stream_record

    content_match_times, content_match_mse = [], []
    for sample in decord_thumbs:
        errors = [float(np.mean((sample - reference) ** 2)) for reference in pyav_thumbs]
        best = int(np.argmin(errors))
        content_match_times.append(decoded_frame_pts[best])
        content_match_mse.append(errors[best])
    backward_jumps = [
        round(content_match_times[i] - content_match_times[i + 1], 4)
        for i in range(len(content_match_times) - 1)
        if content_match_times[i + 1] < content_match_times[i] - 0.2
    ]

    return {
        "idx": item["idx"], "task": item["task"], "gold": item["gold"], "path": item["video_path"],
        "container_duration_s_pyav": duration, "streams_pyav": streams,
        "decord_frame_count": total_frames, "decord_average_fps": fps,
        "decord_first_last_timestamp_s": [float(raw_ts[0][0]), float(raw_ts[-1][1])],
        "decord_sampled_indices": indices,
        "decord_sampled_raw_pts_intervals_s": [[float(a), float(b)] for a, b in raw_ts],
        "sampled_frame_content_match_times_in_pyav_timeline_s": [round(t, 4) for t in content_match_times],
        "sampled_frame_content_match_mse_32x18_rgb": [round(x, 4) for x in content_match_mse],
        "backward_content_timeline_jumps_over_0_2s": backward_jumps,
        "qwen_scalar_sample_fps": sample_fps,
        "qwen_nominal_video_grid_times_s": qwen_grid_times,
        "qwen_temporal_patch_size_frames": temporal_patch,
        "caveat": "Qwen temporal positions are built from scalar FPS; actual Decord PTS are not passed per frame.",
    }


if __name__ == "__main__":
    manifest = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = [audit(x) for x in manifest["splits"]["development"]]
    out = ROOT / "local_qwen/temporal_media_alignment_audit_dev.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for x in rows:
        raw = x["decord_first_last_timestamp_s"]
        audio = x["streams_pyav"].get("audio", {}).get("duration_s")
        print(f"idx={x['idx']} {x['task']} DecordPTS={raw[0]:.3f}..{raw[1]:.3f} "
              f"audio_duration={audio} decord_n={x['decord_frame_count']} "
              f"PyAV_decoded_n={x['streams_pyav'].get('video',{}).get('decoded_frames_pyav')}")
    print("Saved", out)

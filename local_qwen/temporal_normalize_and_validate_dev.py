"""Build normalized dev clips and assert that Qwen's sampled frame order is monotonic."""
import json
import sys
from pathlib import Path

import av
import decord
import numpy as np
import torch
from PIL import Image
from qwen_omni_utils.v2_5.vision_process import smart_nframes

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
sys.path.insert(0, str(ROOT / "local_qwen"))
import temporal_audited_runner as audited


def check(item):
    original = audited.DATA / item["video_path"]
    normalized, norm = audited.normalize_av_clip(original)
    vr = decord.VideoReader(str(normalized))
    total, fps = len(vr), float(vr.get_avg_fps())
    ele = {"video": str(normalized), "max_pixels": 160 * 28 * 28}
    n = smart_nframes(ele, total_frames=total, video_fps=fps)
    indices = torch.linspace(0, total - 1, n).round().long().tolist()
    samples = [np.asarray(Image.fromarray(vr[i].asnumpy()).convert("RGB").resize((32, 18)), dtype=np.float32)
               for i in indices]
    with av.open(str(normalized)) as c:
        frames = [f for f in c.decode(video=0) if f.pts is not None]
        pts = [float(f.pts * f.time_base) for f in frames]
        ref = [np.asarray(f.to_image().convert("RGB").resize((32, 18)), dtype=np.float32) for f in frames]
        duration = float(c.duration / av.time_base) if c.duration else None
        audio_duration = next(float(s.duration * s.time_base) for s in c.streams if s.type == "audio" and s.duration)
    matched, mse = [], []
    for image in samples:
        errs = [float(np.mean((image - item) ** 2)) for item in ref]
        j = int(np.argmin(errs))
        matched.append(pts[j]); mse.append(errs[j])
    jumps = [round(matched[i] - matched[i + 1], 4) for i in range(len(matched) - 1)
             if matched[i + 1] < matched[i] - 0.2]
    return {
        "idx": item["idx"], "task": item["task"], "gold": item["gold"],
        "source_path": item["video_path"], "normalization": norm,
        "normalized_duration_s": duration, "normalized_audio_duration_s": audio_duration,
        "decord_frame_count": total, "decord_fps": fps, "qwen_sampled_frame_count": n,
        "qwen_sampled_content_match_times_s": [round(t, 4) for t in matched],
        "max_rgb_match_mse_32x18": max(mse), "backward_jumps_over_0_2s": jumps,
        "monotonic": not jumps,
    }


if __name__ == "__main__":
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding="utf-8"))
    results = []
    for n, item in enumerate(manifest["splits"]["development"], 1):
        result = check(item); results.append(result)
        print(f"{n}/12 idx={item['idx']} frames={result['qwen_sampled_frame_count']} "
              f"monotonic={result['monotonic']} jumps={result['backward_jumps_over_0_2s']} "
              f"max_rgb_mse={result['max_rgb_match_mse_32x18']:.4f}", flush=True)
    out = ROOT / "local_qwen/temporal_normalization_validation_dev.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All monotonic:", all(x["monotonic"] for x in results), "Saved", out)

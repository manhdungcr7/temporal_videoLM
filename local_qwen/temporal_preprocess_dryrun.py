"""Audit Qwen2.5-Omni official video preprocessing at controlled FPS values."""
import json, sys, time
from pathlib import Path
import decord
import av
import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from qwen_omni_utils.v2_5.vision_process import calculate_video_frame_range, smart_nframes
from temporal_audited_runner import normalize_av_clip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/svhalluc_benchmark"
SPLIT = ROOT / "local_qwen/temporal_pilot_split.json"
OUT = ROOT / "local_qwen/temporal_preprocess_dryrun.json"
FPS_VALUES = (1.0, 2.0, 4.0)
PIXELS = 160 * 28 * 28

def sample_info(item, target_fps):
    source_path = DATA / item["video_path"]
    path, normalization = normalize_av_clip(source_path)
    vr = decord.VideoReader(str(path))
    src_n, src_fps = len(vr), float(vr.get_avg_fps())
    ele = {"video": str(path), "max_pixels": PIXELS, "fps": target_fps}
    start, end, nrange = calculate_video_frame_range(ele, src_n, src_fps)
    n = smart_nframes(ele, nrange, src_fps)
    indices = torch.linspace(start, end, n).round().long().tolist()
    pts = vr.get_frame_timestamp(indices).tolist()
    nominal_fps = n / max(nrange, 1e-6) * src_fps
    conv = [
        {"role":"system","content":[{"type":"text","text":"You are Qwen."}]},
        {"role":"user","content":[{"type":"video","video":str(path),"max_pixels":PIXELS,"fps":target_fps},
                                      {"type":"text","text":""}]},
    ]
    t0 = time.perf_counter()
    # Audit visual reader independently: Qwen's helper expects a separate audio
    # path for the audio modality; the production path builds video-audio tokens
    # separately from the video tensor.
    audios, images, videos, kwargs = process_mm_info(conv, use_audio_in_video=False, return_video_kwargs=True)
    elapsed = time.perf_counter() - t0
    tensor = videos[0]
    with av.open(str(path)) as container:
        astream=next((s for s in container.streams if s.type=="audio"),None)
        if astream is None: audio_samples=0; audio_duration=0.0
        else:
            audio_duration=float(astream.duration*astream.time_base) if astream.duration else float(container.duration/av.time_base)
            audio_samples=round(audio_duration*16000)
    return {
        "idx":item["idx"], "task":item["task"], "gold":item["gold"], "source_id":item["source_id"],
        "source_path":str(source_path), "normalized_path":str(path),
        "normalization":{k:v for k,v in normalization.items() if k!="command"},
        "target_fps":target_fps, "source_duration_est_s":nrange/src_fps, "source_fps":src_fps,
        "source_frame_count":src_n, "selected_frame_count":n,
        "selected_frame_indices":indices,
        "selected_frame_pts_intervals_s":[[float(a),float(b)] for a,b in pts],
        "reported_sampling_fps":nominal_fps, "qwen_video_tensor_shape_tchw":list(tensor.shape),
        "qwen_video_tensor_dtype":str(tensor.dtype), "qwen_video_kwargs":{k:(float(v) if isinstance(v,(float,int)) else str(v)) for k,v in kwargs.items()},
        "source_audio_duration_s":audio_duration, "source_audio_samples_at_16khz_estimate":audio_samples,
        "audio_sample_rate_hz":16000,
        "preprocess_seconds":elapsed,
        "warning":"Preprocess-only audit. It does not establish answer accuracy or inspect hidden-state reasoning."
    }

def main():
    manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
    by_id={x["idx"]:x for split in manifest["splits"].values() for x in split}
    # Two distinct TA diagnostic samples plus a development TA example.
    chosen=[by_id[i] for i in (1961,2081,2310) if i in by_id]
    rows=[]
    for item in chosen:
        for fps in FPS_VALUES:
            rec=sample_info(item,fps); rows.append(rec)
            print(f"idx={item['idx']} fps={fps:g} frames={rec['selected_frame_count']} "
                  f"shape={rec['qwen_video_tensor_shape_tchw']} audio={rec['source_audio_duration_s']:.3f}s "
                  f"prep={rec['preprocess_seconds']:.2f}s",flush=True)
    OUT.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved",OUT)

if __name__=="__main__": main()

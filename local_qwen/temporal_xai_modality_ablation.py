"""One-sample modality-ablation dry run for Qwen2.5-Omni temporal QA.

This is a behavioral intervention, not a faithful dump of the model's private
reasoning. It compares full AV, video-only, and audio-only inputs for one fixed
question, while logging first-token option scores and the generated response.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
sys.path.insert(0, str(ROOT / "local_qwen"))
sys.path.insert(0, str(ROOT / "research_topics_learning/single_sample_prompt_lab"))

import temporal_audited_runner as audited
from runner import load, prepare_video, qwen_inputs, move_inputs
from lab_native_video import conversation_for, VIDEO_MAX_PIXELS
from qwen_omni_utils import process_mm_info

IDX = 2310
MAX_NEW_TOKENS = 64
OUT = ROOT / "local_qwen/temporal_xai_modality_ablation_2310.json"


def prepare_video_only(path):
    convo = conversation_for(path, "", media_mode="vision", max_pixels=VIDEO_MAX_PIXELS)
    _, images, videos, kwargs = process_mm_info(
        convo, use_audio_in_video=False, return_video_kwargs=True
    )
    if images is not None or videos is None or len(videos) != 1:
        raise RuntimeError("Expected one video and no audio/images")
    fps = kwargs["fps"]
    if isinstance(fps, (tuple, list)):
        fps = fps[0]
    return {"path": str(path), "audio": None, "videos": videos,
            "fps": float(fps), "max_pixels_per_frame": VIDEO_MAX_PIXELS}


def prepare_audio_only(path):
    convo = conversation_for(path, "", media_mode="audio", max_pixels=VIDEO_MAX_PIXELS)
    audios, images, videos, _ = process_mm_info(
        convo, use_audio_in_video=False, return_video_kwargs=True
    )
    if audios is None or len(audios) != 1 or images is not None or videos is not None:
        raise RuntimeError("Expected one audio stream and no video/images")
    return {"path": str(path), "audio": audios, "videos": None,
            "fps": 1.0, "max_pixels_per_frame": VIDEO_MAX_PIXELS}


def token_id_if_single(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {"text": text, "ids": ids, "single_token_id": ids[0] if len(ids) == 1 else None}


def run_mode(model, processor, prompt, media, mode):
    inputs, rendered = qwen_inputs(processor, prompt, media, media_mode=mode)
    input_len = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, model)
    eos = processor.tokenizer.eos_token_id
    start = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            use_audio_in_video=(mode == "av"),
            eos_token_id=int(eos),
            pad_token_id=processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    elapsed = time.time() - start
    new_ids = out.sequences[0, input_len:].detach().cpu().tolist()
    response = processor.batch_decode(out.sequences[:, input_len:], skip_special_tokens=True)[0]
    first_scores = out.scores[0][0].float()
    tok = processor.tokenizer
    candidates = {letter: token_id_if_single(tok, letter) for letter in ("A", "B")}
    usable = {
        letter: (float(first_scores[item["single_token_id"]])
                 if item["single_token_id"] is not None else None)
        for letter, item in candidates.items()
    }
    if usable["A"] is not None and usable["B"] is not None:
        norm = max(usable.values())
        ea, eb = math.exp(usable["A"] - norm), math.exp(usable["B"] - norm)
        p_a_given_ab = ea / (ea + eb)
        ab_logit_margin = usable["A"] - usable["B"]
    else:
        p_a_given_ab = None
        ab_logit_margin = None
    top_ids = torch.topk(first_scores, k=8).indices.tolist()
    top_tokens = [{"token_id": int(i), "token": tok.convert_ids_to_tokens(int(i)),
                   "score": float(first_scores[i])} for i in top_ids]
    result = {
        "mode": mode,
        "rendered_prompt": rendered,
        "response": response,
        "generated_token_ids": new_ids,
        "first_token": tok.convert_ids_to_tokens(new_ids[0]) if new_ids else None,
        "first_token_id": int(new_ids[0]) if new_ids else None,
        "first_step_top_tokens": top_tokens,
        "candidate_tokenizations": candidates,
        "first_step_A_minus_B_logit": ab_logit_margin,
        "first_step_P_A_conditioned_on_A_or_B": p_a_given_ab,
        "generation_seconds": round(elapsed, 3),
        "input_token_count": input_len,
        "generated_token_count": len(new_ids),
        "stopped_on_eos": bool(new_ids and new_ids[-1] == int(eos)),
    }
    del out, inputs
    torch.cuda.empty_cache()
    return result


def main():
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding="utf-8"))
    item = next(x for x in manifest["splits"]["development"] if x["idx"] == IDX)
    rows = json.loads((audited.DATA / "samples.json").read_text(encoding="utf-8"))
    row = rows[IDX]
    path, norm = audited.normalize_av_clip(audited.DATA / item["video_path"])
    print(f"Sample {IDX} | {item['task']} | gold={item['gold']}\n{row['question']}", flush=True)
    print("Loading Qwen once; running same question under AV / video-only / audio-only.", flush=True)
    model, processor = load()
    media = {
        "av": prepare_video(path),
        "vision": prepare_video_only(path),
        "audio": prepare_audio_only(path),
    }
    results = {
        "idx": IDX, "task": item["task"], "gold": item["gold"],
        "question": row["question"], "normalization": norm,
        "checkpoint": "Qwen/Qwen2.5-Omni-3B", "revision": audited.REV,
        "quantization": "NF4 4-bit, double quantization, FP16 compute",
        "interpretation_limit": "Modality ablation and next-token scores are behavioral diagnostics, not a faithful explanation of hidden reasoning.",
        "conditions": {},
    }
    for mode in ("av", "vision", "audio"):
        print(f"Running {mode}...", flush=True)
        results["conditions"][mode] = run_mode(
            model, processor, row["question"], media[mode], mode
        )
        print(mode, repr(results["conditions"][mode]["response"]),
              "first=", results["conditions"][mode]["first_token"],
              "A-B=", results["conditions"][mode]["first_step_A_minus_B_logit"], flush=True)
        del media[mode]
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

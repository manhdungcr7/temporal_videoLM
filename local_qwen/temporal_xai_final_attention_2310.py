"""Capture only last-query attention rows for Thinker layers 34-35, sample 2310."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
sys.path.insert(0, str(ROOT / "local_qwen"))
sys.path.insert(0, str(ROOT / "research_topics_learning/single_sample_prompt_lab"))
import temporal_audited_runner as audited
from runner import load, prepare_video, qwen_inputs, move_inputs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

IDX = 2310
LAYERS = {34, 35}
OUT = ROOT / "local_qwen/temporal_xai_final_attention_2310.json"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding="utf-8"))
    item = next(x for x in manifest["splits"]["development"] if x["idx"] == IDX)
    rows = json.loads((audited.DATA / "samples.json").read_text(encoding="utf-8"))
    row = rows[IDX]
    path, norm = audited.normalize_av_clip(audited.DATA / item["video_path"])
    print("Loading Qwen for last-query attention capture.", flush=True)
    model, processor = load()
    media = prepare_video(path)
    inputs, rendered = qwen_inputs(processor, row["question"], media)
    inputs = move_inputs(inputs, model)
    ids = inputs["input_ids"][0].detach().cpu()
    tokenizer = processor.tokenizer
    video_id = int(tokenizer.convert_tokens_to_ids(processor.video_token))
    audio_id = int(tokenizer.convert_tokens_to_ids(processor.audio_token))
    video_pos = torch.where(ids == video_id)[0].tolist()
    audio_pos = torch.where(ids == audio_id)[0].tolist()
    text_pos = sorted(set(range(len(ids))) - set(video_pos) - set(audio_pos))
    captured = {}
    old_sdpa = ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"]

    def capture_last_query(module, query, key, value, attention_mask, **kwargs):
        layer_id = getattr(module, "layer_idx", None)
        if layer_id is not None and int(layer_id) in LAYERS:
            layer = int(layer_id)
            groups = int(module.num_key_value_groups)
            k = key.repeat_interleave(groups, dim=1) if groups > 1 else key
            q_last = query[:, :, -1:, :].float()
            k = k.float()
            scale = kwargs.get("scaling", module.scaling)
            scores = torch.matmul(q_last, k.transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask[..., -1:, :].float()
            window = kwargs.get("sliding_window", module.sliding_window)
            if window is not None and scores.shape[-1] > window:
                scores[..., :-window] = torch.finfo(scores.dtype).min
            probs = torch.softmax(scores, dim=-1)[0, :, 0, :].detach().cpu()
            captured[layer] = probs
        return old_sdpa(module, query, key, value, attention_mask, **kwargs)

    def only_last_position(_module, args):
        return (args[0][:, -1:, :], *args[1:])

    ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = capture_last_query
    head_handle = model.lm_head.register_forward_pre_hook(only_last_position)
    start = time.time()
    try:
        with torch.inference_mode():
            out = model(**inputs, use_audio_in_video=True, use_cache=False,
                        output_attentions=False, output_hidden_states=False, return_dict=True)
    finally:
        ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = old_sdpa
        head_handle.remove()
    elapsed = time.time() - start
    result = {
        "idx": IDX, "gold": item["gold"], "question": row["question"],
        "rendered_prompt": rendered, "normalization": norm,
        "input_length": len(ids), "video_token_id": video_id, "audio_token_id": audio_id,
        "video_token_count": len(video_pos), "audio_token_count": len(audio_pos),
        "attention_query": "last prompt position (the position used to predict the first answer token)",
        "layers": {}, "forward_seconds": round(elapsed, 3),
        "note": "These are attention weights for one query position only. High attention mass shows what is read; it does not establish causal importance or semantic correctness.",
    }
    for layer, weights in captured.items():
        mean = weights.mean(0)
        sums = {
            "video_marker_positions": float(mean[video_pos].sum()) if video_pos else 0.0,
            "audio_marker_positions": float(mean[audio_pos].sum()) if audio_pos else 0.0,
            "question_and_other_text_positions": float(mean[text_pos].sum()) if text_pos else 0.0,
        }
        top_idx = torch.topk(mean, k=min(15, len(mean))).indices.tolist()
        result["layers"][str(layer)] = {
            "mean_head_attention_mass": sums,
            "per_head_attention_mass": {
                "video_markers": [float(weights[h, video_pos].sum()) if video_pos else 0.0 for h in range(weights.shape[0])],
                "audio_markers": [float(weights[h, audio_pos].sum()) if audio_pos else 0.0 for h in range(weights.shape[0])],
                "other_text": [float(weights[h, text_pos].sum()) if text_pos else 0.0 for h in range(weights.shape[0])],
            },
            "top_input_positions": [
                {"position": int(i), "attention": float(mean[i]),
                 "token_id": int(ids[i]), "token": tokenizer.convert_ids_to_tokens(int(ids[i]))}
                for i in top_idx
            ],
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logits = out.logits[0, -1].float()
    result["next_token_A_B_logits"] = {
        "A": float(logits[32]), "B": float(logits[33]),
        "margin_A_minus_B": float(logits[32] - logits[33]),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores": result["next_token_A_B_logits"],
                      "layers": {k: v["mean_head_attention_mass"] for k, v in result["layers"].items()},
                      "saved": str(OUT)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

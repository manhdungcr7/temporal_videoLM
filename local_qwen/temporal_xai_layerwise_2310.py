"""Layerwise logit-lens + same-shape modality interventions for sample 2310.

This does NOT expose a faithful natural-language chain of thought. It records
answer-token margins from each Thinker decoder layer while holding token/layout
positions fixed and zeroing one modality's encoded input at a time.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(r"F:\A_star_paper_NLP_ACL")
sys.path.insert(0, str(ROOT / "local_qwen"))
sys.path.insert(0, str(ROOT / "research_topics_learning/single_sample_prompt_lab"))
import temporal_audited_runner as audited
from runner import load, prepare_video, qwen_inputs, move_inputs

IDX = 2310
OUT = ROOT / "local_qwen/temporal_xai_layerwise_2310.json"


def logit_pair(hidden, model, id_a, id_b):
    # Apply the model's final RMSNorm to an intermediate layer state (logit lens).
    h = model.model.norm(hidden[:, -1:, :]).float()
    w = model.lm_head.weight[[id_a, id_b]].float()
    vals = F.linear(h, w)[0, 0]
    la, lb = float(vals[0]), float(vals[1])
    m = la - lb
    pa = 1.0 / (1.0 + math.exp(max(-80.0, min(80.0, -m))))
    return {"logit_A": la, "logit_B": lb, "margin_A_minus_B": m,
            "p_A_conditioned_on_A_B": pa}


def run_condition(model, processor, inputs, condition, ids):
    saved_audio = inputs["input_features"].clone()
    saved_video = inputs["pixel_values_videos"].clone()
    if condition == "zero_audio_features":
        inputs["input_features"].zero_()
    elif condition == "zero_video_pixels":
        inputs["pixel_values_videos"].zero_()
    elif condition != "full_av":
        raise ValueError(condition)

    layer_records = []
    layer_states_cpu = []
    handles = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        def make_hook(k):
            def hook(_module, _args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                last = h[:, -1:, :].detach()
                record = {"layer": k, **logit_pair(last, model, ids["A"], ids["B"])}
                record["query_state_norm"] = float(last.float().norm())
                layer_records.append(record)
                layer_states_cpu.append(last[0, 0].float().cpu())
            return hook
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    # The standard LM head would materialize [sequence_length, vocabulary_size]
    # logits. Slice its input to the final prompt position to fit the 4 GB GPU.
    def only_last_position(_module, args):
        return (args[0][:, -1:, :], *args[1:])
    head_handle = model.lm_head.register_forward_pre_hook(only_last_position)

    start = time.time()
    with torch.inference_mode():
        out = model(
            **inputs,
            use_audio_in_video=True,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    elapsed = time.time() - start

    for handle in handles:
        handle.remove()
    head_handle.remove()
    layer_records.sort(key=lambda x: x["layer"])
    final_logits = out.logits[0, -1].float()
    # Use actual last-position logits from the regular output for the final layer.
    la, lb = float(final_logits[ids["A"]]), float(final_logits[ids["B"]])
    m = la - lb
    final = {"logit_A": la, "logit_B": lb, "margin_A_minus_B": m,
             "p_A_conditioned_on_A_B": 1.0 / (1.0 + math.exp(max(-80.0, min(80.0, -m))))}
    top = torch.topk(final_logits, k=8).indices.tolist()
    result = {
        "condition": condition,
        "forward_seconds": round(elapsed, 3),
        "final_answer_token_scores": final,
        "top_next_tokens": [{"id": int(i), "token": processor.tokenizer.convert_ids_to_tokens(int(i)),
                              "logit": float(final_logits[i])} for i in top],
        "layerwise_logit_lens": layer_records,
        "_layer_states_cpu": layer_states_cpu,
        "input_sequence_length": int(inputs["input_ids"].shape[-1]),
    }
    del out, final_logits
    inputs["input_features"].copy_(saved_audio)
    inputs["pixel_values_videos"].copy_(saved_video)
    del saved_audio, saved_video
    torch.cuda.empty_cache()
    return result


def main():
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding="utf-8"))
    item = next(x for x in manifest["splits"]["development"] if x["idx"] == IDX)
    rows = json.loads((audited.DATA / "samples.json").read_text(encoding="utf-8"))
    row = rows[IDX]
    path, normalization = audited.normalize_av_clip(audited.DATA / item["video_path"])
    # Keep the exact benchmark wording; the prior forced-letter instruction
    # changed the model's answer and is therefore excluded from this run.
    prompt = row["question"]
    print(f"Loading Qwen once for {IDX}; same sequence positions in all interventions.", flush=True)
    model, processor = load()
    media = prepare_video(path)
    inputs, rendered = qwen_inputs(processor, prompt, media)
    inputs = move_inputs(inputs, model)
    tok = processor.tokenizer
    ids = {letter: tok.encode(letter, add_special_tokens=False) for letter in ("A", "B")}
    if any(len(v) != 1 for v in ids.values()):
        raise RuntimeError(f"Expected A/B to each be one token, got {ids}")
    ids = {k: v[0] for k, v in ids.items()}
    result = {
        "idx": IDX, "task": item["task"], "gold": item["gold"],
        "original_question": row["question"], "diagnostic_prompt": prompt,
        "rendered_prompt": rendered, "normalization": normalization,
        "checkpoint": "Qwen/Qwen2.5-Omni-3B", "revision": audited.REV,
        "quantization": "NF4 4-bit, double quantization, FP16 compute",
        "answer_token_ids": ids,
        "method": "Layerwise logit lens over the final prompt position; counterfactual zeroing of input_features or pixel_values_videos while preserving tensor shapes, special-token sequence, and time-position layout.",
        "caveat": "Logit lens is a probe of intermediate states, not a faithful explanation. Zero-feature interventions are diagnostics, not natural silence/black-video controls. One sample cannot establish a general mechanism.",
        "conditions": {},
    }
    for condition in ("full_av", "zero_audio_features", "zero_video_pixels"):
        print(f"Forward pass: {condition}", flush=True)
        result["conditions"][condition] = run_condition(model, processor, inputs, condition, ids)
        print(result["conditions"][condition]["final_answer_token_scores"], flush=True)
    base = result["conditions"]["full_av"]["layerwise_logit_lens"]
    base_states = result["conditions"]["full_av"]["_layer_states_cpu"]
    for cond in ("zero_audio_features", "zero_video_pixels"):
        other = result["conditions"][cond]["layerwise_logit_lens"]
        other_states = result["conditions"][cond]["_layer_states_cpu"]
        for i, record in enumerate(other):
            record["delta_margin_vs_full_av"] = record["margin_A_minus_B"] - base[i]["margin_A_minus_B"]
            record["cosine_similarity_query_state_vs_full_av"] = float(
                F.cosine_similarity(base_states[i].unsqueeze(0), other_states[i].unsqueeze(0)).item()
            )
        del result["conditions"][cond]["_layer_states_cpu"]
    del result["conditions"]["full_av"]["_layer_states_cpu"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

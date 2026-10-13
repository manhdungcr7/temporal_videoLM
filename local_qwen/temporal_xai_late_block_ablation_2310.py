"""Ablate final Thinker attention vs MLP residual branches on sample 2310."""
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

IDX = 2310
LATE_LAYERS = (34, 35)
OUT = ROOT / "local_qwen/temporal_xai_late_block_ablation_2310.json"


def forward_logits(model, processor, inputs, zero_branch=None):
    handles = []
    if zero_branch:
        branch_name, layer_ids = zero_branch
        for i in layer_ids:
            branch = getattr(model.model.layers[i], branch_name)
            def zero(_module, _args, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]), *output[1:])
                return torch.zeros_like(output)
            handles.append(branch.register_forward_hook(zero))
    def only_last(_module, args):
        return (args[0][:, -1:, :], *args[1:])
    head = model.lm_head.register_forward_pre_hook(only_last)
    start = time.time()
    try:
        with torch.inference_mode():
            out = model(**inputs, use_audio_in_video=True, use_cache=False,
                        output_attentions=False, output_hidden_states=False, return_dict=True)
        logits = out.logits[0, -1].float()
        ids = {"A": 32, "B": 33}
        score_a, score_b = float(logits[ids["A"]]), float(logits[ids["B"]])
        result = {
            "zeroed_branch": zero_branch,
            "seconds": round(time.time() - start, 3),
            "logit_A": score_a,
            "logit_B": score_b,
            "margin_A_minus_B": score_a - score_b,
            "top_next_tokens": [
                {"id": int(i), "token": processor.tokenizer.convert_ids_to_tokens(int(i)),
                 "logit": float(logits[i])}
                for i in torch.topk(logits, k=6).indices.tolist()
            ],
        }
        del out, logits
        torch.cuda.empty_cache()
        return result
    finally:
        head.remove()
        for handle in handles:
            handle.remove()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding="utf-8"))
    item = next(x for x in manifest["splits"]["development"] if x["idx"] == IDX)
    rows = json.loads((audited.DATA / "samples.json").read_text(encoding="utf-8"))
    row = rows[IDX]
    path, norm = audited.normalize_av_clip(audited.DATA / item["video_path"])
    print("Loading Qwen; question text is unchanged.", flush=True)
    model, processor = load()
    media = prepare_video(path)
    inputs, rendered = qwen_inputs(processor, row["question"], media)
    inputs = move_inputs(inputs, model)
    conditions = [
        ("full", None),
        ("zero_attention_layers_34_35", ("self_attn", LATE_LAYERS)),
        ("zero_mlp_layers_34_35", ("mlp", LATE_LAYERS)),
    ]
    result = {
        "idx": IDX, "task": item["task"], "gold": item["gold"],
        "question": row["question"], "rendered_prompt": rendered,
        "normalization": norm, "late_layers": list(LATE_LAYERS),
        "method": "Zero the selected residual branch outputs in the final Thinker blocks; keep input AV tensors and positions fixed; score next-token A/B logits.",
        "warning": "Block ablation is a causal intervention on this one quantized checkpoint/sample, not proof of a general mechanism. Token IDs A=32 and B=33 were verified in the local tokenizer.",
        "conditions": {},
    }
    for name, branch in conditions:
        print(f"Forward pass: {name}", flush=True)
        result["conditions"][name] = forward_logits(model, processor, inputs, branch)
        print(json.dumps(result["conditions"][name], ensure_ascii=True), flush=True)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()

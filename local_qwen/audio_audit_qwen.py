"""Diagnostic only: transcribe selected clips with Qwen audio-only input.
This script is not part of any proposed prompting method or benchmark score.
Usage: python audio_audit_qwen.py 307,8
"""
import json, sys
from pathlib import Path
import torch
import runner
from runner import ROOT, load
from lab_native_video import prepare_video, qwen_inputs, move_inputs

rows = json.loads((ROOT / 'data/svhalluc_benchmark/samples.json').read_text(encoding='utf-8'))
ids = [int(x) for x in sys.argv[1].split(',')]
model, processor = load()
for idx in ids:
    row = rows[idx]
    video_path = ROOT / 'data/svhalluc_benchmark' / row['video_path']
    media = prepare_video(video_path)
    prompt = ('Transcribe the spoken words in the audio as accurately as possible. '
              'Do not describe the video. If a phrase is unintelligible, mark it [unclear].')
    inputs, _ = qwen_inputs(processor, prompt, media, media_mode='audio')
    inputs = move_inputs(inputs, model)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=220, do_sample=False,
                                use_audio_in_video=True,
                                pad_token_id=processor.tokenizer.pad_token_id)
    text = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True)[0]
    print(json.dumps({'idx': idx, 'gold': row['answer'], 'transcript': text},
                     ensure_ascii=False), flush=True)
    del inputs, output, media
    torch.cuda.empty_cache()

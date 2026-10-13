"""CPU ASR alignment aid. Its output is never fed to Qwen or treated as gold."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research_topics_learning/single_sample_prompt_lab'))
from lab_media import decode_audio
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(2)
    model_dir = ROOT / 'research_topics_learning/external_evidence_implementation/models/asr'
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, attn_implementation='eager').eval()
    audio = decode_audio(ROOT / 'local_qwen/temporal_normalized_cfr30/600b36ed8f79cbc4e029_30fps.mp4')
    inputs = processor(audio[:int(5.4 * 16000)], sampling_rate=16000,
                       return_attention_mask=True, return_tensors='pt')
    with torch.inference_mode():
        result = model.generate(**inputs, do_sample=False, num_beams=1,
            max_new_tokens=128, return_timestamps=True, return_token_timestamps=True,
            return_dict_in_generate=True)
    ids = result['sequences'][0].tolist()
    times = result['token_timestamps'][0].tolist()
    record = {'idx': 2310, 'provenance': 'Whisper base.en CPU diagnostic prediction, unverified word alignment; never supplied to Qwen',
        'transcript': processor.tokenizer.decode(ids, skip_special_tokens=True),
        'tokens': [{'id': i, 'text': processor.tokenizer.decode([i]), 'start_s': round(t, 3)}
                   for i, t in zip(ids, times)],
        'late_audio_translation_s': 7.0}
    out = ROOT / 'local_qwen/temporal_clock_speech_audit_2310.json'
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()

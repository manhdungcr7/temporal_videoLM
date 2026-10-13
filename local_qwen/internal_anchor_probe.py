"""Does the model internally 'know' the speech window? (local, NF4, eager attention)
For each short clip and each LLM layer/head: attention mass from the text positions after the media block (question +
assistant prefix) onto video tokens INSIDE vs OUTSIDE the audio-active window, normalised per token.
focus = (mass_in / n_in) / (mass_out / n_out); >1 means the head looks more at the speech-time frames.
Also stores the plain answer (first-token logits) for correlation. Labels used only for scoring."""
import json, sys, time, random
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD
from speech_gate import media_layout
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig

OUT = Path(__file__).with_name('internal_anchor_probe.jsonl')
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))


def load_eager():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-Omni-3B', revision=runner.REV, cache_dir=runner.CACHE,
        dtype=torch.float16, attn_implementation='eager', quantization_config=q, device_map={'': 0}).eval()
    return m, Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-3B', revision=runner.REV, cache_dir=runner.CACHE)


def main():
    maxdur = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    pick = []
    for task, n in (('CMTB', 16), ('TA', 8)):
        pool = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == task and AUD['clips'][r['video_path'].replace(chr(92), '/')]['dur'] <= maxdur]
        random.Random(2).shuffle(pool); pick += pool[:n]
    done = {json.loads(l)['idx'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    model, proc = load_eager()
    layers = model.model.layers
    letters = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    for i, r in pick:
        if i in done: continue
        t0 = time.time()
        p = r['video_path'].replace(chr(92), '/'); c = AUD['clips'][p]; dur = c['dur']; s0, s1 = c['first'] * dur, c['last'] * dur
        media = prepare_video(cut(BASE / p, 0, dur, 'full'))
        inp, _ = qwen_inputs(proc, r['question'], media, 'av'); inp = move_inputs(inp, model)
        pos, sec, is_v, is_a, last_media = media_layout(model, proc, inp)
        spg = float(inp['video_second_per_grid'].reshape(-1)[0])
        L = inp['input_ids'].shape[1]
        dev = inp['input_ids'].device
        rows_q = torch.arange(L, device=dev) > last_media
        is_v = is_v.to(dev); inside = ((sec + spg > s0) & (sec < s1)).to(dev)
        vin, vout = is_v & inside, is_v & ~inside
        n_in, n_out = int(vin.sum()), int(vout.sum())
        stats = []
        def make_hook(li):
            def hook(mod, args, out):
                w = out[1]
                if w is None: return
                wq = w[0][:, rows_q].float().sum(1)          # heads x L : summed over query rows
                mi = wq[:, vin].sum(-1); mo = wq[:, vout].sum(-1); ma = wq[:, is_a.to(dev)].sum(-1); mt = wq.sum(-1)
                stats.append((li, mi.cpu().numpy(), mo.cpu().numpy(), ma.cpu().numpy(), mt.cpu().numpy()))
                return (out[0], None)
            return hook
        hs = [layers[k].self_attn.register_forward_hook(make_hook(k)) for k in range(len(layers))]
        try:
            with torch.no_grad():
                lg = model(**inp, use_audio_in_video=True).logits[0, -1].float()
        finally:
            for h in hs: h.remove()
        k = 2 if r['task_type_abbr'] == 'TA' else 3
        pred = int(torch.argmax(lg[letters[:k]]))
        stats.sort(key=lambda x: x[0])
        mi = np.stack([s[1] for s in stats]); mo = np.stack([s[2] for s in stats]); ma = np.stack([s[3] for s in stats]); mt = np.stack([s[4] for s in stats])
        focus = (mi / max(n_in, 1)) / (mo / max(n_out, 1) + 1e-9)
        rec = dict(idx=i, task=r['task_type_abbr'], gold='ABC'.index(r['answer']), pred=pred, dur=round(dur, 1), win=[round(s0, 1), round(s1, 1)],
                   n_in=n_in, n_out=n_out, L=L, log_focus=np.log(focus + 1e-9).round(3).tolist(),
                   video_share=((mi + mo) / mt).round(4).tolist(), audio_share=(ma / mt).round(4).tolist(), sec=round(time.time() - t0))
        with OUT.open('a') as f: f.write(json.dumps(rec) + '\n')
        lf = np.array(rec['log_focus'])
        print(i, r['task_type_abbr'], 'gold', rec['gold'], 'pred', pred, 'L', L, 'n_in/out', n_in, n_out,
              'median log-focus over heads', round(float(np.median(lf)), 3), 'share heads focus>1', round(float((lf > 0).mean()), 2), rec['sec'], 's', flush=True)
        del inp; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

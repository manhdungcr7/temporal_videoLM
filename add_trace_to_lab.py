"""One-off patch: add the step-by-step dry-run trace to kaggle_temporal_lab.py (idempotent)."""
import ast
from pathlib import Path

p = Path(__file__).with_name('kaggle_temporal_lab.py')
s = p.read_text(encoding='utf-8')
if 'def trace_item' in s:
    print('already patched')
    raise SystemExit

TRACE = r'''

# ----------------------------------------------------------------------------------------------------------------------
# Dry-run trace: input processing -> tokens -> positions -> logits per layer -> attention, for a few items.
# Everything is written to trace/trace_<idx>.json. Failures here never stop the main run.
# ----------------------------------------------------------------------------------------------------------------------
def runs_of_types(n, is_v, is_a, sec, pos_t):
    """Run-length encode the token sequence into text / video / audio runs with their time and TMRoPE ranges."""
    kind = lambda j: 'video' if bool(is_v[j]) else ('audio' if bool(is_a[j]) else 'text')
    out = []; start = 0
    for j in range(1, n + 1):
        if j == n or kind(j) != kind(start):
            seg = dict(kind=kind(start), start=start, end=j - 1, n=j - start,
                       rope_t_min=int(pos_t[start:j].min()), rope_t_max=int(pos_t[start:j].max()))
            if seg['kind'] != 'text':
                seg.update(sec_min=round(float(sec[start:j].min()), 3), sec_max=round(float(sec[start:j].max()), 3))
            out.append(seg); start = j
    return out


def topk_tokens(lg, k=8):
    v, i = torch.topk(torch.log_softmax(lg.float(), -1), k)
    return [dict(token=PROC.tokenizer.decode([int(t)]), id=int(t), logprob=round(float(x), 3)) for x, t in zip(v, i)]


def trace_item(task, r):
    orig = fetch(r['video_path'].replace(BS, '/'))
    full = cut(orig, 0, 10 ** 4, 'full')
    a = decode_audio(full); dur = len(a) / 16000
    if dur > MAX_DUR: return None
    w = speech_window(a)
    if w is None: return None
    s0, s1 = w
    media = prepare_video(full)
    stem, texts = parse_question(r['question']); k = len(texts)
    prompt = render(stem, texts, tuple(range(k)))
    inp, rendered = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    L = inp['input_ids'].shape[1]
    tr = dict(task=task, question=r['question'], gold_letter=r['answer'], prompt=prompt)
    # ---- 1. input processing ---------------------------------------------------------------------------------------
    vshape = list(media['videos'][0].shape)
    tr['preprocess'] = dict(
        file_duration_s=round(dur, 3), speech_window_s=[round(s0, 3), round(s1, 3)], sampled_fps=media['fps'],
        video_tensor_shape_TCHW=vshape, n_frames=vshape[0], frame_times_s=[round(i / media['fps'], 3) for i in range(vshape[0])],
        audio_samples=int(len(media['audio'][0])), audio_seconds=round(len(media['audio'][0]) / 16000, 3),
        max_pixels_per_frame=media['max_pixels_per_frame'], video_grid_thw=inp['video_grid_thw'].tolist(),
        second_per_grid=inp['video_second_per_grid'].reshape(-1).tolist(),
        audio_feature_frames=int(inp['feature_attention_mask'].sum()), input_features_shape=list(inp['input_features'].shape),
        n_input_tokens=L, chat_template_head=rendered[:700], chat_template_tail=rendered[-400:])
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    ids = inp['input_ids'][0].cpu(); pos_t = pos[0, 0].cpu(); sec_c = sec.cpu(); vi = is_v.cpu(); ai = is_a.cpu()
    tr['tokens'] = dict(n_video=int(vi.sum()), n_audio=int(ai.sum()), n_text=int(L - vi.sum() - ai.sum()), last_media_index=last_media,
                        runs=runs_of_types(L, vi, ai, sec_c, pos_t),
                        text_tokens_after_media=[PROC.tokenizer.decode([int(t)]) for t in ids[last_media + 1:]][:300])
    spg = float(inp['video_second_per_grid'].reshape(-1)[0])
    inside = ((sec_c + spg > s0) & (sec_c < s1))
    tr['window_vs_tokens'] = dict(video_tokens_inside=int((vi & inside).sum()), video_tokens_outside=int((vi & ~inside).sum()),
                                  video_tokens_per_second_bin=[int((vi & (sec_c >= b) & (sec_c < b + 1)).sum()) for b in range(int(dur) + 1)])
    # ---- 2. logits for several conditions ---------------------------------------------------------------------------
    with torch.no_grad():
        plain = MODEL_(**inp, use_audio_in_video=True, output_hidden_states=True)
    lg_plain = plain.logits[0, -1].float()
    conds = {'plain': lg_plain,
             'explicit_ids_and_mask_kappa1 (must equal plain)': forward_last(inp, 'ts', kappa=1.0),
             'ts05': forward_last(inp, 'ts', kappa=0.5), 'ts02': forward_last(inp, 'ts', kappa=0.2),
             'gate_all': forward_last(inp, 'gate_all', (s0, s1)), 'gate_late': forward_last(inp, 'gate_late', (s0, s1))}
    if task == 'CMTB':
        conds['audio_block'] = forward_last(inp, 'audio_block', (s0, s1))
        conds['gate_audio'] = forward_last(inp, 'gate_audio', (s0, s1))
    tr['logits'] = {name: dict(letters={L_: round(float(lg[LETTERS[j]]), 3) for j, L_ in enumerate('ABC'[:k])}, top8=topk_tokens(lg),
                               max_abs_diff_vs_plain=round(float((lg.cpu() - lg_plain.cpu()).abs().max()), 4)) for name, lg in conds.items()}
    # ---- 3. logit lens at the last position (plain forward) ----------------------------------------------------------
    hs = plain.hidden_states; head = MODEL_.lm_head; norm = MODEL_.model.norm; lens = []
    with torch.no_grad():
        for li, h in enumerate(hs):
            x = h[0, -1].to(head.weight.device, head.weight.dtype)
            if li < len(hs) - 1: x = norm(x.to(norm.weight.device)).to(head.weight.device)
            lg = head(x).float()
            lens.append(dict(layer=li, letters=[round(float(lg[LETTERS[j]]), 3) for j in range(k)],
                             top1=PROC.tokenizer.decode([int(lg.argmax())]), hidden_norm=round(float(h[0, -1].float().norm()), 2)))
    tr['logit_lens_plain'] = lens
    del plain, hs
    torch.cuda.empty_cache()
    # ---- 4. attention (eager) -----------------------------------------------------------------------------------------
    try:
        MODEL_.set_attn_implementation('eager')
        rows_q = torch.arange(L) > last_media
        vin, vout = (vi & inside), (vi & ~inside)
        bins = [(vi & (sec_c >= b) & (sec_c < b + 1)) for b in range(int(dur) + 1)]
        stats = {}
        def mk(li):
            def hook(mod, args, out):
                wts = out[1]
                if wts is None: return
                dv = wts.device
                qa = wts[0][:, rows_q.to(dv)].float().sum(1); ql = wts[0][:, -1].float()
                rec = dict(video_in=qa[:, vin.to(dv)].sum(-1).cpu().tolist(), video_out=qa[:, vout.to(dv)].sum(-1).cpu().tolist(),
                           audio=qa[:, ai.to(dv)].sum(-1).cpu().tolist(), total=qa.sum(-1).cpu().tolist(),
                           last_video_in=ql[:, vin.to(dv)].sum(-1).cpu().tolist(), last_video_out=ql[:, vout.to(dv)].sum(-1).cpu().tolist(),
                           last_audio=ql[:, ai.to(dv)].sum(-1).cpu().tolist())
                if li in (8, 17, 26, 35):
                    rec['last_row_mass_per_second_bin_mean_over_heads'] = [round(float(ql[:, m.to(dv)].sum(-1).mean()), 5) for m in bins]
                stats[li] = rec
                return (out[0], None)
            return hook
        hooks = [MODEL_.model.layers[li].self_attn.register_forward_hook(mk(li)) for li in range(len(MODEL_.model.layers))]
        try:
            with torch.no_grad(): MODEL_(**inp, use_audio_in_video=True)
        finally:
            for h in hooks: h.remove()
            MODEL_.set_attn_implementation('sdpa')
        tr['attention'] = dict(n_video_in=int(vin.sum()), n_video_out=int(vout.sum()), n_audio=int(ai.sum()),
                               note='per layer and head: attention mass summed over the question rows (video_in/out, audio, total) and from the last position only (last_*)',
                               per_layer=[stats[li] for li in sorted(stats)])
    except Exception as e:
        try: MODEL_.set_attn_implementation('sdpa')
        except Exception: pass
        tr['attention'] = dict(error=repr(e)[:300])
    return tr


def run_traces(mine):
    out_dir = OUT / 'trace'; out_dir.mkdir(exist_ok=True)
    per_task = {}
    for i, r in mine:
        t = r['task_type_abbr']
        if per_task.get(t, 0) >= 2: continue
        f = out_dir / f'trace_{i}.json'
        if f.exists():
            per_task[t] = per_task.get(t, 0) + 1; continue
        try:
            tr = trace_item(t, r)
            if tr is None: continue
            tr['idx'] = i; f.write_text(json.dumps(tr), encoding='utf-8')
            per_task[t] = per_task.get(t, 0) + 1
            print('trace written', f.name, flush=True)
        except Exception as e:
            print('trace failed', i, repr(e)[:200], flush=True)
        torch.cuda.empty_cache()
'''
marker = '\n\ndef main():\n'
assert marker in s
s = s.replace(marker, TRACE + marker, 1)
old = "    print('model ready; shard', SHARD, 'of', NSHARD, 'items', len(mine), 'done', len(done), flush=True)"
assert old in s
s = s.replace(old, old + "\n    if SHARD == 0 and not os.environ.get('LAB_NO_TRACE'): run_traces(mine)", 1)
ast.parse(s)
p.write_text(s, encoding='utf-8')
print('patched; trace code present:', 'def trace_item' in s)

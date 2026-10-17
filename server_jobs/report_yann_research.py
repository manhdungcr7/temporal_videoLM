"""Write a truthful running research report; missing results stay missing."""
import collections,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/yann_temporal_research'
def get(n):return json.loads((D/n).read_text(encoding='utf-8'))
cells=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']
lines=['# Yann — autonomous temporal research, 2026-10-16','',
    '**Chưa tìm được phương pháp mạnh. R1–R4 chưa cải thiện micro8. R5 oracle-trained readout cho tín hiệu cơ chế, nhưng R6/R7 retrieval chưa chuyển được lợi ích đó sang suy luận tự tìm bằng chứng. Xem các vòng bổ sung bên dưới.**','',
    'Mọi micro dùng cùng8 IDs, rotation0; chưa phải full40 với110 rotations. Bộ40 đã là development qua nhiều vòng. Các public Perception Test splits chỉ là train/validation/diagnostic nhỏ theo video, không phải subject-disjoint final test. Không có CVPR/Oral acceptance claim.','',
    '## R1 — media causal topology','']
for phase,key in [('flow','conditions'),('questionflow','conditions'),('probe_eval','mc'),('binding_probe_eval','mc')]:
    if not all((D/f'{phase}_s{s}.json').exists() for s in (1,2)):continue
    rows=[r for s in (1,2) for r in get(f'{phase}_s{s}.json')['rows']]
    conditions=list(rows[0][key]);counts={c:sum(r[key][c]['correct'] for r in rows) for c in conditions}
    if phase=='questionflow':lines+=['## R2 — question-only feedback','']
    if phase=='probe_eval':lines+=['## R3 — learned memory/readout transfer','']
    if phase=='binding_probe_eval':lines+=['## R4 — physical-pair trained memory transfer','']
    lines+=['| Condition | Đúng/sai/8 | Cứu/hại so native |','|---|---|---|']
    for c,n in counts.items():
        rescue=[r['idx'] for r in rows if r[key][c]['correct'] and not r[key]['native']['correct']]
        harm=[r['idx'] for r in rows if not r[key][c]['correct'] and r[key]['native']['correct']]
        lines.append(f'| {c} | {n}/{8-n}/8 | {rescue}/{harm} |')
    lines+=['','| Task/gold | N | '+' | '.join(conditions)+' |','|---|---:|'+'---|'*len(conditions)]
    for cell in cells:
        rr=[r for r in rows if r['task']+'/'+r['gold']==cell];n=len(rr)
        values=[f"{sum(r[key][c]['correct'] for r in rr)}/{n-sum(r[key][c]['correct'] for r in rr)}" if n else 'Chưa đo' for c in conditions]
        lines.append(f'| {cell} | {n} | '+' | '.join(values)+' |')
    lines+=['']
    if phase=='flow':lines+=['Native/explicit-mask parity0. Opening future media edges can change logits without improving QA; do not infer causal mechanism solely from sensitivity. Native media clock is TMRoPE metadata, not independent event ground truth.','']
    if phase=='questionflow':lines+=['New feedback edges exclude MC options; native media-media order and tail causal structure remain. No candidate changes an argmax. Format control has a different edge budget, so no clean semantic attribution claim.','']
    if phase=='probe_eval':
        lines+=['Time/content models selected only by8 public validation human-QA token NLL, both step50.1,350,528 trainable parameters; frozen backbone version checks pass. Same144 prepared cases/48 videos,32train/8validation/8diagnostic.400steps, fixedseed, matched data/budget.','',
            '| Variant | Validation human QA mean NLL | Diagnostic human QA mean NLL | Diagnostic localization top1 |','|---|---:|---:|---|']
        for s in (1,2):
            tr=get(f'trainprobe_s{s}.json');h=next(x for x in tr['history'] if x['step']==tr['selected_step'])
            v=h['validation']['human_qa'];dg=tr['diagnostic']['metrics']
            lines.append(f"| {tr['config']['name']} | {v['mean_lm']:.6f} | {dg['human_qa']['mean_lm']:.6f} | {dg['human_grounding']['ground_top1']}/{dg['human_grounding']['ground_n']} |")
        lines+=['','NLL and teacher-forced token accuracy are not semantic QA accuracy. A zero-content bank still retains time/modality and query features; call it a content ablation, not complete removal of every memory-related signal. No changed MC outcome under address shuffle is evidence against the tested added-address claim, not proof that the entire trained model never uses sensory input.','',
            '### Actual option-free generated text,32-token cap','']
        for r in rows:
            lines.append(f"**{r['idx']} — {r['task']}/{r['gold']}**\n")
            for name,v in r['generated'].items():lines.append(f"- {name}: {v['text']}\n")
        lines+=['No post-hoc regex semantic scores are used. Native MC and option-free generation use different prompts and can disagree. Oversized generated timestamps are visible failures, not correct localization.','']
lines+=['## Data/ontology and execution corrections','',
    'The initial public-data eligibility filter failed to exclude the sound label Muted. That is a mute-state annotation, not an audible event. R3 includes invalid derived sound-grounding questions and must not be called clean event supervision. Preserve all original scores and checkpoints; do not rewrite them after ontology correction. R4v2 excludes Muted endpoints, keeps valid previous pairs and records modality fallbacks/skips before fitting. Original selected videos and benchmark40 are unchanged.','',
    'R3 half-second support for visual mean tokens underestimates native temporal tubelet support (2/native sampled fps). R4 packaging preserves original spans and corrects visual support from actual sampled-frame count and source duration before fitting. This is a known metadata correction, not a QA-dependent parameter choice.','',
    'Preparation guard100 tokens rejected a102-token multi-instance target. Resume increased only that guard to768, keeping full targets/samples. First R4 construction also rejected an assumed exact4s/frame-grid shift: native sampling fps is often1.91–1.99, not exactly2. The corrected transform uses an even number of native frames and matching audio sample delay, preserving video tubelet pairing; source waveform/frame arrays are bitwise retained inside padding. All failed job logs remain.','',
    '## Next hypothesis and limits','',
    'R4 tests a binding-dependent residual identity and actual O/A/V/AV translations with human-derived endpoint-order targets. Paired supervision should make memory necessary under representation interchange; this remains a hypothesis. See round4_preflight.md. Uniform retrieval identity is a mathematical implementation contract, not semantic calibration or an accuracy result. Related IIT/causal abstraction, SyncRA, GroundFormer and LatentOmni constrain novelty claims.','',
    'Artifacts: flow_s*.json, questionflow_s*.json, trainprobe_s*.json, probe_eval_s*.json, frozen public selection manifest, public feature archives with entry SHA checks, validation-selected checkpoints, unit checks, all launch/job/error logs, literature_and_falsification.md.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('Saved running report:',D/'report.md')

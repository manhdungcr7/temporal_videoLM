"""Append completed rounds only; preserve per-query raw evidence and failed controls."""
import json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/yann_temporal_research'
runpy.run_path(str(R/'server_jobs/report_yann_research.py'))
def load(n):return json.loads((D/n).read_text(encoding='utf-8'))
L=['','## R4–R7 — independent physical intervention choices','',
'31 train recordings;8 validation;7 diagnostic. Same original public48 selection, no SVHalluc training. Questions are derived from human category/interval annotations and ask first onset; captions/class names can be coarse. Four O/A/V/AV versions and multiple questions are dependent observations. Report strict recording success and mixed flip-pattern success, not only individual accuracy.','',
'| Round / condition | Selected step | Train choices | Diagnostic choices | Diagnostic strict recordings | Diagnostic mixed-pattern recordings |','|---|---:|---|---|---|---|']
for phase in ('trainbinding','oraclelearn','distillretrieval','cosineretrieval'):
    for s in (1,2):
        p=D/f'{phase}_s{s}.json'
        if not p.exists():continue
        x=load(p.name);tr=x['train']['metrics'];dg=x['diagnostic']['metrics'];name=x['config']['name']
        L.append(f"| {phase}/{name} | {x['selected_step']} | {tr['correct']}/{tr['total']} | {dg['correct']}/{dg['total']} | {dg['joint_recording_correct']}/{dg['n_recordings']} | {dg['mixed_pattern_joint_correct']}/{dg['mixed_flip_pattern_recordings']} |")
L+=['','R5 oraclelearn uses annotation endpoints at input and O context throughout: privileged mechanism diagnosis, not deployment accuracy. Its native O-context diagnostic baseline is28/48, strict0/7. True-address oracle-trained reader reaches42/48, strict3/7, versus shuffled-address-trained30/48, strict0/7. Ordinary/oracle conditions differ from R4 ordinary source context; avoid treating every count as the same native baseline.','',
'R4 full selected checkpoint improves mostly human-QA token NLL; validation binary temporal NLL is worse than untrained. Diagnostic endpoint role0 top1 only1/48, role1 16/48; forcing correct endpoint attention at inference changes27→28/48. R5 separately trains the reader with correct endpoints and instead obtains42/48, showing why inference-only oracle is not an oracle-trained upper bound. R6 freezing that reader and learning retrieval produces21/48(grounding-only) or25/48(answer-distilled); oracle remains42/48. R7 cosine/media-query and cosine/question-text-query both select step0, with25/48 and20/48: no trained retrieval improvement.','',
'R4 micro8 ordinary/full both5/8, rescue2054 and harm1449; address-shuffle6/8, uniform-attention5/8. This does not establish temporal gain. Micro evaluation initially rejected two cached visual-support guesses below.8s; the final separately named clockfix uses each archived spg and extends the native horizon to cover all native cells. Native cache video clocks can exceed container duration (e.g.2150:15.52s start versus12.167s container); this is the known native decoder-clock mismatch, not PTS-calibrated physical time. Original cache/failed logs remain. R3 micro old half-second spans can have negative widths at beyond-container native timestamps; do not interpret it as a clean temporal-address evaluation.','',
'Completion audit checks same input prefix with alternate teacher-forced answer:240/240 query/media features have max difference0. This is finite-precision prefix independence, not proof of independent semantic ground truth. Geometry audit includes duplicate original/extra questions(56 diagnostic versions versus48 after fit dedup); exact-event-center order51/56 versus start-order truth, earliest oracle-cell support56/56. These counts do not prove that the full learned Fourier representation lacks onset information.','',
'## R8 — dense native-feature diagnostic','',
'| Feature head | Diagnostic within-recording macro AUC | Diagnostic macro top1 | Time-only AUC | Independent recordings |','|---|---:|---:|---:|---:|']
for s in (1,2):
    if not(D/f'denseprobe_s{s}.json').exists():continue
    x=load(f'denseprobe_s{s}.json');a=x['outputs']['x']['diagnostic'];b=x['outputs']['prior']['diagnostic']
    L.append(f"| {x['condition']} | {a['macro_recording_auc']:.4f} | {a['macro_recording_top1']:.4f} | {b['macro_recording_auc']:.4f} | {a['recordings']} |")
L+=['','34 training category names; every valid sound/action interval provides cell supervision. Exclude Muted and specified background-other labels. R8 initial max-overlap approximation was corrected to exact merged-interval union before the reported run; old source/job logs retained. Known-category localization is a privileged semantic-label diagnostic, not arbitrary-question QA. Stronger than time-only does not prove all visual/audio semantics or every layer is sufficient.','',
'## R9 — arbitrary-text dense scorer, held category names','',
'| Condition | Selected step | Diagnostic seen AUC/top1 | Diagnostic held AUC/top1 | Held recording/task count |','|---|---:|---|---|---|']
for s in (1,2):
    if not(D/f'semanticfield_s{s}.json').exists():continue
    x=load(f'semanticfield_s{s}.json');a=x['diagnostic']['metrics']['seen'];b=x['diagnostic']['metrics']['unseen']
    L.append(f"| {x['config']['condition']} | {x['selected_step']} | {a['macro_recording_auc']:.4f}/{a['macro_recording_top1']:.4f} | {b['macro_recording_auc']:.4f}/{b['macro_recording_top1']:.4f} | {b['recordings']}/{b['label_variant_tasks']} |")
L+=['','Text alignment helps seen-label localization versus semantic-corruption training, but held-label top1 is0 for the true-semantic branch and its held AUC is below the control on only3 origins/12 tasks. Therefore no open-vocabulary generalization claim is justified. Seen-label selection never uses unseen-label scores. Learned weighted-BCE outputs are not calibrated absence probabilities.','',
'## R10 — native upper-decoder reading path feasibility','']
for s in (1,2):
    if not(D/f'upperparity_s{s}.json').exists():continue
    x=load(f'upperparity_s{s}.json');L.append(f"Server{s}: `{json.dumps(x,ensure_ascii=False)}`\n")
L+=['Initial absolute hidden parity threshold failed. The later named audit reports relative/functional deviations and gradient feasibility separately; it does not retroactively pass the original criterion. Native frozen-prefix versus adapted text-tail decoding must match between training/deployment. This is a decoder-adaptation baseline, not a novelty or accuracy result.','',
'## Additional execution and interpretation notes','',
'Both initial R4 gradient runs stopped before any optimizer update because an overly strict finite-gradient guard prevented normal AMP overflow handling. R4 reruns use standard GradScaler overflow skipping with initial scale1024; both actual runs record0 skipped updates. Loss/data/seed/400-step budget unchanged. All original failures are retained.','',
'Preflight specifications, exact scripts, selected checkpoints, per-query predictions, physical feature archives and hashes, all job/status logs and separate mechanism diagnostics are saved in this directory and server_jobs. Training/validation/diagnostic reuse must be treated as development, with eventual fresh recordings/domains required. No successful QA40 method, novelty guarantee or CVPR/Oral acceptance has been established.']
with (D/'report.md').open('a',encoding='utf-8') as f:f.write('\n'.join(L)+'\n')
print('Appended completed rounds to',D/'report.md')

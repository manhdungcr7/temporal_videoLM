"""Build a benchmark-independent inference namespace from the audited helpers.

Only named functions/constants are loaded; no selection manifests, sample
records, task/gold variables, or experimental loops execute in this namespace.
Input source bundle is uploaded by cq_upload_inference.ps1.
"""
import ast,hashlib,importlib.metadata,sys
try:
    from silero_vad import load_silero_vad,get_speech_timestamps
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','silero-vad==6.2.3'],check=True,timeout=180)
    from silero_vad import load_silero_vad,get_speech_timestamps
FUNCTIONS={
    'native_generate','text_generate','extraction_memory','source_memory','reason_scores',
    'clean_observation','utterance_nodes','hierarchical_memory',
    'resolve_grounded_references','cross_source_checks',
    'parse_reference_plan','verify_audio_reference','measured_graph',
    'proposition_for_choice','grounded_choice_scores','episode_check',
    'neutral_reason_scores','occupancy_memory','choice_meanings','text_choice_scores',
    'interpret_choice','choice_constraints','cq_provenance_answer'
}
CONSTANTS={'EXTRACT_VIDEO','EXTRACT_AUDIO','GENERIC_REASON','DIALOGUE_STOP',
           'REL_MEANINGS','ABSTRACT_RELATIONS','SEMANTIC_CACHE'}
bundle=json.loads((OUT/'uploads/cq_inference_sources.json').read_text())
CQ_NS={name:globals()[name] for name in (
    'MODEL_','PROC','SYSTEM','cut','decode_audio','prepare_video','qwen_inputs','move_inputs',
    'rotations','score','text_only_scores')}
exec('import json, math, re, time\nfrom pathlib import Path\nimport numpy as np\nimport torch',CQ_NS)
CQ_NS.update(TOK=PROC.tokenizer,VAD_MODEL=load_silero_vad(),
             VAD_VERSION=importlib.metadata.version('silero-vad'),
             get_speech_timestamps=get_speech_timestamps)
compiled=[]
for entry in bundle['sources']:
    tree=ast.parse(entry['source'])
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS:
            nodes.append(node)
        elif isinstance(node,ast.Assign):
            names={target.id for target in node.targets if isinstance(target,ast.Name)}
            if names and names<=CONSTANTS:nodes.append(node)
    selected=ast.Module(body=nodes,type_ignores=[])
    exec(compile(selected,entry['path'],'exec'),CQ_NS)
    compiled.append(ast.unparse(selected))
assert FUNCTIONS<=CQ_NS.keys(), sorted(FUNCTIONS-CQ_NS.keys())
assert not ({'DATA','MANIFEST','ITEMS','gold','task'}&CQ_NS.keys())
# The older root-cause scorer handles only ABC. The audited generic scorer
# uses the same rotated-content protocol and also supports a fourth option.
CQ_NS['text_only_scores']=CQ_NS['text_choice_scores']
code='\n\n'.join(compiled)
digest=hashlib.sha256(code.encode()).hexdigest()
CQ_INFERENCE_METADATA=dict(source_sha256=digest,benchmark_variables_absent=True,
    model=MODEL_.config._name_or_path,model_revision=globals().get('MREV'),
    layers=len(MODEL_.model.layers),dtype=str(next(MODEL_.parameters()).dtype),
    vad_version=CQ_NS['VAD_VERSION'])
(OUT/'cq_inference_metadata.json').write_text(json.dumps(CQ_INFERENCE_METADATA,indent=2),encoding='utf-8')
RESULT=CQ_INFERENCE_METADATA
print('CQ_INFERENCE_READY',json.dumps(RESULT),flush=True)

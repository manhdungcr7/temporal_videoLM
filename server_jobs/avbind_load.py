"""Clean inference namespace: reuse perceptual adapters, never old routing."""
import ast,hashlib
allowed=('native_generate','text_generate','extraction_memory','clean_observation',
         'utterance_nodes','text_choice_scores')
AVB_NS={name:CQ_NS[name] for name in ('MODEL_','PROC','SYSTEM','TOK','VAD_MODEL','VAD_VERSION',
                                   'get_speech_timestamps','cut','decode_audio','prepare_video',
                                   'qwen_inputs','move_inputs','rotations','score')}
exec('import json, math, re, time\nfrom pathlib import Path\nimport numpy as np\nimport torch',AVB_NS)
bundle=json.loads((OUT/'uploads/cq_inference_sources.json').read_text())
constants={'EXTRACT_VIDEO','EXTRACT_AUDIO','DIALOGUE_STOP'}
for entry in bundle['sources']:
    tree=ast.parse(entry['source']);nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in allowed:nodes.append(node)
        elif isinstance(node,ast.Assign):
            names={x.id for x in node.targets if isinstance(x,ast.Name)}
            if names and names<=constants:nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),entry['path'],'exec'),AVB_NS)
source=(OUT/'uploads'/globals().get('AVB_SOURCE','avbind_reasoning.py')).read_text()
exec(compile(source,'avbind_reasoning.py','exec'),AVB_NS)
assert not {'DATA','ROWS','task','gold','MANIFEST','interpret_choice','resolve_grounded_references','verify_audio_reference'}&AVB_NS.keys()
AVB_METADATA=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  model=CQ_INFERENCE_METADATA['model'],model_revision=CQ_INFERENCE_METADATA['model_revision'],
                  layers=CQ_INFERENCE_METADATA['layers'],vad_version=CQ_INFERENCE_METADATA['vad_version'],
                  benchmark_namespace_absent=True,old_audio_first_resolver_absent=True)
(OUT/f'avbind_metadata_s{LAB_SERVER}.json').write_text(json.dumps(AVB_METADATA,indent=2),encoding='utf-8')
RESULT=AVB_METADATA
print('AVBIND_READY',json.dumps(RESULT),flush=True)

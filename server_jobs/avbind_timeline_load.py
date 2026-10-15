import hashlib
source = (OUT / 'uploads/avbind_timeline.py').read_text()
TIMELINE_NS = {name: globals()[name] for name in ('MODEL_','PROC','SYSTEM','cut','decode_audio',
                'prepare_video','move_inputs','rotations')}
TIMELINE_NS['TOK'] = PROC.tokenizer
exec(compile(source,'avbind_timeline.py','exec'),TIMELINE_NS)
assert not {'gold','task','DATA','ROWS','MANIFEST','BIND_READER','interpret_choice'} & TIMELINE_NS.keys()
RESULT = dict(api='avbind_timeline_answer',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              condition='raw clock-addressed records; isolated/plain matched ablation',question_type_routing=False)
print('TIMELINE_READY',json.dumps(RESULT),flush=True)

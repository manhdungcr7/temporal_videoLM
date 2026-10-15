source = (OUT / 'uploads/avbind_bounded.py').read_text()
exec(compile(source, 'avbind_bounded.py', 'exec'), NEURAL_NS)
RESULT = dict(api='avbind_bounded_answer', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              diagnostic_only=True, max_external_weight=.25, question_template_routing=False)
print('BOUNDED_MEMORY_READY', json.dumps(RESULT), flush=True)

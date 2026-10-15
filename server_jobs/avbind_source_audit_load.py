source = (OUT / 'uploads/avbind_source_audit.py').read_text()
exec(compile(source, 'avbind_source_audit.py', 'exec'), NEURAL_NS)
RESULT = dict(api='avbind_source_audit_answer', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              diagnostic_only=True, label_or_question_type_routing=False)
print('SOURCE_AUDIT_READY', json.dumps(RESULT), flush=True)

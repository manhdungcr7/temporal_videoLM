"""Recover the exact constructed training corpus after a worker restart."""
import types,sys,ast,random
module=types.ModuleType('avbind_synthetic_runtime');sys.modules[module.__name__]=module
exec(compile((OUT/'uploads/avbind_synthetic.py').read_text(),'avbind_synthetic.py','exec'),module.__dict__)
binding_module=types.ModuleType('avbind_binding_runtime');sys.modules[binding_module.__name__]=binding_module
binding_module.__dict__['SCENES']=module.SCENES
tree=ast.parse((OUT/'uploads/avbind_binding_cases.py').read_text())
tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.level==1)]
exec(compile(tree,'avbind_binding_cases.py','exec'),binding_module.__dict__)
train=module.generate_corpus(256,2026101405)+binding_module.generate_binding_cases(256,2026101405)
random.Random(2026101405).shuffle(train)
for row in train:
    mapping={n['id']:f'n{j}' for j,n in enumerate(row['nodes'])}
    for n in row['nodes']:n['id']=mapping[n['id']]
    row['evidence']={hop:[mapping[x] for x in ids] for hop,ids in row['evidence'].items()}
RESULT=dict(training_corpus_restored=len(train),benchmark_data_read=False)
print('TRAIN_CORPUS_RESTORED',json.dumps(RESULT),flush=True)

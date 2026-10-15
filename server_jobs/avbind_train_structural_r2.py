"""Contextual bank + first-hop clock exclusion + semantic/time counterexamples."""
import ast
source=(OUT/'uploads/avbind_train_structural.py').read_text()
source=source.replace('TRAIN_SEED=2026101403','TRAIN_SEED=2026101404')
source=source.replace('TRAIN_SIZE=256','TRAIN_SIZE=512').replace('STEPS=256','STEPS=768')
source=source.replace('lr=3e-4','lr=1e-4')
addition='''
binding_module=types.ModuleType('avbind_binding_runtime');sys.modules[binding_module.__name__]=binding_module
binding_module.__dict__['SCENES']=module.SCENES
tree=ast.parse((OUT/'uploads/avbind_binding_cases.py').read_text())
tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.level==1)]
exec(compile(tree,'avbind_binding_cases.py','exec'),binding_module.__dict__)
train=module.generate_corpus(TRAIN_SIZE//2,TRAIN_SEED)+binding_module.generate_binding_cases(TRAIN_SIZE//2,TRAIN_SEED)
random.Random(TRAIN_SEED).shuffle(train)
held=module.generate_corpus(38,TRAIN_SEED+1)+binding_module.generate_binding_cases(24,TRAIN_SEED+1)
'''
source=source.replace('train=module.generate_corpus(TRAIN_SIZE,TRAIN_SEED)',addition)
source=source.replace("labels=torch.tensor([[[float(n['id'] in ids) for n in row['nodes']]]],device=device)",
'''if index==0 and ids:
                # Semantic retrieval keeps every occurrence of the referenced
                # content/source before a later temporal selection.
                identities={(n['source'],n['content']) for n in row['nodes'] if n['id'] in ids}
                ids=[n['id'] for n in row['nodes'] if (n['source'],n['content']) in identities]
            labels=torch.tensor([[[float(n['id'] in ids) for n in row['nodes']]]],device=device)''')
source=source.replace('avbind_structural_training.jsonl','avbind_structural_training_r2.jsonl')
source=source.replace('avbind_structural_holdout.json','avbind_structural_holdout_r2.json')
source=source.replace('avbind_reader_checkpoint.pt','avbind_reader_checkpoint_r2.pt')
source=source.replace('avbind_reader_metadata.json','avbind_reader_metadata_r2.json')
exec(compile(source,'avbind_train_structural_r2.py','exec'),globals())

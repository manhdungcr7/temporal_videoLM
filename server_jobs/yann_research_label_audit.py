import collections,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/yann_temporal_research/data'
s=json.loads((D/'sound_localisation_train.json').read_text(encoding='utf-8'))
counts=collections.Counter(e['label'] for r in s.values() for e in r['sound_localisation'])
m=json.loads((D/'perception48_manifest.json').read_text(encoding='utf-8'))
selected=collections.Counter(e['label'] for r in m['rows'] for e in r['sounds'])
out=dict(all_sound_labels=dict(counts),selected_sound_labels=dict(selected))
(D/'sound_ontology_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

"""Read-only, de-duplicated audit of archived development results."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def audit(paths):
    records = {}
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row['idx']
            # A failed later attempt does not invalidate an archived success.
            if key not in records or 'binding' in row or 'binding' not in records[key]:
                records[key] = row
    cells = defaultdict(lambda: {'n': 0, 'base_correct': 0, 'binding_correct': 0,
                                 'errors': 0, 'decision_changes': 0})
    routing = []
    readout = []
    for index, row in records.items():
        cell = cells[row['task'] + '/' + row['gold']]
        cell['n'] += 1
        predictions = {}
        for method in ('base', 'binding'):
            if method in row:
                predictions[method] = 'ABCD'[max(range(len(row[method])), key=row[method].__getitem__)]
                cell[method + '_correct'] += int(predictions[method] == row['gold'])
        cell['errors'] += int('binding' not in row)
        cell['decision_changes'] += int(len(predictions) == 2 and predictions['base'] != predictions['binding'])
        trace = row.get('binding_trace', {})
        measured = trace.get('readout_audits', [])
        if measured:
            readout.append({'idx': index, 'measured_reads': len(measured),
                            'min_external': min(x['external'] for x in measured),
                            'max_external': max(x['external'] for x in measured),
                            'mean_external': sum(x['external'] for x in measured) / len(measured)})
        nodes = trace.get('nodes', [])
        by_source = []
        for item in trace.get('audits', []):
            mass = item.get('memory_mass', [])
            if len(mass) == len(nodes) and mass:
                totals = defaultdict(float)
                for node, weight in zip(nodes, mass):
                    totals[node['source']] += weight
                total = sum(totals.values())
                by_source.append({source: value / total if total else 0 for source, value in totals.items()})
        if by_source:
            routing.append({'idx': index, 'task': row['task'], 'gold': row['gold'],
                            'source_prior_shares': {s: sum(r.get(s, 0) for r in by_source) / len(by_source)
                                                    for s in by_source[0]}})
    return {'unique_samples': len(records), 'groups': dict(sorted(cells.items())),
            'source_prior_shares': routing,
            'measured_readout': readout,
            'warning': 'Reader prior shares are NOT measured final attention weights or causal explanations.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.files), ensure_ascii=False, indent=2))

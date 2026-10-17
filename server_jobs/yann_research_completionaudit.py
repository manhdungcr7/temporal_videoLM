"""Check that sensory/query feature caches do not depend on the future completion."""
import json,torch
YR_DIR=OUT/'yann_temporal_research';rows=[];missing=[]
for folder in ('binding48_v2','binding_evalextra'):
    for path in sorted((YR_DIR/folder).glob('*.pt')):
        if path.stem.endswith('_other'):continue
        c=torch.load(path,map_location='cpu',weights_only=True);foil=None
        for candidate in (path.with_name(path.stem+'_other.pt'),YR_DIR/'binding48'/f'{path.stem}_other.pt'):
            if candidate.exists():
                x=torch.load(candidate,map_location='cpu',weights_only=True)
                if x['question']==c['question'] and x['answer']==c['other_answer']:foil=x;break
        if foil is None:missing.append(str(path.relative_to(YR_DIR)));continue
        assert foil['input_prompt_len']==c['input_prompt_len']
        rows.append(dict(file=str(path.relative_to(YR_DIR)),q_max_error=float((c['q'].float()-foil['q'].float()).abs().max()),
            bank_max_error=float((c['bank'].float()-foil['bank'].float()).abs().max()),different_completion_lengths=len(c['target'])!=len(foil['target'])))
RESULT=dict(server=YR_SERVER,n=len(rows),missing=missing,rows=rows,
    max_query_error=max(r['q_max_error'] for r in rows),max_bank_error=max(r['bank_max_error'] for r in rows),
    note='Same input prefix, different teacher-forced continuation; tests finite-precision feature independence. No claim of independent semantic annotations.')
(YR_DIR/'completionaudit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in RESULT.items() if k not in ('rows','missing')}),flush=True)

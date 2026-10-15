"""Readout-timing ablation: reuse identical optimized slots, no retraining.

The earlier repository audits locate evidence reading around blocks15--24.
Test memory access from15 instead of23, with every fitted slot held fixed.
"""
import json,hashlib,time,gc
from pathlib import Path
import numpy as np
import torch

REPLAY_SERVER=globals().get('REPLAY_SERVER',1)
LO_SERVER=REPLAY_SERVER;LO_LIMIT=None
_source=(OUT/'uploads/latent_slot_opt40.py').read_text()
exec(compile(_source.split('_flags=[p.requires_grad',1)[0],'slot_opt_definitions_only','exec'),globals())
REPLAY_ORIGINAL_DIR=LO_DIR
REPLAY_DIR=OUT/'latent_slot_read15';REPLAY_DIR.mkdir(exist_ok=True)
REPLAY_SHA=hashlib.sha256((OUT/'uploads/latent_slot_read15.py').read_bytes()).hexdigest()
REPLAY_FUSE_FROM=15
_answer='def lo_answer('+_source.split('def lo_answer(',1)[1].split('\ndef lo_run(',1)[0]
assert _answer.count('li>=23')==2
exec(compile(_answer.replace('li>=23','li>=REPLAY_FUSE_FROM'),'only_readout_stage_changed','exec'),globals())
REPLAY_METHODS=['E0_native','E1_unoptimized_read15','E2_optimized_read15','E3_shuffled_read15',
    'E4_optimized_read23_reference','E5_unoptimized_read23_reference']

def replay_run():
    output=REPLAY_DIR/f'results_s{REPLAY_SERVER}.jsonl'
    done={json.loads(x)['idx'] for x in output.read_text().splitlines()} if output.exists() else set()
    original={r['idx']:r for r in [json.loads(x) for x in
        (REPLAY_ORIGINAL_DIR/f'results_s{REPLAY_SERVER}.jsonl').read_text().splitlines() if x.strip()]}
    assert len(original)==20,'Finish the fixed latent-slot fit before readout ablation'
    validated=False
    for sample in SL_SAMPLES:
        idx=sample['idx']
        if idx in done:continue
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        started=time.time();_,texts=parse_question(sample['question']);k=len(texts)
        payload=torch.load(SL_CACHE/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
        _,bank,_=lo_teacher(payload);banks={};items={}
        for s in ('video','audio'):
            _,banks[s],h=lo_teacher(payload,s);items[s]=lo_slots(payload,s,h);del h
        saved=torch.load(REPLAY_ORIGINAL_DIR/f'{idx}_optimized.pt',map_location='cpu',weights_only=False)
        assert saved['runtime_sha256']==LO_SHA
        memories={}
        for method,z in saved['parameters'].items():memories[method],_=lo_memory(payload,banks,items,z)
        del banks;gc.collect();torch.cuda.empty_cache()
        totals={m:np.zeros(k) for m in REPLAY_METHODS};audits=[]
        mapping={'E1_unoptimized_read15':'L1_unoptimized_memory',
                 'E2_optimized_read15':'L2_optimized_memory','E3_shuffled_read15':'L3_time_shuffled_target'}
        for rotation,perm in enumerate(rotations(k)):
            p=torch.load(SL_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
            assert p['manifest_sha256']==SL_SHA
            primary=json.loads((REPLAY_ORIGINAL_DIR/f'{idx}_rot{rotation}.json').read_text())
            scores={'E0_native':primary['scores']['L0_native'],
                'E4_optimized_read23_reference':primary['scores']['L2_optimized_memory'],
                'E5_unoptimized_read23_reference':primary['scores']['L1_unoptimized_memory']}
            parity={}
            if not validated:
                global REPLAY_FUSE_FROM
                REPLAY_FUSE_FROM=23
                try:
                    for method in ('L1_unoptimized_memory','L2_optimized_memory','L3_time_shuffled_target'):
                        lg=lo_answer(p,bank,memories[method],k)
                        lp=torch.log_softmax(lg,dim=-1).numpy()
                        error=float(np.max(abs(lp-np.array(primary['scores'][method]))))
                        assert error<=.06,('read23 reconstruction mismatch',method,error)
                        parity[method]=error
                finally:REPLAY_FUSE_FROM=15
                validated=True
            for method,original_method in mapping.items():
                scores[method]=torch.log_softmax(lo_answer(p,bank,memories[original_method],k),dim=-1).numpy().tolist()
            result=dict(idx=idx,rotation=rotation,perm=list(perm),scores=scores,
                manifest_sha256=SL_SHA,runtime_sha256=REPLAY_SHA,fit_runtime_sha256=LO_SHA,
                fusion_block=15,read23_parity=parity)
            (REPLAY_DIR/f'{idx}_rot{rotation}.json').write_text(json.dumps(result))
            for method,score in scores.items():
                for slot,opt in enumerate(perm):totals[method][opt]+=score[slot]
            audits.append({a:b for a,b in result.items() if a!='scores'})
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=SL_SHA,
            runtime_sha256=REPLAY_SHA,scores={m:s.tolist() for m,s in totals.items()},
            predictions={m:chr(65+int(s.argmax())) for m,s in totals.items()},audit=audits,
            parameters_refitted=False,seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done.add(idx);print('SLOT_READ15_SAMPLE',len(done),'/',len(SL_SAMPLES),json.dumps(rec),flush=True)
        del payload,bank,items,memories;gc.collect();torch.cuda.empty_cache()
    return dict(server=REPLAY_SERVER,completed=len(done),expected=len(SL_SAMPLES))

RESULT=replay_run()
print('SLOT_READ15_COMPLETE',json.dumps(RESULT),flush=True)

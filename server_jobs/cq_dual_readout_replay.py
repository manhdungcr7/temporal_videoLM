"""Replay the two measured readers. All tested weights are reported."""
source=OUT/f'cq_diagnostic_round2_s{LAB_SERVER}.jsonl'
records=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
valid=[r for r in records if 'error' not in r]
out=[]
for r in valid:
    rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'])
    for beta in (0.5,1.0,2.0):
        rec['dual_'+str(beta)]=(np.asarray(r['cq_completed'])+beta*np.asarray(r['cq_memory'])).round(3).tolist()
    out.append(rec)
(OUT/f'cq_dual_readout_replay_s{LAB_SERVER}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out),encoding='utf-8')
RESULT={str(beta):{'accuracy':acc(out,'dual_'+str(beta))[0],'n':acc(out,'dual_'+str(beta))[1]} for beta in (0.5,1.0,2.0)}
print('DUAL_REPLAY_RESULT',json.dumps(RESULT),flush=True)

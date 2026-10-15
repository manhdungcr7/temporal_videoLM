"""Source/time-grounded response verification, shared across arbitrary questions.

Related control: Liu et al., CVPR2025 video-grounded entailment. This is NOT a
reproduction of their entailment tree, nor a novel method claim. MC options are
candidate responses only; no task names, label identities, or question-pattern
router enter inference. A generated open-ended response can use the same critic.
"""
import json,time,gc,math
from pathlib import Path
import numpy as np
import torch
from transformers import DynamicCache

R40_CRITIC_FILE=OUT/f'research40_proposition_critic_s{R40_SERVER}.jsonl'
R40_CRITIC_DIR=OUT/'research40_proposition_critic';R40_CRITIC_DIR.mkdir(exist_ok=True)

def r40_compact_evidence(nodes,links):
    lines=['Independent, question-blind observations of the original recording.',
           'Speech is what was said, not proof that a mentioned event was visible.',
           'Visual intervals bound observations, not exact action onset or duration.',
           'ASR and visual probes can be wrong. Use the original recording to resolve conflicts.']
    for i,v in enumerate(n for n in nodes if n['kind']=='visual'):
        lines.append(f"V{i} silent video [{v['start']:.2f},{v['end']:.2f}]s: {v['text']}")
    for i,a in enumerate(n for n in nodes if n['kind'].startswith('audio')):
        lines.append(f"A{i} spoken episode [{a['start']:.2f},{a['end']:.2f}]s: {a['text']}")
        # Render actual ASR word clocks. Episode envelopes must not replace the
        # timing of each mentioned phrase, especially in long merged utterances.
        groups={}
        for w in a.get('words',[]):
            bucket=int(w['start']//2)
            groups.setdefault(bucket,[]).append(w)
        for ws in groups.values():
            lines.append(f"  word-time evidence [{min(w['start'] for w in ws):.2f},"
                         f"{max(w['end'] for w in ws):.2f}]s: "+''.join(w['text'] for w in ws).strip())
    lines.append('Optional source-only event search probes; scores are not calibrated probabilities.')
    for link in links:
        lines.append(f"A{link['audio_id']} search phrase: {link['event']}; "
                     f"null support={link['null_support']:.3f}; positive detection={link['detected']}.")
        # Show every probe, including negative ones; never force a positive
        # match or assign an occurrence time from a negative-only search.
        for match in link['support']:
            lines.append(f"  V{match['video_id']}: visible-event Yes-minus-No={match['score']:+.2f}")
    return '\n'.join(lines)+'\n\n'

def r40_critic_text_start(prompt):
    conv=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':[{'type':'text','text':prompt}]}]
    rendered=PROC.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    ids=PROC.tokenizer.encode(rendered,add_special_tokens=False)
    empty=DynamicCache(config=MODEL_.model.config)
    device=MODEL_.get_input_embeddings().weight.device
    return r40_cached_text(ids,empty,torch.zeros((3,1,1),dtype=torch.long,device=device),0)

def r40_critic_generate(start,limit=96):
    logits,cache,next_pos,offset=start
    tokens=[];eos={PROC.tokenizer.eos_token_id,PROC.tokenizer.convert_tokens_to_ids('<|im_end|>')}
    for _ in range(limit):
        tok=int(logits.argmax())
        if tok in eos:break
        tokens.append(tok)
        logits,cache,next_pos,offset=r40_cached_text([tok],cache,next_pos,offset)
    return dict(bank=cache.to_legacy_cache(),next_pos=next_pos,offset=offset,
                text=PROC.tokenizer.decode(tokens,skip_special_tokens=True),tokens=len(tokens))

def r40_critic_media_start(payload,original_question,prompt,supplement):
    bank,prefix=r40_prefix(payload)
    native_tail=PROC.tokenizer.decode(payload['input_ids'][0,prefix:].tolist(),skip_special_tokens=False)
    beginning=native_tail[:native_tail.index(original_question)]
    ending=native_tail[native_tail.rindex('<|im_end|>'):]
    ids=PROC.tokenizer.encode(beginning+supplement+prompt+ending,add_special_tokens=False)
    start_pos=payload['decoder']['position_ids'][:,:,prefix:prefix+1]
    cache=DynamicCache(ddp_cache_data=r40_gpu_bank(bank),config=MODEL_.model.config)
    return r40_cached_text(ids,cache,start_pos,prefix)

def r40_response_claim(question,response):
    prompt=('Convert this question and proposed response into a single declarative claim whose truth '
            'would mean that the response correctly answers the question. Preserve all entities, '
            'negation, and reference points exactly. Do not judge its truth, change the proposed '
            'response, or add evidence. Return only the claim.\nQuestion: '+question+
            '\nProposed response: '+response)
    return r40_critic_generate(r40_critic_text_start(prompt),72)['text']

def r40_critic_request(question,response,claim=None,proof=False):
    prompt='Original user question: '+question+'\nProposed response: '+response+'\n'
    if claim:
        prompt+='Candidate claim (a linguistic rewrite, not evidence): '+claim+'\n'
    prompt+=('Determine whether this proposed response correctly answers the original question. '
             'Preserve the question subjects, direction, and reference points. Judge what actually '
             'appears or is heard, rather than what a speaker says will happen. Use any required '
             'source or combination, in any order; do not assume two related events are simultaneous. ')
    if proof:
        prompt+=('Identify the relevant source observations and their actual time bounds, then explain '
                 'whether they support or contradict the candidate. Do not invent evidence. '
                 'Do not yet give the final judgement. Keep the analysis concise.')
    else:
        prompt+='Reply only Yes if it correctly answers the question, or No if it does not.'
    return prompt

def r40_critic_final(state,question,response):
    prompt=('<|im_end|>\n<|im_start|>user\n'+
            r40_critic_request(question,response)+'<|im_end|>\n<|im_start|>assistant\n')
    ids=PROC.tokenizer.encode(prompt,add_special_tokens=False)
    cache=DynamicCache(ddp_cache_data=state['bank'],config=MODEL_.model.config)
    lg,_,_,_=r40_cached_text(ids,cache,state['next_pos'],state['offset'])
    return float(lg[R40_YES[0]]-lg[R40_NO[0]])

evidence_rows=r40_json_rows(OUT/f'research40_verified_graph_v2_s{R40_SERVER}.jsonl')
assert len(evidence_rows)==len(R40_SAMPLES),'Complete verified-v2 extraction first; do not change IDs.'
evidence_by_id={r['idx']:r for r in evidence_rows}
base_by_id={r['idx']:r for r in r40_json_rows(R40_BASE_FILE)}
records=r40_json_rows(R40_CRITIC_FILE);done={r['idx'] for r in records}
print('CRITIC_START',R40_SERVER,len(done),'/',len(R40_SAMPLES),flush=True)
for row in R40_SAMPLES:
    if row['idx'] in done:continue
    started=time.time();question,responses=parse_question(row['question'])
    evidence=evidence_by_id[row['idx']]
    supplement=r40_compact_evidence(evidence['observations'],evidence['links'])
    payload=torch.load(R40_CACHE/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=False)
    payload['sample_idx']=row['idx'];candidates=[]
    for i,response in enumerate(responses):
        path=R40_CRITIC_DIR/f"{row['idx']}_candidate{i}.json"
        if path.exists():candidates.append(json.loads(path.read_text()));continue
        claim=r40_response_claim(question,response)
        direct_prompt=r40_critic_request(question,response)
        # Use the native media bank, not a hard audio->visual pipeline. A single
        # candidate is evaluated independently, without other response options.
        direct_logits,_,_,_=r40_critic_media_start(payload,question,direct_prompt,supplement)
        direct=float(direct_logits[R40_YES[0]]-direct_logits[R40_NO[0]])
        prior_logits,_,_,_=r40_critic_text_start(direct_prompt)
        prior=float(prior_logits[R40_YES[0]]-prior_logits[R40_NO[0]])
        proof_prompt=r40_critic_request(question,response,claim,proof=True)
        state=r40_critic_generate(r40_critic_media_start(payload,question,proof_prompt,supplement),96)
        proof_score=r40_critic_final(state,question,response)
        candidate=dict(response=response,claim=claim,direct=direct,prior=prior,
                       proof_score=proof_score,analysis=state['text'],tokens=state['tokens'])
        path.write_text(json.dumps(candidate,indent=2),encoding='utf-8');candidates.append(candidate)
        del state;gc.collect();torch.cuda.empty_cache()
        print('CRITIC_CANDIDATE',row['idx'],i,json.dumps(candidate),flush=True)
    scores=dict(proposition_direct=[c['direct'] for c in candidates],
                proposition_calibrated=[c['direct']-c['prior'] for c in candidates],
                proposition_proof=[c['proof_score'] for c in candidates])
    rec=dict(idx=row['idx'],task=row['task'],gold=row['gold'],status='ok',
             manifest_sha256=R40_MANIFEST['manifest_sha256'],scores=scores,
             candidates=candidates,seconds=round(time.time()-started,2))
    with R40_CRITIC_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(row['idx'])
    merged=[dict(r,scores=dict(base_by_id[r['idx']]['scores'],**r['scores'])) for r in records]
    r40_snapshot(merged,'proposition_critic')
    print('CRITIC_SAMPLE',row['idx'],json.dumps(scores),rec['seconds'],flush=True)
    del payload;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'proposition_critic_done')
print('CRITIC_SUMMARY',json.dumps(RESULT),flush=True)

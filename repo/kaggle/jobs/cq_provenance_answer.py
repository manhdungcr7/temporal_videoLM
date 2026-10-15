"""Dataset-independent CQ provenance inference in the loaded Omni namespace.

Load with server_jobs/cq_load_inference.py. Default: frozen round15 readout.
No task/gold is accepted. The candidate is validated on diagnostic data only.
"""
def cq_provenance_answer(video_path,question,answer_options,observation_cache=None,mode='episode_constraints'):
    started=time.time()
    src=Path(video_path)
    wave=decode_audio(src)
    dur=len(wave)/16000
    if observation_cache is not None and Path(observation_cache).exists():
        observations=json.loads(Path(observation_cache).read_text())
    else:
        observations=extraction_memory(src,dur)
        if observation_cache is not None:
            Path(observation_cache).write_text(json.dumps(observations,indent=2),encoding='utf-8')
    nodes=utterance_nodes(src,wave,dur)
    _,plan=resolve_grounded_references(observations,nodes,question)
    references=parse_reference_plan(plan,nodes)
    checks,grounded=[],[]
    for ref in references:
        node=verify_audio_reference(ref['event'],nodes)
        grounded.append(node)
        checks.extend(cross_source_checks(src,observations,[node]))
    if mode=='verified_peak':
        memory=measured_graph(checks) if checks else hierarchical_memory(observations,nodes,False)
        plain=text_only_scores(memory+question,answer_options)
        choices,propositions=grounded_choice_scores(memory,question,answer_options)
        final,reasoning=reason_scores(memory,question,answer_options)
        annotated,predicates=answer_options,[]
    elif mode=='episode_constraints':
        checks=[episode_check(c,observations) for c in checks]
        memory=occupancy_memory(checks) if checks else hierarchical_memory(observations,nodes,False)
        annotated,predicates=choice_constraints(question,answer_options)
        # Preserve a broad scene-memory path when no temporal predicate is
        # inferred, instead of imposing an audio-event relation on every QA.
        if predicates and all(op=='unsupported' for op,trace in predicates):
            memory=hierarchical_memory(observations,nodes,False)
        final=text_only_scores(memory+question,annotated)
        plain=choices=propositions=reasoning=None
    else:
        raise ValueError('Unknown CQ readout mode')
    prediction=int(np.argmax(final))
    return dict(dur=round(dur,3),utterances=nodes,plan=plan,references=references,
                audio_grounding=grounded,checks=checks,memory=memory,
                cq_verified_graph=plain,cq_verified_choices=choices,
                cq_verified_reason=final if mode=='verified_peak' else None,
                cq_constraint_graph=final if mode=='episode_constraints' else None,
                propositions=propositions,reasoning=reasoning,
                annotated_options=annotated,predicates=predicates,mode=mode,
                scores=final,prediction=prediction,answer=answer_options[prediction],
                latency_sec=round(time.time()-started,2),vad_version=VAD_VERSION,
                label_or_task_used_for_inference=False)

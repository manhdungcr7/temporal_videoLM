"""Exact textual structural training cases; no LLM or raw AV grounding claim.

Only question, answer_options, and nodes are inputs. Metadata holds PRIVATE
construction identities, never inferred from words, tense, or simultaneity.
Empty evidence lists are null reads. Intervals use strict positive overlap.
"""
import copy
import random

from .avbind_synthetic import SCENES

FAMILIES = ("narration_visible", "caption_visible", "narration_sound", "caption_sound",
            "reverse_words", "before", "after", "duration", "duration_compare",
            "absent_event", "missing_reference", "reverse_silent")
TENSES = ("future", "past", "neutral")
NULL = "The referred event is not observed"


def _phrase(content, tense):
    return {"future": f"Later, the following event will occur: {content}.",
            "past": f"Earlier, the following event occurred: {content}.",
            "neutral": f"The event being described is: {content}."}[tense]


def _interval(node):
    return f'[{node["interval"][0]:g}, {node["interval"][1]:g}] s'


def _event(node):
    return f'{node["content"]} at {_interval(node)}'


def label_binding_case(nodes, query, identities):
    """Construction oracle: identities resolve meaning; intervals resolve time.

    Returns (answer, hop1 IDs, hop2 IDs). Reject ambiguous matches and ties.
    identities maps node IDs to {event, role}; it must never be a model input.
    """
    refs = [n for n in nodes if n["id"] == query["reference"]]
    if not refs:
        return "The reference is not observed", [], []
    ref, op = refs[0], query["op"]
    pool = [n for n in nodes if n["source"] == query["source"] and n["kind"] == query["kind"]]
    if op == "overlap":
        a, b = ref["interval"]
        if any(n["interval"][1] == a or n["interval"][0] == b for n in pool):
            raise ValueError("touching intervals are excluded")
        targets = [n for n in pool if max(a, n["interval"][0]) < min(b, n["interval"][1])]
    else:
        identity = identities[ref["id"]]["event"]
        targets = [n for n in pool if identities[n["id"]] == dict(event=identity, role="occurrence")]
    if len(targets) > 1:
        raise ValueError("ambiguous binding or overlap")
    if not targets:
        return ("No words overlap this visible activity" if op == "overlap" else NULL), [ref["id"]], []
    target, h1 = targets[0], [ref["id"]]
    h2 = [target["id"]]
    if op in ("before", "after"):
        before = op == "before"
        neighbors = [n for n in pool if (n["interval"][1] < target["interval"][0] if before
                                        else n["interval"][0] > target["interval"][1])]
        if not neighbors:
            return "No qualifying event is observed", h1, h2
        edge = 1 if before else 0
        bound = (max if before else min)(n["interval"][edge] for n in neighbors)
        neighbors = [n for n in neighbors if n["interval"][edge] == bound]
        if len(neighbors) != 1:
            raise ValueError("temporal tie")
        return _event(neighbors[0]), h1, h2 + [neighbors[0]["id"]]
    if op == "duration_compare":
        other = next(n for n in nodes if n["id"] == query["comparison"])
        d, e = (n["interval"][1] - n["interval"][0] for n in (target, other))
        if d == e:
            raise ValueError("duration tie")
        return _event(target if d > e else other), h1, h2 + [other["id"]]
    if op == "duration":
        return f'{target["interval"][1] - target["interval"][0]:g} seconds', h1, h2
    if op not in ("semantic", "overlap"):
        raise ValueError("unknown operation")
    return (target["content"] if op == "overlap" else _interval(target)), h1, h2


def _finish(record, rng):
    meta, nodes = record["metadata"], record["nodes"]
    q = meta["query"]
    ref = next((n for n in nodes if n["id"] == q["reference"]), None)
    text = ref["content"] if ref else meta["reference_content"]
    mention = f'{"the displayed caption" if meta["reference_source"] == "video" else "the utterance"} "{text}"'
    noun = "visible activity" if q["source"] == "video" else "non-speech sound"
    question = f"Which interval contains the {noun} referred to by {mention}?"
    pool = [n for n in nodes if n["source"] == q["source"]]
    choices = [_interval(n) for n in pool] + [NULL, "The reference is not observed"]
    if q["op"] == "overlap":
        question = f'Which words are heard during the visible activity "{text}"?'
        choices = [n["content"] for n in pool if n["kind"] == "speech"] + ["No words overlap this visible activity"]
    elif q["op"] in ("before", "after"):
        question = f"Which visible activity ends closest before the activity referred to by {mention} begins?" if q["op"] == "before" else f"Which visible activity starts closest after the activity referred to by {mention} ends?"
        choices = [_event(n) for n in pool if n["kind"] == "action"] + ["No qualifying event is observed", NULL]
    elif q["op"] == "duration":
        question = f"For how many seconds is the activity referred to by {mention} visible?"
        choices = [f"{d} seconds" for d in range(1, 9)] + [NULL]
    elif q["op"] == "duration_compare":
        other = next(n for n in nodes if n["id"] == q["comparison"])
        question = f'Which visible activity lasts longer: the one referred to by {mention}, or "{other["content"]}"?'
        choices = [_event(n) for n in pool if n["kind"] == "action"] + [NULL]
    answer, h1, h2 = label_binding_case(nodes, q, meta["identities"])
    distractors = sorted(set(choices) - {answer})
    options = [answer] + rng.sample(distractors, rng.randint(1, min(3, len(distractors))))
    rng.shuffle(options)
    record.update(question=question, answer_options=options, gold=options.index(answer), evidence=dict(hop1=h1, hop2=h2))
    return record


def edit_binding_case(record, seed=0, time_shift=0, audio_shift=0, tense=None):
    """Copy, edit clocks/phrasing, permute nodes/options, remap IDs, and relabel.

    Uniform translation changes interval-valued answers, never their identity.
    Audio-only edits can change overlap answers; ambiguous edits are rejected.
    """
    result, rng = copy.deepcopy(record), random.Random(seed)
    meta, q = result["metadata"], result["metadata"]["query"]
    if tense is not None and tense not in TENSES:
        raise ValueError("unknown tense")
    keys = set(meta["identities"]) | {q["reference"]}
    mapping = {i: f"n{rng.getrandbits(128):032x}" for i in sorted(keys)}
    for n in result["nodes"]:
        n["interval"] = [t + time_shift + (audio_shift if n["source"] == "audio" else 0) for t in n["interval"]]
        if tense is not None and n["id"] in meta["mentions"]:
            n["content"] = _phrase(meta["mentions"][n["id"]], tense)
        n["id"] = mapping[n["id"]]
    for key in ("identities", "mentions"):
        meta[key] = {mapping[i]: value for i, value in meta[key].items()}
    for key in ("reference", "comparison"):
        if key in q:
            q[key] = mapping[q[key]]
    if tense is not None:
        meta.update(tense=tense, reference_content=_phrase(meta["action"], tense))
    meta["time_shift"] += time_shift
    meta["audio_shift"] += audio_shift
    rng.shuffle(result["nodes"])
    return _finish(result, rng)


def generate_binding_cases(count=512, seed=0):
    """Deterministic supplemental corpus; every cycle covers all families.

    Narration tense, narration clock, visual clock, and scene are independent.
    Preparation/holding and completed results cannot certify a performance.
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    rng, records, schedule = random.Random(seed), [], []
    while len(records) < count:
        if not schedule:
            schedule = list(FAMILIES) + ["narration_visible"] * 5
            rng.shuffle(schedule)
        family, scene, tense = schedule.pop(), rng.choice(list(SCENES)), rng.choice(TENSES)
        action, other, last = rng.sample(SCENES[scene].split("|"), 3)
        sound = family in ("narration_sound", "caption_sound")
        source, kind = ("audio", "sound") if sound else ("video", "action")
        action, other, last = ("a bell ringing", "a whistle blowing", "a knock sounding") if sound else (action, other, last)
        reference_source = "video" if family.startswith("caption") else "audio"
        nodes, identities, mentions = [], {}, {}
        def add(name, src, interval, content, typ, event, role):
            nodes.append(dict(id=name, source=src, interval=list(interval), content=content, kind=typ))
            identities[name] = dict(event=event, role=role)
        # Randomize temporal rank and duration independently of event identity.
        # The previous middle/longest target was a construction shortcut.
        starts = [10 + rng.randint(0, 3), 30 + rng.randint(0, 3), 50 + rng.randint(0, 3)]
        rng.shuffle(starts)
        durations = [rng.randint(1, 8) for _ in range(3)]
        if family == "duration_compare" and durations[0] == durations[1]:
            durations[1] = durations[1] % 8 + 1
        spans = [(a, a + d) for a, d in zip(starts, durations)]
        target_span, other_span, last_span = spans
        add("target", source, target_span, action, kind, "target-event", "occurrence")
        add("other", source, other_span, other, kind, "other-event", "occurrence")
        add("last", source, last_span, last, kind, "last-event", "occurrence")
        add("ready", source, (target_span[0]-6, target_span[0]-3), f"Preparing or holding ready for: {action}; no performance", "preparation", "target-event", "preparation")
        add("result", source, (target_span[1]+3, target_span[1]+6), f"Completed result of: {action}; no ongoing performance", "result", "target-event", "result")
        mention_span = rng.choice(spans)
        narration_overlaps = family == "reverse_words" and rng.random() < .5
        if family in ("reverse_words", "reverse_silent"):
            mention_span = target_span if narration_overlaps else rng.choice((other_span, last_span))
        add("mention", reference_source, mention_span, _phrase(action, tense), "caption" if reference_source == "video" else "speech", "target-event", "mention")
        mentions["mention"] = action
        word_span = (other_span[0]+.25, other_span[0]+.75) if narration_overlaps else (target_span[0]+.25, target_span[0]+.75)
        add("words", "audio", word_span, rng.choice(("Please wait here", "The blue box is mine", "We can leave soon")), "speech", "unrelated-words", "occurrence")
        op = family if family in ("before", "after", "duration", "duration_compare") else "semantic"
        reverse = family in ("reverse_words", "reverse_silent")
        q = dict(op="overlap" if reverse else op, reference="target" if reverse else "mention", source="audio" if reverse else source, kind="speech" if reverse else kind)
        if op == "duration_compare":
            q["comparison"] = "other"
        missing = {"absent_event": "target", "missing_reference": "mention", "reverse_silent": "words"}.get(family)
        if missing:
            nodes = [n for n in nodes if n["id"] != missing]
        meta = dict(family=family, scene=scene, query=q, identities=identities, mentions=mentions,
                    action=action, tense=tense, reference_source="video" if reverse else reference_source,
                    reference_content=_phrase(action, tense), time_shift=0, audio_shift=0, construction_only=True)
        records.append(edit_binding_case(dict(nodes=nodes, metadata=meta), rng.getrandbits(64), rng.randint(0, 200)))
    return records

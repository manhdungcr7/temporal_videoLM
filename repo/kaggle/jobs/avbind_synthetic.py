"""Textual structural supervision for AVBindReader, using exact synthetic events.

No teacher, benchmark inputs, inference routing, or training is involved. This
corpus cannot establish raw audiovisual grounding or general AVQA performance.
Only question, answer_options and nodes are model inputs; metadata is for corpus
construction/stratification and evidence contains supervised occurrence targets.
"""

import copy
import random


SCENES = {
    "park": "a cyclist rings a bell|a dog jumps|a runner stretches|a child waves|a gardener waters flowers|a pigeon lands",
    "station": "a train arrives|a traveler lifts a suitcase|a guard waves|a passenger reads a map|a porter pushes a cart|a commuter sits",
    "workshop": "a mechanic turns a wrench|a worker sands wood|a painter opens a can|a technician tests a lamp|a carpenter measures a board|a welder lowers a visor",
    "library": "a reader turns a page|a librarian stacks books|a visitor opens a bag|a student writes notes|a child points at a shelf|a clerk stamps a card",
    "beach": "a surfer lifts a board|a swimmer waves|a child builds a castle|a lifeguard raises a flag|a gull lands|a visitor unfolds a chair",
    "gym": "an athlete lifts a weight|a coach claps|a player bounces a ball|a runner ties a shoe|a gymnast stretches|a referee raises a hand",
    "gallery": "a visitor studies a painting|an artist draws a line|a guide points|a guard opens a door|a photographer kneels|a curator moves a frame",
    "street": "a bus stops|a pedestrian crosses|a courier parks a bicycle|a driver opens a door|a vendor raises an umbrella|a musician bows",
    "office": "a worker types|a colleague picks up a folder|a manager points at a screen|a visitor sits|a clerk opens a drawer|a designer sketches",
    "forest": "a hiker steps over a log|a deer walks|a bird takes flight|a ranger lifts binoculars|a camper unfolds a map|a squirrel climbs",
    "theater": "an actor bows|a dancer spins|a stagehand moves a ladder|a conductor raises a baton|a performer waves|a technician adjusts a spotlight",
    "playground": "a child slides|a parent pushes a swing|a teenager skips rope|a child climbs a ladder|a visitor kicks a ball|a teacher waves",
}
WORDS = ("please wait here", "the blue box is mine", "we can leave soon",
         "look over there", "bring the next folder", "that was wonderful")
SOUNDS = ("a whistle", "a dog barking", "a phone ringing", "a loud knock", "a horn")
FAMILIES = ("visual_audio", "audio_visual", "visual_sound",
            "video_before", "video_after", "audio_before", "audio_after",
            "video_count", "audio_count", "video_duration", "audio_duration",
            "video_first", "video_last", "audio_first", "audio_last",
            "video_absent", "audio_absent", "general", "arithmetic")
TEMPLATES = {
    "speech": ("What was said during {r}?", "Which words were spoken while {r} took place?", "What did someone say at the time of {r}?"),
    "sound": ("Which non-speech sound was heard during {r}?", "What sound, excluding speech, accompanied {r}?"),
    "visible": ("What was visible during {r}?", "Which action could be seen while {r} took place?", "What visual event overlapped {r}?"),
    "before": ("Which {s} event finished most recently before {r} began?", "What was the last {s} event to end before the start of {r}?"),
    "after": ("Which {s} event began soonest after {r} ended?", "What was the next {s} event to start following the end of {r}?"),
    "count": ("How many distinct, separate occurrences of {r} are recorded?", "How many noncontiguous times does {r} occur?"),
    "duration": ("Which {s} event lasted longer: {r} or {b}?", "Between {r} and {b}, which {s} event had the greater duration?"),
}


def make_scene(rng):
    """Two independent renewal timelines; repetitions keep separate identities."""
    scene = rng.choice(list(SCENES))
    nodes = []
    visual = [(c, "action") for c in SCENES[scene].split("|")]
    audio = [(c, "speech") for c in rng.sample(WORDS, 4)]
    audio += [(c, "sound") for c in rng.sample(SOUNDS, 3)] + [("instrumental music", "music")]
    for source, vocabulary in (("video", visual), ("audio", audio)):
        occurrences = [item for item in vocabulary for _ in range(rng.randint(1, 3))]
        rng.shuffle(occurrences)
        time = rng.randint(0, 8)
        for content, kind in occurrences:
            time += rng.randint(2, 8)
            end = time + rng.randint(1, 7)
            nodes.append(dict(id=f"n{rng.getrandbits(128):032x}", source=source,
                              interval=[time, end], content=content, kind=kind))
            time = end
    rng.shuffle(nodes)
    return scene, nodes


def _matches(nodes, reference):
    return [n for n in nodes if all(n[k] in v if isinstance(v, list) else n[k] == v
                                   for k, v in reference.items())]


def _ref(node):
    return {k: node[k] for k in ("source", "content", "kind")}


def label_query(nodes, query):
    """Construction-only oracle over explicit query specs, never parsed questions.

    Empty target lists supervise null reads. Ambiguous occurrence or temporal
    ties are rejected; equal durations have their own single, explicit answer.
    """
    op = query["op"]
    if op == "retention":
        return query["answer"], [], []
    refs = _matches(nodes, query["reference"])
    if op == "count":
        ordered = sorted(refs, key=lambda n: n["interval"])
        if any(a["interval"][1] >= b["interval"][0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("count requires noncontiguous occurrences")
        return str(len(refs)), [n["id"] for n in ordered], []
    selector = query.get("selector")
    if selector and refs:
        edge = 0 if selector == "first" else 1
        value = (min if selector == "first" else max)(n["interval"][edge] for n in refs)
        refs = [n for n in refs if n["interval"][edge] == value]
    if len(refs) > 1:
        raise ValueError("ambiguous reference")
    null = query["null"]
    if not refs:
        return null, [], []
    anchor = refs[0]
    candidates = _matches(nodes, query["target"])
    if op == "overlap":
        a, b = anchor["interval"]
        if any(n["interval"][1] == a or n["interval"][0] == b for n in candidates):
            raise ValueError("touching boundaries are excluded")
        targets = [n for n in candidates if max(a, n["interval"][0]) < min(b, n["interval"][1])]
    elif op in ("before", "after"):
        before = op == "before"
        targets = [n for n in candidates if (n["interval"][1] < anchor["interval"][0]
                   if before else n["interval"][0] > anchor["interval"][1])]
        if targets:
            edge = 1 if before else 0
            value = (max if before else min)(n["interval"][edge] for n in targets)
            targets = [n for n in targets if n["interval"][edge] == value]
    elif op == "duration":
        if len(candidates) != 1:
            raise ValueError("duration needs one comparison occurrence")
        other = candidates[0]
        d = anchor["interval"][1] - anchor["interval"][0]
        e = other["interval"][1] - other["interval"][0]
        answer = "They lasted equally long" if d == e else (anchor if d > e else other)["content"]
        return answer, [anchor["id"]], [other["id"]]
    else:
        raise ValueError("unknown construction operation")
    if len(targets) > 1:
        raise ValueError("ambiguous target or temporal tie")
    return (targets[0]["content"] if targets else null,
            [anchor["id"]], [n["id"] for n in targets])


def make_record(nodes, scene, family, rng):
    """Choose a supervision question; family branches only construct labels."""
    source = "audio" if family.startswith("audio") else "video"
    pool = [n for n in nodes if n["source"] == source]
    unique = [n for n in pool if len(_matches(nodes, _ref(n))) == 1]
    if family == "audio_visual":
        unique = [n for n in unique if n["kind"] == "speech"]
    repeated = [n for n in pool if len(_matches(nodes, _ref(n))) > 1]
    query = dict(op="overlap", target={"source": "audio", "kind": "speech"},
                 null="No speech during this action")
    if family in ("general", "arithmetic"):
        if family == "general":
            question, answer, distractors = rng.choice((
                ("How many sides does a triangle have?", "3", ["2", "4", "5"]),
                ("Which animal is a mammal?", "A dolphin", ["A trout", "A beetle", "A sparrow"])))
        else:
            a, b = rng.randint(2, 25), rng.randint(2, 25)
            question = rng.choice((f"What is {a} plus {b}?", f"Calculate the sum of {a} and {b}."))
            answer, distractors = str(a + b), [str(a + b + d) for d in (-2, -1, 1)]
        query = dict(op="retention", answer=answer)
    else:
        suffix = family.split("_")[-1]
        reference = rng.choice(repeated if suffix in ("first", "last", "count") else unique)
        query["reference"] = _ref(reference)
        selector = suffix if suffix in ("first", "last") else None
        noun = "action" if source == "video" else ("words" if reference["kind"] == "speech" else "sound")
        r = f'the {noun} “{reference["content"]}”'
        if suffix != "count":
            r = f'the {selector + " " if selector else ""}occurrence of {r}'
        mode = "speech"
        if source == "audio":
            query.update(target={"source": "video"}, null="No visual action during this audio event")
            mode = "visible"
        if family == "visual_sound":
            query.update(target={"source": "audio", "kind": ["sound", "music"]},
                         null="No matching non-speech sound")
            mode = "sound"
        if suffix in ("before", "after", "count", "duration"):
            mode = suffix
            query.update(op=suffix, target={"source": source}, null=f"No {source} event qualifies")
        if selector:
            query["selector"] = selector
        if suffix == "absent":
            query["reference"]["content"] = ("a visitor releases a purple kite" if source == "video"
                                               else "the silver ticket is missing")
            query["reference"]["kind"] = "action" if source == "video" else "speech"
            query["null"] = "The referenced event is absent"
            r = f'the occurrence of “{query["reference"]["content"]}”'
        other = rng.choice([n for n in unique if n["id"] != reference["id"]]) if mode == "duration" else reference
        if mode == "duration":
            query["target"] = _ref(other)
        question = rng.choice(TEMPLATES[mode]).format(r=r, b=f'“{other["content"]}”',
                                                      s="visual" if source == "video" else "audio")
        answer, _, _ = label_query(nodes, query)
        distractors = ([str(i) for i in range(5)] if mode == "count" else
                       [n["content"] for n in _matches(nodes, query["target"])])
        distractors += [reference["content"], "They lasted equally long"] if mode == "duration" else [query["null"]]
    answer, hop1, hop2 = label_query(nodes, query)
    options = [answer] + rng.sample(sorted(set(distractors) - {answer}), rng.randint(1, min(3, len(set(distractors) - {answer}))))
    rng.shuffle(options)
    return dict(question=question, answer_options=options, gold=options.index(answer),
                nodes=copy.deepcopy(nodes), evidence=dict(hop1=hop1, hop2=hop2),
                metadata=dict(family=family, scene=scene, query=query, time_shift=0))


def augment_record(record, seed, time_shift=0):
    """Preserve answers while permuting storage, translating time and renaming IDs."""
    result, rng = copy.deepcopy(record), random.Random(seed)
    mapping = {n["id"]: f"n{rng.getrandbits(128):032x}" for n in result["nodes"]}
    for node in result["nodes"]:
        node["id"] = mapping[node["id"]]
        node["interval"] = [t + time_shift for t in node["interval"]]
    rng.shuffle(result["nodes"])
    result["evidence"] = {hop: [mapping[i] for i in ids] for hop, ids in result["evidence"].items()}
    result["metadata"]["time_shift"] += time_shift
    return result


def generate_corpus(count=512, seed=0):
    """Return count deterministic records (e.g. 384, 512 or 768).

    gold indexes shuffled answer_options. evidence['hop1'] / ['hop2'] contain
    node IDs: address targets are ID membership in the current node order;
    each hop's null target is int(not evidence[hop]). Retention and missing
    references have both hops null. Metadata is never a model input. Each
    complete family cycle covers all families; no inference API is provided.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    rng, records, schedule = random.Random(seed), [], []
    while len(records) < count:
        if not schedule:
            schedule = list(FAMILIES)
            rng.shuffle(schedule)
        family = schedule[-1]
        scene, nodes = make_scene(rng)
        try:
            record = make_record(nodes, scene, family, rng)
        except (ValueError, IndexError):
            continue  # Reject ambiguous queries or scenes without suitable references.
        records.append(augment_record(record, rng.getrandbits(64), rng.randint(0, 200)))
        schedule.pop()
    return records

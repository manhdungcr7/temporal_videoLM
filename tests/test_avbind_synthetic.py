"""CPU-only structural checks; these do not measure audiovisual grounding.

Run: python -B -m unittest discover -s tests -p test_avbind_synthetic.py
"""

import copy
import json
import random
import unittest

from repo.kaggle.jobs.avbind_synthetic import (
    FAMILIES, SCENES, augment_record, generate_corpus, label_query, make_scene,
)


def node(identity, source, interval, content, kind):
    return dict(id=identity, source=source, interval=interval, content=content, kind=kind)


def reference(n):
    return {k: n[k] for k in ("source", "content", "kind")}


class SyntheticCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_corpus(384, seed=73)

    def test_seed_and_schema(self):
        self.assertEqual(self.records, generate_corpus(384, seed=73))
        self.assertNotEqual(self.records[:5], generate_corpus(5, seed=74))
        self.assertEqual(generate_corpus(0), [])
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                generate_corpus(invalid)
        positions, sizes = set(), set()
        for record in self.records:
            self.assertEqual(set(record), {"question", "answer_options", "gold", "nodes", "evidence", "metadata"})
            options = record["answer_options"]
            self.assertTrue(2 <= len(options) <= 4)
            self.assertEqual(len(options), len(set(options)))
            self.assertIs(type(record["gold"]), int)
            self.assertTrue(0 <= record["gold"] < len(options))
            positions.add(record["gold"])
            sizes.add(len(options))
            ids = {n["id"] for n in record["nodes"]}
            self.assertEqual(len(ids), len(record["nodes"]))
            for n in record["nodes"]:
                self.assertEqual(set(n), {"id", "source", "interval", "content", "kind"})
                self.assertIn(n["source"], ("video", "audio"))
                self.assertLess(*n["interval"])
                self.assertIn(n["kind"], ("action", "speech", "sound", "music"))
            for hop in ("hop1", "hop2"):
                self.assertTrue(set(record["evidence"][hop]) <= ids)
        self.assertEqual(sizes, {2, 3, 4})
        self.assertEqual(positions, {0, 1, 2, 3})
        for question in ("How many sides does a triangle have?", "Which animal is a mammal?"):
            self.assertGreater(len({r["gold"] for r in self.records if r["question"] == question}), 1)
        json.dumps(self.records)  # Records are directly serializable.

    def test_every_label_against_independent_interval_calculation(self):
        for record in self.records:
            q, nodes = record["metadata"]["query"], record["nodes"]
            options, evidence = record["answer_options"], record["evidence"]
            answer, h1, h2 = label_query(nodes, q)
            self.assertEqual(options[record["gold"]], answer)
            self.assertEqual(evidence, dict(hop1=h1, hop2=h2))
            if q["op"] == "retention":
                self.assertEqual(evidence, dict(hop1=[], hop2=[]))
                continue
            refs = [n for n in nodes if reference(n) == q["reference"]]
            if q["op"] == "count":
                self.assertEqual(answer, str(len(refs)))
                self.assertEqual(set(h1), {n["id"] for n in refs})
                ordered = sorted(refs, key=lambda n: n["interval"])
                self.assertTrue(all(a["interval"][1] < b["interval"][0] for a, b in zip(ordered, ordered[1:])))
                continue
            if not refs:
                self.assertEqual(answer, "The referenced event is absent")
                self.assertEqual(h1 + h2, [])
                continue
            if q.get("selector"):
                first = q["selector"] == "first"
                anchor = (min if first else max)(refs, key=lambda n: n["interval"][0 if first else 1])
            else:
                self.assertEqual(len(refs), 1)
                anchor = refs[0]
            self.assertEqual(h1, [anchor["id"]])
            targets = [n for n in nodes if all(n[k] in v if isinstance(v, list) else n[k] == v
                                             for k, v in q["target"].items())]
            if q["op"] == "duration":
                self.assertEqual(len(targets), 1)
                other = targets[0]
                d, e = (n["interval"][1] - n["interval"][0] for n in (anchor, other))
                expected = "They lasted equally long" if d == e else (anchor if d > e else other)["content"]
                self.assertEqual(answer, expected)
                self.assertEqual(h2, [other["id"]])
                continue
            a, b = anchor["interval"]
            if q["op"] == "overlap":
                self.assertTrue(all(n["interval"][1] != a and n["interval"][0] != b for n in targets))
                targets = [n for n in targets if n["interval"][0] < b and n["interval"][1] > a]
            elif q["op"] == "before":
                targets = sorted([n for n in targets if n["interval"][1] < a], key=lambda n: -n["interval"][1])[:1]
            else:
                targets = sorted([n for n in targets if n["interval"][0] > b], key=lambda n: n["interval"][0])[:1]
            self.assertLessEqual(len(targets), 1)
            self.assertEqual(h2, [n["id"] for n in targets])
            self.assertEqual(answer, targets[0]["content"] if targets else q["null"])

    def test_permutation_translation_and_identity(self):
        for record in self.records[:80]:
            original = copy.deepcopy(record)
            shuffled = copy.deepcopy(record)
            random.Random(9).shuffle(shuffled["nodes"])
            self.assertEqual(label_query(shuffled["nodes"], shuffled["metadata"]["query"]),
                             label_query(record["nodes"], record["metadata"]["query"]))
            changed = augment_record(record, seed=98, time_shift=10000.125)
            self.assertEqual(record, original)
            self.assertEqual(changed["question"], record["question"])
            self.assertEqual(changed["answer_options"], record["answer_options"])
            self.assertEqual(changed["gold"], record["gold"])
            self.assertEqual(changed["metadata"]["time_shift"], record["metadata"]["time_shift"] + 10000.125)
            self.assertTrue({n["id"] for n in changed["nodes"]}.isdisjoint(n["id"] for n in record["nodes"]))
            key = lambda n: (n["source"], n["content"], n["kind"], n["interval"])
            before, after = sorted(record["nodes"], key=key), sorted(changed["nodes"], key=key)
            mapping = {a["id"]: b["id"] for a, b in zip(before, after)}
            for a, b in zip(before, after):
                self.assertEqual(b["interval"], [t + 10000.125 for t in a["interval"]])
                self.assertEqual(reference(a), reference(b))
            for hop in ("hop1", "hop2"):
                self.assertEqual(changed["evidence"][hop], [mapping[i] for i in record["evidence"][hop]])
            answer, h1, h2 = label_query(changed["nodes"], changed["metadata"]["query"])
            self.assertEqual(answer, changed["answer_options"][changed["gold"]])
            self.assertEqual(changed["evidence"], dict(hop1=h1, hop2=h2))

    def test_causal_time_edit_changes_answer_without_identity_edit(self):
        nodes = [node("action", "video", [10, 15], "a child waves", "action"),
                 node("word-a", "audio", [11, 12], "hello", "speech"),
                 node("word-b", "audio", [21, 22], "goodbye", "speech")]
        q = dict(op="overlap", reference=reference(nodes[0]), target={"source": "audio", "kind": "speech"}, null="No speech during this action")
        self.assertEqual(label_query(nodes, q), ("hello", ["action"], ["word-a"]))
        edited = copy.deepcopy(nodes)
        edited[0]["interval"] = [20, 25]
        self.assertEqual(label_query(edited, q), ("goodbye", ["action"], ["word-b"]))
        self.assertEqual([reference(n) for n in nodes], [reference(n) for n in edited])
        self.assertEqual([n["id"] for n in nodes], [n["id"] for n in edited])
        q.update(op="duration", target=reference(nodes[2]))
        self.assertEqual(label_query(nodes, q)[0], "a child waves")
        edited[0]["interval"] = [10, 10.5]
        self.assertEqual(label_query(edited, q)[0], "goodbye")

    def test_multiple_occurrences_first_last_and_temporal_relations(self):
        nodes = [node("v1", "video", [1, 3], "a child waves", "action"),
                 node("v2", "video", [11, 14], "a child waves", "action"),
                 node("a1", "audio", [1.5, 2.5], "hello", "speech"),
                 node("a2", "audio", [12, 13], "goodbye", "speech")]
        q = dict(op="count", reference=reference(nodes[0]))
        self.assertEqual(label_query(nodes, q), ("2", ["v1", "v2"], []))
        q.update(op="overlap", target={"source": "audio", "kind": "speech"}, null="No speech during this action", selector="first")
        self.assertEqual(label_query(nodes, q), ("hello", ["v1"], ["a1"]))
        q["selector"] = "last"
        self.assertEqual(label_query(nodes, q), ("goodbye", ["v2"], ["a2"]))
        q.update(op="before", target={"source": "video"})
        self.assertEqual(label_query(nodes, q), ("a child waves", ["v2"], ["v1"]))
        q.update(op="after", selector="first")
        self.assertEqual(label_query(nodes, q), ("a child waves", ["v1"], ["v2"]))

    def test_music_sound_speech_nulls_and_ambiguity(self):
        nodes = [node("v", "video", [10, 15], "a child waves", "action"),
                 node("m", "audio", [11, 14], "instrumental music", "music")]
        q = dict(op="overlap", reference=reference(nodes[0]), target={"source": "audio", "kind": "speech"}, null="No speech during this action")
        self.assertEqual(label_query(nodes, q), (q["null"], ["v"], []))
        q["target"]["kind"] = ["sound", "music"]
        self.assertEqual(label_query(nodes, q), ("instrumental music", ["v"], ["m"]))
        nodes[1].update(content="a whistle", kind="sound")
        self.assertEqual(label_query(nodes, q)[0], "a whistle")
        nodes.append(node("extra", "audio", [12, 13], "a knock", "sound"))
        with self.assertRaises(ValueError):
            label_query(nodes, q)
        nodes.pop()
        nodes[1]["interval"] = [15, 17]
        with self.assertRaises(ValueError):
            label_query(nodes, q)
        q.update(op="duration", target=reference(nodes[1]))
        nodes[1]["interval"] = [20, 25]
        self.assertEqual(label_query(nodes, q)[0], "They lasted equally long")
        q.update(op="before", target={"source": "audio"})
        nodes[1]["interval"] = [2, 5]
        nodes.append(node("tie", "audio", [3, 5], "a knock", "sound"))
        with self.assertRaises(ValueError):
            label_query(nodes, q)

    def test_family_scene_paraphrase_and_clean_input_coverage(self):
        self.assertEqual({r["metadata"]["family"] for r in self.records}, set(FAMILIES))
        self.assertEqual({r["metadata"]["scene"] for r in self.records}, set(SCENES))
        for family in FAMILIES:
            rows = [r for r in self.records if r["metadata"]["family"] == family]
            self.assertGreater(len({r["question"] for r in rows}), 1)
            if family.endswith("absent"):
                self.assertTrue(all(r["evidence"] == dict(hop1=[], hop2=[]) for r in rows))
            if family == "audio_visual":
                self.assertTrue(all(r["metadata"]["query"]["reference"]["kind"] == "speech" for r in rows))
        inputs = json.dumps([{k: r[k] for k in ("question", "answer_options", "nodes")} for r in self.records]).lower()
        banned = ("sv" + "halluc", "past", "present", "future", "task_id", "kitchen", "cooking", "recipe")
        for term in banned:
            self.assertNotIn(term, inputs)
        for family in FAMILIES:
            self.assertNotIn(family, inputs)
        overlaps = [r for r in self.records if r["metadata"]["family"] == "visual_audio"]
        self.assertTrue(any(r["evidence"]["hop2"] for r in overlaps))
        self.assertTrue(any(not r["evidence"]["hop2"] for r in overlaps))
        for r in self.records:
            if r["metadata"]["family"] == "general":
                self.assertEqual(r["answer_options"][r["gold"]], r["metadata"]["query"]["answer"])
            if r["metadata"]["family"] == "arithmetic":
                numbers = [int(w) for w in r["question"].replace("?", "").replace(".", "").split() if w.isdigit()]
                self.assertEqual(r["answer_options"][r["gold"]], str(sum(numbers)))

    def test_timeline_generation_has_separate_occurrence_ids(self):
        _, nodes = make_scene(random.Random(18))
        self.assertNotEqual(nodes, sorted(nodes, key=lambda n: n["interval"]))
        for source in ("video", "audio"):
            rows = sorted([n for n in nodes if n["source"] == source], key=lambda n: n["interval"])
            self.assertTrue(all(a["interval"][1] < b["interval"][0] for a, b in zip(rows, rows[1:])))
            self.assertLess(len({n["content"] for n in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()

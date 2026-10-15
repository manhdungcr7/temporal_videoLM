"""Exact textual checks, not a measurement of raw audiovisual grounding.

Run: .venv/Scripts/python.exe -B -m unittest discover -s tests -p test_avbind_binding_cases.py
"""
import ast
import copy
import json
import unittest
from pathlib import Path

from repo.kaggle.jobs import avbind_binding_cases as binding


def answer(record):
    return record["answer_options"][record["gold"]]


def role_nodes(record, role, event=None):
    identities = record["metadata"]["identities"]
    return [n for n in record["nodes"] if identities[n["id"]]["role"] == role
            and (event is None or identities[n["id"]]["event"] == event)]


def interval(node):
    return f'[{node["interval"][0]:g}, {node["interval"][1]:g}] s'


class BindingCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = binding.generate_binding_cases(408, seed=73)

    def test_determinism_schema_and_coverage(self):
        self.assertEqual(self.records, binding.generate_binding_cases(408, seed=73))
        self.assertEqual(self.records[:17], binding.generate_binding_cases(17, seed=73))
        self.assertNotEqual(self.records[:17], binding.generate_binding_cases(17, seed=74))
        self.assertEqual(binding.generate_binding_cases(0), [])
        for bad in (-1, True, 1.5, "3"):
            with self.assertRaises(ValueError):
                binding.generate_binding_cases(bad)
        sizes, positions = set(), set()
        for r in self.records:
            self.assertEqual(set(r), {"question", "answer_options", "gold", "nodes", "evidence", "metadata"})
            opts = r["answer_options"]
            self.assertTrue(2 <= len(opts) <= 4)
            self.assertEqual(len(opts), len(set(opts)))
            self.assertIs(type(r["gold"]), int)
            self.assertIn(r["gold"], range(len(opts)))
            sizes.add(len(opts)); positions.add(r["gold"])
            ids = {n["id"] for n in r["nodes"]}
            self.assertEqual(len(ids), len(r["nodes"]))
            for n in r["nodes"]:
                self.assertEqual(set(n), {"id", "source", "interval", "content", "kind"})
                self.assertIn(n["source"], ("video", "audio"))
                self.assertLess(*n["interval"])
            self.assertTrue(all(set(h) <= ids for h in r["evidence"].values()))
            q, identities = r["metadata"]["query"], r["metadata"]["identities"]
            expected, h1, h2 = binding.label_binding_case(r["nodes"], q, identities)
            self.assertEqual((answer(r), r["evidence"]), (expected, dict(hop1=h1, hop2=h2)))
        self.assertEqual(sizes, {2, 3, 4}); self.assertEqual(positions, {0, 1, 2, 3})
        self.assertEqual({r["metadata"]["family"] for r in self.records}, set(binding.FAMILIES))
        self.assertEqual({r["metadata"]["scene"] for r in self.records}, set(binding.SCENES))
        json.dumps(self.records)

    def test_semantics_versus_simultaneity_and_stages(self):
        for r in self.records:
            q = r["metadata"]["query"]
            if q["op"] != "semantic" or r["metadata"]["family"] in ("absent_event", "missing_reference"):
                continue
            ref = next(n for n in r["nodes"] if n["id"] == q["reference"])
            target = role_nodes(r, "occurrence", "target-event")[0]
            self.assertEqual(answer(r), interval(target))
            self.assertEqual(r["evidence"], dict(hop1=[ref["id"]], hop2=[target["id"]]))
            # Concurrent and nonconcurrent semantic references are both valid.
            decoys = [n for n in role_nodes(r, "occurrence") if n["interval"] == ref["interval"]]
            self.assertEqual(len(decoys), 1)
            for stage in ("preparation", "result"):
                self.assertEqual(len(role_nodes(r, stage, "target-event")), 1)
                self.assertNotEqual(answer(r), interval(role_nodes(r, stage)[0]))

    def test_reverse_overlap_is_not_narration_and_audio_edit_changes_it(self):
        r = next(r for r in self.records if r["metadata"]["family"] == "reverse_words"
                 and answer(r) == role_nodes(r, "occurrence", "unrelated-words")[0]["content"])
        original = copy.deepcopy(r)
        target = role_nodes(r, "occurrence", "target-event")[0]
        narration = role_nodes(r, "mention")[0]
        words = role_nodes(r, "occurrence", "unrelated-words")[0]
        self.assertEqual(answer(r), words["content"])
        offset = target["interval"][0] - narration["interval"][0] + .125
        edited = binding.edit_binding_case(r, seed=8, audio_shift=offset)
        self.assertEqual(answer(edited), narration["content"])
        self.assertEqual(interval(role_nodes(edited, "occurrence", "target-event")[0]), interval(target))
        self.assertEqual(r, original)
        for row in self.records:
            if row["metadata"]["family"] == "narration_visible":
                shifted = binding.edit_binding_case(row, seed=9, audio_shift=61.25)
                self.assertEqual(answer(row), answer(shifted))

    def test_tense_is_independent_of_observed_timestamp(self):
        signs = {t: set() for t in binding.TENSES}
        for r in self.records:
            if r["metadata"]["family"] != "narration_visible":
                continue
            ref, target = role_nodes(r, "mention")[0], role_nodes(r, "occurrence", "target-event")[0]
            signs[r["metadata"]["tense"]].add(target["interval"][0] < ref["interval"][0])
            for tense in binding.TENSES:
                edited = binding.edit_binding_case(r, seed=9, tense=tense)
                self.assertEqual(answer(r), answer(edited))
                self.assertEqual(role_nodes(edited, "mention")[0]["interval"], ref["interval"])
                self.assertEqual(role_nodes(edited, "occurrence", "target-event")[0]["interval"], target["interval"])
                self.assertIn(role_nodes(edited, "mention")[0]["content"], edited["question"])
        self.assertTrue(all(v == {True, False} for v in signs.values()))

    def test_nullable_hops_and_temporal_comparisons(self):
        for r in self.records:
            family, e = r["metadata"]["family"], r["evidence"]
            if family == "missing_reference":
                self.assertEqual(e, dict(hop1=[], hop2=[]))
            elif family in ("absent_event", "reverse_silent"):
                self.assertEqual(len(e["hop1"]), 1); self.assertEqual(e["hop2"], [])
            elif family in ("before", "after", "duration", "duration_compare"):
                target = role_nodes(r, "occurrence", "target-event")[0]
                if family == "duration":
                    self.assertEqual(answer(r), f'{target["interval"][1] - target["interval"][0]:g} seconds')
                    continue
                candidates = [n for n in r["nodes"] if n["kind"] == "action" and n != target]
                if family == "before":
                    pool = [n for n in candidates if n["interval"][1] < target["interval"][0]]
                    expected = max(pool, key=lambda n: n["interval"][1]) if pool else None
                elif family == "after":
                    pool = [n for n in candidates if n["interval"][0] > target["interval"][1]]
                    expected = min(pool, key=lambda n: n["interval"][0]) if pool else None
                else:
                    other = next(n for n in r["nodes"] if n["id"] == r["metadata"]["query"]["comparison"])
                    expected = max((target, other), key=lambda n: n["interval"][1] - n["interval"][0])
                self.assertEqual(answer(r), f'{expected["content"]} at {interval(expected)}' if expected else "No qualifying event is observed")

    def test_translation_permutation_and_private_identity_remapping(self):
        for r in self.records[:34]:
            original = copy.deepcopy(r)
            shifted = binding.edit_binding_case(r, seed=17, time_shift=1000.125)
            before = {n["id"] for n in r["nodes"]}
            self.assertTrue(before.isdisjoint(n["id"] for n in shifted["nodes"]))
            mapping = {}
            for role in ("mention", "preparation", "result", "occurrence"):
                old = sorted(role_nodes(r, role), key=lambda n: (n["source"], n["content"]))
                new = sorted(role_nodes(shifted, role), key=lambda n: (n["source"], n["content"]))
                for a, b in zip(old, new):
                    mapping[a["id"]] = b["id"]
                    self.assertEqual(b["interval"], [t + 1000.125 for t in a["interval"]])
            self.assertEqual(shifted["evidence"], {hop: [mapping[i] for i in ids] for hop, ids in r["evidence"].items()})
            shuffled = copy.deepcopy(shifted)
            shuffled["nodes"].reverse()
            q, ids = shifted["metadata"]["query"], shifted["metadata"]["identities"]
            expected = binding.label_binding_case(shuffled["nodes"], q, ids)
            self.assertEqual((answer(shifted), shifted["evidence"]["hop1"], shifted["evidence"]["hop2"]), expected)
            self.assertEqual(r, original)

    def test_no_fixed_temporal_rank_duration_or_concurrence_shortcut(self):
        ranks=set();duration_order=set();concurrence=set()
        for r in self.records:
            if r['metadata']['family']!='narration_visible':continue
            target=role_nodes(r,'occurrence','target-event')[0]
            events=[n for n in r['nodes'] if n['kind']=='action']
            ordered=sorted(events,key=lambda n:n['interval'][0])
            ranks.add(ordered.index(target))
            other=role_nodes(r,'occurrence','other-event')[0]
            durations=[n['interval'][1]-n['interval'][0] for n in (target,other)]
            duration_order.add((durations[0]>durations[1])-(durations[0]<durations[1]))
            ref=role_nodes(r,'mention')[0]
            concurrence.add(max(ref['interval'][0],target['interval'][0])<min(ref['interval'][1],target['interval'][1]))
        self.assertEqual(ranks,{0,1,2})
        self.assertEqual(duration_order,{-1,0,1})
        self.assertEqual(concurrence,{False,True})

    def test_boundary_and_identity_and_duration_ties_rejected(self):
        for family in ("reverse_words", "before", "duration_compare", "narration_visible"):
            r = copy.deepcopy(next(r for r in self.records if r["metadata"]["family"] == family))
            target = role_nodes(r, "occurrence", "target-event")[0]
            other = role_nodes(r, "occurrence", "other-event")[0]
            if family == "reverse_words":
                role_nodes(r, "occurrence", "unrelated-words")[0]["interval"] = [target["interval"][1], target["interval"][1] + 2]
            elif family == "before":
                other["interval"] = [target["interval"][0]-5, target["interval"][0]-1]
                role_nodes(r, "occurrence", "last-event")[0]["interval"] = [target["interval"][0]-4, target["interval"][0]-1]
            elif family == "duration_compare":
                other["interval"] = [4, 4 + target["interval"][1] - target["interval"][0]]
            else:
                r["metadata"]["identities"][other["id"]] = dict(event="target-event", role="occurrence")
            with self.assertRaises(ValueError):
                binding.label_binding_case(r["nodes"], r["metadata"]["query"], r["metadata"]["identities"])

    def test_ambiguity_rejected_and_no_llm_truth(self):
        r = copy.deepcopy(next(r for r in self.records if r["metadata"]["family"] == "reverse_words"))
        target = role_nodes(r, "occurrence", "target-event")[0]
        role_nodes(r, "mention")[0]["interval"] = target["interval"][:]
        role_nodes(r, "occurrence", "unrelated-words")[0]["interval"] = target["interval"][:]
        with self.assertRaises(ValueError):
            binding.label_binding_case(r["nodes"], r["metadata"]["query"], r["metadata"]["identities"])
        tree = ast.parse(Path(binding.__file__).read_text(encoding="utf-8"))
        imports = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
        self.assertEqual(imports, {"copy", "random"})
        from_imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertEqual([(n.module, [a.name for a in n.names]) for n in from_imports], [("avbind_synthetic", ["SCENES"])])
        for r in self.records:
            self.assertTrue(r["metadata"]["construction_only"])
            self.assertFalse(set(r["answer_options"]) & {"Yes", "No", "Past", "Present", "Future"})
            inputs = json.dumps({k: r[k] for k in ("question", "answer_options", "nodes")})
            self.assertNotIn("target-event", inputs)
            self.assertNotIn("SVHalluc", inputs)


if __name__ == "__main__":
    unittest.main()

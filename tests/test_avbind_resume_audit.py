import unittest
from server_jobs.avbind_result_audit import audit
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np


class ArchivedAuditTests(unittest.TestCase):
    def test_timeline_resume_skips_completed_condition_not_just_completed_sample(self):
        source=Path('server_jobs/avbind_timeline_pilot.py').read_text(encoding='utf-8')
        indices=[1454,2396,1727,1647,2345,2348,2141]
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[None]*(max(indices)+1)
            previous=[];existing=[]
            for index in indices:
                rows[index]=dict(video_path=f'{index}.mp4',question='arbitrary q',task_type_abbr='TA',answer='B')
                previous.append(dict(idx=index,base=[1,2]))
                record=dict(idx=index,task='TA',gold='B',base=[1,2],memory=[1,2],
                            memory_trace={'audits':[{'tokens':14}]})
                if index!=1454:
                    record.update(binding=[1,2],binding_trace={'audits':[{'tokens':14}]})
                existing.append(record)
            (root/'samples.json').write_text(json.dumps(rows),encoding='utf-8')
            (root/'avbind_independent_kv_r7_s1.jsonl').write_text('\n'.join(map(json.dumps,previous)),encoding='utf-8')
            (root/'avbind_timeline_r11_s1.jsonl').write_text('\n'.join(map(json.dumps,existing))+'\n',encoding='utf-8')
            def infer(src,*args,isolated):
                calls.append((int(src.stem),isolated))
                return dict(scores=[1,2],audits=[{'tokens':14}])
            ns=dict(LAB_SERVER=1,OUT=root,json=json,np=np,BS='\\',fetch=lambda name:root/name,
                    cut=lambda src,*args:src,parse_question=lambda _:('q',['a','b']),
                    TIMELINE_NS={'avbind_timeline_answer':infer},
                    torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
            with patch('builtins.print'):
                exec(compile(source,'timeline_resume','exec'),ns)
                self.assertEqual(calls,[(1454,True)])
                self.assertEqual(ns['RESULT']['completed'],{'memory':7,'binding':7})
                calls.clear();exec(compile(source,'timeline_resume','exec'),ns)
                self.assertEqual(calls,[])

    def test_remote_resume_skips_successes_recovers_smoke_and_retries_only_missing(self):
        source = Path('server_jobs/avbind_counterfactual_pilot.py').read_text(encoding='utf-8')
        indices = [1454,2396,1727,1647,2345,2348,2141]
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'uploads').mkdir()
            prior = [dict(idx=i, task='TA', gold='B', base=[1,2]) for i in indices]
            (root / 'avbind_independent_kv_r7_s1.jsonl').write_text('\n'.join(map(json.dumps, prior)), encoding='utf-8')
            smoke = dict(id='8cea2842d1',status='done',result=dict(idx=1454,scores=[2,1],gain=1,fixed_coefficient=1))
            (root / 'uploads/avbind_r8_smoke_s1.json').write_text(json.dumps(smoke), encoding='utf-8')
            samples = [None] * (max(indices)+1)
            for i in indices:
                samples[i] = dict(video_path=f'{i}.mp4',question='q',task_type_abbr='TA',answer='B')
            (root / 'samples.json').write_text(json.dumps(samples), encoding='utf-8')
            already = [dict(idx=2396,task='TA',gold='B',base=[1,2],binding=[1,2]),
                       dict(idx=1727,task='TA',gold='B',base=[1,2],error='prior interruption')]
            (root / 'avbind_counterfactual_r8_s1.jsonl').write_text('\n'.join(map(json.dumps, already))+'\n',encoding='utf-8')
            def infer(src,*args,**kwargs):
                calls.append(int(src.stem))
                return dict(scores=[1,2])
            namespace = dict(LAB_SERVER=1,OUT=root,json=json,np=np,BS='\\',
                             fetch=lambda name: root/name,cut=lambda src,*args: src,
                             parse_question=lambda _: ('q',['a','b']),
                             NEURAL_NS={'avbind_counterfactual_answer':infer},
                             torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
            with patch('builtins.print'):
                exec(compile(source,'resume','exec'),namespace)
                self.assertEqual(calls,[1727,1647,2345,2348,2141])
                self.assertEqual(namespace['RESULT']['completed'],7)
                calls.clear()
                exec(compile(source,'resume','exec'),namespace)
                self.assertEqual(calls,[])
                self.assertEqual(namespace['RESULT']['errors'],0)

    def test_success_not_replaced_by_error_or_double_counted(self):
        good = {'idx': 1, 'task': 'TA', 'gold': 'B', 'base': [2, 1], 'binding': [1, 2]}
        error = {'idx': 1, 'task': 'TA', 'gold': 'B', 'error': 'later diagnostic error'}
        with patch('pathlib.Path.read_text', return_value='\n'.join(map(json.dumps, [good, error, good]))):
            result = audit(['unused'])
        self.assertEqual(result['unique_samples'], 1)
        self.assertEqual(result['groups']['TA/B']['binding_correct'], 1)
        self.assertEqual(result['groups']['TA/B']['errors'], 0)

    def test_source_shares_are_normalized_and_not_called_attention(self):
        row = {'idx': 1, 'task': 'TF', 'gold': 'C', 'base': [1, 2, 0], 'binding': [1, 2, 0],
               'binding_trace': {'nodes': [{'source': 'video'}, {'source': 'audio'}],
                                 'audits': [{'memory_mass': [.002, .998]}]}}
        with patch('pathlib.Path.read_text', return_value=json.dumps(row)):
            result = audit(['unused'])
        self.assertAlmostEqual(result['source_prior_shares'][0]['source_prior_shares']['audio'], .998)
        self.assertIn('NOT measured final attention', result['warning'])

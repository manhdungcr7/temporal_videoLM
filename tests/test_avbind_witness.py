import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch


class WitnessTests(unittest.TestCase):
    def namespace(self):
        ns={}
        exec(compile(Path('repo/kaggle/jobs/avbind_witness.py').read_text(encoding='utf-8'),'witness','exec'),ns)
        return ns

    def test_draft_prompt_has_no_options_or_fixed_source_order(self):
        ns=self.namespace()
        text=ns['witness_tail_text']('What did he say while sweeping?')+ns['WITNESS_INSTRUCTION']
        self.assertNotIn('(A)',text)
        self.assertIn('any evidence order',text)
        self.assertNotIn('past',ns['WITNESS_INSTRUCTION'])
        self.assertNotIn('TF',ns['WITNESS_INSTRUCTION'])

    def test_fallible_draft_is_not_an_observation(self):
        ns=self.namespace()
        text=ns['witness_tail_text']('q',['x','y'],(1,0),'possibly heard x')
        self.assertIn('may be wrong',text)
        self.assertTrue(text.endswith('(A) y (B) x'))

    def test_provenance_changes_for_media_question_and_condition(self):
        ns=self.namespace();ns.update(GUIDE_METADATA={'model_revision':'v','source_sha256':'a'},WITNESS_SOURCE_SHA='s')
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'video';video.write_bytes(b'one')
            first=ns['witness_identity'](video,'q',False)
            self.assertFalse(first['options_used'])
            self.assertNotEqual(first,ns['witness_identity'](video,'q2',False))
            self.assertNotEqual(first,ns['witness_identity'](video,'q',True))
            video.write_bytes(b'two')
            self.assertNotEqual(first,ns['witness_identity'](video,'q',False))

    def test_resume_only_missing_condition_then_zero_calls(self):
        source=Path('server_jobs/avbind_witness_pilot.py').read_text(encoding='utf-8')
        indices=[1454,2396,1727,1647,2345,2348,2141];calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[None]*(max(indices)+1);prior=[];existing=[]
            for index in indices:
                rows[index]=dict(video_path=f'{index}.mp4',question='q',task_type_abbr='TA',answer='B')
                prior.append(dict(idx=index,base=[1,2]))
                r=dict(idx=index,task='TA',gold='B',base=[1,2],memory=[1,2])
                if index!=1454:r['binding']=[1,2]
                existing.append(r)
            (root/'samples.json').write_text(json.dumps(rows),encoding='utf-8')
            (root/'avbind_independent_kv_r7_s1.jsonl').write_text('\n'.join(map(json.dumps,prior)),encoding='utf-8')
            (root/'avbind_witness_r15_s1.jsonl').write_text('\n'.join(map(json.dumps,existing))+'\n',encoding='utf-8')
            def infer(video,*args,guided,**kwargs):
                calls.append((int(video.stem),guided));return dict(scores=[1,2])
            ns=dict(LAB_SERVER=1,OUT=root,json=json,np=np,BS='\\',fetch=lambda name:root/name,
                cut=lambda src,*args:src,parse_question=lambda _:('q',['a','b']),
                TIMELINE_NS={'avbind_witness_answer':infer},
                torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
            with patch('builtins.print'):
                exec(compile(source,'resume','exec'),ns)
                self.assertEqual(calls,[(1454,True)])
                calls.clear();exec(compile(source,'resume','exec'),ns);self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()

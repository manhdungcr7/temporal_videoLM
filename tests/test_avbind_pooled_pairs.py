import ast
import unittest
from pathlib import Path
import torch
from repo.kaggle.jobs.avbind_pooled_pairs import pool_media_states


class PooledMediaTests(unittest.TestCase):
    def test_text_and_assistant_states_cannot_enter_pool(self):
        hidden=torch.randn(1,6,8);mask=torch.tensor([[False,True,True,False,True,False]])
        first=pool_media_states(hidden,mask);changed=hidden.clone();changed[0,~mask[0]]+=1000
        torch.testing.assert_close(first,pool_media_states(changed,mask))
        self.assertEqual(first.shape,(8,))

    def test_empty_or_multi_source_batch_rejected(self):
        with self.assertRaises(ValueError):pool_media_states(torch.zeros(1,3,8),torch.zeros(1,3,dtype=torch.bool))
        with self.assertRaises(ValueError):pool_media_states(torch.zeros(2,3,8),torch.ones(2,3,dtype=torch.bool))

    def test_extraction_reuses_teacher_data_without_regeneration(self):
        text=Path('server_jobs/avbind_pooled_pair_extract.py').read_text(encoding='utf-8')
        self.assertNotIn('native_pair_description(',text)
        self.assertNotIn('native_pair_text_feature(',text)
        self.assertIn('example=dict(old)',text)
        tree=ast.parse(Path('repo/kaggle/jobs/avbind_pooled_pairs.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pooled_native_feature')
        self.assertEqual([a.arg for a in fn.args.args],['media','mode'])
        names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
        self.assertFalse(names & {'question','options','gold','task','start','end'})


if __name__=='__main__':unittest.main()

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from repo.kaggle.jobs.avbind_alignment import occurrence_token_bias
import repo.kaggle.jobs.avbind_native_guidance as native
import repo.kaggle.jobs.avbind_grounded_guidance as grounded


class GroundedQueryTests(unittest.TestCase):
    def test_native_prefill_and_each_answer_state_refresh_routes(self):
        native.occurrence_token_bias=occurrence_token_bias
        capture=torch.nn.Identity()
        model=SimpleNamespace(model=SimpleNamespace(layers=[None]*27+[capture]+[SimpleNamespace(self_attn=object()) for _ in range(8)]))
        current={}
        @contextlib.contextmanager
        def reads(modules,rows,bias,answer_bias,indices,audit):
            current.update(bias=bias,answer_bias=answer_bias);yield
        inp={'input_ids':torch.zeros(1,7,dtype=torch.long)};trace=[]
        with patch.object(grounded,'MODEL_',model,create=True),patch.object(grounded,'grounded_question_rows',lambda *a:[4,5]), \
             patch.object(grounded,'indexed_occurrence_bias',native.indexed_occurrence_bias,create=True), \
             patch.object(grounded,'native_late_guidance',reads,create=True):
            with grounded.grounded_occurrence_reads(inp,'arbitrary reference',torch.eye(3),[torch.tensor([i]) for i in range(3)],trace,[]):
                hidden=torch.zeros(1,7,3);hidden[0,4]=torch.tensor([2.,0.,0.]);hidden[0,5]=torch.tensor([0.,2.,0.]);hidden[0,6]=torch.tensor([0.,0.,2.])
                capture(hidden);first=current['answer_bias'].clone()
                capture(torch.tensor([[[2.,0.,0.]]]))
                self.assertFalse(torch.equal(first,current['answer_bias']))
                self.assertEqual([t['kind'] for t in trace],['native_multimodal_question','current_answer_state','current_answer_state'])
                self.assertEqual(current['bias'].shape,(2,7))
            self.assertEqual(len(capture._forward_hooks),0)

    def test_current_answer_hook_is_removed_on_failure(self):
        capture=torch.nn.Identity()
        model=SimpleNamespace(model=SimpleNamespace(layers=[None]*27+[capture]+[SimpleNamespace(self_attn=object()) for _ in range(8)]))
        @contextlib.contextmanager
        def reads(*args,**kwargs):yield
        with patch.object(grounded,'MODEL_',model,create=True),patch.object(grounded,'grounded_question_rows',lambda *a:[4]), \
             patch.object(grounded,'native_late_guidance',reads,create=True):
            with self.assertRaises(RuntimeError):
                with grounded.grounded_occurrence_reads({'input_ids':torch.zeros(1,7)},'q',torch.eye(3),[],[],[]):
                    raise RuntimeError('interruption')
            self.assertEqual(len(capture._forward_hooks),0)


if __name__=='__main__':unittest.main()

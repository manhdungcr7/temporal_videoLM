import ast
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
from repo.kaggle.jobs.avbind_timeline import local_record_allowed, timeline_isolation, timeline_grid_seconds


class TimelineIsolationTests(unittest.TestCase):
    def test_actual_fps_metadata_is_tensor_for_native_rope(self):
        result=timeline_grid_seconds([{'media':{'fps':2.0}},{'media':{'fps':1.25}}],2)
        torch.testing.assert_close(result,torch.tensor([1.,1.6]))
        self.assertEqual(result.dtype,torch.float32)
        self.assertTrue(torch.isfinite(result.cpu().float()).all())

    def test_records_cannot_see_other_records_question_or_future(self):
        rows=torch.tensor([7,8,9]);keys=torch.arange(16)
        mask=local_record_allowed(rows,keys,3,7,10)
        self.assertTrue(mask[:,:3].all())
        self.assertFalse(mask[:,3:7].any())
        self.assertFalse(mask[:,10:].any())
        self.assertTrue(mask[0,7]);self.assertFalse(mask[0,8])

    def test_prefix_and_source_records_have_no_question_label_access(self):
        tree=ast.parse(Path('repo/kaggle/jobs/avbind_timeline.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='timeline_records')
        names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
        self.assertFalse(names & {'question','answer_options','gold','task'})
        all_names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertFalse(all_names & {'gold','task','task_type_abbr','TA','TF','CMTB'})

    def test_prefix_and_record_indices_partition_native_sequence(self):
        prefix=3;segments=[(3,7),(7,10)];tail=16
        indices=list(range(prefix))
        for a,b in segments:indices.extend(range(a,b))
        indices.extend(range(segments[-1][1],tail))
        self.assertEqual(indices,list(range(tail)))

    def test_actual_dispatch_matches_dense_block_mask_with_gqa(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        torch.manual_seed(9)
        q=torch.randn(1,4,13,8);k=torch.randn(1,2,13,8);v=torch.randn_like(k)
        prefix=3;segments=[(3,7),(7,10)]
        keys=torch.arange(13);allowed=keys[None,:]<=keys[:,None]
        for a,b in segments:
            allowed[a:b]=local_record_allowed(keys[a:b],keys,prefix,a,b)
        for supplied in (None, torch.ones(1,1,13,13,dtype=torch.bool), torch.zeros(1,1,13,13)):
            with timeline_isolation(segments,prefix):
                actual,_=ALL_ATTENTION_FUNCTIONS['sdpa'](None,q,k,v,supplied,scaling=.3)
            expected=F.scaled_dot_product_attention(q,k.repeat_interleave(2,-3),v.repeat_interleave(2,-3),
                attn_mask=allowed,scale=.3).transpose(1,2).contiguous()
            torch.testing.assert_close(actual,expected)
            self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],original)

    def test_guidance_changes_only_tail_rows_and_preserves_local_record_masks(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        torch.manual_seed(11)
        q=torch.randn(1,4,13,8);k=torch.randn(1,2,13,8);v=torch.randn_like(k)
        keys=torch.arange(13);allowed=keys[None,:]<=keys[:,None]
        for a,b in [(3,7),(7,10)]:allowed[a:b]=local_record_allowed(keys[a:b],keys,3,a,b)
        bias=torch.zeros(13,13);bias[10:,:7]=.6
        def provider(module,rows,keylen):return bias[rows,:keylen]
        with timeline_isolation([(3,7),(7,10)],3,tail_bias=provider):
            actual,_=ALL_ATTENTION_FUNCTIONS['sdpa'](None,q,k,v,None)
        mask=bias.masked_fill(~allowed,float('-inf'))
        expected=F.scaled_dot_product_attention(q,k.repeat_interleave(2,-3),v.repeat_interleave(2,-3),attn_mask=mask).transpose(1,2)
        torch.testing.assert_close(actual,expected)

    def test_attention_registry_restores_on_error(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa'];q=torch.ones(1,1,5,8)
        with self.assertRaises(ValueError):
            with timeline_isolation([(1,3)],1):
                ALL_ATTENTION_FUNCTIONS['sdpa'](None,q,q,q,None,dropout=.1)
        self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],original)

    def test_sensor_encoder_bypasses_decoder_mask(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa'];decoder=object();encoder=object();calls=[]
        def ordinary(module,q,k,v,mask,**kwargs):
            calls.append(module);return q.transpose(1,2),None
        ALL_ATTENTION_FUNCTIONS.register('sdpa',ordinary)
        try:
            # Encoder has a different length than the decoder's record map.
            q=torch.ones(1,1,2,8)
            with timeline_isolation([(3,7)],3,decoder_modules=[decoder]):
                actual,_=ALL_ATTENTION_FUNCTIONS['sdpa'](encoder,q,q,q,None)
                torch.testing.assert_close(actual,q.transpose(1,2))
            self.assertEqual(calls,[encoder])
            self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],ordinary)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

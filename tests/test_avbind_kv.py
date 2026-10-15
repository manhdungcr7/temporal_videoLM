"""Math/interface tests of separate keys/values, not performance claims."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
from repo.kaggle.jobs.avbind_kv import inverse_text_rope,joint_memory_read

class IndependentKVTests(unittest.TestCase):
    def test_inverse_rotary_recovers_content_for_each_query_row(self):
        torch.manual_seed(3);q=torch.randn(1,4,5,8)
        phase=torch.randn(1,1,5,4);cos=torch.cat((phase.cos(),phase.cos()),-1)*1.2
        sin=torch.cat((phase.sin(),phase.sin()),-1)*1.2
        half=torch.cat((-q[...,4:],q[...,:4]),-1)
        rotated=q*cos+half*sin
        torch.testing.assert_close(inverse_text_rope(rotated,cos,sin),q,rtol=1e-5,atol=1e-6)

    def test_null_memory_matches_native_masked_attention(self):
        torch.manual_seed(7)
        q=torch.randn(1,4,3,8);k=torch.randn(1,2,9,8);v=torch.randn_like(k)
        mk=torch.randn(1,2,5,8);mv=torch.randn_like(mk)
        mask=torch.ones(3,9,dtype=torch.bool);mask[:,-1]=False
        actual=joint_memory_read(q,k,v,q,mk,mv,torch.full((1,1,1,5),float('-inf')),mask)
        expected=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,attn_mask=mask)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_memory_keys_values_are_distinct_and_permutation_equivariant(self):
        torch.manual_seed(7)
        q=torch.randn(1,4,3,8);k=torch.randn(1,2,9,8);v=torch.randn_like(k)
        mk=torch.randn(1,2,5,8);mv=torch.randn_like(mk);mass=torch.randn(1,1,1,5)
        original=joint_memory_read(q,k,v,q,mk,mv,mass)
        order=torch.tensor([4,1,3,0,2])
        permuted=joint_memory_read(q,k,v,q,mk[:,:,order],mv[:,:,order],mass[:,:,:,order])
        torch.testing.assert_close(original,permuted,rtol=1e-5,atol=1e-6)
        changed=joint_memory_read(q,k,v,q,mk,mv+2,mass)
        self.assertGreater(float((changed-original).abs().max()),.1)

    def test_address_values_do_not_receive_question_options_or_generation(self):
        tree=ast.parse(Path('repo/kaggle/jobs/avbind_kv.py').read_text(encoding='utf-8'))
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ('latent_address_value','addressed_value_bank'):
            names={n.id for n in ast.walk(functions[name]) if isinstance(n,ast.Name)}
            self.assertFalse(names&{'question','answer_options','gold','task'})
            methods={n.func.attr for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            self.assertNotIn('generate',methods)

if __name__=='__main__':unittest.main()

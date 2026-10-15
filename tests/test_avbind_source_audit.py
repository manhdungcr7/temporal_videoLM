import ast
from pathlib import Path
import unittest
import torch
from repo.kaggle.jobs.avbind_source_audit import equalize_source_log_mass, final_row_memory_audit


class SourceBudgetAuditTests(unittest.TestCase):
    def test_preserves_total_null_mass_and_within_source_ratios(self):
        mass = torch.tensor([.001, .003, .199, .597]).view(1,1,1,-1)
        ids = torch.tensor([0,0,1,1])
        corrected = equalize_source_log_mass(mass.log(), ids).exp()
        torch.testing.assert_close(corrected.sum(-1), mass.sum(-1))
        torch.testing.assert_close(corrected[..., :2].sum(-1), corrected[..., 2:].sum(-1))
        self.assertAlmostEqual(float(corrected[..., 1] / corrected[..., 0]), 3, places=5)

    def test_all_null_and_absent_source_remain_null(self):
        ids = torch.tensor([0,1])
        null = torch.full((1,1,1,2), float('-inf'))
        self.assertTrue(torch.isneginf(equalize_source_log_mass(null, ids)).all())
        mass = torch.tensor([.8,0.]).view(1,1,1,2)
        torch.testing.assert_close(equalize_source_log_mass(mass.log(), ids).exp(), mass)

    def test_source_label_and_record_order_equivariance(self):
        mass = torch.tensor([.01,.03,.2,.56]).view(1,1,1,4).log()
        ids = torch.tensor([0,0,1,1]);order = torch.tensor([3,0,2,1])
        actual = equalize_source_log_mass(mass[..., order], 7-ids[order])
        torch.testing.assert_close(actual, equalize_source_log_mass(mass, ids)[..., order])

    def test_measured_mass_sums_to_one_and_zero_prior_blocks_bank(self):
        torch.manual_seed(7)
        q = torch.randn(1,4,3,8);key = torch.randn(1,2,9,8);mk = torch.randn(1,2,4,8)
        ids = torch.tensor([0,1,0,1]);mass = torch.zeros(1,1,1,4)
        mask = torch.ones(3,9,dtype=torch.bool);mask[:,-1] = False
        audit = final_row_memory_audit(q,key,q,mk,mass,ids,mask)
        self.assertAlmostEqual(audit['native']+audit['external'],1,places=6)
        self.assertAlmostEqual(sum(audit['external_by_source'].values()),audit['external'],places=6)
        null = final_row_memory_audit(q,key,q,mk,torch.full_like(mass,float('-inf')),ids,mask)
        self.assertEqual(null['external'],0)

    def test_intervention_has_no_task_label_or_template_access(self):
        tree = ast.parse(Path('repo/kaggle/jobs/avbind_source_audit.py').read_text(encoding='utf-8'))
        names = {n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertFalse(names & {'task','task_type_abbr','gold','TA','TF','CMTB'})

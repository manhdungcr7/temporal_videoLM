import unittest
import torch
from repo.kaggle.jobs.avbind_context_clock import source_record_indices,serialize_record_positions


class ContextPreservingClockTests(unittest.TestCase):
    def test_all_native_features_covered_once_and_modality_order_preserved(self):
        ids=torch.tensor([0,9,9,8,8,9,9,8,8,0])
        time=torch.tensor([0,10,35,10,35,60,85,60,85,86])
        pos=time.view(1,1,-1).expand(3,1,-1)
        records,origin=source_record_indices(ids,pos,{'video':9,'audio':8},4)
        self.assertEqual(len(records),4)
        covered=torch.cat([r['indices'] for r in records])
        self.assertEqual(sorted(covered.tolist()),list(range(1,9)))
        self.assertEqual(len(covered.unique()),8)
        for source,token in [('video',9),('audio',8)]:
            actual=torch.cat([r['indices'] for r in records if r['source']==source])
            torch.testing.assert_close(actual,torch.where(ids==token)[0])

    def test_serial_control_preserves_local_geometry_and_changes_only_clock_translation(self):
        pos=torch.tensor([0,1,2,10,11,12,13,10,11,12,13,20,21]).view(1,1,-1).expand(3,1,-1).clone()
        result=serialize_record_positions(pos,[(3,7),(7,11)],3)
        for a,b in [(3,7),(7,11)]:
            torch.testing.assert_close(result[:,:,a:b]-result[:,:,a:a+1],pos[:,:,a:b]-pos[:,:,a:a+1])
        self.assertGreater(int(result[:,:,7:11].min()),int(result[:,:,3:7].max()))
        torch.testing.assert_close(result[:,:,:3],pos[:,:,:3])

    def test_uniform_native_clock_translation_does_not_change_partition(self):
        ids=torch.tensor([0,9,8,9,8,0]);pos=torch.tensor([0,10,10,70,70,71]).view(1,1,-1).expand(3,1,-1)
        first,_=source_record_indices(ids,pos,{'video':9,'audio':8},4)
        shifted,_=source_record_indices(ids,pos+1000,{'video':9,'audio':8},4)
        self.assertEqual([(r['source'],r['start'],r['end'],r['indices'].tolist()) for r in first],
                         [(r['source'],r['start'],r['end'],r['indices'].tolist()) for r in shifted])

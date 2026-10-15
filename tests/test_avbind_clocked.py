import unittest
import torch
from repo.kaggle.jobs.avbind_clocked import canonical_record_positions


class SharedOccurrenceClockTests(unittest.TestCase):
    def example(self):
        ids=torch.tensor([0,0,0,0,9,9,9,0,0,8,8,8,0,0,0])
        pos=torch.arange(15).view(1,1,-1).expand(3,1,-1).clone()
        pos[1,0,5]+=2;pos[2,0,6]+=3
        return pos,ids,[(3,8),(8,13)]

    def test_concurrent_modalities_share_physical_origin_not_serial_order(self):
        pos,ids,segments=self.example()
        clocked,audit=canonical_record_positions(pos,ids,segments,[2,2],{8,9},3)
        self.assertEqual(int(clocked[0,0,4]),int(clocked[0,0,9]))
        self.assertEqual(audit['first_media_coordinates'][0],audit['clock_origin']+50)
        torch.testing.assert_close(clocked[:,:,:3],pos[:,:,:3])

    def test_native_spatial_temporal_differences_preserved_inside_each_record(self):
        pos,ids,segments=self.example()
        clocked,_=canonical_record_positions(pos,ids,segments,[0,4],{8,9},3)
        for a,b in segments:
            torch.testing.assert_close(clocked[:,:,a:b]-clocked[:,:,a:a+1],pos[:,:,a:b]-pos[:,:,a:a+1])

    def test_uniform_clock_translation_preserves_all_record_question_relative_positions(self):
        pos,ids,segments=self.example()
        first,_=canonical_record_positions(pos,ids,segments,[0,4],{8,9},3)
        shifted,_=canonical_record_positions(pos,ids,segments,[10,14],{8,9},3)
        torch.testing.assert_close(shifted[:,:,3:]-first[:,:,3:],torch.full_like(first[:,:,3:],250))
        self.assertTrue(torch.equal(first[:,0,13],first[0,0,13].expand(3)))

    def test_independent_source_shift_changes_only_its_record_and_query_origin(self):
        pos,ids,segments=self.example()
        first,_=canonical_record_positions(pos,ids,segments,[0,4],{8,9},3)
        shifted,_=canonical_record_positions(pos,ids,segments,[0,8],{8,9},3)
        torch.testing.assert_close(first[:,:,3:8],shifted[:,:,3:8])
        torch.testing.assert_close(shifted[:,:,8:13]-first[:,:,8:13],torch.full_like(first[:,:,8:13],100))

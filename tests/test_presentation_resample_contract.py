"""Actual PTS select the displayed frame even for reordered/duplicate metadata."""
import ast
from pathlib import Path
import unittest
import numpy as np

class PresentationSamplingContract(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'server_jobs/presentation_resample40.py'
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pc_select']
        ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        self.select=ns['pc_select']

    def test_no_future_frame_at_uneven_intervals(self):
        pts=np.array([0.,.1,.8,1.2]);times=np.array([0.,.3,.9,1.1,1.8])
        indices=self.select(pts,times)
        np.testing.assert_equal(indices,[0,1,2,2,3])
        self.assertTrue(np.all(pts[indices]<=times))

    def test_reordered_pts_preserve_their_picture_indices(self):
        np.testing.assert_equal(self.select([0.,.8,.4,1.2],[.5,.9,1.3]),[2,1,3])

    def test_black_before_first_and_last_duplicate_wins(self):
        np.testing.assert_equal(self.select([.2,.8,.8,1.2],[0.,.2,.8,5.]),[-1,0,2,3])

if __name__=='__main__':unittest.main()

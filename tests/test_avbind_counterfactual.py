"""Control integrity and clock/question independence, not QA performance."""
import ast
from pathlib import Path
import unittest
import numpy as np
import torch
from repo.kaggle.jobs import avbind_counterfactual as cf

class SensorControls(unittest.TestCase):
    def test_video_freeze_preserves_shape_and_original_media(self):
        # The runtime injects its shared torch/np helpers into the clean namespace.
        cf.torch=torch;cf.np=np
        frames=torch.arange(4*3*2*2).reshape(4,3,2,2);audio=np.arange(12,dtype=np.float32)
        media=dict(videos=[frames],audio=[audio],fps=2,path='unchanged.mp4')
        before=frames.clone();control=cf.sensor_control(media,'video')
        self.assertEqual(control['videos'][0].shape,frames.shape)
        for frame in control['videos'][0]:torch.testing.assert_close(frame,frames[0])
        torch.testing.assert_close(frames,before)
        self.assertIs(control['audio'],media['audio'])
        self.assertEqual(control['fps'],media['fps'])

    def test_silence_keeps_duration_and_does_not_change_visuals(self):
        cf.torch=torch;cf.np=np
        wave=np.arange(16000,dtype=np.float32);media=dict(audio=[wave],videos=[torch.ones(4,3,2,2)],fps=2)
        control=cf.sensor_control(media,'audio')
        self.assertEqual(control['audio'][0].shape,wave.shape)
        self.assertEqual(control['audio'][0].dtype,wave.dtype)
        self.assertEqual(np.count_nonzero(control['audio'][0]),0)
        self.assertEqual(np.count_nonzero(wave),15999)
        self.assertIs(control['videos'],media['videos'])

    def test_keys_use_fixed_coefficient_without_question_or_clock_prompt(self):
        tree=ast.parse(Path(cf.__file__).read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='differential_sensor_bank')
        names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
        self.assertFalse(names&{'question','answer_options','gold','task'})
        key_calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='latent_sensor_feature']
        self.assertEqual(len(key_calls),2)
        self.assertTrue(all(len(n.args)==2 for n in key_calls))

if __name__=='__main__':unittest.main()

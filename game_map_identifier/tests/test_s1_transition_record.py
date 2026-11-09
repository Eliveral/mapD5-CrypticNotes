import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.s1_transition_record import TransitionRecorder

class Inline:
    def __init__(self,target,**kwargs):self.target=target
    def start(self):self.target()

class TransitionEvidenceTests(unittest.TestCase):
    def test_bounded_slots_and_explicit_missing_frame_flags(self):
        with tempfile.TemporaryDirectory() as folder, patch('src.s1_transition_record.threading.Thread',Inline):
            rec=TransitionRecorder(Path(folder))
            source=dict(floor={'id':'s1_13_B2'},at=1.,sample_at=1.,matrix=np.eye(2,3),zoom=.7,
                player_hd=np.array([10.,20.]),diagnostic_frame=np.zeros((50,100,3),np.uint8),diagnostic_scale=.5)
            sample=dict(revision=1,selected_floor='B3',frame=np.zeros((100,200,3),np.uint8),
                at=2.,zoom=.7,shared_attempt={'reason':'target_pose_unverified'})
            for n in range(9):
                source['sample_at']=n
                rec.observe(sample,source)
            files=list(rec.root.glob('*.json'));self.assertEqual(len(files),6)
            for file in files:
                doc=json.loads(file.read_text('utf8'))
                self.assertTrue(doc['has_source']);self.assertFalse(doc['has_clear'])
                self.assertEqual(doc['source'],'s1_13_B2')
            count=rec.count
            rec.observe(sample,source);self.assertEqual(rec.count,count)

    def test_frame_scale_retains_aspect(self):
        image,scale=TransitionRecorder.picture(np.zeros((1000,2000,3),np.uint8))
        self.assertEqual(image.shape,(640,1280,3));self.assertEqual(scale,.64)

if __name__=='__main__':unittest.main()

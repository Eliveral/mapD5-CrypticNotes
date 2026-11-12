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
            for n in range(27):
                source['sample_at']=n
                rec.observe(sample,source)
            files=list(rec.root.glob('*.json'));self.assertEqual(len(files),24)
            for file in files:
                doc=json.loads(file.read_text('utf8'))
                self.assertTrue(doc['has_source']);self.assertFalse(doc['has_clear'])
                self.assertEqual(doc['source'],'s1_13_B2')
            count=rec.count
            rec.observe(sample,source);self.assertEqual(rec.count,count)

    def test_async_record_owns_attempt_snapshot(self):
        pending=[]
        class Deferred:
            def __init__(self,target,**kwargs):pending.append(target)
            def start(self):pass
        with tempfile.TemporaryDirectory() as folder, patch('src.s1_transition_record.threading.Thread',Deferred):
            rec=TransitionRecorder(Path(folder))
            source=dict(floor={'id':'s1_11_B2'},at=1.,matrix=np.eye(2,3),zoom=.7,player_hd=np.array([2.,3.]))
            sample=dict(selected_floor='B3',frame=np.zeros((20,30,3),np.uint8),
                at=2.,zoom=.7,shared_attempt={'reason':'accepted'})
            rec.observe(sample,source)
            sample.update(selected_floor=None,zoom=None)
            sample['frame'][:]=255;sample['shared_attempt']['reason']='changed'
            source['matrix'][:]=0
            pending[0]()
            doc=json.loads((rec.root/'transition-0.json').read_text('utf8'))
            self.assertEqual(doc['target'],'B3')
            self.assertEqual(doc['attempt']['reason'],'accepted')
            self.assertEqual(doc['source_matrix'],[[1.,0.,0.],[0.,1.,0.]])
            import cv2,hashlib
            data=(rec.root/'transition-0-target.png').read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),doc['files']['transition-0-target.png'])
            self.assertFalse(cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR).any())

    def test_frame_preserves_native_pixels_and_owns_memory(self):
        source=np.random.default_rng(17).integers(0,256,(1000,2000,3),dtype=np.uint8)
        image,scale=TransitionRecorder.picture(source)
        self.assertEqual(image.shape,source.shape);self.assertEqual(scale,1.)
        self.assertTrue(np.array_equal(image,source))
        self.assertFalse(np.shares_memory(image,source))

if __name__=='__main__':unittest.main()

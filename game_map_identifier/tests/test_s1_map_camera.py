import unittest
from src.s1_map_camera import MapPointer
from src.s1_round_lock import RoundLock

class PointerTests(unittest.TestCase):
    def test_floor_click_and_drag_without_changing_scale(self):
        p=MapPointer();p.configure((10,(100,200,2560,1440)))
        before=p.snapshot();x=100+2560-.22*1440;y=200+.20*1440
        p.observe(0x201,(x,y),99)
        self.assertEqual(before,p.snapshot())
        p.observe(0x201,(x,y),10);p.observe(0x202,(x,y),10)
        click=p.snapshot();self.assertEqual(click.epoch,before.epoch);self.assertGreater(click.click_at,0)
        p.observe(0x201,(1500,800),10);p.observe(0x200,(1610,765),10);p.observe(0x202,(1610,765),10)
        self.assertEqual(p.snapshot().pan,(110.,-35.))
        self.assertEqual(p.snapshot().click_at,click.click_at)
        p.observe(0x200,(1900,850),10);self.assertEqual(p.snapshot().pan,(110.,-35.))
    def test_zoom_focus_and_window_change_invalidate_continuity(self):
        p=MapPointer();p.configure((10,(0,0,2560,1440)));before=p.snapshot()
        p.observe(0x20A,(1500,600),10);self.assertGreater(p.snapshot().epoch,before.epoch)
        p.configure(None);self.assertIsNone(p.snapshot())
        p.configure((10,(0,0,2560,1440)));self.assertGreater(p.snapshot().epoch,before.epoch)
        old=p.snapshot();p.configure((10,(50,0,2560,1440)));self.assertGreater(p.snapshot().epoch,old.epoch)
    def test_partial_or_conflicting_layouts_never_share_camera(self):
        s=RoundLock();s.candidate='s1_13';s.count=2;self.assertIsNone(s.shared_layout)
        s.layout='s1_13';self.assertEqual(s.shared_layout,'s1_13')
        for field in ('uncertain','b1_conflict','b2_arrival','recheck_b2_after_landing'):
            setattr(s,field,True);self.assertIsNone(s.shared_layout);setattr(s,field,False)
        s.landing.layout='s1_03';self.assertIsNone(s.shared_layout)
        s.landing.layout='s1_13';self.assertEqual(s.shared_layout,'s1_13')
        s.landing.waiting=True;self.assertIsNone(s.shared_layout)


class CameraBoundaryTests(unittest.TestCase):
    def test_source_frame_after_click_keeps_open_session_reference(self):
        from unittest.mock import Mock,patch
        import numpy as np
        from src.s1_map_camera import shared_floor,PointerSnapshot
        matcher=Mock()
        floor=dict(id='s1_13_B3',layout_id='s1_13',floor='B3')
        matcher.handoffs.floors={'s1_13_B3':floor}
        matcher.coordinates.predict.return_value=(np.eye(2,3),None)
        matcher.template.return_value=(None,np.zeros((240,400),np.uint8))
        matcher.handoffs.result.side_effect=lambda floor,mat,*args:dict(floor=floor,matrix=mat)
        pointer=PointerSnapshot(1,(0,0),2.,70.)
        visit=dict(floor=dict(id='s1_13_B2',layout_id='s1_13',floor='B2'),
                   matrix=np.eye(2,3),unique_layout=True,zoom=.7,at=2.2,sample_at=2.2,
                   open_since=1.,pointer=pointer)
        image=np.zeros((240,400,3),np.uint8)
        area=lambda im:np.full(im.shape[:2],255,np.uint8)
        with patch('src.s1_map_camera.selected_button',return_value=('B3',350,70)):
            hit=shared_floor(matcher,image,area,visit,'B3',zoom=.7,pointer=pointer,continuous=True)
            self.assertEqual(hit['camera_mode'],'floor_click')
            self.assertTrue(hit['camera_only'])
            self.assertIsNone(shared_floor(matcher,image,area,visit,'B3',zoom=.7,pointer=pointer,continuous=False))
            self.assertIsNone(shared_floor(matcher,image,area,visit,'B3',zoom=.5,pointer=pointer,continuous=True))

if __name__=='__main__':unittest.main()

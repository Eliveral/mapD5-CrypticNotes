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

if __name__=='__main__':unittest.main()

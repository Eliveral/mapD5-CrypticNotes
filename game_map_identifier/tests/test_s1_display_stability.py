"""Transient observations retain a verified display only within one open view."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.s1_map_camera import PointerSnapshot
from src.s1_round_lock import RoundLock
from src.s1_trigger_live import LiveController


@pytest.fixture
def showing(monkeypatch):
    controller=LiveController.__new__(LiveController)
    controller.active=True
    controller.hidden=controller.suspended=False
    controller.generation=4
    controller.session_revision=1
    controller.game=SimpleNamespace(hwnd=100)
    controller.target=100
    controller.sampling=False
    controller.round_lock=RoundLock()
    controller.round_lock.layout='s1_12'
    controller.observed_floor='B2'
    controller.app=Mock()
    controller.app.layout_limit.get.return_value='自动'
    controller.overlay=Mock(_visible=True)
    controller.overlay.hide.side_effect=lambda:setattr(controller.overlay,'_visible',False)
    controller.note=Mock()
    controller.begin_proof=Mock()
    controller.begin_landing=Mock()
    controller.check_pose=Mock()
    controller.view_cache=None
    controller.pending=None
    controller.proof_job=controller.landing_job=None
    controller.burst_until=0.
    controller.due=0.
    controller.display_recorder=None
    pointer=PointerSnapshot(2,(0.,0.),-1.,-1.)
    sample=dict(epoch=4,revision=1,game=controller.game,at=10.,selected_floor='B2',
                presence={'visible':True},pointer=pointer,zoom=.75,zoom_changed=False,
                frame=np.zeros((20,30,3),np.uint8),capture_ms=10.,
                probing=True,observed_probe=object(),stationary=False)
    controller.latest=dict(sample,stationary=True)
    controller.display=dict(epoch=4,selected_floor='B2',pointer=pointer,zoom=.75,
                            label='s1_12_B2',uploaded=True,provisional=False)
    clock=[10.]
    monkeypatch.setattr('src.s1_trigger_live.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('src.s1_live.foreground',lambda:100)
    return controller,sample,clock


def test_single_unsteady_frame_then_stationary_keeps_overlay(showing):
    controller,sample,clock=showing
    display=controller.display
    controller.sample_received(sample)
    assert controller.display is display
    assert controller.overlay._visible
    clock[0]+=.12
    controller.sample_received(dict(sample,stationary=True,at=clock[0]))
    assert controller.display is display
    assert 'unsteady_since' not in display
    controller.overlay.hide.assert_not_called()


def test_repeated_unverified_frames_expire_instead_of_freezing_overlay(showing):
    controller,sample,clock=showing
    controller.sample_received(sample)
    clock[0]+=.26
    controller.sample_received(dict(sample,at=clock[0]))
    assert controller.display is None
    controller.overlay.hide.assert_called_once()


@pytest.mark.parametrize('change',['pan','wheel','zoom','closed','other_floor','no_input','provisional','conflict'])
def test_real_changes_and_untrusted_display_do_not_wait(showing,change):
    controller,sample,_=showing
    if change=='pan':sample['pointer']=replace(sample['pointer'],pan=(20.,0.))
    elif change=='wheel':sample['pointer']=replace(sample['pointer'],epoch=3)
    elif change=='zoom':sample.update(zoom=.65,zoom_changed=True)
    elif change=='closed':sample['presence']={'visible':False}
    elif change=='other_floor':sample['selected_floor']='B4'
    elif change=='no_input':sample['pointer']=None
    elif change=='provisional':controller.display['provisional']=True
    elif change=='conflict':controller.round_lock.uncertain=True
    controller.sample_received(sample)
    assert controller.display is None
    controller.overlay.hide.assert_called_once()


def test_unreadable_floor_can_recover_before_hiding(showing):
    controller,sample,clock=showing
    display=controller.display
    controller.sample_received(dict(sample,selected_floor=None))
    assert controller.display is display
    clock[0]+=.1
    controller.sample_received(dict(sample,stationary=True))
    controller.overlay.hide.assert_not_called()


def test_floor_click_does_not_bridge_an_unreadable_floor(showing):
    controller,sample,_=showing
    sample.update(selected_floor=None,pointer=replace(sample['pointer'],click_at=10.,click_y=450.))
    controller.sample_received(sample)
    assert controller.display is None
    controller.overlay.hide.assert_called_once()


def test_stale_capture_cannot_start_or_extend_a_retry(showing):
    controller,sample,_=showing
    controller.sample_received(dict(sample,epoch=3))
    assert 'unsteady_since' not in controller.display
    controller.overlay.hide.assert_not_called()


def test_failed_new_pose_does_not_invalidate_a_still_stationary_old_view(showing,monkeypatch):
    """Exercise the two-capture path: candidate fails, original view recovers."""
    import queue
    from src import s1_trigger_live as live
    controller,sample,_=showing
    raw1=np.zeros((20,30,3),np.uint8)
    raw2=np.full_like(raw1,7)
    controller.capture_bounds=lambda:(0,0,30,20)
    controller.capture_excluded=False
    controller.ensure_overlay=lambda:None
    controller.roi=None
    controller.pose_fixes={};controller.floor_visits={};controller.review_signatures={}
    controller.warming=False
    controller.proof_due=controller.b1_due=0.
    controller.app.layout_limit.get.return_value='12'
    controller.app.floor_limit.get.return_value='自动'
    controller.app.overlay_alpha.get.return_value=18
    controller.layout_limit=lambda *args:'s1_12'
    controller.valid_visit=lambda v,*args:bool(v)
    controller.scale_agrees=lambda result,*args:bool(result)
    controller.events=queue.Queue()
    controller.capture_worker=SimpleNamespace(submit=lambda job:job())
    controller.hotkeys=SimpleNamespace(mouse_ready=True,pointer=SimpleNamespace(snapshot=lambda:sample['pointer']))
    controller.visibility=SimpleNamespace(inspect=lambda *args,**kwargs:{'visible':True})
    probe=Mock()
    probe.zoom=.75
    probe.stationary.side_effect=[False,True]
    probe.uncovered_frame.side_effect=lambda frame:frame.copy()+1
    probe.unpainted=np.full((20,30),255,np.uint8)
    controller.painted=dict(selected_floor='B2',probe=probe)
    floor=dict(id='s1_12_B2',layout_id='s1_12',floor='B2')
    controller.view_cache=dict(floor=floor,selected_floor='B2',at=9.,zoom=.75,
                              matrix=np.float64([[1,0,0],[0,1,0]]),signature=raw1)
    features=Mock()
    features.camera_motion.return_value=features.motion_to.return_value=None
    features.same_pixels.return_value=False
    candidate=dict(search='camera_pan',floor=floor,matrix=controller.view_cache['matrix'],view_features=features)
    controller.matcher=Mock()
    controller.matcher.follow_known.return_value=candidate
    monkeypatch.setattr(live.base,'refresh_game',lambda game:game)
    monkeypatch.setattr(live.base,'capture',Mock(side_effect=[raw1,raw2]))
    monkeypatch.setattr(live,'selected_floor',lambda frame:'B2')
    monkeypatch.setattr(live,'zoom_position',lambda frame:.75)
    monkeypatch.setattr(live,'signature',lambda frame:frame)
    monkeypatch.setattr(live,'same_view',lambda *args:False)
    monkeypatch.setattr(live,'ViewFeatures',lambda *args:features)
    controller.launch()
    kind,observed=controller.events.get_nowait()
    assert kind=='sample',observed
    assert 'handoff' not in observed  # No verified replacement pose.
    assert observed['stationary'] is True  # Existing display independently verified.
    assert probe.stationary.call_count==2
    assert probe.stationary.call_args.args[0] is raw2  # Before masking.
    assert observed['capture_frame'] is raw2
    controller.sample_received(observed)
    controller.overlay.hide.assert_not_called()


def test_display_diagnostics_skip_clear_closes_but_keep_open_map_failure(showing):
    controller,sample,_=showing
    controller.view_cache=dict(floor={'id':'s1_12_B2'})
    controller.display_recorder=Mock()
    controller.record_display_loss(dict(sample,selected_floor=None,presence={'visible':False}),'map_closed')
    controller.display_recorder.observe.assert_not_called()
    controller.record_display_loss(sample,'unconfirmed_motion')
    controller.display_recorder.observe.assert_called_once()

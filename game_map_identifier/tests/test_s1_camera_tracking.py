"""Regression checks for the once-per-round opening and camera evidence."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_round_lock import RoundLock
from src.s1_trigger_live import LiveController
from src.s1_view_motion import ViewFeatures


@pytest.mark.parametrize("first",["B1","B2","B3","B4"])
def test_only_first_readable_floor_is_a_random_landing(first):
    state=RoundLock()
    assert not state.observe_opening(None,0)
    assert state.observe_opening(first,1)
    cutoff=(state.b2_since,state.landing.since)
    for index,floor in enumerate((first,"B3","B2","B1","B3",first),2):
        assert not state.observe_opening(floor,index)
        assert (state.b2_since,state.landing.since)==cutoff
    assert state.opening_floor==first
    assert state.b2_arrival==(first=="B2")
    assert state.landing.waiting==(first=="B3")


def test_verified_opening_b2_can_immediately_handoff_to_b3():
    state=RoundLock()
    state.observe_opening("B2",1)
    for at in (2,3,4):
        state.observe(dict(at=at,unique_layout=True,floor=dict(floor="B2",layout_id="s1_05")))
    assert not state.observe_opening("B3",5)
    assert state.permits_overlay("B3","s1_05",["s1_05"])
    assert not state.permits_overlay("B3","s1_09",["s1_09"])


def test_matching_room_colour_does_not_hide_changed_texture():
    mask=np.full((48,80),255,np.uint8)
    original=np.zeros((48,80,3),np.uint8)
    original[12:30,24:50]=(25,85,140)
    changed=original.copy()
    changed[12:30,24:50]=(48,147,230)
    a=ViewFeatures(original,lambda _:mask.copy())
    b=ViewFeatures(changed,lambda _:mask.copy())
    # Same confirmed room outline, but the underlying screen has changed.
    for view in (a,b):
        view.geom[:]=0
        view.geom[12:30,24:50]=127
    assert a.same_pixels(a)
    assert not a.same_pixels(b)


def test_long_search_yields_to_newer_clean_camera_only():
    first=np.zeros((40,80),np.uint8)
    first[8:24,8:24]=255
    moved=np.roll(first,16,axis=1)
    capture=dict(at=1.,selected_floor="B2",presence={"visible":True},signature=first)
    current=dict(capture,at=2.,signature=moved)
    assert LiveController.camera_superseded(capture,current)
    assert not LiveController.camera_superseded(capture,dict(current,signature=first))
    assert not LiveController.camera_superseded(capture,dict(current,probing=True))
    assert not LiveController.camera_superseded(capture,dict(current,presence={"visible":False}))

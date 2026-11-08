"""Locked-floor motion is accepted only from consistent screen geometry."""
import sys
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_view_motion import ViewFeatures,zoom_allows_scale
from src.s1_original_map import OriginalMap


def rooms(seed):
    rng=np.random.default_rng(seed)
    im=np.full((300,520,3),(30,28,24),np.uint8)
    for _ in range(22):
        x,y=rng.integers([20,20],[420,225])
        w,h=rng.integers(16,54,2)
        texture=rng.integers(0,80,(h,w,1),dtype=np.uint8)
        im[y:y+h,x:x+w]=texture+np.uint8([22,70,108])
    return im


def features(image):
    return ViewFeatures(image,lambda im:np.full(im.shape[:2],255,np.uint8))


@pytest.mark.parametrize("scale,shift",[(1.,(54,-25)),(.73,(80,36))])
def test_wall_motion_recovers_known_camera(scale,shift):
    image=rooms(14)
    mat=np.float64([[scale,0,shift[0]],[0,scale,shift[1]]])
    moved=cv2.warpAffine(image,mat,image.shape[1::-1],borderValue=(30,28,24))
    a,b=features(image),features(moved)
    a.zoom=.45
    b.zoom=.45 if scale==1 else .58
    hit=a.camera_motion(b)
    assert hit is not None
    points=np.float64([[120,64,1],[410,195,1]]).T
    assert np.linalg.norm((hit[0]-mat)@points,axis=0).max()<2.


@pytest.mark.parametrize('before,after,ratio,allowed',[
    (.58631,.75528,4.2106236/4.2106256,False),
    (.58631,.75528,1.84194/4.21062,True),
    (.75528,.58631,1.00004,False),
    (.75528,.58631,4.21062/1.84194,True),
    (.586,.587,1.,True),
    (None,.755,1.,True),
])
def test_slider_direction_and_magnitude(before,after,ratio,allowed):
    assert bool(zoom_allows_scale(before,after,ratio)) is allowed


def test_transition_frame_cannot_stamp_old_scale_on_new_thumb():
    a=features(rooms(14))
    # The HUD has already moved while the rooms are still at the old scale.
    b=features(cv2.warpAffine(rooms(14),np.float64([[1,0,1],[0,1,2]]),(520,300)))
    a.zoom=.58631;b.zoom=.75528
    assert a.camera_motion(b) is None
    assert a.motion_to(b) is None


def test_blank_or_different_rooms_cannot_carry_the_old_camera():
    first=features(rooms(14))
    assert first.camera_motion(features(np.zeros((300,520,3),np.uint8))) is None
    assert first.camera_motion(features(rooms(939))) is None


def test_stationary_team_list_is_excluded_while_map_moves():
    image=rooms(14)
    target=cv2.warpAffine(image,np.float64([[1,0,33],[0,1,-41]]),
                          image.shape[1::-1],borderValue=(30,28,24))
    panel=np.random.default_rng(23).integers(0,255,(84,165,3),dtype=np.uint8)
    image[:84,:165]=panel
    target[:84,:165]=panel
    a,b=features(image),features(target)
    assert not a.valid[:84,:165].any()
    got=a.camera_motion(b)
    assert got is not None
    np.testing.assert_allclose(got[0][:,2],(33,-41),atol=.9)


def test_clipped_original_draws_only_the_requested_rectangle():
    original=OriginalMap.__new__(OriginalMap)
    original.source=Mock(return_value=(rooms(14),np.eye(2,3),'art'))
    floor={'id':'local'}
    hit=dict(floor=floor,matrix=np.float64([[.5,0,30.],[0,.5,20.]]))
    full=original.render(hit,(400,250))
    cropped=original.render(hit,(400,250),clip=(37,41,200,180))
    expected=np.zeros_like(full)
    expected[41:180,37:200]=full[41:180,37:200]
    np.testing.assert_array_equal(expected,cropped)

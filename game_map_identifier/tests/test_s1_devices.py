"""Only uniquely positioned fixtures can supply a single-symbol camera."""
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_devices import FixedDevices,device_points,glyph
from src.s1_live_match import map_area

def test_device_detector_ignores_legend_and_repeated_anvils():
    im=np.full((1377,2560,3),(73,83,106),np.uint8)
    icon=cv2.normalize(glyph('anvil',22),None,30,210,cv2.NORM_MINMAX).astype(np.uint8)
    def mark(x,y):im[y:y+22,x:x+22]=icon[:,:,None]
    mark(468,974)
    assert not device_points(im,map_area)
    mark(1460,704)
    assert 'anvil' in device_points(im,map_area)
    clear=np.full(im.shape[:2],255,np.uint8);clear[700:733,1456:1490]=0
    assert 'anvil' not in device_points(im,map_area,clear)
    mark(1590,802)
    assert 'anvil' not in device_points(im,map_area)


def test_device_observations_require_motion_and_reset_at_round_boundary():
    floor=dict(id='s1_08_B2',floor='B2',layout_id='s1_08')
    mat=np.eye(2,3)
    matcher=SimpleNamespace(catalog=SimpleNamespace(by_id={'hd_08_B2':{'sha256':'example'}}),
                            coordinates=SimpleNamespace(matrices={'s1_08_B2':mat}))
    devices=FixedDevices(matcher);labels=SimpleNamespace(points={})
    hit=dict(floor=floor,inliers=28,possible_layouts=['s1_08'],matrix=mat.copy())
    im=np.zeros((800,1200,3),np.uint8);epoch=devices.generation
    with patch('src.s1_devices.device_points',return_value={'anvil':(np.array([150.,200.]),.9)}):
        devices.observe(im,map_area,None,hit,epoch);devices.observe(im,map_area,None,hit,epoch)
    assert not devices.sites(floor,'anvil',labels)[1]
    hit['matrix'][0,2]=90
    with patch('src.s1_devices.device_points',return_value={'anvil':(np.array([240.,200.]),.9)}):
        devices.observe(im,map_area,None,hit,epoch)
    assert devices.sites(floor,'anvil',labels)[1]
    with patch('src.s1_devices.device_points',return_value={'anvil':(np.array([380.,200.]),.9)}):
        devices.observe(im,map_area,None,hit,epoch)
    points,exact=devices.sites(floor,'anvil',labels)
    assert len(points)==2 and not exact
    devices.reset();devices.observe(im,map_area,None,hit,epoch)
    assert devices.sites(floor,'anvil',labels)==([],False)

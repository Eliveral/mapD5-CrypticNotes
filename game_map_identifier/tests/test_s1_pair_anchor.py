"""A single shared room can register a verified pair in either direction."""
from pathlib import Path
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_floor_transition import FloorCoordinates
from src.s1_handoff import FloorHandoff
from src.s1_live_match import LiveMatcher


def paired_matcher():
    match=LiveMatcher.__new__(LiveMatcher)
    coords=FloorCoordinates.__new__(FloorCoordinates)
    coords.matrices={'test_B2':np.float64([[1,0,8],[0,1,14]]),
                     'test_B3':np.float64([[.8,0,22],[0,.8,35]])}
    a,b=np.array([75.,65.]),np.array([92.,56.])
    coords.offsets={('test','B2','B3'):b-a,('test','B3','B2'):a-b}
    coords.passages={('test','B2','B3'):[('维修站',a,b)],('test','B3','B2'):[('维修站',b,a)]}
    floors={};images={}
    for level,start in [('B2',a),('B3',b)]:
        fid='test_'+level
        floors[fid]=dict(id=fid,layout_id='test',floor=level)
        cx,cy=np.rint(coords.matrices[fid]@[*start,1]).astype(int)
        g=np.zeros((180,220),np.uint8)
        cv2.rectangle(g,(cx-28,cy-20),(cx+28,cy+20),127,-1)
        images[fid]=g
    match.template=lambda floor:(None,images[floor['id']])
    hands=FloorHandoff.__new__(FloorHandoff)
    hands.coords=match.coordinates=coords;hands.floors=floors
    hands.matcher=match;hands.sites={};match.handoffs=hands
    match.corridors=Mock();match.corridors.locate.return_value=None
    return match


@pytest.mark.parametrize('level,target',[('B2','B3'),('B3','B2')])
def test_current_marker_corrects_a_recentered_single_room(level,target):
    matcher=paired_matcher();h=matcher.handoffs;c=matcher.coordinates
    lf,tf=h.floors['test_'+level],h.floors['test_'+target]
    _,start,end=c.passages['test',level,target][0]
    player=start+np.array([6.,-3.])
    source=dict(floor=lf,matrix=np.float64([[1.8,0,75],[0,1.8,50]]),unique_layout=True,player_hd=player,zoom=.5)
    truth,_=c.predict(source,target)
    marker=np.array([560,275])
    truth[:,2]=marker-truth[:,:2]@(c.matrices[tf['id']]@[* (player+end-start),1])
    mask=cv2.warpAffine(matcher.template(tf)[1],truth,(960,540),flags=cv2.INTER_NEAREST)
    im=np.full((540,960,3),(32,27,23),np.uint8);im[mask>0]=(47,82,131)
    cv2.circle(im,tuple(marker),7,(0,223,255),-1)
    hit=matcher.paired_floor(im,source,target,'test',zoom=.5)
    assert hit and hit['search']=='paired_player'
    np.testing.assert_allclose(hit['matrix'],truth,atol=1.3)
    # An unverified source, or a blank target, cannot authorize a projection.
    assert matcher.paired_floor(im,dict(source,unique_layout=False),target,'test') is None
    assert matcher.paired_floor(np.zeros_like(im),source,target,'test') is None
    # A remembered target view can work after the player icon disappears.
    cv2.circle(im,tuple(marker),8,(47,82,131),-1)
    back=dict(floor=tf,matrix=truth,player_hd=player+end-start,zoom=.6)
    hit=matcher.paired_floor(im,source,target,'test',return_views=[back],zoom=.6)
    assert hit and hit['search']=='paired_return'
    np.testing.assert_array_equal(hit['matrix'],truth)


def test_repeated_ordinary_room_needs_current_marker_or_registered_view():
    m=paired_matcher();c=m.coordinates
    source,target=m.handoffs.floors['test_B2'],m.handoffs.floors['test_B3']
    player=np.array([137.,102.])
    hd=player+c.offsets['test','B2','B3']
    xy=np.rint(c.matrices['test_B3']@[*hd,1]).astype(int)
    ref=m.template(target)[1]
    for pt in (xy,np.array([178,54])):
        cv2.rectangle(ref,tuple(pt-[13,11]),tuple(pt+[13,11]),127,-1)
    source_point=np.rint(c.matrices['test_B2']@[*player,1]).astype(int)
    cv2.rectangle(m.template(source)[1],tuple(source_point-12),tuple(source_point+12),127,-1)
    previous=dict(floor=source,matrix=np.float64([[1.8,0,75],[0,1.8,50]]),
                  unique_layout=True,player_hd=player,zoom=.5)
    truth=c.predict(previous,'B3')[0]
    marker=np.array([560,275])
    truth[:,2]=marker-truth[:,:2]@(c.matrices['test_B3']@[*hd,1])
    room=m.handoffs.site(target,hd)[0]
    visible=cv2.warpAffine(room,truth,(960,540),flags=cv2.INTER_NEAREST)>0
    im=np.full((540,960,3),(32,27,23),np.uint8);im[visible]=(47,82,131)
    assert m.paired_floor(im,previous,'B3','test',zoom=.5) is None
    cv2.circle(im,tuple(marker),7,(0,223,255),-1)
    got=m.paired_floor(im,previous,'B3','test',zoom=.5)
    assert got and got['search']=='paired_room'
    np.testing.assert_allclose(got['matrix'],truth,atol=1.5)

"""Share one registered screen camera across calibrated floor images."""
from dataclasses import dataclass
from threading import RLock
import time

import cv2
import numpy as np

from .s1_alignment import geometry
from .s1_floor_buttons import selected_button
from .s1_view_motion import exclude_map_hud, zoom_changed


@dataclass(frozen=True)
class PointerSnapshot:
    epoch: int
    pan: tuple
    click_at: float
    click_y: float


class MapPointer:
    """Read only map gestures in the selected game window. No input is sent."""
    def __init__(self):
        self.lock=RLock()
        self.target=None
        self.epoch=0
        self.pan=(0.,0.)
        self.click_at=-1.
        self.click_y=-1.
        self.drag=None

    def configure(self,target):
        with self.lock:
            if target!=self.target:
                self.target=target
                self.epoch+=1
                self.drag=None
                self.click_at=-1.

    def snapshot(self):
        with self.lock:
            if self.target is None:return None
            return PointerSnapshot(self.epoch,self.pan,self.click_at,self.click_y)

    def observe(self,message,point,hwnd):
        with self.lock:
            if not self.target or hwnd!=self.target[0]:return
            _,(left,top,w,h)=self.target
            x,y=point[0]-left,point[1]-top
            if message==0x201:
                self.drag=None
                if abs(x-(w-.22*h))<.032*h and .125*h<y<.478*h:
                    self.click_at=time.monotonic()
                    self.click_y=y
                elif .17*w<x<.86*w and .125*h<y<.83*h and not (x<.35*w and y>.56*h):
                    self.drag=(x,y)
                else:
                    self.epoch+=1
            elif message==0x200 and self.drag is not None:
                dx,dy=x-self.drag[0],y-self.drag[1]
                self.pan=(self.pan[0]+dx,self.pan[1]+dy)
                self.drag=(x,y)
            elif message==0x202:
                self.drag=None
            elif message in (0x20A,0x20E,0x204,0x207,0x20B):
                self.epoch+=1
                self.drag=None


def shared_floor(matcher,image,area,visit,target,clear=None,zoom=None,pointer=None,continuous=False):
    if not visit or not visit.get('unique_layout') or visit.get('provisional'):return None
    source=visit['floor']
    if target not in ('B1','B2','B3'):return None
    before=visit.get('zoom')
    if before is None or zoom is None or zoom_changed(before,zoom):return None
    floor=matcher.handoffs.floors.get(source['layout_id']+'_'+target)
    matrix,_=matcher.coordinates.predict(visit,target)
    if floor is None or matrix is None:return None
    previous=visit.get('pointer')
    same_input=continuous and previous is not None and pointer is not None and previous.epoch==pointer.epoch
    if same_input:
        matrix=matrix.copy()
        matrix[:,2]+=np.asarray(pointer.pan)-previous.pan
    button=selected_button(image) if same_input else None
    clicked=(button and button[0]==target and abs(button[2]-pointer.click_y)<image.shape[0]*.042
             and pointer.click_at>visit.get('sample_at',visit['at']))
    continuing=same_input and source['floor']==target and visit.get('shared_camera')
    if not (clicked or continuing):return None

    factor=min(1.,800/image.shape[1])
    small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    valid=exclude_map_hud(area(small))
    if clear is not None:
        visible=cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)
        valid &= np.uint8(visible>=252)*255
    observed=geometry(small)&valid
    counts=[np.count_nonzero(observed==kind) for kind in (127,255)]
    expected=cv2.warpAffine(matcher.template(floor)[1],matrix*factor,small.shape[1::-1],flags=cv2.INTER_NEAREST)
    rates=[]
    for kind,count in zip((127,255),counts):
        if count<24:continue
        near=cv2.dilate(np.uint8(expected==kind),np.ones((5,5),np.uint8))
        rates.append(float(near[observed==kind].mean()))
    if rates and min(rates)<.87:return None
    hit=matcher.handoffs.result(floor,matrix,'三层共用坐标',min(rates,default=1.),False)
    hit.update(search='shared_camera',shared_camera=True,camera_only=not rates,pointer=pointer)
    return hit

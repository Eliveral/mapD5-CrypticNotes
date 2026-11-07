"""Check a visible overlay without periodically blanking the screen.

Calculated pixels only establish stationarity. Registrations use captured
pixels in regions untouched by the overlay, or a full clean capture.
"""
import cv2
import numpy as np
from copy import copy
from .s1_alignment import geometry
from .s1_live_match import map_area


class LayerProbe:
    def __init__(self, clean, rendered, signature, label_tiles=(),known_pixels=None):
        self.size = (signature.shape[1], signature.shape[0])
        self.clean = cv2.resize(clean,self.size).astype(np.float32)
        opaque = cv2.bitwise_not(cv2.inRange(rendered,(0,0,0),(16,16,16)))
        self.unpainted=cv2.bitwise_not(opaque)
        self.coverage = (cv2.resize(opaque,self.size).astype(np.float32)/255)[:,:,None]
        self.ink = cv2.resize(cv2.bitwise_and(rendered,rendered,mask=opaque),self.size).astype(np.float32)
        # Compare pixels near observed structures; dark, animated fog occupies
        # much of the rest of the window and does not locate the map.
        self.label_tiles=label_tiles
        self.label_area=np.zeros(signature.shape,np.uint8)
        for x,y,mask in label_tiles:
            h,w=mask.shape
            sx,sy=self.size[0]/clean.shape[1],self.size[1]/clean.shape[0]
            cv2.rectangle(self.label_area,(int(x*sx)-1,int(y*sy)-1),
                          (int((x+w)*sx)+1,int((y+h)*sy)+1),255,-1)
        self.area=(cv2.dilate(signature,np.ones((13,13),np.uint8))>0) & (self.label_area==0)
        self.rooms=signature>0
        self.known=(cv2.resize(known_pixels,self.size,interpolation=cv2.INTER_AREA)>=252) if known_pixels is not None else np.ones(signature.shape,bool)
        self.area &= self.known
        self.update_edges()

    def update_edges(self):
        g=geometry(self.clean.astype(np.uint8))
        self.edge_area=(self.coverage[:,:,0]<.02)&(self.label_area==0)&self.area
        self.walls=(cv2.morphologyEx(g,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0)&self.edge_area

    def uncovered_frame(self,frame):
        # Use actual screen pixels only. Anything painted by the previous
        # layer is excluded while we prepare its replacement, without blinking it off.
        return cv2.bitwise_and(frame,frame,mask=self.unpainted)

    def with_reference(self, clean, signature):
        # The same rendered map has the same ink and coverage. On a verified
        # reopen only the tiny clean reference changes, never the full-size mask.
        result = copy(self)
        result.clean = cv2.resize(clean,self.size).astype(np.float32)
        result.area = (cv2.dilate(signature,np.ones((13,13),np.uint8)) > 0) & (self.label_area==0)
        result.rooms=signature>0
        result.known=np.ones(signature.shape,bool)
        result.update_edges()
        return result

    def stationary(self, frame, alpha):
        if np.count_nonzero(self.area) < 60:
            return False
        raw = cv2.resize(frame,self.size).astype(np.float32)
        hsv=cv2.cvtColor(raw.astype(np.uint8),cv2.COLOR_BGR2HSV)
        rooms=cv2.inRange(hsv,(4,25,45),(33,200,255))>0
        clear=self.known&(self.coverage[:,:,0]<.02)&(self.label_area==0)&(map_area(hsv)>0)
        if np.count_nonzero(rooms & ~self.rooms & clear)>max(40,self.rooms.sum()*.055):
            return False
        a = round(max(.15,min(.90,alpha))*255)/255
        transmitted = 1-self.coverage*a
        restored = np.clip((raw-self.ink*a)/transmitted,0,255)
        for sample in (raw, restored):
            delta = np.max(np.abs(sample-self.clean),axis=2)[self.area]
            if np.mean(delta) < 5 and np.mean(delta > 16) < .06:
                return True
        # Animated fog, highlights and the player icon may change RGB without
        # moving the room walls. Check structure in untouched pixels separately.
        if self.walls.sum()<70:return False
        current=geometry(raw.astype(np.uint8))
        edges=(cv2.morphologyEx(current,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0)&self.edge_area
        if edges.sum()<70:return False
        kernel=np.ones((3,3),np.uint8)
        agreement=(cv2.dilate(np.uint8(edges),kernel)[self.walls].mean(),
                   cv2.dilate(np.uint8(self.walls),kernel)[edges].mean())
        return min(agreement)>.92

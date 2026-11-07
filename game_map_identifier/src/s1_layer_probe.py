"""Check a visible overlay without periodically blanking the screen.

The calculated pixels are used only to test if the view moved. New registrations
always require an actual clean frame after removing an outdated projection.
"""
import cv2
import numpy as np
from copy import copy


class LayerProbe:
    def __init__(self, clean, rendered, signature, label_tiles=()):
        self.size = (signature.shape[1], signature.shape[0])
        self.clean = cv2.resize(clean,self.size).astype(np.float32)
        opaque = cv2.bitwise_not(cv2.inRange(rendered,(0,0,0),(16,16,16)))
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

    def with_reference(self, clean, signature):
        # The same rendered map has the same ink and coverage. On a verified
        # reopen only the tiny clean reference changes, never the full-size mask.
        result = copy(self)
        result.clean = cv2.resize(clean,self.size).astype(np.float32)
        result.area = (cv2.dilate(signature,np.ones((13,13),np.uint8)) > 0) & (self.label_area==0)
        return result

    def stationary(self, frame, alpha):
        if np.count_nonzero(self.area) < 60:
            return False
        raw = cv2.resize(frame,self.size).astype(np.float32)
        a = round(max(.15,min(.90,alpha))*255)/255
        transmitted = 1-self.coverage*a
        restored = np.clip((raw-self.ink*a)/transmitted,0,255)
        for sample in (raw, restored):
            delta = np.max(np.abs(sample-self.clean),axis=2)[self.area]
            if np.mean(delta) < 5 and np.mean(delta > 16) < .06:
                return True
        return False

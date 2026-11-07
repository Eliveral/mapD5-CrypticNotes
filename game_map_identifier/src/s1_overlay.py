"""One native window with separate opacity for floor art and room lettering."""
import cv2
import numpy as np
from .gui import RouteMapOverlayWindow


def room_surface(raw, tiles, opacity):
    # UpdateLayeredWindow expects premultiplied BGRA with constant alpha 255.
    surface=cv2.convertScaleAbs(raw,alpha=opacity)
    valid=cv2.extractChannel(raw,3)
    surface=cv2.bitwise_and(surface,surface,mask=valid)
    for x,y,mask in tiles:
        h,w=mask.shape
        region=np.s_[y:y+h,x:x+w]
        source=raw[region]
        strength=opacity+(max(.86,opacity)-opacity)*(mask.astype(np.float32)/255)
        surface[region]=np.rint(source*strength[:,:,None]).astype(np.uint8)
        surface[region][source[:,:,3]==0]=0
    return surface


class S1Overlay(RouteMapOverlayWindow):
    label_tiles=()
    _room_surface=None

    def _submit_native_bgra(self, bgra, screen_roi):
        if not self.label_tiles:
            self._room_surface=None
            return super()._submit_native_bgra(bgra,screen_roi)
        cached=self._room_surface
        if cached is None or cached[0] is not bgra or cached[1] is not self.label_tiles or cached[2]!=self.alpha:
            cached=bgra,self.label_tiles,self.alpha,room_surface(bgra,self.label_tiles,self.alpha)
            self._room_surface=cached
        super()._submit_native_bgra(cached[3],screen_roi,constant_alpha=1.)

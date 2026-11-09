"""Locate the shared B1/B2 elevator, then compare the visible B2 neighbourhood."""
from functools import lru_cache
import cv2
import numpy as np

from .s1_alignment import geometry
from .s1_catalog import read_image
from .s1_view_motion import exclude_map_hud,zoom_changed


def cameras_agree(first,independent,size):
    """Compare projected positions across the current visible map area."""
    w,h=size
    points=np.float64([[w*.42,h*.35,1],[w*.68,h*.68,1],[w*.55,h*.5,1]]).T
    native=np.linalg.inv(np.vstack((independent,[0,0,1])))@points
    return bool(np.max(np.linalg.norm((first-independent)@native,axis=0))<=max(6.,h*.009))


class ElevatorMatcher:
    grid=.25

    def __init__(self,matcher):
        self.matcher=matcher

    @lru_cache(maxsize=15)
    def reference(self,layout):
        m=self.matcher
        links=m.coordinates.passages.get((layout,'B1','B2'),())
        if len(links)!=1:return None
        floor=m.handoffs.floors[layout+'_B2']
        center=links[0][2]
        art=read_image(m.catalog.path(m.catalog.by_id['hd_'+layout[3:]+'_B2']))
        x,y=np.rint(center).astype(int)
        seed=cv2.cvtColor(art[y-78:y+78,x-78:x+78],cv2.COLOR_BGR2GRAY)
        if seed.shape!=(156,156):return None
        small=cv2.resize(art,None,fx=self.grid,fy=self.grid,interpolation=cv2.INTER_AREA)
        hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        room=cv2.inRange(hsv,(3,16,33),(38,255,255))>0
        hall=cv2.inRange(hsv,(91,15,33),(141,255,255))>0
        fields=np.zeros(room.shape,np.uint16)
        for j,field in enumerate((room,hall,*m.pose_check.sides(room),*m.pose_check.sides(hall))):
            fields |= np.uint16(cv2.dilate(np.uint8(field),np.ones((5,5),np.uint8)))<<j
        return floor,center,seed,fields,cv2.cvtColor(art[y-101:y+101,x-101:x+101],cv2.COLOR_BGR2GRAY)

    @staticmethod
    def find_room(image,valid,seed,scales):
        factor=min(1.,800/image.shape[1])
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        view=cv2.resize(gray,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        clean=cv2.resize(valid,view.shape[1::-1],interpolation=cv2.INTER_AREA)==255
        # Search only visible brown room pixels, excluding the HUD and overlay.
        small=cv2.resize(image,view.shape[1::-1],interpolation=cv2.INTER_AREA)
        ink=(geometry(small)==127)&clean
        if not ink.any():return None
        x,y,w,h=cv2.boundingRect(np.uint8(ink));pad=round(max(scales)*80*factor)+2
        x,y,x2,y2=max(0,x-pad),max(0,y-pad),min(view.shape[1],x+w+pad),min(view.shape[0],y+h+pad)
        roi=view[y:y2,x:x2];candidates=[]
        for size in sorted({round(156*z*factor) for z in scales}):
            if size<10 or size>=min(roi.shape):continue
            tile=cv2.resize(seed,(size,size),interpolation=cv2.INTER_AREA)
            scores=cv2.matchTemplate(roi,tile,cv2.TM_CCOEFF_NORMED)
            ok=cv2.erode(np.uint8(clean[y:y2,x:x2]),np.ones((size,size),np.uint8))
            half=size//2
            scores[~((ok>0)&ink[y:y2,x:x2])[half:half+scores.shape[0],half:half+scores.shape[1]]]=-1
            for _ in range(2):
                _,v,_,(u,t)=cv2.minMaxLoc(scores)
                candidates.append((v,np.array([x+u+size*.5,y+t+size*.5])/factor,size/(156*factor)))
                cv2.rectangle(scores,(u-size//2,t-size//2),(u+size//2,t+size//2),-1,-1)
        if not candidates:return None
        score,center,scale=max(candidates,key=lambda row:row[0])
        rival=max((v for v,p,_ in candidates if np.linalg.norm(p-center)>max(12.,scale*60)),default=0.)
        if score<.82 or score-rival<.075:return None
        # Refine the measured centre and scale on original pixels.
        pad=round(scale*15)+3;l,t=np.maximum(0,np.floor(center-90*scale-pad).astype(int))
        rr,bb=np.minimum(gray.shape[::-1],np.ceil(center+90*scale+pad).astype(int))
        crop=gray[t:bb,l:rr];best=None
        for n in sorted({round(156*scale*z) for z in np.linspace(.87,1.14,15)}):
            if n<10 or n>min(crop.shape):continue
            tile=cv2.resize(seed,(n,n),interpolation=cv2.INTER_AREA if n<156 else cv2.INTER_LINEAR)
            _,v,_,(u,y0)=cv2.minMaxLoc(cv2.matchTemplate(crop,tile,cv2.TM_CCOEFF_NORMED))
            if not np.all(valid[t+y0:t+y0+n,l+u:l+u+n]):continue
            if best is None or v>best[0]:best=(v,np.array([l+u+n*.5,t+y0+n*.5]),n/156.)
        return best if best and best[0]>=.81 else None

    @staticmethod
    def fit_outer_walls(image,valid,tile,pixel,measured):
        # The wider crop includes the fixed side walls. Their separation
        # measures zoom more accurately than the texture in the lift interior.
        if tile.shape!=(202,202):return None
        x,y=np.rint(pixel).astype(int);radius=round(115*measured)+5
        l,t=max(0,x-radius),max(0,y-radius)
        crop=cv2.cvtColor(image[t:y+radius+1,l:x+radius+1],cv2.COLOR_BGR2GRAY)
        best=None
        for size in range(round(202*measured*.93),round(202*measured*1.08)+1):
            if size<10 or size>min(crop.shape):continue
            ref=cv2.resize(tile,(size,size),interpolation=cv2.INTER_AREA)
            _,value,_,(u,v)=cv2.minMaxLoc(cv2.matchTemplate(crop,ref,cv2.TM_CCOEFF_NORMED))
            center=np.array([l+u+size*.5,t+v+size*.5])
            if np.linalg.norm(center-pixel)>max(4.,measured*8) or not np.all(valid[t+v:t+v+size,l+u:l+u+size]):continue
            if best is None or value>best[0]:best=value,center,size/202.
        return best if best and best[0]>=.87 else None

    def evidence(self,image,valid,pixel):
        factor=min(1.,840/image.shape[1])
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        clean=cv2.resize(valid,small.shape[1::-1],interpolation=cv2.INTER_AREA)>=252
        clean=cv2.erode(np.uint8(clean),np.ones((5,5),np.uint8))>0
        geom=geometry(small)
        sharp=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),24,65)
        sharp=(cv2.dilate(sharp,np.ones((3,3),np.uint8))>0)&clean
        groups=[]
        for value in (127,255):
            pos=np.column_stack(np.nonzero((geom==value)&clean))[:,::-1]
            groups.append(pos[::max(1,len(pos)//480)]/factor-pixel)
        for value in (127,255):
            for i,side in enumerate(self.matcher.pose_check.sides(geom==value)):
                kernel=np.ones((6,1) if i<2 else (1,6),np.uint8)
                lines=cv2.morphologyEx(np.uint8(side&sharp),cv2.MORPH_OPEN,kernel)
                pos=np.column_stack(np.nonzero(lines))[:,::-1]
                groups.append(pos[::max(1,len(pos)//120)]/factor-pixel)
        return groups

    def score(self,ref,groups,scale):
        _,center,_,fields,_=ref;rates=[];walls=[]
        for i,p in enumerate(groups):
            if len(p)<8:continue
            x,y=np.rint((center+p/scale)*self.grid).astype(int).T
            ok=(x>=0)&(x<fields.shape[1])&(y>=0)&(y<fields.shape[0])
            rate=np.count_nonzero(fields[y[ok],x[ok]]&(1<<i))/len(p)
            (rates if i<2 else walls).append(rate)
        if not rates or not walls:return 0.,0.
        return .65*np.mean(walls)+.35*np.mean(rates),min(rates)

    def locate(self,image,area,source,clear=None,zoom=None):
        if (not source or source['floor']['floor']!='B1' or not source.get('unique_layout')
                or source.get('provisional') or source.get('alternatives')):return None
        layout=source['floor']['layout_id'];m=self.matcher
        ref=self.reference(layout)
        if ref is None:return None
        valid=exclude_map_hud(area(image))
        if clear is not None:valid &= np.uint8(clear>=252)*255
        pred,_=m.coordinates.predict(source,'B2')
        if pred is None:return None
        reg=np.asarray(m.coordinates.matrices[ref[0]['id']])
        hint=np.linalg.norm((pred@np.vstack((reg,[0,0,1])))[0,:2])
        unchanged=zoom is not None and source.get('zoom') is not None and not zoom_changed(source['zoom'],zoom)
        scales=np.geomspace(.2,2.65,21)*image.shape[0]/1377.
        found=self.find_room(image,valid,ref[2],hint*np.array([.96,1.,1.04])) if unchanged else None
        if found is None:found=self.find_room(image,valid,ref[2],scales)
        if found is None:return None
        texture,pixel,measured=found
        groups=self.evidence(image,valid,pixel)
        island=all(not len(p) or np.max(np.abs(p))<=150*measured for p in groups)
        precise=None if island else self.fit_outer_walls(image,valid,ref[4],pixel,measured)
        if precise:
            _,pixel,scale=precise
            groups=self.evidence(image,valid,pixel)
            strength,coverage=self.score(ref,groups,scale)
        else:
            options=[measured]
            if not island:options.extend(measured*z for z in np.linspace(.96,1.04,7))
            (strength,coverage),scale=max(((self.score(ref,groups,z),z) for z in options),key=lambda row:row[0][0])
        if unchanged and abs(hint/scale-1)<.025 and self.score(ref,groups,hint)[0]>=strength-.023:
            scale=hint;strength,coverage=self.score(ref,groups,scale)
        if strength<.68 or coverage<.90:return None
        rivals=[]
        # Every calibrated layout contains this same elevator. Its distinctive
        # texture fixes position, but cannot identify the round by itself.
        for other in m.floors:
            if island:break
            if other['floor']!='B2' or other['layout_id']==layout:continue
            alt=self.reference(other['layout_id'])
            if alt is None:return None
            rivals.append(max(self.score(alt,groups,scale*z)[0] for z in (.96,1.,1.04)))
        rival=max(rivals,default=1.)
        if not island and rival>strength+.055:return None
        native=np.float64([[scale,0,pixel[0]-ref[1][0]*scale],[0,scale,pixel[1]-ref[1][1]*scale]])
        mat=native@np.linalg.inv(np.vstack((reg,[0,0,1])))
        result=m.handoffs.result(ref[0],mat,'升降梯',float(strength),False)
        beyond=sum(np.count_nonzero(np.max(np.abs(p),axis=1)>180*scale) for p in groups[2:] if len(p))
        result.update(search='b1_b2_elevator',segmentation='elevator',reason='已对齐负二升降梯',rival=float(rival),
                      elevator_verified=bool(not island and strength>.70 and strength-rival>=.10 and beyond>=18),
                      texture=float(texture))
        return result

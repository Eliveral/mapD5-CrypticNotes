"""Use the registered repair station and its doorway when changing to B3."""
import cv2
import numpy as np
from .s1_alignment import geometry
from .s1_entrance import turns


class StationMatcher:
    def __init__(self,matcher):
        self.matcher=matcher
        self.refs={}

    def reference(self,floor,anchor):
        key=floor['id']
        if key not in self.refs:
            ref=self.matcher.template(floor)[1]
            rooms=np.uint8(ref==127)
            _,labels=cv2.connectedComponents(rooms)
            x,y=np.rint(anchor).astype(int)
            if not (0<=x<ref.shape[1] and 0<=y<ref.shape[0]) or not labels[y,x]:return None
            station=labels==labels[y,x]
            wall=cv2.morphologyEx(ref,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
            self.refs[key]=(rooms.astype(np.float32),np.float32(ref==255),np.float32(wall>0),
                            station,*turns(ref,4))
        return self.refs[key]

    def locate(self,image,area,floor,scale,anchor):
        """Return a verified pose, or whether a single corner still lacks evidence."""
        if not .8<=scale<=6:return None,False
        factor=min(1.,1000/image.shape[1])
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        geom=geometry(small)&area(small)
        _,_,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        if np.sum(stats[1:,4]>80)!=1:return None,False
        _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom>0))
        main=1+int(np.argmax(stats[1:,4]));geom[labels!=main]=0
        x,y,w,h=cv2.boundingRect(geom);crop=geom[y:y+h,x:x+w]
        if min(w,h)<12:return None,True
        ref=self.reference(floor,anchor)
        if ref is None:return None,False
        edge=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),32,76)
        border=cv2.morphologyEx(geom,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
        sharp=(edge[y:y+h,x:x+w]>0)&(border[y:y+h,x:x+w]>0)
        observed=[np.float32(q) for q in (crop==127,crop==255,sharp)]
        if observed[1].sum()<max(35,75*(scale*factor)**2):return None,True
        pt,tag=turns(geom,max(3.,scale*factor*3.5));base,bt=ref[4:]
        rows=[]
        for s in (scale,scale*.97,scale*1.03,scale*.94,scale*1.06):
            self.matcher.check_search()
            size=round(w/(s*factor)),round(h/(s*factor))
            if min(size)<8 or size[0]>=ref[0].shape[1] or size[1]>=ref[0].shape[0]:continue
            fields=[cv2.resize(q,size,interpolation=cv2.INTER_AREA) for q in observed]
            if any(q.sum()<25 for q in fields):continue
            rates=[cv2.matchTemplate(r,q,cv2.TM_CCORR)/q.sum() for r,q in zip(ref,fields)]
            joint=.26*rates[0]+.24*rates[1]+.5*rates[2]
            for _ in range(2):
                _,score,_,(px,py)=cv2.minMaxLoc(joint)
                center=np.array([px+size[0]/2,py+size[1]/2])
                rows.append((score,center,s,px,py,[float(r[py,px]) for r in rates],fields[0]))
                joint[max(0,py-14):py+15,max(0,px-14):px+15]=-1
        if not rows:return None,False
        rows.sort(key=lambda r:-r[0]);best,center,s,px,py,(rr,rh,rw),room=rows[0]
        rival=max((row[0] for row in rows if np.linalg.norm(row[1]-center)>15),default=0.)
        corners=0
        if len(pt) and len(base):
            reference_points=(pt-[x,y])/(s*factor)+[px,py]
            delta=reference_points[:,None,:]-base[None,:,:]
            dist=np.sum(delta*delta,axis=2);dist[tag[:,None]!=bt]=1e8
            pick=dist.argmin(axis=1);ok=dist.min(axis=1)<=4**2
            corners=len(set(pick[ok]))
        if not (best-rival>=.10 and rr>=.94 and rh>=.80 and rw>=.50 and corners>=4):
            return None,observed[1].sum()<550*(scale*factor)**2 and corners<3
        station=ref[3][py:py+room.shape[0],px:px+room.shape[1]]
        if (station*room).sum()<.94*room.sum():return None,False
        mat=np.array([[s,0,x/factor-px*s],[0,s,y/factor-py*s]])
        return dict(accepted=True,floor=floor,matrix=mat,scale=s,inliers=corners,score=best,
                    consistency=min(rr,rh),rival=rival,reason='已核对维修站出口',error_px=None,
                    segmentation='station_doorway',search='known_station',possible_layouts=[floor['layout_id']],
                    candidates=[]),False

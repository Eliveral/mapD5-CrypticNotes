"""Carry a registered map camera through a shared passage on the first target frame."""
import cv2
import numpy as np
from .s1_alignment import geometry
from .s1_floor_transition import FloorCoordinates,selected_floor


class FloorHandoff:
    def __init__(self,matcher):
        self.matcher=matcher
        self.coords=FloorCoordinates(matcher.catalog)
        self.floors={f['id']:f for f in matcher.floors}
        self.sites={}

    def site(self,floor,point):
        key=(floor['id'],tuple(point))
        if key not in self.sites:
            ref=self.matcher.template(floor)[1]
            pos=np.rint(np.asarray(self.coords.matrices[floor['id']])@[*point,1]).astype(int)
            if not (0<=pos[0]<ref.shape[1] and 0<=pos[1]<ref.shape[0]):return None
            _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(ref==127))
            index=labels[pos[1],pos[0]]
            if not index:return None
            room=np.uint8(labels==index)
            self.sites[key]=(room,stats[index,:4],pos)
        return self.sites[key]

    def inherit(self,image,visit,target,area):
        source=visit['floor'];layout=source['layout_id'];level=source['floor']
        sites=self.coords.passages.get((layout,level,target),[])
        if not sites:return None
        pred,_=self.coords.predict(visit,target)
        if pred is None:return None
        floor=self.floors[layout+'_'+target]
        factor=min(1.,800/image.shape[1]);small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        geom=geometry(small)&area(small);ref=self.matcher.template(floor)[1]
        ys,xs=np.nonzero(geom[::2,::2]);xy=np.stack((xs*2,ys*2),axis=1)
        observed=geom[ys*2,xs*2]
        if len(xy)<18:
            return self.result(floor,pred,'继承视角',0.,True) if selected_floor(image)==target else None
        check=cv2.invertAffineTransform(pred*factor)
        rx,ry=np.rint(xy@check[:,:2].T+check[:,2]).astype(int).T
        valid=(rx>=0)&(ry>=0)&(rx<ref.shape[1])&(ry<ref.shape[0])
        samples=np.zeros(len(xy),np.uint8);samples[valid]=ref[ry[valid],rx[valid]]
        parts=[float(np.mean(samples[observed==v]==v)) for v in (127,255) if np.sum(observed==v)>=15]
        edges=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),30,72)
        border=cv2.morphologyEx(geom,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
        ey,ex=np.nonzero((edges>0)&(border>0))
        if len(ex)>=10:
            ex,ey=np.rint(np.stack((ex,ey),axis=1)@check[:,:2].T+check[:,2]).astype(int).T
            valid=(ex>=0)&(ey>=0)&(ex<ref.shape[1])&(ey<ref.shape[0])
            outlines=cv2.morphologyEx(ref,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
            parts.append(float(np.count_nonzero(outlines[ey[valid],ex[valid]]))/len(ex))
        if parts and min(parts)>.84:return self.result(floor,pred,'重叠视角',min(parts),False)
        # A descent can recenter the camera. Only search the paired passage's
        # room, at the scale already known from the previous floor.
        _,ids,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        if len(stats)<2:return None
        main=1+int(np.argmax(stats[1:,4]));qx,qy,w,h=stats[main,:4]
        if stats[main,4]<70 or np.sum(stats[1:,4]>max(80,stats[main,4]*.12))>1:return None
        obs=np.uint8(ids==main)*255
        sign_edges=self.straight_walls(obs,edges)
        active=[i for i,c in enumerate(sign_edges) if np.any(c)]
        if len(active)!=2:return None
        scale=float(np.linalg.norm(pred[0,:2]));s=scale*factor
        height,width=geom.shape;left,top=max(0,qx-3),max(0,qy-3)
        right,bottom=min(width,qx+w+3),min(height,qy+h+3)
        box=np.s_[top:bottom,left:right];size=(round((right-left)/s),round((bottom-top)/s))
        if min(size)<5:return None
        fields=[cv2.resize(c[box].astype(np.float32),size,interpolation=cv2.INTER_AREA) for c in [obs,*sign_edges]]
        choices=[]
        player=visit.get('player_hd')
        for name,start,end in sites:
            site=self.site(floor,end)
            if site is None:continue
            room,(x,y,ww,hh),center=site
            # Limit the target site to the actual shared room. Padding lets
            # observed wall edges land on the boundary of the known room.
            l,t=max(0,x-5),max(0,y-5)
            rr=room[t:y+hh+5,l:x+ww+5]
            if rr.shape[1]<size[0] or rr.shape[0]<size[1]:continue
            sides=self.sides(rr*255)
            channels=[rr,*[cv2.dilate(c,np.ones((3,3),np.uint8)) for c in sides]]
            def correlation(index):
                q=fields[index]
                return cv2.matchTemplate(channels[index].astype(np.float32),q,cv2.TM_CCORR)/max(float(q.sum()),1.)
            occupancy=correlation(0)
            rates=[correlation(i+1) for i in active]
            score=occupancy*.32+sum(rates)*(.68/len(rates))
            _,val,_,(px,py)=cv2.minMaxLoc(score)
            if occupancy[py,px]<.89 or min(rate[py,px] for rate in rates)<.48 or val<.80:continue
            score[max(0,py-7):py+8,max(0,px-7):px+8]=-1
            if val-float(score.max())<.035:continue
            mat=np.float64([[scale,0,left/factor-(px+l)*scale],[0,scale,top/factor-(py+t)*scale]])
            near=False
            if player is not None:
                src=self.site(source,start)
                if src is not None:
                    sx,sy=np.rint(np.asarray(self.coords.matrices[source['id']])@[*player,1]).astype(int)
                    mask=src[0]
                    if 0<=sx<mask.shape[1] and 0<=sy<mask.shape[0]:
                        near=bool(mask[max(0,sy-8):sy+9,max(0,sx-8):sx+9].any())
            choices.append((near,val,name,mat))
        choices.sort(key=lambda row:(row[0],row[1]),reverse=True)
        if not choices:return None
        near,val,name,mat=choices[0]
        if len(choices)>1 and near==choices[1][0] and val-choices[1][1]<.07:return None
        # The B1 elevator is unique. For B2/B3, a lone corner of an
        # ordinary room must not be forced onto a known site without a source
        # passage observation. Let the independent station matcher try instead.
        if 'B1' not in (level,target) and not near:return None
        return self.result(floor,mat,name,val,False)

    @staticmethod
    def straight_walls(room,edges):
        # Fog has a rounded edge. On each axis choose the longest straight
        # stretch of boundary, with image gradients breaking close ties.
        fields=[np.zeros_like(room) for _ in range(4)]
        sharp=cv2.dilate(edges,np.ones((3,3),np.uint8))>0
        for axis in (0,1):
            mask=room.T if axis else room
            rows=np.flatnonzero(mask.any(axis=1))
            best=None
            for reverse in (0,1):
                scan=mask[:,::-1] if reverse else mask
                locations=np.argmax(scan[rows],axis=1)
                frequencies=np.bincount(locations,minlength=scan.shape[1])
                center=int(np.argmax(np.convolve(frequencies,[1,1,1],mode='same')))
                sel=np.abs(locations-center)<=1
                if sel.sum()<max(11,len(rows)*.40):continue
                ys,xs=rows[sel],locations[sel]
                if reverse:xs=mask.shape[1]-1-xs
                if axis:xs,ys=ys,xs
                quality=sel.sum()*(.75+.25*sharp[ys,xs].mean())
                if best is None or quality>best[0]:best=(quality,reverse,ys,xs)
            if best is not None:
                _,reverse,ys,xs=best
                fields[2*axis+reverse][ys,xs]=1
        return fields

    @staticmethod
    def sides(room,edges=None):
        gx=cv2.Sobel(room,cv2.CV_16S,1,0);gy=cv2.Sobel(room,cv2.CV_16S,0,1)
        sides=[(gx>180)&(gx>np.abs(gy)),(gx<-180)&(-gx>np.abs(gy)),
               (gy>180)&(gy>np.abs(gx)),(gy<-180)&(-gy>np.abs(gx))]
        if edges is not None:
            sharp=cv2.dilate(edges,np.ones((3,3),np.uint8))>0
            sides=[a&sharp for a in sides]
        return [a.astype(np.uint8) for a in sides]

    @staticmethod
    def result(floor,mat,link,score,provisional):
        return dict(accepted=True,floor=floor,matrix=mat,scale=float(np.linalg.norm(mat[0,:2])),
                    possible_layouts=[floor['layout_id']],inliers=0,error_px=None,score=score,
                    consistency=score,segmentation='floor_handoff',search='floor_handoff',
                    passage=link,provisional=provisional,candidates=[],reason='继承楼层位置')

    def player_position(self,image,floor,mat):
        if floor['id'] not in self.coords.matrices:return None
        h,w=image.shape[:2];factor=min(1.,640/w)
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        mask=cv2.inRange(cv2.cvtColor(small,cv2.COLOR_BGR2HSV),(16,110,170),(39,255,255))
        _,_,stats,points=cv2.connectedComponentsWithStats(mask)
        candidates=[]
        for stat,point in zip(stats[1:],points[1:]):
            if not 6<=stat[4]<=160:continue
            center=point/factor;distance=np.linalg.norm(center-[w*.58,h*.50])
            if distance<h*.20:candidates.append((float(distance),center))
        candidates.sort(key=lambda a:a[0])
        if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<h*.02):return None
        xf=mat@np.vstack((self.coords.matrices[floor['id']],[0,0,1]))
        return cv2.invertAffineTransform(xf)@[*candidates[0][1],1]

"""Check a camera against the walls in the art actually being projected."""
from functools import lru_cache
from types import SimpleNamespace
from .s1_view_motion import exclude_map_hud
import cv2
import numpy as np
from .s1_alignment import geometry
from .s1_fixed_walls import FixedWalls
from .s1_original_map import OriginalMap
from .s1_entrance import turns


class PoseCheck:
    def __init__(self,matcher):
        self.matcher=matcher
        self.hd_walls=FixedWalls(matcher,self.hd_geometry)
        self.atlas_walls=FixedWalls(matcher)

    @lru_cache(maxsize=45)
    def _hd_geometry(self,fid):
        floor=self.matcher.handoffs.floors[fid]
        if not hasattr(self.matcher,'original'):self.matcher.original=OriginalMap(self.matcher)
        art,registration,_=self.matcher.original.source(floor)
        h,w=self.matcher.template(floor)[1].shape
        image=cv2.warpAffine(art,registration,(w,h),flags=cv2.INTER_AREA)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        rooms=cv2.inRange(hsv,(3,16,33),(38,255,255))
        halls=cv2.inRange(hsv,(91,15,33),(141,255,255))
        geom=np.maximum(rooms//2,halls)
        return cv2.morphologyEx(geom,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

    def hd_geometry(self,floor):
        return self._hd_geometry(floor['id'])

    @staticmethod
    def sides(mask):
        binary=np.uint8(mask)*255
        dx=cv2.Sobel(binary,cv2.CV_16S,1,0)
        dy=cv2.Sobel(binary,cv2.CV_16S,0,1)
        vertical=np.abs(dx)>np.abs(dy)
        return [(dx>0)&vertical,(dx<0)&vertical,(dy>0)&~vertical,(dy<0)&~vertical]

    @lru_cache(maxsize=6)
    def distances(self,fid,hd):
        floor=self.matcher.handoffs.floors[fid]
        ref=self.hd_geometry(floor) if hd else self.matcher.template(floor)[1]
        return [cv2.distanceTransform(np.uint8(~edge),cv2.DIST_L2,3)
                for code in (127,255) for edge in self.sides(ref==code)]

    @staticmethod
    def residual(field,xy,mat):
        inv=cv2.invertAffineTransform(mat)
        coords=xy@inv[:,:2].T+inv[:,2]
        indices=np.rint(coords).astype(int);h,w=field.shape
        ok=(indices[:,0]>=0)&(indices[:,0]<w)&(indices[:,1]>=0)&(indices[:,1]<h)
        out=np.full(len(xy),60.,np.float32)
        out[ok]=np.minimum(60,field[indices[ok,1],indices[ok,0]]*np.linalg.norm(mat[0,:2]))
        return out

    @staticmethod
    def translation(finder,features,floor,prior):
        fid=floor['id']
        if fid not in finder.refs:finder.refs[fid]=turns(finder.reference(floor),4)
        ref,types=finder.refs[fid];factor=features.factor
        points,tags=turns(features.geom,max(3,np.linalg.norm(prior[0,:2])*factor*3))
        if len(points)<4:return None
        iy,ix=np.rint(points).astype(int)[:,::-1].T
        valid=cv2.erode(features.valid,np.ones((7,7),np.uint8))[iy,ix]>0
        points,tags=points[valid],tags[valid]
        q,r=np.nonzero(tags[:,None]==types[None,:])
        if len(q)<4:return None
        differences=points[q]-ref[r]@prior[:,:2].T*factor
        candidates=[]
        for shift in (0.,2.):
            bins=np.floor((differences+shift)/4).astype(int)
            _,ids,counts=np.unique(bins,axis=0,return_inverse=True,return_counts=True)
            for peak in np.argsort(counts)[-6:]:
                if counts[peak]<3:continue
                offset=np.median(differences[ids==peak],axis=0)
                err=np.linalg.norm(differences-offset,axis=1)
                compatible=np.flatnonzero(err<3.2)
                used=set();seen=set();matches=[]
                for i in compatible[np.argsort(err[compatible])]:
                    if q[i] in seen or r[i] in used:continue
                    used.add(r[i]);seen.add(q[i]);matches.append(i)
                if len(matches)<4:continue
                x,y=ref[r[matches]],points[q[matches]]/factor
                if np.ptp(y,axis=0).min()<32:continue
                pos=np.median(y-x@prior[:,:2].T,axis=0)
                candidates.append((len(matches),pos,x,y))
        if not candidates:return None
        candidates.sort(key=lambda row:-row[0])
        n,pos,x,y=candidates[0]
        rival=max((v[0] for v in candidates if np.linalg.norm(pos-v[1])*factor>7),default=0)
        if n<.35*len(points) or n-rival<max(2,n*.14):return None
        mat=prior.copy();mat[:,2]=pos
        return dict(accepted=True,floor=floor,matrix=mat,scale=float(np.linalg.norm(mat[0,:2])),
                    inliers=n,score=n,error_px=None,possible_layouts=[floor['layout_id']],candidates=[],
                    rival=rival,wall_pairs=(x,y),adjustment='translation')

    def check(self,frame,area,clear,floor,prior,style,confirmed):
        factor=min(1.,1024/frame.shape[1])
        small=cv2.resize(frame,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        valid=exclude_map_hud(area(small).copy())
        if clear is not None:
            mask=cv2.resize(clear,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_AREA)
            valid &= cv2.erode(np.uint8(mask==255)*255,np.ones((5,5),np.uint8))
        geom=geometry(small)&valid
        hd=style!='structure'
        finder=self.hd_walls if hd else self.atlas_walls
        features=SimpleNamespace(factor=factor,geom=geom,valid=valid)
        hit=self.translation(finder,features,floor,prior)
        if hit:
            trustworthy,result=self.evaluate(small,features,floor,prior,hd,hit)
            if trustworthy:return result
        hit=finder.locate(features,floor,float(np.linalg.norm(prior[0,:2])),confirmed,with_pairs=True)
        if hit is not None:
            hit['adjustment']='scale_and_translation'
            return self.evaluate(small,features,floor,prior,hd,hit)[1]
        return None

    def evaluate(self,small,features,floor,prior,hd,hit):
        geom,valid,factor=features.geom,features.valid,features.factor
        x,y=hit.pop('wall_pairs')
        mat=hit['matrix']
        if hit['adjustment']=='scale_and_translation':
            cx,cy=x.mean(axis=0),y.mean(axis=0)
            zoom=np.sum((x-cx)*(y-cy))/np.sum((x-cx)**2)
            mat=np.eye(2,3)*zoom;mat[:,2]=cy-zoom*cx
        span=np.ptp(y,axis=0)
        height=small.shape[0]/factor
        if min(span)<max(28,height*.032):return False,None
        corners=np.rint(y*factor).astype(int)
        neighborhood=np.zeros(geom.shape,np.uint8)
        for pt in corners:cv2.circle(neighborhood,tuple(pt),16,255,-1)
        for pt in corners:cv2.circle(neighborhood,tuple(pt),3,0,-1)
        sharp=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),24,65)
        exposed=(cv2.erode(valid,np.ones((5,5),np.uint8))>0)&(cv2.dilate(sharp,np.ones((3,3),np.uint8))>0)
        usable=exposed&(neighborhood>0)
        old_samples=[];new_samples=[];axes=set()
        outside=[]
        directions=[e for code in (127,255) for e in self.sides(geom==code)]
        for i,(edge,dist) in enumerate(zip(directions,self.distances(floor['id'],hd))):
            iy,ix=np.nonzero(edge&usable)
            if len(ix)<8:continue
            xy=np.stack((ix,iy),axis=1)/factor
            old_samples.append(self.residual(dist,xy,prior))
            new_samples.append(self.residual(dist,xy,mat))
            axes.add((i%4)//2)
            y2,x2=np.nonzero(edge&exposed&(neighborhood==0))
            if len(x2)>20:
                xy2=np.stack((x2[::3],y2[::3]),axis=1)/factor
                outside.append((self.residual(dist,xy2,prior),self.residual(dist,xy2,mat)))
        if len(axes)<2:return False,None
        old=np.concatenate(old_samples);new=np.concatenate(new_samples)
        if len(old)<50:return False,None
        before,after=np.percentile(old,75),np.percentile(new,75)
        difference=np.linalg.norm(x@(mat[:,:2]-prior[:,:2]).T+mat[:,2]-prior[:,2],axis=1)
        moved=float(np.percentile(difference,80))
        tolerance=max(2.5,height*.0025)
        if after>2*tolerance or np.mean(new<=tolerance)<.60:return False,None
        if outside:
            a,b=(np.concatenate(v) for v in zip(*outside))
            limit=4*tolerance
            if np.mean(np.minimum(b,limit)-np.minimum(a,limit))>tolerance*.15:return False,None
        if moved<2*tolerance or before-after<tolerance:return True,None
        hit.update(matrix=mat,scale=float(np.linalg.norm(mat[0,:2])),search='wall_realign',segmentation='wall_realign',
                   before_px=round(float(before),2),after_px=round(float(after),2),shift_px=round(moved,2))
        return True,hit


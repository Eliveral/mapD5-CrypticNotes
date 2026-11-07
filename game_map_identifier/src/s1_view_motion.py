"""Measure pan/zoom from explored game pixels, independent of the painted overlay."""
from functools import cached_property
import cv2
import numpy as np
from .s1_alignment import geometry


class ViewFeatures:
    def __init__(self,image,area,unpainted=None):
        self.factor=min(1.,720/image.shape[1])
        small=cv2.resize(image,None,fx=self.factor,fy=self.factor,interpolation=cv2.INTER_AREA)
        valid=area(small)
        if unpainted is not None:
            clear=cv2.resize(unpainted,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_AREA)
            valid=cv2.bitwise_and(valid,np.uint8(clear>=252)*255)
        self.valid=cv2.erode(valid,np.ones((5,5),np.uint8))
        self.geom=geometry(small)&self.valid
        self.gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)

    @cached_property
    def keypoints(self):
        # The first overlay needs only a pose. Compute tracking descriptors
        # when comparing two views, after the first pose can already be shown.
        mask=cv2.dilate(np.uint8(self.geom>0)*255,np.ones((5,5),np.uint8))&self.valid
        features,desc=cv2.SIFT_create(nfeatures=370,contrastThreshold=.009).detectAndCompute(self.gray,mask)
        return np.float32([p.pt for p in features]).reshape(-1,2),desc

    @property
    def points(self):
        return self.keypoints[0]

    @property
    def descriptors(self):
        return self.keypoints[1]

    def motion_to(self,other):
        if len(self.points)<20:
            patch=self.patch_motion(other)
            if patch is not None:return patch
        keypoints=self.keypoint_motion(other)
        if keypoints is not None:return keypoints
        if len(other.points)<20:
            reverse=other.patch_motion(self)
            if reverse is not None:return cv2.invertAffineTransform(reverse[0]),reverse[1],reverse[2]
        return self.patch_motion(other) if len(self.points)>=20 else None

    def keypoint_motion(self,other):
        if self.descriptors is None or other.descriptors is None or min(len(self.points),len(other.points))<5:return None
        pairs=cv2.BFMatcher().knnMatch(self.descriptors,other.descriptors,k=2)
        chosen={}
        for pair in pairs:
            if len(pair)<2:continue
            a,b=pair
            if a.distance<.76*b.distance and (a.trainIdx not in chosen or a.distance<chosen[a.trainIdx].distance):
                chosen[a.trainIdx]=a
        if len(chosen)<5:return None
        src=np.float32([self.points[p.queryIdx] for p in chosen.values()])
        dst=np.float32([other.points[p.trainIdx] for p in chosen.values()])
        mat,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=400)
        if mat is None:return None
        ok=inliers.ravel().astype(bool)
        if ok.sum()<5 or (ok.mean()<.65 and ok.sum()<10):return None
        # Multiple SIFT orientations at one point are one physical landmark.
        points=np.unique(np.rint(src[ok]/3),axis=0)
        if len(points)<4 or cv2.contourArea(cv2.convexHull(np.float32(points)))<10:return None
        scale=mat[0,0]
        if not .53<scale<1.85 or abs(mat[1,0])>.018*scale:return None
        # The game has no map rotation. Fit the camera the game actually uses.
        x,y=src[ok],dst[ok]
        cx,cy=x.mean(axis=0),y.mean(axis=0)
        scale=np.sum((x-cx)*(y-cy))/max(np.sum((x-cx)**2),1)
        offset=cy-scale*cx
        if not .53<scale<1.85 or np.median(np.linalg.norm(x*scale+offset-y,axis=1))>1.5:return None
        mat=np.array([[scale,0,offset[0]],[0,scale,offset[1]]],dtype=np.float64)
        h,w=other.geom.shape
        warped=cv2.warpAffine(self.geom,mat,(w,h),flags=cv2.INTER_NEAREST)
        valid=(cv2.warpAffine(self.valid,mat,(w,h),flags=cv2.INTER_NEAREST)>0)&(other.valid>0)
        # A hidden room can be newly visible, replaced, or attached to a
        # corridor. Verify matched neighborhoods on separate pieces of wall.
        neighborhoods=np.zeros((h,w),np.uint8)
        for point in y:cv2.circle(neighborhoods,tuple(np.rint(point).astype(int)),12,255,-1)
        valid &= neighborhoods>0
        agreement=[]
        for label in (127,255):
            a=(warped==label)&valid;b=(other.geom==label)&valid
            if min(a.sum(),b.sum())<35:continue
            k=np.ones((5,5),np.uint8)
            agreement.extend((float(cv2.dilate(np.uint8(a),k)[b].mean()),float(cv2.dilate(np.uint8(b),k)[a].mean())))
        if not agreement or min(agreement)<.76:return None
        mat[:,:2]*=self.factor/other.factor
        mat[:,2]/=other.factor
        return mat,len(points),min(agreement)

    def patch_motion(self,other):
        # A partially revealed room may have fewer than four distinct SIFT
        # points. Compare its actual walls and texture against the new screen.
        _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(self.geom>0))
        if len(stats)<2:return None
        largest=1+np.argmax(stats[1:,4])
        x,y,w,h=stats[largest,:4]
        if min(w,h)<12 or max(w,h)>130 or w*h>9500:return None
        mask=np.uint8(labels==largest)*255
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8))&self.valid
        x,y,w,h=cv2.boundingRect(mask)
        if min(w,h)<12:return None
        roi=np.s_[y:y+h,x:x+w]
        template=self.gray[roi];keep=mask[roi]
        corners=cv2.goodFeaturesToTrack(template,16,.06,4,mask=keep)
        if corners is None or len(corners)<3 or np.ptp(corners[:,0,:],axis=0).min()<6:return None
        H,W=other.gray.shape
        _,ids,stats,_=cv2.connectedComponentsWithStats(np.uint8(other.geom>0))
        real=np.flatnonzero(stats[1:,4]>24)+1
        if not len(real):return None
        pieces=stats[real];left,top=np.maximum(0,pieces[:,:2].min(axis=0)-[w,h])
        right,bottom=np.minimum([W,H],(pieces[:,:2]+pieces[:,2:4]).max(axis=0)+[w,h])
        search=other.gray[top:bottom,left:right]
        candidates=[]
        for zoom in (1.,.8,1.2):
            size=round(w*zoom),round(h*zoom)
            if min(size)<8 or size[0]>=search.shape[1] or size[1]>=search.shape[0]:continue
            q=cv2.resize(template,size);k=cv2.resize(keep,size,interpolation=cv2.INTER_NEAREST)
            if q[k>0].size<60 or q[k>0].std()<7:continue
            scores=cv2.matchTemplate(search,q,cv2.TM_CCOEFF_NORMED,mask=k)
            scores[~np.isfinite(scores)]=-1
            scores[scores>1.001]=-1
            for _ in range(2):
                _,v,_,(lx,ly)=cv2.minMaxLoc(scores)
                if v<.72:break
                px,py=lx+left,ly+top
                region=np.s_[py:py+size[1],px:px+size[0]]
                valid=other.valid[region][k>0].mean()/255
                g=cv2.resize(self.geom[roi],size,interpolation=cv2.INTER_NEAREST)
                observed=(g>0)&(k>0)
                agree=(other.geom[region][observed]==g[observed]).mean() if observed.any() else 0.
                if valid>.86 and agree>.81:candidates.append((v,zoom,px,py))
                scores[max(0,ly-6):ly+7,max(0,lx-6):lx+7]=-1
        if not candidates:return None
        candidates.sort(reverse=True);best,zoom,px,py=candidates[0]
        center=np.array([px+w*zoom/2,py+h*zoom/2])
        rival=max((v for v,z,i,j in candidates[1:] if np.linalg.norm(center-[i+w*z/2,j+h*z/2])>7),default=0.)
        if best<.87 or best-rival<.09:return None
        factor=self.factor/other.factor
        mat=np.float64([[zoom*factor,0,(px-x*zoom)/other.factor],[0,zoom*factor,(py-y*zoom)/other.factor]])
        return mat,len(corners),best


"""Measure pan/zoom from explored game pixels, independent of the painted overlay."""
from functools import cached_property
import cv2
import numpy as np
from .s1_alignment import geometry


def zoom_position(image):
    """Read the slider thumb in the map HUD. A missing thumb supplies no hint."""
    h,w=image.shape[:2]
    if h<240 or w<h*.8:return None
    x=w-round(h*.096);radius=round(h*.030)
    top,bottom=round(h*.382),round(h*.765)
    strip=cv2.cvtColor(image[top:bottom,x-radius:x+radius+1],cv2.COLOR_BGR2GRAY)
    wing=max(1,round(h*.008));pad=max(1,round(h*.002))
    left=strip[:,radius-wing-pad:radius-wing+pad+1].mean(axis=1)
    right=strip[:,radius+wing-pad:radius+wing+pad+1].mean(axis=1)
    outer=(strip[:,:pad+1].mean(axis=1)+strip[:,-pad-1:].mean(axis=1))*.5
    contrast=np.minimum(left,right)-outer
    bright=np.flatnonzero(contrast>max(23.,contrast.max()*.60))
    runs=np.split(bright,np.flatnonzero(np.diff(bright)>1)+1)
    thumbs=[run for run in runs if h*.004<len(run)<h*.032]
    if len(thumbs)!=1:return None
    thumb=thumbs[0]
    return float((top+np.average(thumb,weights=contrast[thumb]))/h)


def zoom_changed(before,after):
    return before is not None and after is not None and abs(before-after)>.0035


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
        self.zoom=zoom_position(image)

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

    @cached_property
    def pieces(self):
        _,ids,stats,_=cv2.connectedComponentsWithStats(np.uint8(self.geom>0))
        pieces=[]
        for idx in np.argsort(stats[1:,4])[::-1][:6]+1:
            x,y,w,h,area=stats[idx]
            if area<60 or min(w,h)<8:continue
            # The observed silhouette is stable across zoom only if its edges
            # are inside the valid capture, away from HUD or overlay cuts.
            box=np.s_[max(0,y-2):min(ids.shape[0],y+h+2),max(0,x-2):min(ids.shape[1],x+w+2)]
            mask=np.uint8(ids[box]==idx)*255
            if not self.valid[box][mask>0].all():continue
            keep=cv2.dilate(mask,np.ones((3,3),np.uint8))
            gray=self.gray[box]
            if np.std(gray[keep>0])<5:continue
            pieces.append((x,y,w,h,box,gray,keep))
        return pieces

    def zoom_motion(self,other):
        # Pair visible shapes, then solve scale and translation jointly from
        # their texture. Slider position signals motion but never sets the pose.
        proposals=[]
        for x,y,w,h,box,gray,mask in self.pieces:
            if min(w,h)<13:continue
            for tx,ty,tw,th,*_ in other.pieces:
                sx,sy=tw/w,th/h
                guess=(sx+sy)*.5
                if not .20<guess<4.8 or abs(sx-sy)>.30*guess:continue
                if zoom_changed(self.zoom,other.zoom) and (other.zoom-self.zoom)*(guess-1)>.007:continue
                if .96<guess<1.04:continue
                # Search near the paired component, including rasterisation error.
                center=np.array([tx+tw*.5,ty+th*.5])
                for z in np.unique(np.r_[sx,sy,guess*np.array([.92,.96,.98,1.,1.02,1.04,1.08])]):
                    size=tuple(np.rint(np.array(gray.shape[::-1])*z).astype(int))
                    if min(size)<8:continue
                    start=np.maximum(0,np.floor(center-np.array(size)/2-5).astype(int))
                    stop=np.minimum(other.gray.shape[::-1],np.ceil(center+np.array(size)/2+5).astype(int))
                    if np.any(stop-start<size):continue
                    region=np.s_[start[1]:stop[1],start[0]:stop[0]]
                    pattern=cv2.resize(gray,size,interpolation=cv2.INTER_AREA if z<1 else cv2.INTER_LINEAR)
                    keep=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
                    if keep.sum()<255*45 or pattern[keep>0].std()<5:continue
                    scores=cv2.matchTemplate(other.gray[region],pattern,cv2.TM_CCOEFF_NORMED,mask=keep)
                    scores[(~np.isfinite(scores))|(scores>1.001)]=-1
                    _,score,_,pos=cv2.minMaxLoc(scores)
                    if score<.83:continue
                    xy=start+pos;zz=np.mean(np.array(size)/gray.shape[::-1])
                    offset=xy-zz*np.array([box[1].start,box[0].start])
                    proposals.append((score,zz,offset,box))
        if not proposals:return None
        proposals.sort(key=lambda v:-v[0])
        accepted=[]
        for score,z,offset,box in proposals:
            if score<proposals[0][0]-.11:break
            mat=np.array([[z,0,offset[0]],[0,z,offset[1]]])
            # Verify actual room/walkway pixels on the whole visible source,
            # separately from the intensity patch used to propose the match.
            h,w=other.geom.shape
            warp=cv2.warpAffine(self.geom,mat,(w,h),flags=cv2.INTER_NEAREST)
            valid=other.valid>0
            agreements=[]
            for color in (127,255):
                pattern=(warp==color)&valid
                if pattern.sum()<25:continue
                near=cv2.dilate(np.uint8(other.geom==color),np.ones((3,3),np.uint8))
                agreements.append(float(near[pattern].mean()))
            if agreements and min(agreements)>.86:
                accepted.append((score,mat,min(agreements),box))
        if not accepted:return None
        score,mat,agreement,box=accepted[0]
        source_center=np.array([self.gray.shape[1]*.5,self.gray.shape[0]*.5,1])
        if any(score-rival<.05 and np.linalg.norm((mat-alt)@source_center)>6
               for rival,alt,_,_ in accepted[1:]):return None
        # The component span is rounded to pixels. Refine its pose using
        # subpixel corners after both images are at nearly the same magnification.
        height,width=other.gray.shape
        area=np.zeros_like(self.geom)
        area[box]=np.uint8(self.geom[box]>0)*255
        mask=cv2.warpAffine(area,mat,(width,height),flags=cv2.INTER_NEAREST)
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8)) & other.valid
        projected=cv2.warpAffine(self.gray,mat,(width,height))
        points=cv2.goodFeaturesToTrack(projected,70,.025,3,mask=mask)
        if points is not None and len(points)>=4:
            moved,status,_=cv2.calcOpticalFlowPyrLK(projected,other.gray,points,None,winSize=(11,11),maxLevel=1)
            back,valid,_=cv2.calcOpticalFlowPyrLK(other.gray,projected,moved,None,winSize=(11,11),maxLevel=1)
            ok=status.ravel().astype(bool)&valid.ravel().astype(bool)
            ok&=(np.linalg.norm(back-points,axis=2).ravel()<.6)&(np.linalg.norm(moved-points,axis=2).ravel()<4)
            a,b=points[ok,0],moved[ok,0]
            if len(a)>=4 and np.ptp(a,axis=0).min()>7:
                ca,cb=a.mean(axis=0),b.mean(axis=0)
                correction=float(np.sum((a-ca)*(b-cb))/np.sum((a-ca)**2))
                shift=cb-correction*ca
                residual=np.linalg.norm(a*correction+shift-b,axis=1)
                if .90<correction<1.10 and np.median(residual)<.55:
                    mat=mat*correction
                    mat[:,2]+=shift
        mat[:,:2]*=self.factor/other.factor
        mat[:,2]/=other.factor
        return mat,1,agreement

    def scale_match(self,other):
        # When zooming in, match the sharper, larger patch down to its smaller
        # counterpart, then invert. Enlarging a tiny source loses detail.
        if self.zoom is not None and other.zoom is not None and self.zoom>other.zoom+.0035:
            reverse=other.zoom_motion(self)
            if reverse is not None:return cv2.invertAffineTransform(reverse[0]),reverse[1],reverse[2]
        direct=self.zoom_motion(other)
        if direct is not None:return direct
        if not zoom_changed(self.zoom,other.zoom):
            reverse=other.zoom_motion(self)
            if reverse is not None:return cv2.invertAffineTransform(reverse[0]),reverse[1],reverse[2]
        return None

    def motion_to(self,other):
        if zoom_changed(self.zoom, other.zoom):
            hit=self.keypoint_motion(other)
            if hit is not None and (other.zoom-self.zoom)*(hit[0][0,0]-1)<0:return hit
            return self.scale_match(other)
        if len(self.points)<20:
            patch=self.patch_motion(other)
            if patch is not None:return patch
        keypoints=self.keypoint_motion(other)
        if keypoints is not None:return keypoints
        if len(other.points)<20:
            reverse=other.patch_motion(self)
            if reverse is not None:return cv2.invertAffineTransform(reverse[0]),reverse[1],reverse[2]
        hit=self.patch_motion(other) if len(self.points)>=20 else None
        return hit if hit is not None else self.scale_match(other)

    def same_pixels(self, other):
        if self.geom.shape!=other.geom.shape:
            return False
        valid=(self.valid>0)&(other.valid>0)
        support=((self.geom>0)|(other.geom>0))&valid
        if support.sum()<100 or (self.geom[support]!=other.geom[support]).mean()>.035:
            return False
        delta=cv2.absdiff(self.gray,other.gray)[support]
        return delta.mean()<4 and (delta>15).mean()<.06

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
        if not .20<scale<5. or abs(mat[1,0])>.018*scale:return None
        # The game has no map rotation. Fit the camera the game actually uses.
        x,y=src[ok],dst[ok]
        cx,cy=x.mean(axis=0),y.mean(axis=0)
        scale=np.sum((x-cx)*(y-cy))/max(np.sum((x-cx)**2),1)
        offset=cy-scale*cx
        if not .20<scale<5. or np.median(np.linalg.norm(x*scale+offset-y,axis=1))>1.5:return None
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


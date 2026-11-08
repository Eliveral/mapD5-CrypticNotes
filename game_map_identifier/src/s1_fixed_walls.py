"""Register the known target floor from fixed wall junctions despite a variable room."""
import cv2
import numpy as np
from .s1_entrance import turns


class FixedWalls:
    def __init__(self,matcher,reference=None):
        self.matcher=matcher
        self.reference=reference if reference is not None else lambda floor:matcher.template(floor)[1]
        self.refs={}

    def locate(self,features,floor,scale,confirmed=False,with_pairs=False,ratios=None,compare_layouts=False):
        fid=floor['id']
        if fid not in self.refs:
            self.refs[fid]=turns(self.reference(floor),4)
        ref,rt=self.refs[fid]
        query,qt=turns(features.geom,max(3,scale*features.factor*3))
        if len(query):
            xy=np.rint(query).astype(int)
            interior=cv2.erode(features.valid,np.ones((7,7),np.uint8))
            reliable=interior[xy[:,1],xy[:,0]]>0
            query,qt=query[reliable],qt[reliable]
        if len(query)<7 or len(ref)<7:return None
        qi,ri=np.nonzero(qt[:,None]==rt[None,:])
        if len(qi)<7:return None
        candidates=[]
        for ratio in ((1.,.95,1.05,.90,1.10,.85,1.17,.79,1.25) if ratios is None else ratios):
            zoom=scale*features.factor*ratio
            offsets=query[qi]-ref[ri]*zoom
            # Two grids avoid splitting a correct vote at a bin boundary.
            for shift in (0.,3.):
                cells=np.floor((offsets+shift)/6).astype('int32')
                # A scalar bin key preserves lexicographic cell order and ties
                # without sorting structured (x,y) rows for every zoom ratio.
                y=cells[:,1]
                stride=int(y.max())-int(y.min())+1
                keys=cells[:,0].astype(np.int64)*stride+y
                _,ids,counts=np.unique(keys,return_inverse=True,return_counts=True)
                for idx in np.argsort(counts)[-3:]:
                    if counts[idx]<4:continue
                    offset=np.median(offsets[ids==idx],axis=0)
                    delta=np.linalg.norm(offsets-offset,axis=1)
                    matches=np.flatnonzero(delta<5)
                    seen_q=set();seen_r=set();a=[];b=[]
                    for i in matches[np.argsort(delta[matches])]:
                        if qi[i] in seen_q or ri[i] in seen_r:continue
                        seen_q.add(qi[i]);seen_r.add(ri[i]);a.append(ri[i]);b.append(qi[i])
                    if len(a)<6:continue
                    x,y=ref[a],query[b]
                    cx,cy=x.mean(axis=0),y.mean(axis=0)
                    fitted=np.sum((x-cx)*(y-cy))/max(1.,np.sum((x-cx)**2))
                    if abs(fitted/zoom-1)>.12:continue
                    off=cy-fitted*cx
                    good=np.linalg.norm(x*fitted+off-y,axis=1)<3.3
                    y=y[good];tags=qt[np.asarray(b)[good]]
                    n=len(y)
                    if n<6 or np.ptp(y,axis=0).min()<18 or np.sum(tags//64==255)<2 or np.sum(tags//64==127)<2:continue
                    if len(set(map(tuple,np.floor(y/22).astype(int))))<3:continue
                    candidates.append((n,fitted,off,y,x[good] if with_pairs else None))
        if not candidates:return None
        candidates.sort(key=lambda row:-row[0])
        n,zoom,off,points,reference_points=candidates[0]
        center=np.mean(points,axis=0)
        rival=max((c[0] for c in candidates[1:] if np.linalg.norm((center-off)/zoom-(center-c[2])/c[1])*zoom>10),default=0)
        if (not confirmed and n<.34*len(query)) or (not compare_layouts and n-rival<max(2,n*.12)):return None
        mat=np.float64([[zoom,0,off[0]],[0,zoom,off[1]]])
        h,w=features.geom.shape
        refgeom=self.reference(floor)
        model=cv2.warpAffine(refgeom,mat,(w,h),flags=cv2.INTER_NEAREST)
        # Verify the neighborhoods that voted for the placement. Unexplained
        # rooms elsewhere in the frame are not evidence against these walls.
        neighborhoods=np.zeros((h,w),np.uint8)
        for xy in points:cv2.circle(neighborhoods,tuple(np.rint(xy).astype(int)),10,255,-1)
        valid=(neighborhoods>0)&(features.valid>0)
        rates=[]
        for label in (127,255):
            a=(features.geom==label)&valid;b=model==label
            if a.sum()<25:return None
            supported=cv2.dilate(np.uint8(b),np.ones((3,3),np.uint8))
            rates.append(float(supported[a].mean()))
            # Optional rooms can explain localized disagreement. Contradictions
            # across several distinct areas are evidence against this layout.
            observed=np.uint8((features.geom==label)&(features.valid>0))
            if not confirmed and supported[observed>0].mean()<.88:
                count,ids,stats,_=cv2.connectedComponentsWithStats(observed)
                support=np.bincount(ids.ravel(),weights=supported.ravel(),minlength=count)
                large=stats[1:,4]>max(35,observed.sum()*.005)
                if np.count_nonzero(large&(support[1:]/stats[1:,4]<.78))>2:return None
        if min(rates)<.85:return None
        mat/=features.factor
        result=dict(accepted=True,floor=floor,matrix=mat,scale=mat[0,0],inliers=n,error_px=None,
                    possible_layouts=[floor['layout_id']],score=n,consistency=min(rates),rival=rival,
                    candidates=[],segmentation='fixed_walls',search='fixed_walls',reason='已对齐固定房间和走廊')
        if with_pairs:result['wall_pairs']=(reference_points,points/features.factor)
        return result


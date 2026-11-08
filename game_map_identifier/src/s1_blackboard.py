"""Locate the B3 repair station from its map symbol and current map geometry."""
from functools import lru_cache
import cv2
import numpy as np

from .s1_alignment import geometry
from .s1_view_motion import exclude_map_hud

# Quantized, high-pass B channel of the map symbol from a registered B3 capture.
# Including the feet distinguishes it from the chest and rectangular HUD buttons.
_GLYPH = (
    '55444333322222222211111', '55444333222222221112211', '5544433333444555777aa61',
    '444448ccddffeeeecbbae92', '44434da7764443332223d70', '44433c42233333322212d60',
    '44434d43333333332213b50', '44334d43333334332215d50', '44334c43333333332225d40',
    '44335b43333333332215c30', '44326a33333333222114c30', '43326b65567565545447b30',
    '43238b777797899abbaba30', '43259a88aab999897668a60', '43228837a4445558cbaa981',
    '43339984222222117826940', '43349a52222322211577920', '4446a95333333332116b931',
    '4458a843344444332129a52', '44444434445554433239963', '44444444455555444357642',
    '55444445555555544343332', '55544555555555554433333',
)
_ICON=np.float32([[int(c,16) for c in row] for row in _GLYPH])-5


@lru_cache(maxsize=48)
def glyph(size):
    mode=cv2.INTER_AREA if size<23 else cv2.INTER_LINEAR
    return cv2.resize(_ICON,(size,size),interpolation=mode)


def board_point(image,area,clear=None):
    """Return a single upright board on exposed map pixels, in screen coordinates."""
    h,w=image.shape[:2]
    valid=exclude_map_hud(area(image))
    if clear is not None:valid=cv2.bitwise_and(valid,np.uint8(clear>=252)*255)
    x0,y0,bw,bh=cv2.boundingRect(valid)
    if not bw or not bh:return None
    ys,xs=slice(y0,y0+bh),slice(x0,x0+bw)
    blue,_,red=cv2.split(image[ys,xs])
    sigma=max(1.,h*.002)
    contrast=cv2.subtract(blue,cv2.GaussianBlur(blue,(0,0),sigma))
    ink=np.uint8((contrast>18)&(cv2.subtract(blue,red)>4)&(valid[ys,xs]>0))
    ink=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(ink)
    rows=[]
    for x,y,ww,hh,n in stats[1:]:
        if not (h*.004<=min(ww,hh) and max(ww,hh)<=h*.035 and .65<=ww/hh<=1.45 and n>=8):continue
        pad=max(4,round(max(ww,hh)*.75))
        l,t=max(0,x-pad),max(0,y-pad)
        rr,bb=min(bw,x+ww+pad),min(bh,y+hh+pad)
        roi=blue[t:bb,l:rr].astype(np.float32)
        roi-=cv2.GaussianBlur(roi,(0,0),sigma)
        for size in {round(max(ww,hh)*z) for z in (1.12,1.28,1.45)}:
            if size<9 or size>min(roi.shape):continue
            _,score,_,(u,v)=cv2.minMaxLoc(cv2.matchTemplate(roi,glyph(size),cv2.TM_CCOEFF_NORMED))
            u+=l;v+=t
            if score>=.60 and np.all(valid[y0+v:y0+v+size,x0+u:x0+u+size]):
                rows.append((score,x0+u+(size-1)*.5,y0+v+(size-1)*.5))
    if not rows:return None
    best=max(rows)
    if best[0]<.70 or any(s>best[0]-.085 and np.hypot(x-best[1],y-best[2])>h*.017 for s,x,y in rows):return None
    return np.array(best[1:])


# The board can be on either side of the station. These orientations were
# measured in real registered game frames. Other art needs both-axis wall proof.
ORIENTATIONS={
    '040b510c2f8b3f7854ebd0d3bc4280f1d1bce1da667d71b6dde816c83cbc638b':-1,
    '08a716def7a93603993cf2724b62ae381af2b5fa94e80af9b2dd71657b22a025':-1,
    '850c1463baa8e963dc289caa6ccbf4ecc069176546a37884f9ebcbba12fe5001':-1,
    '3e5dd81260468ff199de0a50505d8c5d4845a90f8bcab618d24e5d4e20a1950f':1,
}


def board_pose(matcher,image,area,floor,station,clear=None,scale_hint=None):
    """The station offset proposes a translation. Current walls verify the pose."""
    point=board_point(image,area,clear)
    if point is None:return None
    reg=matcher.coordinates.matrices.get(floor['id'])
    if reg is None:return None
    art=matcher.catalog.by_id.get('hd_'+floor['layout_id'][3:]+'_B3',{})
    orientation=ORIENTATIONS.get(art.get('sha256'))
    fixed=orientation is not None and scale_hint is not None
    if scale_hint is not None and (not np.isfinite(scale_hint) or scale_hint<=0):return None
    factor=min(1.,950/image.shape[1])
    small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    valid=exclude_map_hud(area(small))
    if clear is not None:
        clean=cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)
        valid &= np.uint8(clean>=252)*255
    geom=geometry(small)&valid
    x,y=np.rint(point*factor).astype(int)
    radius=round(small.shape[0]*.30)
    left,top=max(0,x-radius),max(0,y-radius)
    roi=np.s_[top:y+radius+1,left:x+radius+1]
    crop=geom[roi];mask=valid[roi]
    edges=cv2.Canny(cv2.cvtColor(small[roi],cv2.COLOR_BGR2GRAY),24,60)
    sharp=(cv2.dilate(edges,np.ones((3,3),np.uint8))>0)&(cv2.erode(mask,np.ones((5,5),np.uint8))>0)
    fields=matcher.pose_check.distances(floor['id'],True)
    points=[];axes=set()
    for i,side in enumerate(edge for k in (127,255) for edge in matcher.pose_check.sides(crop==k)):
        axis=(i%4)//2
        kernel=np.ones((6,1) if axis==0 else (1,6),np.uint8)
        iy,ix=np.nonzero(cv2.morphologyEx(np.uint8(side&sharp),cv2.MORPH_OPEN,kernel))
        if len(ix)<8:continue
        delta=np.stack((ix+left,iy+top),axis=1)-point*factor
        points.append((fields[i],delta[::max(1,len(ix)//65)],axis));axes.add(axis)
    if len(axes)<2 and not fixed:return None
    near=geom[max(0,y-7):y+8,max(0,x-7):x+8]
    if near.size<160 or np.mean(near==127)<.62:return None
    occupied=np.column_stack(np.nonzero(crop[::3,::3])).astype(float)
    if not len(occupied):return None
    oy,ox=(occupied*3).astype(int).T
    expected=crop[oy,ox]
    positions=np.stack((ox+left,oy+top),axis=1)-point*factor
    model=matcher.pose_check.hd_geometry(floor)

    def evaluate(anchor,scale):
        def indices(delta):
            px,py=np.rint(anchor+delta/(scale*factor)).astype(int).T
            return np.clip(py,0,model.shape[0]-1),np.clip(px,0,model.shape[1]-1)
        iy,ix=indices(positions)
        agreement=np.mean(model[iy,ix]==expected)
        errors=[dist[indices(delta)]*(scale*factor) for dist,delta,_ in points]
        rates=[np.mean(d<=2.3) for d in errors]
        if not errors:return float(agreement),1.,agreement
        fit=np.mean([np.mean(np.maximum(0,1-d/4)) for d in errors])
        weakest=min(np.mean([rate for rate,(_,_,axis) in zip(rates,points) if axis==used]) for used in axes)
        return float(.75*fit+.25*agreement),weakest,agreement

    # Screen scale changes with game zoom and client size. Same-zoom measured
    # scale has priority, but must still fit the current frame.
    unit=image.shape[0]/1377/np.linalg.norm(np.asarray(reg)[0,:2])
    trials=np.geomspace(.18*unit,2.7*unit,47) if scale_hint is None else [scale_hint]
    trials=list(trials)
    candidates=[]
    for sign in (orientation,) if orientation else (-1,1):
        anchor=np.asarray(reg)@[*np.add(station,np.array([125,31])*sign),1]
        ranked=[(evaluate(anchor,z),z) for z in trials]
        quality,scale=max(ranked)
        if scale_hint is None:
            fine=np.linspace(scale*.95,scale*1.05,13)
            quality,scale=max((evaluate(anchor,z),z) for z in fine)
        candidates.append((quality,scale,anchor,ranked))
    candidates.sort(key=lambda row:row[0],reverse=True)
    (score,weakest,coverage),scale,anchor,ranked=candidates[0]
    if fixed:
        # A little explored island has artificial fog edges. A measured board
        # orientation and unchanged zoom already determine its camera; check
        # that the entire visible patch fits, even with only one real wall.
        island=coverage>=.98 and np.ptp(positions,axis=0).max()<small.shape[0]*.17
        if coverage<.94 or (not island and (score<.64 or weakest<.32)):return None
    elif score<.72 or weakest<.65 or coverage<.90:return None
    if len(candidates)>1 and candidates[1][0][0]>score-.08:return None
    if scale_hint is None and any(abs(z/scale-1)>.15 and q[0]>score-.045 for q,z in ranked):return None
    mat=np.eye(2,3)*scale
    mat[:,2]=point-scale*anchor
    hit=matcher.handoffs.result(floor,mat,'预言黑板',score,False)
    hit.update(search='b3_blackboard',segmentation='station_board',reason='黑板与当前负三墙线已对齐')
    return hit

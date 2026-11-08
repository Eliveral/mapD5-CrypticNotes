"""Compare B2 itself with every layout, without borrowing identity from B1."""
import cv2
import numpy as np
from .s1_alignment import geometry
from .s1_view_motion import ViewFeatures,exclude_map_hud
from .s1_entrance import turns


def compare_walls(matcher,image,area,clear,full=None,level='B2',incumbent=None):
    features=ViewFeatures(image,area,clear)
    g=features.geom
    _,_,stats,centers=cv2.connectedComponentsWithStats(np.uint8(g==127))
    rooms=centers[1:][stats[1:,4]>=40]
    if len(rooms)<2 or np.ptp(rooms,axis=0).max()<28:return None,{}
    corners,_=turns(g,3)
    if len(corners)<10:return None,{}
    finder=matcher.pose_check.hd_walls
    floors=[f for f in matcher.floors if f['floor']==level]
    rows=[]
    # Compare scales in HD pixels; different atlas templates use different crops/resolutions.
    if full:
        base=full['scale']*np.linalg.norm(np.asarray(matcher.coordinates.matrices[full['floor']['id']])[0,:2])
        ratios=(.88,.95,1.,1.05,1.13)
    else:
        base=1.
        ratios=np.geomspace(.16,2.8,44)
    for floor in floors:
        atlas=matcher.coordinates.matrices.get(floor['id'])
        if atlas is None:continue
        zoom=base/np.linalg.norm(np.asarray(atlas)[0,:2])
        hit=finder.locate(features,floor,zoom,confirmed=True,with_pairs=True,ratios=ratios,compare_layouts=True)
        if not hit:continue
        x,y=hit.pop('wall_pairs')
        if np.ptp(y,axis=0).min()<45 or hit['consistency']<.88:continue
        hit['cells']=frozenset(map(tuple,np.floor(x/24).astype(int)))
        rows.append(hit)
    if not rows:return None,{}
    rows.sort(key=lambda h:-h['inliers'])
    score={h['floor']['layout_id']:h['inliers'] for h in rows}
    best=rows[0];n=best['inliers']
    rival=max(best['rival'],rows[1]['inliers'] if len(rows)>1 else 0)
    if n<max(10,len(corners)*.27) or best['consistency']<.90 or n-rival<max(3,n*.20):
        # Several distributed alternatives can disprove an incumbent even
        # though they cannot yet identify a unique replacement.
        if (incumbent and n>=max(14,len(corners)*.15) and best['consistency']>=.88 and score.get(incumbent,0)<n*.55 and
                sum(v>=max(10,n*.70) for k,v in score.items() if k!=incumbent)>=2):
            best['conflicting_models']=True
        else:return None,score
    if level=='B2':
        best['region_verified']=(not best.get('conflicting_models') and len(rooms)>=3 and
                                 n>=18 and len(best['cells'])>=10 and best['consistency']>=.90
                                 and n-rival>=max(6,n*.35))
    method=level.lower()+'_wall_review'
    best.update(search=method,segmentation=method,proof_kind='partial')
    return best,score


def check_partial_b2_edges(matcher,image,area,clear,wall,support):
    """Independently locate visible directional edges throughout all B2 layouts.
    The corner fit supplies only the common art scale, never the translation.
    """
    winner=wall['floor']['layout_id']
    n=wall['inliers']
    rival=max((score for layout,score in support.items() if layout!=winner),default=0)
    if n<16 or len(wall['cells'])<10 or wall['consistency']<.94 or max(rival,wall['rival'])>.65*n:
        return False
    factor=min(1.,520/image.shape[1])
    small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    valid=exclude_map_hud(area(small).copy())
    if clear is not None:
        visible=cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)
        valid &= np.uint8(visible>=252)*255
    valid=cv2.erode(valid,np.ones((5,5),np.uint8))>0
    geom=geometry(small)
    sharp=cv2.dilate(cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),22,54),np.ones((3,3),np.uint8))>0
    # Fog cuts and the overlay boundary cannot supply directional wall pixels.
    channels=np.stack([edge & valid & sharp for v in (127,255)
                       for edge in matcher.pose_check.sides(geom==v)],axis=2)
    x,y,w,h=cv2.boundingRect(np.uint8(channels.any(axis=2)))
    if min(w,h)<15:return False
    query=channels[y:y+h,x:x+w].astype(np.float32)
    counts=query.sum(axis=(0,1))
    if counts.min()<5:return False
    query/=counts
    fid=wall['floor']['id'];mat=wall['matrix']
    native_scale=wall['scale']*np.linalg.norm(matcher.coordinates.matrices[fid][0][:2])
    corners=np.float64([[x,y,1],[x+w,y+h,1]])
    corners[:,:2]/=factor
    native=np.linalg.solve(np.vstack((mat,[0,0,1])),corners.T)
    winner_score=rival_score=0.
    for floor in matcher.floors:
        if floor['floor']!='B2':continue
        other=floor['id'];ref=matcher.pose_check.hd_geometry(floor)
        scale=native_scale*factor/np.linalg.norm(matcher.coordinates.matrices[other][0][:2])
        for z in (scale*.95,scale,scale*1.05):
            model=cv2.resize(ref,None,fx=z,fy=z,interpolation=cv2.INTER_NEAREST)
            model=cv2.copyMakeBorder(model,3,3,3,3,cv2.BORDER_CONSTANT)
            if model.shape[0]<h or model.shape[1]<w:continue
            edges=np.stack([cv2.dilate(np.uint8(edge),np.ones((3,3),np.uint8))
                            for v in (127,255) for edge in matcher.pose_check.sides(model==v)],axis=2)
            scores=cv2.matchTemplate(edges.astype(np.float32),query,cv2.TM_CCORR)/8
            _,score,_,(u,v)=cv2.minMaxLoc(scores)
            estimate=np.float64([[z,0,x-u+3],[0,z,y-v+3]])/factor
            if other==fid:
                if np.linalg.norm((estimate-mat)@native,axis=0).max()<image.shape[0]*.012:
                    winner_score=max(winner_score,score)
                # Neighbouring scales over the same room are one site. Suppress
                # only that site, even when another placement won this trial.
                center=native[:2].mean(axis=1)
                px,py=np.rint(z*center - [w*.5,h*.5] + 3).astype(int)
                cv2.rectangle(scores,(px-4,py-4),(px+4,py+4),-1,-1)
                score=cv2.minMaxLoc(scores)[1]
            rival_score=max(rival_score,score)
    return winner_score>=.70 and winner_score-rival_score>=.14

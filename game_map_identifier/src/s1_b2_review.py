"""Compare B2 itself with every layout, without borrowing identity from B1."""
import cv2
import numpy as np
from .s1_view_motion import ViewFeatures
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
    method=level.lower()+'_wall_review'
    best.update(search=method,segmentation=method,proof_kind='partial')
    return best,score


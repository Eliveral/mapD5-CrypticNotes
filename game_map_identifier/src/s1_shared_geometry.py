"""Use the confirmed floor and inherited scale on the first target sample."""
from types import SimpleNamespace
import cv2
import numpy as np
from .s1_alignment import geometry
from .s1_view_motion import exclude_map_hud

def refine_shared(matcher,image,area,visit,target,clear=None,audit=None):
    report=audit if audit is not None else {}
    def reject(reason):
        report['geometry_reason']=reason
        return None
    source=visit['floor'];layout=source['layout_id']
    predicted,_=matcher.coordinates.predict(visit,target)
    if predicted is None:return reject('missing_calibration')
    floor=matcher.handoffs.floors[layout+'_'+target]
    factor=min(1.,1000/image.shape[1])
    small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    valid=exclude_map_hud(area(small))
    if clear is not None:
        valid &= np.uint8(cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)>=252)*255
    geom=geometry(small)&valid
    features=SimpleNamespace(factor=factor,geom=geom,valid=valid)
    scale=float(np.linalg.norm(predicted[0,:2]))
    # Current corners supply translation only; distributed image evidence must
    # validate it. This does not run the layout library or search other scales.
    pose_check=matcher.pose_check
    candidate=pose_check.translation(pose_check.hd_walls,features,floor,predicted)
    hit=None
    if candidate:
        fallback={k:v for k,v in candidate.items() if k!='wall_pairs'}
        trustworthy,corrected=pose_check.evaluate(small,features,floor,predicted,True,candidate)
        if trustworthy:
            hit=corrected or fallback
    if hit is None:
        hit=matcher.corridors.locate(geom,layout,target,scale*factor,quick=True,
            predicted=predicted*factor,zoom_ratios=(1.,),paired=True,fixed_scale=True)
        if hit:
            hit['matrix']/=factor;hit['scale']/=factor
    if hit is None:return reject('insufficient_unique_current_geometry')
    # Keep a fresh target registration close to the calibrated source view or
    # its current-player recentering. Larger resets use the independent paths.
    seeds=[predicted]
    marker=matcher.handoffs.player_screen(image);point=visit.get('player_hd')
    if marker is not None and point is not None:
        native=np.asarray(matcher.coordinates.matrices[source['id']])@[*point,1]
        old=visit['matrix']@[*native,1]
        seed=predicted.copy();seed[:,2]+=marker-old;seeds.append(seed)
    center=np.array([image.shape[1]/2,image.shape[0]/2,1.])
    native=np.r_[cv2.invertAffineTransform(hit['matrix'])@center,1.]
    shift=min(np.linalg.norm((seed-hit['matrix'])@native) for seed in seeds)
    if shift>image.shape[0]*.28:return reject('outside_shared_neighborhood')
    if abs(float(np.linalg.norm(hit['matrix'][0,:2]))/scale-1)>.005:return reject('scale_not_inherited')
    report.update(geometry_reason='accepted',geometry_shift_px=round(float(shift),2))
    hit.update(search='shared_geometry',passage='共用比例·当前房间和走廊校正')
    return hit

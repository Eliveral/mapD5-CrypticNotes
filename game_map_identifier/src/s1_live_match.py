"""Live-client matching: exclude HUD, validate separate rooms, retain atlas rules."""
from __future__ import annotations
import cv2
import numpy as np
import threading
from .s1_view_motion import ViewFeatures,exclude_map_hud,zoom_position,zoom_changed,zoom_allows_scale
from .s1_alignment import FloorMatcher, geometry, register
from .s1_floor_transition import CorridorMatcher, FloorCoordinates
from .s1_original_map import OriginalMap
from .s1_landmarks import RoomLabels
from .s1_entrance import EntranceMatcher
from .s1_station import StationMatcher
from .s1_blackboard import board_pose
from .s1_handoff import FloorHandoff
from .s1_fixed_walls import FixedWalls
from .s1_pose_check import PoseCheck
from .s1_b2_review import compare_walls,check_partial_b2_edges


def map_area(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[int(h*.125):int(h*.90), int(w*.17):int(w*.86)] = 255
    mask[int(h*.56):, :int(w*.35)] = 0
    return mask


def signature(image):
    small = cv2.resize(image, (320, round(image.shape[0]*320/image.shape[1])))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Only room colour: UI animations and translucent backgrounds are excluded.
    return cv2.inRange(hsv, (4, 25, 45), (33, 200, 255)) & map_area(small)


def same_view(a, b):
    if a is None or b is None or a.shape != b.shape:
        return False
    union = np.count_nonzero(a | b)
    return union > 60 and np.count_nonzero(a ^ b)/union < .035


class LiveMatcher(FloorMatcher):
    def __init__(self,catalog,suppress_gradient=True):
        super().__init__(catalog,suppress_gradient)
        self._search_thread=threading.local()
        self.handoffs=FloorHandoff(self)
        self.coordinates=self.handoffs.coords
        self.fixed_walls=FixedWalls(self)
        self.pose_check=PoseCheck(self)
        self.corridors = CorridorMatcher(self)
        self.original = OriginalMap(self)

    def prepare_paired_floor(self, layout):
        for level,other in (('B2','B3'),('B3','B2')):
            floor=self.handoffs.floors.get(layout+'_'+level)
            sites=self.coordinates.passages.get((layout,level,other))
            if floor and sites:
                self.original.source(floor)
                self.corridors.reference_channels(floor)
                if level=='B3':self.pose_check.distances(floor['id'],True)
                for _,point,_ in sites:self.handoffs.site(floor,point)

    def locate_board(self,image,layout,clear=None,visit=None,zoom=None):
        floor=self.handoffs.floors.get(layout+'_B3') if layout else None
        if not floor:return None
        if not hasattr(self,'room_labels'):self.room_labels=RoomLabels(self.catalog)
        station=self.room_labels.points.get(floor['id'],{}).get('supply')
        if station is None:return None
        scale=None
        if (visit and visit['floor']['layout_id']==layout and visit.get('unique_layout')
                and not visit.get('provisional') and zoom is not None):
            old_zoom=visit.get('zoom')
            if old_zoom is None and visit.get('view_features'):old_zoom=visit['view_features'].zoom
            if old_zoom is not None and not zoom_changed(old_zoom,zoom):
                mat,_=self.coordinates.predict(visit,'B3')
                if mat is not None:scale=float(np.linalg.norm(mat[0,:2]))
        return board_pose(self,image,map_area,floor,station,clear,scale)

    def paired_b3(self,image,visit,locked,clear=None):
        return self.paired_floor(image,visit,'B3',locked,clear)

    def paired_floor(self,image,visit,target,layout,clear=None,return_views=(),zoom=None,source_history=None):
        if not visit or not layout or not visit.get('unique_layout') or visit.get('provisional'):
            return None
        source=visit['floor'];level=source['floor']
        if ({level,target}!={'B2','B3'} or source['layout_id']!=layout
                or (layout,level,target) not in self.coordinates.passages):
            return None
        predicted,_=self.coordinates.predict(visit,target)
        if predicted is None:return None
        floor=self.handoffs.floors[layout+'_'+target]
        factor=min(1.,1000/image.shape[1])
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        valid=exclude_map_hud(map_area(small))
        if clear is not None:
            mask=cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)
            valid &= np.uint8(mask>=252)*255
        geom=geometry(small)&valid
        if np.count_nonzero(geom)<100:return None
        returned=[v for v in return_views if v and v['floor']['id']==floor['id']
                  and not zoom_changed(v.get('zoom'),zoom)]
        player=visit.get('player_hd')
        if player is None and source_history is not None:
            player=source_history.get('player_hd')
        passage=self.handoffs.passage_at(source,target,player)
        candidates=[]
        source_zoom=visit.get('zoom')
        if source_zoom is None and visit.get('view_features'):source_zoom=visit['view_features'].zoom
        same_scale=not zoom_changed(source_zoom,zoom)
        if returned and (not same_scale or source_zoom is None):
            basis=returned[0]['matrix']
        else:basis=predicted if same_scale else None
        marker=self.handoffs.player_screen(image) if player is not None else None
        if passage is not None and basis is not None:
            if marker is not None:
                name,start,end=passage
                target_hd=player+end-start
                template_point=np.asarray(self.coordinates.matrices[floor['id']])@[*target_hd,1]
                mat=basis.copy()
                mat[:,2]=marker-mat[:,:2]@template_point
                candidates.append(('paired_player',name,mat,passage))
        for old in returned:
            known=self.handoffs.passage_at(floor,level,old.get('player_hd'))
            reverse=(known[0],known[2],known[1]) if known else None
            candidates.append(('paired_return','上次目标层位置',old['matrix'],passage or (reverse if player is None else None)))
        if same_scale:candidates.append(('paired_camera','上下层对应位置',predicted,passage))
        checked=[]
        for method,name,mat,site in candidates:
            if any(np.allclose(mat,old,rtol=0,atol=.02) and (site is not None)==has_site for old,has_site in checked):continue
            checked.append((mat,site is not None))
            hit=self.handoffs.paired_edges(small,geom,valid,floor,mat,factor,visit,site)
            if hit:
                hit.update(search=method,passage=name)
                return hit
        for method,name,mat,site in candidates:
            if site is None or method not in ('paired_player','paired_return'):continue
            aligned=self.handoffs.refine_passage(small,geom,valid,floor,mat,factor,site)
            if aligned is not None:
                hit=self.handoffs.paired_edges(small,geom,valid,floor,aligned,factor,visit,site)
                if hit:
                    hit.update(search=method+'_adjust',passage=name)
                    return hit
        if basis is not None and visit.get('player_hd') is not None:
            offset=passage[2]-passage[1] if passage else self.coordinates.offsets.get((layout,level,target))
            if offset is not None:
                hit=self.handoffs.whole_room_camera(small,geom,valid,floor,basis,factor,visit,
                                                     visit['player_hd']+offset,marker,passage is not None)
                if hit:return hit
        # Even away from the shared passage, a previously seen target room can
        # supply a camera. Verify both the measured screen motion and atlas walls.
        features=None
        for old in return_views:
            if not old or old['floor']['id']!=floor['id'] or not old.get('view_features'):continue
            if features is None:features=ViewFeatures(image,map_area,clear)
            relative=old['view_features'].camera_motion(features)
            if relative is None:continue
            mat=relative[0]@np.vstack((old['matrix'],[0,0,1]))
            hit=self.handoffs.paired_edges(small,geom,valid,floor,mat,factor,visit,matched_return=True)
            if hit:
                hit.update(search='paired_return_view',passage='上次目标层墙角',view_features=features)
                return hit
        if basis is None:return None
        # Keep the old wider search available for a camera reset or a missing marker.
        hit=self.corridors.locate(geom,layout,target,float(np.linalg.norm(basis[0,:2]))*factor,
                                  quick=True,predicted=basis*factor,zoom_ratios=(1.,.90,1.10,.80,1.20),paired=True)
        if hit is not None:
            hit['matrix']/=factor;hit['scale']/=factor
            hit.update(search='paired_corridor',passage='锁定布局·当前层墙线')
        return hit

    @staticmethod
    def follow_anchors(features,floor,anchors,quick=False):
        for anchor in anchors:
            if quick:
                relative=anchor['view_features'].camera_motion(features)
            else:
                relative=anchor['view_features'].motion_to(features)
            if relative is None:continue
            change,count,agreement=relative
            if not quick and abs(change[0,0]-1)>.06 and len(anchors)>1:
                # Correct accumulated scale error against an independently registered,
                # more detailed view if the current map still overlaps it.
                reference=max((v for v in anchors if v.get('independent') or v.get('result')),
                              key=lambda v:v['matrix'][0,0],default=None)
                if reference is not None and reference['matrix'][0,0]>anchor['matrix'][0,0]*1.25:
                    direct=reference['view_features'].motion_to(features)
                    if direct is not None:
                        change,count,agreement=direct
                        anchor=reference
            mat=change@np.vstack((anchor['matrix'],[0,0,1]))
            return dict(accepted=True,floor=floor,matrix=mat,scale=float(np.linalg.norm(mat[0,:2])),
                        inliers=count,score=agreement,consistency=agreement,error_px=None,
                        possible_layouts=[floor['layout_id']],candidates=[],segmentation='view_motion',
                        search=('camera_zoom' if abs(change[0,0]-1)>.06 else 'camera_pan') if quick else
                               ('view_zoom' if abs(change[0,0]-1)>.06 else 'view_motion'),
                        handoff_origin=anchor.get('handoff_origin'))
        return None

    def follow_known(self,features,prior,target,anchors,image=None,source_anchor=None,locked_layout=None):
        """Reuse a measured view of this floor, including one seen before a close."""
        if not prior.get('unique_layout') or prior.get('provisional'):return None
        layout=prior['floor']['layout_id']
        floor=self.handoffs.floors.get(layout+'_'+target)
        if floor is None:return None
        tracked=self.follow_anchors(features,floor,anchors)
        if tracked is not None:return tracked
        predicted,_=self.coordinates.predict(prior,target)
        if predicted is None:return None
        scale=float(np.linalg.norm(predicted[0,:2]))
        fixed=self.fixed_walls.locate(features,floor,scale,confirmed=(layout==locked_layout))
        if fixed:return fixed
        if image is not None and source_anchor and prior['floor']['floor']!=target:
            # A descent plus a pan can move the shared room off the old screen
            # position. Measure its motion and verify the target atlas independently.
            relative=source_anchor['view_features'].motion_to(features)
            if relative is not None:
                mat=relative[0]@np.vstack((source_anchor['matrix'],[0,0,1]))
                transfer=self.handoff(image,dict(prior,matrix=mat),target)
                if transfer and transfer.get('passage')=='重叠视角' and not transfer.get('provisional'):
                    transfer['search']='moving_passage'
                    return transfer
        return None

    def locate_at_new_zoom(self,image,layout,level,reference):
        """Reacquire the known floor after a slider move, independently of the old position."""
        fid,before,old_scale=reference
        floor=self.handoffs.floors.get(fid)
        if not floor or floor['layout_id']!=layout or floor['floor']!=level:return None
        features=ViewFeatures(image,map_area)
        if not zoom_changed(before,features.zoom):return None
        shrinking=features.zoom>before
        bounds=(.16,1.) if shrinking else (1.,6.)
        ratios=np.geomspace(*bounds,26)
        nominal=old_scale*np.sqrt(bounds[0]*bounds[1])
        hit=self.pose_check.hd_walls.locate(features,floor,nominal,confirmed=True,
                                            ratios=ratios*old_scale/nominal)
        if hit and zoom_allows_scale(before,features.zoom,hit['scale']/old_scale):
            hit.update(search='zoom_wall_reset',view_features=features)
            return hit
        return None

    def handoff(self,image,visit,target):
        if not visit or not target or visit['floor']['floor']==target:return None
        poses=visit.get('alternatives') or [visit]
        rows=[]
        for pose in poses:
            found=self.handoffs.inherit(image,pose,target,map_area)
            if not found:return None
            rows.append(found)
        result=dict(rows[0])
        if 'revision' in visit:
            # Keep the source pose, never retain its full-size frame or a chain of visits.
            keys=('floor','matrix','player_hd','at','selected_floor','revision','game','roi','alternatives')
            result['handoff_origin']={key:visit[key] for key in keys if key in visit}
        if len(rows)>1:
            result.update(common_floors=rows,possible_layouts=[a['floor']['layout_id'] for a in rows])
        return result

    def check_search(self):
        callback=getattr(self,'search_guard',None)
        if callback:callback()

    @property
    def search_guard(self):
        return getattr(self._search_thread,'guard',None)

    @search_guard.setter
    def search_guard(self,value):
        self._search_thread.guard=value

    def review_b1(self,image):
        if not hasattr(self,'entrances'):self.entrances=EntranceMatcher(self)
        for width in (960,800):
            hit=self.entrances.locate(image,map_area,None,width)
            if hit is not None:return hit
        compact=self.entrances.compact(image,map_area)
        if compact is not None:return compact
        hit=self._locate_live(image,None,'B1',1280,1024)
        # Feature matching may find several equivalent layouts without
        # registering each row for a consensus overlay.
        return hit if hit['accepted'] and len(hit['possible_layouts'])==1 else None

    def render(self, match, size, style='original_clean'):
        if style == 'structure':
            return super().render(match, size, 'structure')
        if not hasattr(self, 'original'):
            self.original = OriginalMap(self)
        # Older callers may still request routes. All S1 original-art requests
        # now resolve to the clean HD source for the selected floor.
        return self.original.render(match, size)

    def render_live(self, match, frame, style='original_clean', manual=False):
        height,width = frame.shape[:2]
        allowed=None if manual else map_area(frame)
        if allowed is not None and style!='structure':
            x,y,w,h=cv2.boundingRect(allowed)
            rendered=self.original.render(match,(width,height),clip=(x,y,x+w,y+h))
        else:
            rendered=self.render(match,(width,height),style)
        # Leave the actual explored area and its player/item icons visible.
        factor = min(1., 900/max(width,height))
        small = cv2.resize(frame,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        seen = np.uint8(geometry(small)>0)*255
        markers = cv2.inRange(cv2.cvtColor(small,cv2.COLOR_BGR2HSV),(0,115,105),(179,255,255))
        seen = cv2.bitwise_or(seen,markers)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        seen = cv2.morphologyEx(seen,cv2.MORPH_CLOSE,kernel)
        seen = cv2.dilate(seen,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
        keep=cv2.bitwise_not(seen)
        if match.get('common_floors'):
            consensus=None
            for row in match['common_floors']:
                ref=self.template(row['floor'])[1]
                geom=cv2.warpAffine(ref,row['matrix']*factor,(small.shape[1],small.shape[0]),flags=cv2.INTER_NEAREST)
                if consensus is None:consensus=geom
                else:consensus[geom!=consensus]=0
            keep=cv2.bitwise_and(keep,np.uint8(consensus>0)*255)
        keep=cv2.resize(keep,(width,height),interpolation=cv2.INTER_NEAREST)
        if allowed is not None:keep=cv2.bitwise_and(keep,allowed)
        rendered=cv2.bitwise_and(rendered,rendered,mask=keep)
        if match.get('common_floors'):
            match['label_tiles']=[]
            match['visible_rooms']=[]
            return rendered
        if not hasattr(self,'room_labels'): self.room_labels=RoomLabels(self.catalog)
        reserved=cv2.dilate(markers,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        self.room_labels.on_live(rendered,match,allowed,cv2.resize(reserved,(width,height),interpolation=cv2.INTER_NEAREST))
        return rendered

    def locate_selected(self, image, layout, floor_name, scale_hint=None, visit=None):
        if (visit and layout and visit.get('zoom') is not None and visit.get('unique_layout')
                and not visit.get('provisional') and floor_name in ('B2','B3')
                and visit['floor']['id']==layout+'_'+floor_name
                and zoom_changed(visit['zoom'],zoom_position(image))):
            reference=(visit['floor']['id'],visit['zoom'],float(np.linalg.norm(visit['matrix'][0,:2])))
            return self.locate_resized(image,layout,floor_name,reference)
        return self._locate_selected(image,layout,floor_name,scale_hint,visit)

    def locate_resized(self,image,layout,level,reference):
        result=self.locate_at_new_zoom(image,layout,level,reference)
        if result:return result
        self.check_search()
        # Match current walls without the obsolete prior position or scale.
        result=self._locate_selected(image,layout,level)
        if result['accepted'] and not zoom_allows_scale(reference[1],zoom_position(image),
                                                         result['scale']/reference[2]):
            return dict(accepted=False,reason='缩放已变化，等待当前尺寸的固定墙线。')
        return result

    def _locate_selected(self, image, layout, floor_name, scale_hint=None, visit=None):
        if floor_name not in ('B1', 'B2', 'B3'):
            return dict(accepted=False, candidates=[], reason=f'图库未收录 {floor_name}。')
        if floor_name=='B3' and layout:
            hit=self.locate_board(image,layout,visit=visit,zoom=zoom_position(image))
            if hit:return hit
        if floor_name=='B1' and not layout:
            return self.review_b1(image) or dict(accepted=False,candidates=[],reason='当前负一缺少可独立定位的固定墙线。')
        if floor_name=='B2' and not layout and visit and visit['floor']['floor']!='B2':
            scale_hint=float(np.linalg.norm(visit['matrix'][0,:2]))
            visit=None
        if visit and (not layout or layout==visit['floor']['layout_id']):
            transfer=self.handoff(image,visit,floor_name)
            if transfer:return transfer
        if not hasattr(self, 'corridors'):
            self.corridors = CorridorMatcher(self)
        if floor_name=='B1':
            if not hasattr(self,'entrances'):self.entrances=EntranceMatcher(self)
            result=self.entrances.locate(image,map_area,layout)
            if result is None:result=self.entrances.locate(image,map_area,layout,800)
            if result and visit and layout and result.get('common_floors'):
                preferred=visit['floor']['layout_id']
                verified=next((row for row in result['common_floors'] if row['floor']['layout_id']==preferred),None)
                if verified and verified['consistency']>.95:
                    result.update(verified,possible_layouts=[preferred],search='known_entrance')
                    del result['common_floors']
            if result is not None:return result
        if visit and (not layout or layout == visit['floor']['layout_id']):
            if not hasattr(self, 'coordinates'):
                self.coordinates = FloorCoordinates(self.catalog)
            predicted, relation = self.coordinates.predict(visit, floor_name)
            basis = predicted if predicted is not None else np.asarray(visit['matrix'])
            native_scale = float(np.linalg.norm(basis[0,:2]))
            factor = min(1.,1000/image.shape[1])
            small = cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
            geom = geometry(small) & map_area(small)
            preferred = visit['floor']['layout_id']
            # Previous identity is a search preference. A user-selected layout
            # stays a hard limit; automatic discovery can recover a different ID.
            for candidate in ([preferred] if layout else [preferred,None]):
                prior = predicted*factor if predicted is not None and candidate==preferred else None
                result = self.corridors.locate(geom, candidate, floor_name, native_scale*factor,
                                               quick=True, predicted=prior)
                if result is None and candidate==preferred and floor_name=='B3':
                    if not hasattr(self,'room_labels'):self.room_labels=RoomLabels(self.catalog)
                    if not hasattr(self,'stations'):self.stations=StationMatcher(self)
                    target=next(f for f in self.floors if f['layout_id']==preferred and f['floor']=='B3')
                    xy=self.room_labels.points.get(target['id'],{}).get('supply')
                    if xy:
                        anchor=np.asarray(self.room_labels.hd_matrices[target['id']]) @ [*xy,1]
                        station,incomplete=self.stations.locate(image,map_area,target,native_scale,anchor)
                        if station is not None:return station
                        if incomplete:return dict(accepted=False,candidates=[],reason='当前仅露出一处墙角，需要再显示一段负三走廊。')
                zoomed = False
                if result is None and candidate == preferred:
                    result = self.corridors.locate(geom,candidate,floor_name,native_scale*factor,
                                                   quick=True,zoom_ratios=(.68,.78,.86,1.16,1.28,1.42))
                    zoomed = result is not None
                if result is not None:
                    result['matrix'] /= factor
                    result['scale'] /= factor
                    result['search'] = (relation if result['near_prediction'] else 'known_layout') if candidate else 'scale_recovery'
                    if zoomed: result['search'] = 'known_layout_zoom'
                    return result
                if candidate == preferred:
                    feature = self._locate_live(image,preferred,floor_name,1280,1024)
                    if feature['accepted'] and feature['inliers'] >= 16 and feature['consistency'] >= .88:
                        feature['search'] = 'known_layout_features'
                        return feature
        result = self._locate_live(image, layout, floor_name, 1280, 1024)
        if result['accepted'] or result.get('candidates'):
            return result
        geom = geometry(image) & map_area(image)
        sparse = self.corridors.locate(geom, layout, floor_name, scale_hint)
        if sparse is not None:
            return sparse
        for width, features in ((1600,1280), (1920,1600), (2560,1600)):
            result = self._locate_live(image, layout, floor_name, width, features)
            if result['accepted'] or result.get('candidates'): break
        return result

    def locate_live(self, image, layout=None, floor_name=None):
        result = self._locate_live(image, layout, floor_name, 1280, 1024)
        if not result['accepted'] and not result.get('candidates'):
            for width, features in ((1600, 1280), (1920, 1600), (2560, 1600)):
                result = self._locate_live(image, layout, floor_name, width, features)
                if result['accepted'] or result.get('candidates'): break
        return result

    def review_landing(self,image,clear=None,incumbent=None,pending=None):
        # A B2 passage and a previous B3 visit are position hints only. Compare
        # all B3 images on real unpainted pixels, without supplying their IDs.
        full=self._locate_live(image,None,'B3',1280,1024,verification=True,clear=clear)
        full=full if self.broad_b2(full,image.shape) else None
        hit,support=compare_walls(self,image,map_area,clear,full,level='B3',incumbent=incumbent)
        if hit and hit.get('conflicting_models'):return hit
        if not hit or (full and hit['floor']['id']!=full['floor']['id']):return None
        if incumbent not in (None,hit['floor']['layout_id']) and support.get(incumbent,0)>=.70*hit['inliers']:
            return None
        if (full and pending==hit['floor']['layout_id'] and full['verified_rooms']>=6
                and hit['inliers']>=20):
            # A stationary, well-explored view may not reveal new cells. Require
            # each of two disjoint regions to independently beat all 14 rivals.
            axis=int(np.argmax(np.ptp(full['points'],axis=0)))
            cut=round(float(np.median(full['points'][:,axis])))
            mask=map_area(image)
            if clear is not None:mask=cv2.bitwise_and(mask,clear)
            domain=np.indices(mask.shape,sparse=True)[1-axis] < cut
            pts=np.c_[full['points'],np.ones(len(full['points']))].T
            verified=[]
            for side in (domain,~domain):
                region=np.where(side,mask,0).astype(np.uint8)
                local,_=compare_walls(self,image,map_area,region,full,level='B3')
                if not local or local['floor']['id']!=hit['floor']['id']:break
                delta=(local['matrix']-hit['matrix'])@np.linalg.inv(np.vstack((hit['matrix'],[0,0,1])))@pts
                if np.percentile(np.linalg.norm(delta,axis=0),75)>image.shape[0]*.007:break
                verified.append(local['inliers'])
            hit['disjoint_proof']=len(verified)==2
        return hit

    def confirm_b2(self,image,clear=None,incumbent=None,entry=False,local_layout=None):
        # A pose suggested by a prior, a manual selection or an overlay must
        # never count as an independent identification of the round.
        hit=self._locate_live(image,None,'B2',1280,1024,verification=True,clear=clear)
        full=self.broad_b2(hit,image.shape)
        if full and not incumbent and not entry:
            full['proof_kind']='full'
            return full
        wall,support=compare_walls(self,image,map_area,clear,full,incumbent=incumbent)
        if wall and wall.get('conflicting_models'):return wall
        if not wall:
            if full and not entry and full['floor']['layout_id']==incumbent:
                full['proof_kind']='full'
                return full
            return None
        if full and full['floor']['layout_id']!=wall['floor']['layout_id']:
            if incumbent:
                wall['conflicting_models']=True
                return wall
            return None
        if full:
            full.update(proof_kind='full',cells=wall['cells'])
            result=full
        else:result=wall
        result['region_verified']=bool(wall.get('region_verified'))
        if full:
            # The descriptor match and HD wall registration must independently
            # select one layout and agree on where the visible rooms actually are.
            points=np.c_[full['points'],np.ones(len(full['points']))].T
            delta=(wall['matrix']-full['matrix'])@np.linalg.inv(np.vstack((full['matrix'],[0,0,1])))@points
            errors=np.linalg.norm(delta,axis=0)
            result['region_verified'] &= bool(np.percentile(errors,75)<image.shape[0]*.008)
            result['local_verified']=bool(full['verified_rooms']>=5 and wall['inliers']>=18
                                          and len(wall['cells'])>=10 and np.percentile(errors,90)<image.shape[0]*.008)
        result['contradicts']=bool(incumbent and incumbent!=wall['floor']['layout_id'] and
                                   support.get(incumbent,0)<.70*wall['inliers'])
        if not full and wall['floor']['layout_id']!=local_layout and (result['contradicts'] or not incumbent):
            result['local_verified']=check_partial_b2_edges(self,image,map_area,clear,wall,support)
        return result

    @staticmethod
    def broad_b2(hit,shape):
        if not hit['accepted'] or len(hit['possible_layouts'])!=1:return None
        h,w=shape[:2]
        span=np.ptp(hit['points'],axis=0)
        rival=max((v['score'] for v in hit['candidates'][1:]),default=0.)
        # A distributed six-room registration can survive small differences
        # in atlas rasterization even with one fewer descriptor correspondence.
        broad=(hit['verified_rooms']>=6 and hit['consistency']>=.94 and
               hit['error_px']<=2. and rival<hit['score']*.5)
        if (hit['verified_rooms']<5 or (hit['inliers']<14 and not broad) or hit['consistency']<.88 or
                span[0]<w*.14 or span[1]<h*.16):return None
        return hit

    def _locate_live(self, image, layout, floor_name, width, features,verification=False,clear=None):
        factor = min(1., width/image.shape[1])
        small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        geom = geometry(small, True)
        area = map_area(small)
        if clear is not None:
            mask=cv2.resize(clear,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_AREA)
            area &= cv2.erode(np.uint8(mask==255)*255,np.ones((7,7),np.uint8))
        room=((geom==127)&(area>0)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(room,8)
        if verification and np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>=150)<5:
            return dict(accepted=False,candidates=[],reason='可核对的负二固定房间不足。')
        feature_factor = min(1., features/max(geom.shape))
        fg = cv2.resize(geom, None, fx=feature_factor, fy=feature_factor)
        fm = cv2.resize(area, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST)
        kp, des = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.015).detectAndCompute(fg, fm)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 2)/feature_factor
        # Validate on independent pixels, including multiple separate rooms.
        # Live partially explored maps contain many unmatched fog/UI corners;
        # use a lower RANSAC fraction ONLY with the stronger checks below.
        rows = []
        for floor in self.floors:
            if layout and floor['layout_id'] != layout: continue
            if floor_name and floor['floor'] != floor_name: continue
            if not verification:self.check_search()
            _, ref, (rp, rd) = self.template(floor)
            match = register(rp, rd, pts, des, min_fraction=.20)
            if match is None or match['inliers'] < 12: continue
            pred = cv2.warpAffine(ref, match['matrix'], (geom.shape[1], geom.shape[0]), flags=cv2.INTER_NEAREST)
            near = cv2.dilate((pred > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            observed = (geom > 0) & near & (area > 0)
            agreement = (abs(pred.astype(np.int16)-geom.astype(np.int16)) < 35) & observed
            consistency = np.count_nonzero(agreement)/max(1, np.count_nonzero(observed))
            if consistency < .80: continue
            verified_rooms = 0
            for i in range(1, count):
                if stats[i, cv2.CC_STAT_AREA] < max(150, 300*match['scale']**2): continue
                component = labels == i
                if np.count_nonzero(component & (pred == 127))/stats[i, cv2.CC_STAT_AREA] > .72:
                    verified_rooms += 1
            if verified_rooms < 3: continue
            match.update(floor=floor, consistency=float(consistency), verified_rooms=verified_rooms)
            match['score'] *= consistency
            rows.append(match)
        rows.sort(key=lambda r:r['score'], reverse=True)
        if not rows:
            return dict(accepted=False, candidates=[], reason='局部结构不足以可靠确认布局。')
        best = rows[0]
        equivalent = []
        for other in rows[1:]:
            if other['score'] <= best['score']*.72: continue
            if other['floor']['floor'] != best['floor']['floor'] or not self.equivalent(best, other):
                return dict(accepted=False, candidates=rows[:5], reason='相似布局尚未区分。')
            equivalent.append(other)
        # One conversion for every row; best is also rows[0].
        for row in rows:
            row['matrix'] /= factor
            row['points'] /= factor
            row['scale'] /= factor
            row['error_px'] /= factor
        best.update(accepted=True, candidates=rows[:5], reason='已匹配',
                    segmentation='live_hud_masked',
                    possible_layouts=[r['floor']['layout_id'] for r in [best]+equivalent])
        return best

"""Live-client matching: exclude HUD, validate separate rooms, retain atlas rules."""
from __future__ import annotations
import cv2
import numpy as np
from .s1_alignment import FloorMatcher, geometry, register
from .s1_floor_transition import CorridorMatcher, FloorCoordinates
from .s1_original_map import OriginalMap
from .s1_landmarks import RoomLabels
from .s1_entrance import EntranceMatcher


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
    def check_search(self):
        callback=getattr(self,'search_guard',None)
        if callback:callback()

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
        rendered = self.render(match,(width,height),style)
        allowed = None if manual else map_area(rendered)
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
        if not hasattr(self,'room_labels'): self.room_labels=RoomLabels(self.catalog)
        reserved=cv2.dilate(markers,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        self.room_labels.on_live(rendered,match,allowed,cv2.resize(reserved,(width,height),interpolation=cv2.INTER_NEAREST))
        return rendered

    def locate_selected(self, image, layout, floor_name, scale_hint=None, visit=None):
        if floor_name not in ('B1', 'B2', 'B3'):
            return dict(accepted=False, candidates=[], reason=f'图库未收录 {floor_name}。')
        if not hasattr(self, 'corridors'):
            self.corridors = CorridorMatcher(self)
        if floor_name=='B1':
            if not hasattr(self,'entrances'):self.entrances=EntranceMatcher(self)
            result=self.entrances.locate(image,map_area,layout)
            if result is None:result=self.entrances.locate(image,map_area,layout,800)
            if result and visit and result.get('common_floors'):
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

    def _locate_live(self, image, layout, floor_name, width, features):
        factor = min(1., width/image.shape[1])
        small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        geom = geometry(small, True)
        area = map_area(small)
        feature_factor = min(1., features/max(geom.shape))
        fg = cv2.resize(geom, None, fx=feature_factor, fy=feature_factor)
        fm = cv2.resize(area, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST)
        kp, des = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.015).detectAndCompute(fg, fm)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 2)/feature_factor
        # Validate on independent pixels, including multiple separate rooms.
        # Live partially explored maps contain many unmatched fog/UI corners;
        # use a lower RANSAC fraction ONLY with the stronger checks below.
        room = ((geom == 127) & (area > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(room, 8)
        rows = []
        for floor in self.floors:
            if layout and floor['layout_id'] != layout: continue
            if floor_name and floor['floor'] != floor_name: continue
            self.check_search()
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

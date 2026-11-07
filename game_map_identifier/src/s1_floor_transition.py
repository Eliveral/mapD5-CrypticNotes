"""Read the selected floor and align a fog-limited corridor against floor maps."""
from __future__ import annotations

import cv2
import numpy as np
import json
from .s1_passage_data import PASSAGES


class FloorCoordinates:
    """Transform between templates only for calibrated pairs of atlas floors."""

    def __init__(self, catalog):
        config = catalog.root.parent.parent/'config'
        self.matrices = {}
        self.offsets = {}
        self.passages = {}
        path = config/'s1_floor_relations.json'
        registration = config/'s1_hd_registration.json'
        if not path.is_file() or not registration.is_file():
            return  # An older installation can still match without positional hints.
        self.matrices = json.loads(registration.read_text(encoding='utf-8'))
        document=json.loads(path.read_text(encoding='utf-8'))
        for pair in document['pairs']:
            layout, a, b = pair['layout'], pair['source'], pair['target']
            first = catalog.by_id.get(f"hd_{layout[3:]}_{a}", {})
            second = catalog.by_id.get(f"hd_{layout[3:]}_{b}", {})
            if first.get('sha256') != pair['source_sha'] or second.get('sha256') != pair['target_sha']:
                continue
            anchors = np.asarray(pair['anchors_hd'], dtype=np.float64)
            offset = np.median(anchors[:,1]-anchors[:,0], axis=0)
            self.offsets[layout,a,b] = offset
            self.offsets[layout,b,a] = -offset
        for layout,group in {**PASSAGES,**document.get('passages',{})}.items():
            for row in group['pairs']:
                a,b=row['source'],row['target']
                if any(catalog.by_id.get('hd_'+layout[3:]+'_'+lv,{}).get('sha256')!=group['sha'].get(lv) for lv in (a,b)):continue
                pa,pb=np.asarray(row['a'],float),np.asarray(row['b'],float)
                for src,dst,start,end in [(a,b,pa,pb),(b,a,pb,pa)]:
                    self.passages.setdefault((layout,src,dst),[]).append((row['name'],start,end))
                    self.offsets.setdefault((layout,src,dst),end-start)

    def predict(self, visit, target):
        source = visit['floor']
        matrix = np.float64(visit['matrix'])
        if source['floor'] == target:
            return matrix, 'visited_floor'
        layout, level = source['layout_id'], source['floor']
        offset = self.offsets.get((layout,level,target))
        if offset is None:
            for middle in ('B1','B2','B3'):
                first = self.offsets.get((layout,level,middle))
                second = self.offsets.get((layout,middle,target))
                if first is not None and second is not None:
                    offset = first + second
                    break
        if offset is None or source['id'] not in self.matrices or f'{layout}_{target}' not in self.matrices:
            return None, None
        # Coordinates are in native HD pixels, not in the differently cropped
        # atlas templates. The game's camera may move, so this is only a proposal.
        prev = np.vstack([self.matrices[source['id']], [0,0,1]])
        nxt = np.vstack([self.matrices[f'{layout}_{target}'], [0,0,1]])
        translate = np.eye(3); translate[:2,2] = -offset
        predicted = np.vstack([matrix,[0,0,1]]) @ prev @ translate @ np.linalg.inv(nxt)
        return predicted[:2], 'atlas_relation'


def selected_floor(image):
    height, width = image.shape[:2]
    center = width - round(.22 * height)
    if center < 0 or height < 240:
        return None
    votes = []
    # Sample the panel on either side of the number, outside the glyph.
    for offset in (-.014, .014):
        x = center + round(offset * height)
        scores = []
        for y_fraction in (.180, .259, .340, .420):
            y = round(height * y_fraction)
            radius = max(2, round(.005 * height))
            patch = image[y-radius:y+radius, x-radius:x+radius]
            if not patch.size:
                return None
            scores.append(float(np.median(patch[:, :, 0])))
        order = np.argsort(scores)
        top, second = int(order[-1]), int(order[-2])
        if scores[top] - scores[second] < 18 or scores[top] < 1.35 * scores[second]:
            return None
        votes.append(top)
    return f'B{votes[0]+1}' if votes[0] == votes[1] else None


class CorridorMatcher:
    grid = .40

    def __init__(self, matcher):
        self.matcher = matcher
        self.reference = {}

    def reference_channels(self, floor):
        if floor['id'] not in self.reference:
            geom = self.matcher.template(floor)[1]
            grid = cv2.resize(geom, None, fx=self.grid, fy=self.grid, interpolation=cv2.INTER_NEAREST)
            self.reference[floor['id']] = [(grid == value).astype(np.float32) for value in (127, 255, 0)]
        return self.reference[floor['id']]

    def locate(self, geom, layout, floor_name, scale_hint=None, *, quick=False, predicted=None,
               zoom_ratios=(.94,1.,1.06), paired=False):
        if floor_name not in ('B1', 'B2', 'B3'):
            return None
        _, labels, stats, _ = cv2.connectedComponentsWithStats((geom > 0).astype(np.uint8))
        if len(stats) < 2:
            return None
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, _ = stats[main]
        if min(width, height) < (30 if paired else 65):
            return None
        crop = geom[y:y+height, x:x+width].copy()
        crop[labels[y:y+height, x:x+width] != main] = 0
        hull = np.zeros_like(crop)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(crop)), 1)
        masks = [crop == 127, crop == 255, (hull != 0) & (crop == 0)]
        if any(np.count_nonzero(mask) < max(45 if paired else 150, crop.size * .045) for mask in masks):
            return None
        channels = [mask.astype(np.float32) for mask in masks]
        floors = [f for f in self.matcher.floors if f['floor'] == floor_name and (not layout or f['layout_id'] == layout)]
        if quick and (not scale_hint or not .10 < scale_hint < 5):
            return None
        scales = [scale_hint*d for d in zoom_ratios] if quick else list(np.geomspace(.45, 3.8, 14))
        if not quick and scale_hint and .15 < scale_hint < 5:
            scales.extend(scale_hint * d for d in (.94, 1., 1.06))
        placements = []

        def search(scale, candidates, center=None):
            size = (round(width*self.grid/scale), round(height*self.grid/scale))
            if min(size) < 12:
                return
            query = [cv2.resize(channel, size, interpolation=cv2.INTER_AREA) for channel in channels]
            norms = [float(q.sum()) for q in query]
            for floor in candidates:
                self.matcher.check_search()
                refs = self.reference_channels(floor)
                left = top = 0
                if center is not None:
                    # Try a neighbourhood around the corresponding point on the
                    # previous floor before expanding to the whole target floor.
                    origin = center*self.grid - np.array(size)/2
                    margin = max(8,round(min(size)*.22))
                    left, top = np.maximum(0,np.floor(origin-margin).astype(int))
                    right, bottom = np.ceil(origin+np.array(size)+margin).astype(int)
                    if right <= left or bottom <= top: continue
                    refs = [r[top:bottom,left:right] for r in refs]
                if size[0] > refs[0].shape[1] or size[1] > refs[0].shape[0]:
                    continue
                scores = [cv2.matchTemplate(r, q, cv2.TM_CCORR) / n for r,q,n in zip(refs,query,norms)]
                joint = scores[0]*.28 + scores[1]*.44 + scores[2]*.28
                for _ in range(2):
                    _, best, _, (px,py) = cv2.minMaxLoc(joint)
                    if best < .69: break
                    agreement = tuple(float(s[py,px]) for s in scores)
                    placements.append((best, floor, scale, px+left, py+top, agreement))
                    margin = max(4, round(min(size)*.18))
                    joint[max(0,py-margin):py+margin+1,max(0,px-margin):px+margin+1] = -1

        local_center = None
        if quick and predicted is not None:
            center = cv2.invertAffineTransform(predicted) @ np.float64([x+width/2,y+height/2,1])
            search(scale_hint, floors, center)
            if placements and max(row[0] for row in placements) >= .85:
                local_center = center
                # One full-floor search at this scale checks for a repeated
                # location outside the proposed neighbourhood before acceptance.
                scales = [scale_hint]
            placements.clear()
        for scale in scales:
            search(float(scale), floors)
        if not placements:
            return None
        placements.sort(key=lambda row:row[0], reverse=True)
        # Refine the best scale of each competitive floor. A previous floor
        # supplies scale only. Its layout and its translation are never forced.
        refined = set()
        for score, floor, scale, *_ in list(placements):
            if score < placements[0][0]-.09: break
            if floor['id'] not in refined:
                refined.add(floor['id'])
                factors = () if local_center is not None else ((.97,1.03) if quick else (.94,.97,1.03,1.06))
                for factor in factors: search(scale*factor, [floor])
        placements.sort(key=lambda row:row[0], reverse=True)
        score, floor, scale, px, py, agreement = placements[0]
        # Compare different placements, including two sites within one floor.
        def placement_center(row):
            return np.array([row[3],row[4]])/self.grid + np.array([width,height])/(2*row[2])
        winner_center = placement_center(placements[0])
        rival = max((r[0] for r in placements[1:] if r[1]['id'] != floor['id'] or
                     np.linalg.norm(placement_center(r)-winner_center) > max(14, .18*min(width,height)/scale)), default=0.)
        # Fog in the convex hull can conceal a legitimate room. Require strong
        # observed room and corridor agreement; keep the existing void threshold.
        weak_structure = quick and (agreement[0] < .84 or agreement[1] < .84)
        if score < (.85 if quick else .82) or score-rival < (.065 if quick else .055) or min(agreement) < .62 or weak_structure:
            return None
        matrix = np.float64([[scale,0,x-scale*px/self.grid], [0,scale,y-scale*py/self.grid]])
        return dict(accepted=True, floor=floor, matrix=matrix, scale=scale,
                    score=score, consistency=score, rival=rival, inliers=0, error_px=None,
                    possible_layouts=[floor['layout_id']], candidates=[], reason='已匹配局部走廊',
                    segmentation='corridor_alignment',
                    near_prediction=bool(local_center is not None and
                        np.linalg.norm(winner_center-local_center) < max(20,.24*min(width,height)/scale)))

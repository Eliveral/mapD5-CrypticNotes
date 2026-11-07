"""S1 floor-level registration. All transforms map native template pixels to capture pixels."""
from __future__ import annotations
import json
import math
import cv2
import numpy as np
from .s1_catalog import read_image


def geometry(image, suppress_gradient=True):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    room = cv2.inRange(hsv, (4, 25, 45), (33, 200, 255))
    hall = cv2.inRange(hsv, (95, 25, 45), (135, 230, 255))
    # Live UI has a broad blue-grey lighting gradient, not the atlas's flat
    # background. Its saturation exceeds its value; actual stone corridors
    # are brighter and less saturated. Brightness-only Otsu admits the entire
    # gradient as a corridor. Verified against the user's live-map failure.
    if suppress_gradient:
        hall[hsv[:, :, 1] > hsv[:, :, 2]] = 0
    # Atlas backgrounds and explored blue corridors share hue. Separate the
    # dark background population before feature extraction (brightness varies).
    blue_values = hsv[:, :, 2][(hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 1] <= 230)]
    if blue_values.size > 100:
        values = blue_values[blue_values < 125]
        if values.size > 100 and np.percentile(values, 99)-np.percentile(values, 1) > 16:
            threshold, _ = cv2.threshold(values.reshape(-1,1), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            hall[hsv[:, :, 2] <= threshold] = 0
    k = np.ones((5, 5), np.uint8)
    room = cv2.morphologyEx(room, cv2.MORPH_CLOSE, k)
    hall = cv2.morphologyEx(hall, cv2.MORPH_CLOSE, k)
    result = cv2.morphologyEx(np.maximum(room//2, hall), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) if suppress_gradient else result


def descriptors(image, suppress_gradient=True):
    gray = geometry(image, suppress_gradient)
    factor = min(1., 1600/max(gray.shape))
    if factor < 1:
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.015).detectAndCompute(gray, None)
    # Return native coordinates, never allow resized feature coordinates to leak.
    pts = np.float32([k.pt for k in kp]).reshape(-1, 2)/factor
    return pts, des


def register(ref_pts, ref_des, query_pts, query_des, min_fraction=.40):
    if ref_des is None or query_des is None or min(len(ref_pts), len(query_pts)) < 12:
        return None
    pairs = cv2.BFMatcher().knnMatch(query_des, ref_des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .70*p[1].distance]
    # SIFT produces multiple orientations at one corner; count unique positions.
    unique = {}
    for m in sorted(good, key=lambda m: m.distance):
        key = tuple(np.round(ref_pts[m.trainIdx]/3).astype(int))
        if key not in unique:
            unique[key] = m
    good = list(unique.values())
    if len(good) < 10:
        return None
    src = np.float32([ref_pts[m.trainIdx] for m in good])
    dst = np.float32([query_pts[m.queryIdx] for m in good])
    mat, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=3, maxIters=4000, confidence=.995)
    if mat is None or mask is None:
        return None
    keep = mask.ravel().astype(bool)
    n = int(keep.sum())
    scale = float(np.hypot(mat[0, 0], mat[1, 0]))
    angle = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))
    if n < 10 or n/len(good) < min_fraction or not .15 < scale < 5 or abs(angle) > 4:
        return None
    hull = cv2.convexHull(dst[keep])
    extent = np.ptp(dst[keep], axis=0)
    if min(extent) < 65 or cv2.contourArea(hull) < 6000:
        return None
    err = float(np.median(np.linalg.norm(src[keep]@mat[:, :2].T+mat[:, 2]-dst[keep], axis=1)))
    return dict(matrix=mat, inliers=n, error_px=err,
                score=n/(1+err), points=dst[keep], scale=scale)


class FloorMatcher:
    def __init__(self, catalog, suppress_gradient=True):
        self.catalog = catalog
        self.suppress_gradient = suppress_gradient
        self.flat_matcher = None
        self.spec = json.loads((catalog.root/'floors.json').read_text(encoding='utf-8'))
        self.floors = self.spec['floors']
        self.refs = {}

    def template(self, floor):
        fid = floor['id']
        if fid not in self.refs:
            e = self.catalog.by_id[floor['entry_id']]
            x0, y0, x1, y1 = floor['box']
            im = read_image(self.catalog.path(e))[y0:y1, x0:x1]
            self.refs[fid] = (im, geometry(im, self.suppress_gradient), descriptors(im, self.suppress_gradient))
        return self.refs[fid]

    def warmup(self):
        for floor in self.floors:
            self.template(floor)

    def locate(self, image, layout=None, floor_name=None):
        if not self.suppress_gradient:
            result = self._locate(image, layout, floor_name)
            result['segmentation'] = 'flat_background'
            return result
        if self.flat_matcher is None:
            self.flat_matcher = FloorMatcher(self.catalog, suppress_gradient=False)
        # Keep the existing flat-background representation for flat atlases.
        # Switch when a substantial part of the "corridor" mask is actually
        # dark saturated blue background. This avoids erasing valid atlas
        # details merely to correct the live client's gradient.
        old_mask = geometry(image, False)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gradient_fraction = float(np.mean((old_mask == 255) & (hsv[:, :, 1] > hsv[:, :, 2])))
        primary, secondary = (self, self.flat_matcher) if gradient_fraction > .01 else (self.flat_matcher, self)
        result = primary._locate(image, layout, floor_name)
        result['segmentation'] = 'gradient_background' if primary is self else 'flat_background'
        # Never resolve an AMBIGUOUS result through a different representation.
        if not result['accepted'] and not result.get('candidates'):
            fallback = secondary._locate(image, layout, floor_name)
            fallback['segmentation'] = 'gradient_background' if secondary is self else 'flat_background'
            if fallback['accepted']:
                result = fallback
        result['gradient_fraction'] = gradient_fraction
        return result

    def _locate(self, image, layout=None, floor_name=None):
        pts, des = descriptors(image, self.suppress_gradient)
        rows = []
        qgeom = geometry(image, self.suppress_gradient)
        for floor in self.floors:
            if layout and floor['layout_id'] != layout:
                continue
            if floor_name and floor['floor'] != floor_name:
                continue
            im, geom, (rp, rd) = self.template(floor)
            match = register(rp, rd, pts, des)
            if match is None:
                continue
            mat = match['matrix']
            pred = cv2.warpAffine(geom, mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            # Verify observed structure near the matched floor, tolerating icons
            # and fog but rejecting repeated corners on a different topology.
            bounds = cv2.dilate((pred > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            observed = (qgeom > 0) & bounds
            agreement = (abs(pred.astype(np.int16)-qgeom.astype(np.int16)) < 35) & observed
            consistency = float(agreement.sum()/max(1, observed.sum()))
            if consistency < .60:
                continue
            # A single shared entrance chamber is not enough evidence. Require
            # room-coloured observations outside that chamber as well.
            observed_rooms = (pred == 127) & (qgeom == 127)
            if int(observed_rooms.sum()) < max(400, 1000*match['scale']**2):
                continue
            match.update(floor=floor, consistency=consistency)
            match['score'] *= consistency
            rows.append(match)
        rows.sort(key=lambda r: r['score'], reverse=True)
        if not rows:
            return dict(accepted=False, reason='未找到足够的地图结构；请打开地图或露出更多走廊。', candidates=[])
        best = rows[0]
        competitors = [r for r in rows[1:] if r['score'] > best['score']*.72]
        equivalents = []
        for r in competitors:
            # Identical shared floors are useful even when layout ID is unknown.
            # Compare both entire projected floor masks, not just the seen crop.
            if r['floor']['floor'] != best['floor']['floor'] or not self.equivalent(best, r):
                return dict(accepted=False, reason='相似布局证据不足，已隐藏叠加；请露出更多结构或限定已知布局。', candidates=rows[:5])
            equivalents.append(r)
        best.update(accepted=True, candidates=rows[:5],
                    possible_layouts=[r['floor']['layout_id'] for r in [best]+equivalents],
                    reason='已匹配')
        return best

    def equivalent(self, a, b):
        ga = self.template(a['floor'])[1]
        gb = self.template(b['floor'])[1]
        am = np.vstack((a['matrix'], [0, 0, 1]))
        bm = np.vstack((b['matrix'], [0, 0, 1]))
        rel = (np.linalg.inv(am)@bm)[:2]
        # Use padded full template space so off-screen differences still count.
        pad = 1000
        rel[:, 2] += pad
        shape = (ga.shape[1]+2*pad, ga.shape[0]+2*pad)
        other = cv2.warpAffine(gb, rel, shape, flags=cv2.INTER_NEAREST)
        own = np.pad(ga, pad)
        union = (own > 0) | (other > 0)
        same = (abs(own.astype(np.int16)-other.astype(np.int16)) < 35) & (own > 0) & (other > 0)
        return float(same.sum()/max(1, union.sum())) > .96

    def render(self, match, size, style='structure'):
        floor = match['floor']
        im, geom, _ = self.template(floor)
        drawn = np.zeros_like(im)
        drawn[geom == 127] = (80, 175, 245)
        drawn[geom == 255] = (235, 190, 80)
        return cv2.warpAffine(drawn, match['matrix'], tuple(size), flags=cv2.INTER_LINEAR)

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .deep_matcher import deep_embedding_score
from .map_database import MapDatabase, MapTemplate
from .preprocess import PreprocessConfig, crop_visible_structure, locate_door_anchor, preprocess_image

LOGGER = logging.getLogger(__name__)


@dataclass
class MatchResult:
    map_id: str
    map_name: str
    color_score: float
    ssim_score: float
    deep_score: float
    candidate_score: float
    template_score: float
    feature_score: float
    structure_score: float
    partial_structure_score: float
    entrance_patch_score: float
    entrance_shape_score: float
    entrance_direction_score: float
    entrance_corridor_pair_score: float
    anchored_subset_score: float
    corridor_component_score: float
    entry_corridor_score: float
    door_anchor_score: float
    final_score: float
    cnn_score: float = 0.0
    cnn_rank: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    best_scale: float = 1.0
    is_reliable: bool = False
    reason: str = ""
    deep_backend: str = ""
    sample_id: str = ""
    source: str = "map_template"
    roi_version: str = ""
    support_count: int = 0
    sample_count: int = 0
    structure_evaluated: bool = False


@dataclass
class MatchSummary:
    results: List[MatchResult]
    top3: List[MatchResult]
    top_candidates: List[MatchResult]
    best: Optional[MatchResult]
    is_reliable: bool
    status: str
    current_preprocessed: np.ndarray
    best_template_preprocessed: Optional[np.ndarray] = None
    current_cropped_roi: Optional[np.ndarray] = None
    current_crop_bbox: Optional[tuple[int, int, int, int]] = None


def _scale_values(min_scale: float, max_scale: float, step: float) -> List[float]:
    if step <= 0:
        return [1.0]
    values: List[float] = []
    s = min_scale
    while s <= max_scale + 1e-9:
        values.append(round(s, 4))
        s += step
    if 1.0 not in values:
        values.append(1.0)
    return sorted(set(values))


def template_match_score(current: np.ndarray, template: np.ndarray, app_config: Mapping[str, Any]) -> tuple[float, float]:
    """Return (score, best_scale). Uses normalized correlation over multi-scale templates."""
    if current.size == 0 or template.size == 0:
        return 0.0, 1.0
    scales = [1.0]
    if bool(app_config.get("enable_multiscale", True)):
        scales = _scale_values(
            float(app_config.get("scale_min", 0.8)),
            float(app_config.get("scale_max", 1.2)),
            float(app_config.get("scale_step", 0.05)),
        )

    best = -1.0
    best_scale = 1.0
    cur_h, cur_w = current.shape[:2]
    for scale in scales:
        resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tpl_h, tpl_w = resized.shape[:2]
        if tpl_h < 8 or tpl_w < 8:
            continue
        # matchTemplate requires the search image to be at least as large as template.
        if cur_h < tpl_h or cur_w < tpl_w:
            search = resized
            tpl = current
        else:
            search = current
            tpl = resized
        if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        score = float(max(0.0, max_val))
        if score > best:
            best = score
            best_scale = scale
    return max(0.0, best), best_scale


def _prepare_color_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (3, 3), 0)


def color_pixel_score(current: np.ndarray, template: np.ndarray, app_config: Mapping[str, Any]) -> tuple[float, float]:
    """Compare color/texture similarity without binarizing map structure.

    This is intentionally useful for the new full3 templates, which resemble the
    in-game minimap more closely. It keeps color and soft edges, then blends
    direct pixel distance, HSV histogram overlap, and normalized correlation.
    """
    if current.size == 0 or template.size == 0:
        return 0.0, 1.0
    size_raw = app_config.get("color_match_size", [160, 160])
    size = (int(size_raw[0]), int(size_raw[1]))
    cur = _prepare_color_image(current, size)
    scales = [1.0]
    if bool(app_config.get("enable_multiscale", True)):
        scales = _scale_values(
            float(app_config.get("scale_min", 0.8)),
            float(app_config.get("scale_max", 1.2)),
            float(app_config.get("scale_step", 0.05)),
        )

    cur_lab = cv2.cvtColor(cur, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    best = 0.0
    best_scale = 1.0
    for scale in scales:
        tpl_size = (max(16, int(size[0] * scale)), max(16, int(size[1] * scale)))
        tpl = _prepare_color_image(template, tpl_size)
        if tpl.shape[:2] != cur.shape[:2]:
            if tpl.shape[0] <= cur.shape[0] and tpl.shape[1] <= cur.shape[1]:
                res = cv2.matchTemplate(cur, tpl, cv2.TM_CCOEFF_NORMED)
                _, corr, _, loc = cv2.minMaxLoc(res)
                x, y = loc
                cur_cmp = cur[y : y + tpl.shape[0], x : x + tpl.shape[1]]
            else:
                tpl_cmp = _prepare_color_image(template, size)
                cur_cmp = cur
                corr = float(cv2.matchTemplate(cur_gray, cv2.cvtColor(tpl_cmp, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)[0, 0])
                tpl = tpl_cmp
        else:
            cur_cmp = cur
            corr = float(cv2.matchTemplate(cur_gray, cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)[0, 0])

        tpl_lab = cv2.cvtColor(tpl, cv2.COLOR_BGR2LAB).astype(np.float32)
        if tpl_lab.shape != cur_lab.shape:
            cur_lab_cmp = cv2.cvtColor(cur_cmp, cv2.COLOR_BGR2LAB).astype(np.float32)
        else:
            cur_lab_cmp = cur_lab
        mean_delta = float(np.mean(np.linalg.norm(cur_lab_cmp - tpl_lab, axis=2))) / 255.0
        pixel_score = float(np.exp(-mean_delta / 0.28))

        tpl_hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
        cur_hsv_cmp = cv2.cvtColor(cur_cmp, cv2.COLOR_BGR2HSV) if cur_cmp.shape[:2] != cur_hsv.shape[:2] else cur_hsv
        cur_hist = cv2.calcHist([cur_hsv_cmp], [0, 1], None, [24, 16], [0, 180, 0, 256])
        tpl_hist = cv2.calcHist([tpl_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(cur_hist, cur_hist, alpha=1.0, norm_type=cv2.NORM_L1)
        cv2.normalize(tpl_hist, tpl_hist, alpha=1.0, norm_type=cv2.NORM_L1)
        hist_score = float(max(0.0, cv2.compareHist(cur_hist, tpl_hist, cv2.HISTCMP_INTERSECT)))

        corr_score = max(0.0, min(1.0, float(corr)))
        score = 0.50 * pixel_score + 0.25 * hist_score + 0.25 * corr_score
        if score > best:
            best = score
            best_scale = scale
    return float(max(0.0, min(1.0, best))), best_scale


def ssim_score(current: np.ndarray, template: np.ndarray, app_config: Mapping[str, Any]) -> float:
    """Structural similarity on grayscale map texture without hard binarization."""
    if current.size == 0 or template.size == 0:
        return 0.0
    size_raw = app_config.get("ssim_match_size", app_config.get("color_match_size", [160, 160]))
    size = (int(size_raw[0]), int(size_raw[1]))
    cur = _prepare_color_image(current, size)
    tpl = _prepare_color_image(template, size)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = (7, 7)
    mu_cur = cv2.GaussianBlur(cur_gray, kernel, 1.5)
    mu_tpl = cv2.GaussianBlur(tpl_gray, kernel, 1.5)
    sigma_cur = cv2.GaussianBlur(cur_gray * cur_gray, kernel, 1.5) - mu_cur * mu_cur
    sigma_tpl = cv2.GaussianBlur(tpl_gray * tpl_gray, kernel, 1.5) - mu_tpl * mu_tpl
    sigma_cur_tpl = cv2.GaussianBlur(cur_gray * tpl_gray, kernel, 1.5) - mu_cur * mu_tpl
    numerator = (2.0 * mu_cur * mu_tpl + c1) * (2.0 * sigma_cur_tpl + c2)
    denominator = (mu_cur * mu_cur + mu_tpl * mu_tpl + c1) * (sigma_cur + sigma_tpl + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-6)
    raw = float(np.mean(ssim_map))
    return max(0.0, min(1.0, (raw + 1.0) * 0.5))


def feature_match_score(
    current: np.ndarray,
    template: MapTemplate,
    orb: Optional[cv2.ORB],
    app_config: Mapping[str, Any],
) -> tuple[float, int, int, float]:
    if orb is None or template.descriptors is None or len(template.keypoints) < 2:
        return 0.0, 0, 0, 0.0

    kp1, des1 = orb.detectAndCompute(current, None)
    if des1 is None or not kp1 or len(kp1) < 2:
        return 0.0, 0, 0, 0.0

    try:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(des1, template.descriptors, k=2)
    except cv2.error:
        return 0.0, 0, 0, 0.0

    ratio = float((app_config.get("orb", {}) or {}).get("ratio_test", 0.75))
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    min_good = int((app_config.get("orb", {}) or {}).get("min_good_matches", 8))
    if len(good) < min_good:
        # Still give a small continuous score so weak but non-zero evidence is visible.
        return min(0.35, len(good) / max(min_good, 1) * 0.35), len(good), 0, 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([template.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    inliers = 0
    inlier_ratio = 0.0
    if len(good) >= 4:
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.ravel().sum())
            inlier_ratio = inliers / max(len(good), 1)

    # Combine amount and geometric consistency. Saturates so maps with many features don't dominate excessively.
    count_score = min(1.0, len(good) / 45.0)
    inlier_score = min(1.0, inlier_ratio)
    score = 0.45 * count_score + 0.55 * inlier_score
    return float(score), len(good), inliers, float(inlier_ratio)


def edge_distance_score(current: np.ndarray, template: np.ndarray) -> float:
    """Compare binary edge maps using symmetric distance transform similarity."""
    if current.size == 0 or template.size == 0:
        return 0.0
    if current.shape != template.shape:
        template = cv2.resize(template, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)
    cur = (current > 0).astype(np.uint8)
    tpl = (template > 0).astype(np.uint8)
    if int(cur.sum()) < 8 or int(tpl.sum()) < 8:
        return 0.0

    # Distance to nearest edge. For each edge in A, ask how far it is from B.
    dt_tpl = cv2.distanceTransform(1 - tpl, cv2.DIST_L2, 3)
    dt_cur = cv2.distanceTransform(1 - cur, cv2.DIST_L2, 3)
    d1 = float(np.mean(dt_tpl[cur > 0]))
    d2 = float(np.mean(dt_cur[tpl > 0]))
    diag = float(np.hypot(*current.shape[:2]))
    norm_d = (d1 + d2) / 2.0 / max(diag * 0.08, 1.0)
    distance_score = float(np.exp(-norm_d))

    overlap = float(np.logical_and(cur > 0, tpl > 0).sum()) / max(float(np.logical_or(cur > 0, tpl > 0).sum()), 1.0)
    return max(0.0, min(1.0, 0.75 * distance_score + 0.25 * overlap))


def partial_structure_score(current: np.ndarray, template: np.ndarray, app_config: Mapping[str, Any]) -> float:
    """Score whether the visible current structure appears as a sub-structure of template.

    This is important for fog-of-war maps: the current screenshot often contains
    only a long corridor and the first junction, while the template ROI contains
    additional rooms that have not been explored yet. Symmetric full-ROI matching
    would wrongly penalize those extra template structures.
    """
    if current.size == 0 or template.size == 0:
        return 0.0
    cur = (current > 0).astype(np.uint8) * 255
    tpl = (template > 0).astype(np.uint8) * 255
    if int((cur > 0).sum()) < 8 or int((tpl > 0).sum()) < 8:
        return 0.0

    scales = app_config.get("partial_structure_scales", [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25])
    best_corr = 0.0
    for scale in scales:
        scale = float(scale)
        if scale <= 0:
            continue
        resized_cur = cv2.resize(cur, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if resized_cur.shape[0] < 8 or resized_cur.shape[1] < 8:
            continue
        if tpl.shape[0] >= resized_cur.shape[0] and tpl.shape[1] >= resized_cur.shape[1]:
            res = cv2.matchTemplate(tpl, resized_cur, cv2.TM_CCOEFF_NORMED)
            best_corr = max(best_corr, float(cv2.minMaxLoc(res)[1]))
        elif resized_cur.shape[0] >= tpl.shape[0] and resized_cur.shape[1] >= tpl.shape[1]:
            res = cv2.matchTemplate(resized_cur, tpl, cv2.TM_CCOEFF_NORMED)
            best_corr = max(best_corr, float(cv2.minMaxLoc(res)[1]))

    return max(0.0, min(1.0, best_corr))


@dataclass(frozen=True)
class CorridorCandidate:
    """A structural candidate for the first corridor after the side door.

    The values are deliberately geometry-first and color-free.  This is the
    closest code equivalent of the visual reasoning "I see a narrow vertical
    corridor from the side door, with a small opening/room at the far end".
    """

    orientation: str
    x: float
    y: float
    width: float
    height: float
    length: float
    thickness: float
    aspect: float
    profile: tuple[float, ...]
    end_density_a: float
    end_density_b: float
    confidence: float


def _safe_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(aa, bb) / denom)))


def _edge_patch_profile(edge_image: np.ndarray, bbox: tuple[int, int, int, int], orientation: str) -> tuple[tuple[float, ...], float, float, float]:
    x, y, w, h = bbox
    patch = (edge_image[y : y + h, x : x + w] > 0).astype(np.uint8)
    if patch.size == 0:
        return tuple([0.0] * 8), 0.0, 0.0, 0.0

    bins = 8
    profile: list[float] = []
    if orientation == "vertical":
        parts = np.array_split(patch, bins, axis=0)
        end_a = parts[0]
        end_b = parts[-1]
    else:
        parts = np.array_split(patch, bins, axis=1)
        end_a = parts[0]
        end_b = parts[-1]

    for part in parts:
        profile.append(float(part.sum()) / max(float(part.size), 1.0))
    max_v = max(profile) if profile else 0.0
    if max_v > 1e-6:
        profile = [v / max_v for v in profile]

    end_density_a = float(end_a.sum()) / max(float(end_a.size), 1.0)
    end_density_b = float(end_b.sum()) / max(float(end_b.size), 1.0)
    fill = float(patch.sum()) / max(float(patch.size), 1.0)
    return tuple(profile), end_density_a, end_density_b, fill


def _line_corridor_candidates(edge_image: np.ndarray) -> list[CorridorCandidate]:
    """Find narrow corridor candidates from paired wall lines in a binary edge map."""
    if edge_image.size == 0:
        return []
    h, w = edge_image.shape[:2]
    edges = (edge_image > 0).astype(np.uint8) * 255
    if int((edges > 0).sum()) < 12:
        return []

    # A tiny dilation connects stippled wall edges from the game UI without
    # turning the whole ROI into a filled blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    line_img = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        line_img,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=max(18, int(min(h, w) * 0.12)),
        maxLineGap=max(6, int(min(h, w) * 0.04)),
    )
    if lines is None:
        return []

    vertical: list[tuple[float, int, int, float]] = []
    horizontal: list[tuple[float, int, int, float]] = []
    for item in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in item]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < max(18, min(h, w) * 0.10):
            continue
        if abs(dx) <= max(3, abs(dy) * 0.35):
            vertical.append(((x1 + x2) / 2.0, min(y1, y2), max(y1, y2), length))
        elif abs(dy) <= max(3, abs(dx) * 0.35):
            horizontal.append(((y1 + y2) / 2.0, min(x1, x2), max(x1, x2), length))

    candidates: list[CorridorCandidate] = []

    def add_candidate(orientation: str, a: tuple[float, int, int, float], b: tuple[float, int, int, float]) -> None:
        pos_a, start_a, end_a, len_a = a
        pos_b, start_b, end_b, len_b = b
        gap = abs(pos_a - pos_b)
        if orientation == "vertical":
            if gap < max(8, w * 0.035) or gap > w * 0.55:
                return
        else:
            if gap < max(8, h * 0.035) or gap > h * 0.55:
                return
        ov_start = max(start_a, start_b)
        ov_end = min(end_a, end_b)
        overlap = ov_end - ov_start
        if overlap < max(20, min(h, w) * 0.13):
            return
        if overlap < min(len_a, len_b) * 0.25:
            return

        if orientation == "vertical":
            x0 = int(max(0, min(pos_a, pos_b) - gap * 0.12))
            x1 = int(min(w, max(pos_a, pos_b) + gap * 0.12))
            y0 = int(max(0, ov_start - overlap * 0.06))
            y1 = int(min(h, ov_end + overlap * 0.06))
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            length_norm = bh / max(h, 1)
            thickness_norm = bw / max(w, 1)
        else:
            x0 = int(max(0, ov_start - overlap * 0.06))
            x1 = int(min(w, ov_end + overlap * 0.06))
            y0 = int(max(0, min(pos_a, pos_b) - gap * 0.12))
            y1 = int(min(h, max(pos_a, pos_b) + gap * 0.12))
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            length_norm = bw / max(w, 1)
            thickness_norm = bh / max(h, 1)

        aspect = length_norm / max(thickness_norm, 1e-6)
        if aspect < 1.25:
            return
        profile, end_a_density, end_b_density, fill = _edge_patch_profile(edges, (x0, y0, bw, bh), orientation)
        # Corridors are wall-edge dominated: too empty is noise, too filled is
        # likely an icon/text blob instead of a corridor.
        if fill < 0.015 or fill > 0.45:
            return
        confidence = (
            0.30 * min(1.0, aspect / 4.0)
            + 0.25 * min(1.0, length_norm / 0.55)
            + 0.20 * min(1.0, (min(len_a, len_b) / max(max(h, w), 1)) / 0.45)
            + 0.15 * min(1.0, fill / 0.16)
            + 0.10
        )
        candidates.append(
            CorridorCandidate(
                orientation=orientation,
                x=float(x0 / max(w, 1)),
                y=float(y0 / max(h, 1)),
                width=float(bw / max(w, 1)),
                height=float(bh / max(h, 1)),
                length=float(length_norm),
                thickness=float(thickness_norm),
                aspect=float(aspect),
                profile=profile,
                end_density_a=float(end_a_density),
                end_density_b=float(end_b_density),
                confidence=float(max(0.0, min(1.0, confidence))),
            )
        )

    for i in range(len(vertical)):
        for j in range(i + 1, len(vertical)):
            add_candidate("vertical", vertical[i], vertical[j])
    for i in range(len(horizontal)):
        for j in range(i + 1, len(horizontal)):
            add_candidate("horizontal", horizontal[i], horizontal[j])

    # Deduplicate very similar candidates, keeping the most confident one.
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    kept: list[CorridorCandidate] = []
    for cand in candidates:
        duplicate = False
        for old in kept:
            if cand.orientation != old.orientation:
                continue
            if abs(cand.x - old.x) + abs(cand.y - old.y) + abs(cand.width - old.width) + abs(cand.height - old.height) < 0.18:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
        if len(kept) >= 12:
            break
    return kept


def _component_corridor_candidates(edge_image: np.ndarray) -> list[CorridorCandidate]:
    """Fallback candidates from elongated connected components."""
    if edge_image.size == 0:
        return []
    h, w = edge_image.shape[:2]
    mask = (edge_image > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[CorridorCandidate] = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if bw < 10 or bh < 10 or area < 30:
            continue
        aspect_raw = bh / max(bw, 1)
        if aspect_raw >= 1.25:
            orientation = "vertical"
            length_norm = bh / max(h, 1)
            thickness_norm = bw / max(w, 1)
            aspect = length_norm / max(thickness_norm, 1e-6)
        elif aspect_raw <= 0.8:
            orientation = "horizontal"
            length_norm = bw / max(w, 1)
            thickness_norm = bh / max(h, 1)
            aspect = length_norm / max(thickness_norm, 1e-6)
        else:
            continue
        if aspect < 1.25:
            continue
        profile, end_a_density, end_b_density, fill = _edge_patch_profile(edge_image, (x, y, bw, bh), orientation)
        if fill < 0.012 or fill > 0.50:
            continue
        confidence = 0.20 + 0.35 * min(1.0, aspect / 4.0) + 0.25 * min(1.0, length_norm / 0.55) + 0.20 * min(1.0, fill / 0.18)
        candidates.append(
            CorridorCandidate(
                orientation=orientation,
                x=float(x / max(w, 1)),
                y=float(y / max(h, 1)),
                width=float(bw / max(w, 1)),
                height=float(bh / max(h, 1)),
                length=float(length_norm),
                thickness=float(thickness_norm),
                aspect=float(aspect),
                profile=profile,
                end_density_a=float(end_a_density),
                end_density_b=float(end_b_density),
                confidence=float(max(0.0, min(1.0, confidence * 0.82))),
            )
        )
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:8]


def extract_corridor_candidates(edge_image: np.ndarray) -> list[CorridorCandidate]:
    candidates = _line_corridor_candidates(edge_image)
    candidates.extend(_component_corridor_candidates(edge_image))
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:16]


def _corridor_pair_score(current: CorridorCandidate, template: CorridorCandidate) -> float:
    if current.orientation != template.orientation:
        return 0.0

    aspect_score = float(np.exp(-abs(np.log((current.aspect + 1e-6) / (template.aspect + 1e-6))) / 0.75))
    thickness_score = float(np.exp(-abs(current.thickness - template.thickness) / 0.18))

    # A template may contain more structure than the fogged current screenshot.
    # Penalize a shorter template strongly, but allow a longer template.
    if template.length >= current.length:
        length_score = float(np.exp(-abs(template.length - current.length) / 0.55))
    else:
        length_score = float(np.exp(-abs(template.length - current.length) / 0.25))

    profile_score = _safe_cosine(current.profile, template.profile)
    end_current = sorted([current.end_density_a, current.end_density_b])
    end_template = sorted([template.end_density_a, template.end_density_b])
    end_score = float(np.exp(-(abs(end_current[0] - end_template[0]) + abs(end_current[1] - end_template[1])) / 0.35))

    raw = (
        0.32 * aspect_score
        + 0.24 * thickness_score
        + 0.20 * length_score
        + 0.16 * profile_score
        + 0.08 * end_score
    )
    return max(0.0, min(1.0, raw * current.confidence * template.confidence))


def corridor_component_score(current: np.ndarray, template: np.ndarray) -> float:
    """Compare side-door corridor topology in black/white structure maps.

    Unlike matchTemplate, this does not care whether the template has extra
    unexplored rooms. It asks: do both maps contain the same kind of entrance
    corridor component?
    """
    current_candidates = extract_corridor_candidates(current)
    template_candidates = extract_corridor_candidates(template)
    if not current_candidates or not template_candidates:
        return 0.0
    best = 0.0
    for cur in current_candidates:
        for tpl in template_candidates:
            best = max(best, _corridor_pair_score(cur, tpl))
    return float(best)


def _template_yellow_label_anchor(template_roi: np.ndarray, preprocess_config: PreprocessConfig) -> Optional[tuple[float, float]]:
    """Locate the template-side side-door label/icon area using yellow text.

    This is only used on full template assets, where the yellow "侧门" label is a
    useful annotation. The actual comparison still uses cleaned black/white
    structure images, so the yellow text itself is not matched against gameplay.
    """
    if template_roi.ndim != 3 or template_roi.size == 0:
        return None
    h, w = template_roi.shape[:2]
    hsv = cv2.cvtColor(template_roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mask = (
        (hue >= preprocess_config.yellow_hue_min)
        & (hue <= preprocess_config.yellow_hue_max)
        & (sat >= preprocess_config.overlay_saturation_min)
        & (val >= preprocess_config.overlay_value_min)
    ).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    components: list[tuple[float, float, float, float, float]] = []
    area_norm = float(w * h)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(40, area_norm * 0.002):
            continue
        if bw < w * 0.04 or bh < h * 0.04:
            continue
        if bw > w * 0.75 or bh > h * 0.55:
            continue
        cx, cy = centroids[i]
        components.append((float(area), float(cx), float(cy), float(bw), float(bh)))
    if not components:
        return None

    # The "侧门" label is normally two adjacent yellow text components. Prefer a
    # cluster of the largest yellow components near each other; this avoids small
    # route digits and line fragments.
    components.sort(key=lambda c: c[0], reverse=True)
    best_cluster: list[tuple[float, float, float, float, float]] = []
    best_score = 0.0
    for seed in components[:6]:
        sx, sy = seed[1], seed[2]
        cluster = [
            comp
            for comp in components[:10]
            if abs(comp[1] - sx) <= w * 0.34 and abs(comp[2] - sy) <= h * 0.28
        ]
        if not cluster:
            continue
        area_sum = sum(c[0] for c in cluster)
        span_x = max(c[1] for c in cluster) - min(c[1] for c in cluster)
        span_y = max(c[2] for c in cluster) - min(c[2] for c in cluster)
        text_like = min(1.0, len(cluster) / 2.0) * min(1.0, area_sum / max(area_norm * 0.018, 1.0))
        compact = float(np.exp(-(span_x / max(w * 0.42, 1.0) + span_y / max(h * 0.35, 1.0))))
        score = 0.7 * text_like + 0.3 * compact
        if score > best_score:
            best_score = score
            best_cluster = cluster
    if not best_cluster:
        best_cluster = [components[0]]
    total_area = sum(c[0] for c in best_cluster)
    cx = sum(c[0] * c[1] for c in best_cluster) / max(total_area, 1.0)
    cy = sum(c[0] * c[2] for c in best_cluster) / max(total_area, 1.0)
    return float(cx / max(w, 1)), float(cy / max(h, 1))


def _template_side_door_icon_anchor(template_roi: np.ndarray, preprocess_config: PreprocessConfig) -> Optional[tuple[float, float]]:
    """Find the white side-door icon in a template ROI.

    The yellow label tells us the neighborhood; the white door icon is the true
    anchor for cropping. This avoids cropping around the text itself or around
    unrelated yellow route numbers.
    """
    if template_roi.ndim != 3 or template_roi.size == 0:
        return None
    try:
        from .door_icon_detector import detect_side_door_icon

        detection = detect_side_door_icon(template_roi, preprocess_config, kind="template")
        label_anchor = _template_yellow_label_anchor(template_roi, preprocess_config)
        if detection is not None and detection.confidence >= 0.68:
            if label_anchor is None:
                return detection.anchor
            # Template icon should be near the yellow "侧门" annotation. This
            # prevents generic wall corners from winning template matching.
            if float(np.hypot(detection.anchor[0] - label_anchor[0], detection.anchor[1] - label_anchor[1])) <= 0.45:
                return detection.anchor
    except Exception:
        pass
    h, w = template_roi.shape[:2]
    label_anchor = _template_yellow_label_anchor(template_roi, preprocess_config)
    hsv = cv2.cvtColor(template_roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY)
    # The icon is bright/white and low-saturation. Keep the threshold moderate
    # because some template icons are anti-aliased gray.
    mask = np.where((gray >= 125) & (sat < 105), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_anchor: Optional[tuple[float, float]] = None
    best_score = 0.0
    img_area = float(w * h)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 6 or area > img_area * 0.025:
            continue
        if bw > w * 0.22 or bh > h * 0.25:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue
        cx, cy = centroids[i]
        density = area / max(bw * bh, 1)
        compact = min(1.0, max(0.0, 1.0 - abs(np.log(max(aspect, 1e-3))) / 2.2))
        if label_anchor is not None:
            dx = cx / max(w, 1) - label_anchor[0]
            dy = cy / max(h, 1) - label_anchor[1]
            near_label = float(np.exp(-float(np.hypot(dx, dy)) / 0.42))
            # The icon is usually at the side/below the label; far above is less
            # plausible but not impossible due to ROI framing.
            relative_bonus = 0.45 if dy < -0.35 else 1.0
        else:
            near_label = 0.50
            relative_bonus = 1.0
        border_dist = min(cx, cy, w - cx, h - cy) / max(min(w, h), 1)
        border_bonus = 1.0 - min(1.0, border_dist * 3.5)
        score = 0.42 * near_label + 0.22 * density + 0.18 * compact + 0.10 * relative_bonus + 0.08 * border_bonus
        if score > best_score:
            best_score = float(score)
            best_anchor = (float(cx / max(w, 1)), float(cy / max(h, 1)))
    if best_anchor is not None:
        return best_anchor
    return label_anchor


def _clamp_anchor(anchor: tuple[float, float]) -> tuple[float, float]:
    return (
        max(0.0, min(1.0, float(anchor[0]))),
        max(0.0, min(1.0, float(anchor[1]))),
    )


def _mirror_anchor(anchor: tuple[float, float], orientation: str) -> tuple[float, float]:
    if orientation == "vertical":
        return _clamp_anchor((anchor[0], 1.0 - anchor[1]))
    if orientation == "horizontal":
        return _clamp_anchor((1.0 - anchor[0], anchor[1]))
    return _clamp_anchor((1.0 - anchor[0], 1.0 - anchor[1]))


def _dedupe_anchors(anchors: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for anchor in anchors:
        anchor = _clamp_anchor(anchor)
        if any(abs(anchor[0] - old[0]) + abs(anchor[1] - old[1]) < 0.03 for old in deduped):
            continue
        deduped.append(anchor)
    return deduped


@dataclass(frozen=True)
class TemplateAnchorCandidate:
    anchor: tuple[float, float]
    confidence: float
    source: str


def _template_anchor_candidates_for_match(template_roi: np.ndarray, preprocess_config: PreprocessConfig) -> list[TemplateAnchorCandidate]:
    """Return several plausible template anchors for the entrance patch.

    The earlier pipeline trusted one template anchor too much.  In practice the
    door icon, the yellow label, and the structure-derived entrance position can
    disagree slightly, so we keep a small set and let the current screenshot
    decide which one actually lines up.
    """
    candidates: list[TemplateAnchorCandidate] = []
    icon_anchor = _template_side_door_icon_anchor(template_roi, preprocess_config)
    if icon_anchor is not None:
        candidates.append(TemplateAnchorCandidate(icon_anchor, 0.95, "icon"))

    template_pre = preprocess_image(template_roi, preprocess_config)
    template_features = extract_entry_corridor_features(template_pre)
    if template_features is not None:
        structure_anchor = (template_features.door_x, template_features.door_y)
        candidates.append(TemplateAnchorCandidate(structure_anchor, 0.80, "structure"))
        if icon_anchor is not None:
            candidates.append(TemplateAnchorCandidate(_mirror_anchor(icon_anchor, template_features.orientation), 0.72, "mirror"))
            if template_features.orientation == "block":
                candidates.append(TemplateAnchorCandidate(_mirror_anchor(icon_anchor, "vertical"), 0.68, "mirror_v"))
                candidates.append(TemplateAnchorCandidate(_mirror_anchor(icon_anchor, "horizontal"), 0.68, "mirror_h"))

    label_anchor = _template_yellow_label_anchor(template_roi, preprocess_config)
    if label_anchor is not None:
        candidates.append(TemplateAnchorCandidate(label_anchor, 0.62, "label"))

    unique: list[TemplateAnchorCandidate] = []
    for candidate in candidates:
        anchor = _clamp_anchor(candidate.anchor)
        if any(abs(anchor[0] - old.anchor[0]) + abs(anchor[1] - old.anchor[1]) < 0.03 for old in unique):
            continue
        unique.append(TemplateAnchorCandidate(anchor, candidate.confidence, candidate.source))
    return unique


def _anchor_variant_patches(
    image: np.ndarray,
    anchor: tuple[float, float],
    patch_fraction: float,
) -> list[tuple[np.ndarray, tuple[float, float]]]:
    return [
        _crop_anchor_patch_with_anchor(image, anchor, patch_fraction, prefer_below=True),
        _crop_anchor_patch_with_anchor(image, anchor, patch_fraction, prefer_below=False),
    ]


def _score_patch_variants(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
    scorer,
) -> float:
    cur_anchor, cur_conf = locate_door_anchor(current_roi, preprocess_config)
    if cur_anchor is None:
        cur_anchor = (0.5, 0.20)
        cur_conf = 0.35

    template_candidates = _template_anchor_candidates_for_match(template_roi, preprocess_config)
    if not template_candidates:
        return 0.0

    patch_fraction = float(app_config.get("entrance_patch_fraction", 0.52))
    cur_variants = _anchor_variant_patches(current_roi, cur_anchor, patch_fraction)
    best = 0.0
    for cur_patch, cur_patch_anchor in cur_variants:
        for tpl_candidate in template_candidates:
            tpl_variants = _anchor_variant_patches(template_roi, tpl_candidate.anchor, patch_fraction)
            for tpl_patch, tpl_patch_anchor in tpl_variants:
                score = scorer(cur_patch, cur_patch_anchor, tpl_patch, tpl_patch_anchor, cur_conf, tpl_candidate.confidence, tpl_candidate.source)
                if score > best:
                    best = score
    return float(best)


def _crop_anchor_patch(
    image: np.ndarray,
    anchor: tuple[float, float],
    patch_fraction: float,
    prefer_below: bool = False,
) -> np.ndarray:
    h, w = image.shape[:2]
    patch_fraction = max(0.18, min(0.95, float(patch_fraction)))
    pw = max(16, int(w * patch_fraction))
    ph = max(16, int(h * patch_fraction))
    cx = int(anchor[0] * w)
    cy = int(anchor[1] * h)
    x = int(cx - pw * 0.50)
    y = int(cy - ph * (0.25 if prefer_below else 0.50))
    x = max(0, min(max(0, w - pw), x))
    y = max(0, min(max(0, h - ph), y))
    return image[y : y + ph, x : x + pw].copy()


def _crop_anchor_patch_with_anchor(
    image: np.ndarray,
    anchor: tuple[float, float],
    patch_fraction: float,
    prefer_below: bool = False,
) -> tuple[np.ndarray, tuple[float, float]]:
    h, w = image.shape[:2]
    patch_fraction = max(0.18, min(0.95, float(patch_fraction)))
    pw = max(16, int(w * patch_fraction))
    ph = max(16, int(h * patch_fraction))
    cx = int(anchor[0] * w)
    cy = int(anchor[1] * h)
    x = int(cx - pw * 0.50)
    y = int(cy - ph * (0.25 if prefer_below else 0.50))
    x = max(0, min(max(0, w - pw), x))
    y = max(0, min(max(0, h - ph), y))
    patch_anchor = (
        float((anchor[0] * w - x) / max(pw, 1)),
        float((anchor[1] * h - y) / max(ph, 1)),
    )
    patch_anchor = (max(0.0, min(1.0, patch_anchor[0])), max(0.0, min(1.0, patch_anchor[1])))
    return image[y : y + ph, x : x + pw].copy(), patch_anchor


def entrance_patch_score(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> float:
    """Compare only the immediate side-door entrance patch.

    Templates may include much more map structure than the fogged current view.
    This score first finds the template's annotated side-door label, crops the
    nearby entrance patch, removes annotations, then compares black/white
    structure. It is intentionally narrow: "where did I step in from 1-Door?"
    """
    if current_roi.size == 0 or template_roi.size == 0:
        return 0.0

    def scorer(
        cur_patch: np.ndarray,
        _cur_patch_anchor: tuple[float, float],
        tpl_patch: np.ndarray,
        _tpl_patch_anchor: tuple[float, float],
        cur_conf: float,
        tpl_conf: float,
        _source: str,
    ) -> float:
        cur_pre = preprocess_image(cur_patch, preprocess_config)
        tpl_pre = preprocess_image(tpl_patch, preprocess_config)
        edge_score = edge_distance_score(cur_pre, tpl_pre)
        tmpl_score, _ = template_match_score(cur_pre, tpl_pre, app_config)
        corridor_score = corridor_component_score(cur_pre, tpl_pre)
        score = 0.50 * edge_score + 0.25 * tmpl_score + 0.25 * corridor_score
        confidence = min(1.0, 0.65 + 0.20 * cur_conf + 0.15 * tpl_conf)
        return float(max(0.0, min(1.0, score * confidence)))

    return _score_patch_variants(current_roi, template_roi, app_config, preprocess_config, scorer)


def _visible_area_mask(image: np.ndarray, preprocess_config: PreprocessConfig, is_template: bool) -> np.ndarray:
    """Binary mask of visible walkable/map area for topology comparison."""
    if image.size == 0:
        return np.zeros((220, 220), dtype=np.uint8)
    cleaned = preprocess_image  # keep linters from confusing imported helpers below
    del cleaned
    from .preprocess import remove_overlay_annotations, resize_normalized, suppress_dark_regions, to_gray

    raw = remove_overlay_annotations(image, preprocess_config)
    gray = to_gray(raw)
    gray = resize_normalized(gray, preprocess_config.target_size)
    gray = suppress_dark_regions(gray, preprocess_config.dark_mask_threshold)

    # Template backgrounds are pure black; gameplay fog is gray-blue. Use
    # different thresholds so both become a comparable "visible map body" mask.
    if is_template:
        threshold = max(22, int(np.percentile(gray, 52)))
    else:
        threshold = max(55, int(np.percentile(gray, 68)))
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    # Keep connected components that are close to the largest one. This preserves
    # the first room/corridor while dropping small text/icon islands.
    largest_idx = max(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    lx, ly, lw, lh, _ = [int(v) for v in stats[largest_idx]]
    keep = labels == largest_idx
    for i in range(1, n):
        if i == largest_idx:
            continue
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 35:
            continue
        near = not (x > lx + lw + 12 or lx > x + w + 12 or y > ly + lh + 12 or ly > y + h + 12)
        if near:
            keep |= labels == i
    return keep.astype(np.uint8) * 255


@dataclass(frozen=True)
class AreaShapeFeatures:
    orientation: str
    aspect: float
    width: float
    height: float
    fill: float
    main_axis_profile: tuple[float, ...]
    cross_axis_profile: tuple[float, ...]
    end_expansion_a: float
    end_expansion_b: float
    anchor_x: float
    anchor_y: float
    confidence: float


def _area_shape_features(mask: np.ndarray, anchor: Optional[tuple[float, float]] = None) -> Optional[AreaShapeFeatures]:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) < 50:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary * 255, 8)
    if n <= 1:
        return None
    idx = max(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    x, y, w, h, area = [int(v) for v in stats[idx]]
    if w < 8 or h < 8:
        return None
    patch = (labels[y : y + h, x : x + w] == idx).astype(np.uint8)
    raw_aspect = h / max(w, 1)
    orientation = "vertical" if raw_aspect >= 1.12 else "horizontal" if raw_aspect <= 0.90 else "block"
    if orientation == "vertical":
        main_parts = np.array_split(patch, 8, axis=0)
        cross_parts = np.array_split(patch, 6, axis=1)
        longness = h / max(w, 1)
        end_a = main_parts[0]
        end_b = main_parts[-1]
    elif orientation == "horizontal":
        main_parts = np.array_split(patch, 8, axis=1)
        cross_parts = np.array_split(patch, 6, axis=0)
        longness = w / max(h, 1)
        end_a = main_parts[0]
        end_b = main_parts[-1]
    else:
        main_parts = np.array_split(patch, 8, axis=0)
        cross_parts = np.array_split(patch, 6, axis=1)
        longness = max(h / max(w, 1), w / max(h, 1))
        end_a = main_parts[0]
        end_b = main_parts[-1]
    main_profile = [float(part.sum()) / max(float(part.size), 1.0) for part in main_parts]
    cross_profile = [float(part.sum()) / max(float(part.size), 1.0) for part in cross_parts]
    max_main = max(main_profile) if main_profile else 0.0
    max_cross = max(cross_profile) if cross_profile else 0.0
    if max_main > 1e-6:
        main_profile = [v / max_main for v in main_profile]
    if max_cross > 1e-6:
        cross_profile = [v / max_cross for v in cross_profile]
    fill = float(area) / max(float(w * h), 1.0)
    if anchor is None:
        anchor_x = 0.5
        anchor_y = 0.0 if orientation == "vertical" else 0.5
    else:
        img_h, img_w = mask.shape[:2]
        ax = anchor[0] * img_w
        ay = anchor[1] * img_h
        anchor_x = float((ax - x) / max(w, 1))
        anchor_y = float((ay - y) / max(h, 1))
        anchor_x = max(-0.5, min(1.5, anchor_x))
        anchor_y = max(-0.5, min(1.5, anchor_y))
    confidence = 0.30 + 0.30 * min(1.0, longness / 2.4) + 0.25 * min(1.0, fill / 0.75) + 0.15 * min(1.0, area / max(mask.size * 0.18, 1))
    return AreaShapeFeatures(
        orientation=orientation,
        aspect=float(raw_aspect),
        width=float(w / max(mask.shape[1], 1)),
        height=float(h / max(mask.shape[0], 1)),
        fill=float(fill),
        main_axis_profile=tuple(main_profile),
        cross_axis_profile=tuple(cross_profile),
        end_expansion_a=float(end_a.sum()) / max(float(end_a.size), 1.0),
        end_expansion_b=float(end_b.sum()) / max(float(end_b.size), 1.0),
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        confidence=float(max(0.0, min(1.0, confidence))),
    )


def _area_shape_score(current_features: AreaShapeFeatures, template_features: AreaShapeFeatures) -> float:
    orientation_score = 1.0 if current_features.orientation == template_features.orientation else 0.15 if "block" in {current_features.orientation, template_features.orientation} else 0.0
    aspect_score = float(np.exp(-abs(np.log((current_features.aspect + 1e-6) / (template_features.aspect + 1e-6))) / 0.75))
    size_score = float(np.exp(-(abs(current_features.width - template_features.width) + abs(current_features.height - template_features.height)) / 0.70))
    main_profile_score = _safe_cosine(current_features.main_axis_profile, template_features.main_axis_profile)
    cross_profile_score = _safe_cosine(current_features.cross_axis_profile, template_features.cross_axis_profile)
    cur_ends = sorted([current_features.end_expansion_a, current_features.end_expansion_b])
    tpl_ends = sorted([template_features.end_expansion_a, template_features.end_expansion_b])
    end_score = float(np.exp(-(abs(cur_ends[0] - tpl_ends[0]) + abs(cur_ends[1] - tpl_ends[1])) / 0.65))
    if current_features.orientation == "vertical" and template_features.orientation == "vertical":
        anchor_score = float(np.exp(-(abs(current_features.anchor_y - template_features.anchor_y) / 0.32 + abs(current_features.anchor_x - template_features.anchor_x) / 0.85)))
    elif current_features.orientation == "horizontal" and template_features.orientation == "horizontal":
        anchor_score = float(np.exp(-(abs(current_features.anchor_x - template_features.anchor_x) / 0.32 + abs(current_features.anchor_y - template_features.anchor_y) / 0.85)))
    else:
        anchor_score = float(np.exp(-(abs(current_features.anchor_x - template_features.anchor_x) + abs(current_features.anchor_y - template_features.anchor_y)) / 0.70))
    raw = (
        0.24 * orientation_score
        + 0.16 * aspect_score
        + 0.10 * size_score
        + 0.16 * main_profile_score
        + 0.08 * cross_profile_score
        + 0.08 * end_score
        + 0.18 * anchor_score
    )
    return float(max(0.0, min(1.0, raw * current_features.confidence * template_features.confidence)))


def entrance_shape_score(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> float:
    def scorer(
        cur_patch: np.ndarray,
        cur_patch_anchor: tuple[float, float],
        tpl_patch: np.ndarray,
        tpl_patch_anchor: tuple[float, float],
        _cur_conf: float,
        _tpl_conf: float,
        _source: str,
    ) -> float:
        cur_mask = _visible_area_mask(cur_patch, preprocess_config, is_template=False)
        tpl_mask = _visible_area_mask(tpl_patch, preprocess_config, is_template=True)
        cur_features = _area_shape_features(cur_mask, cur_patch_anchor)
        tpl_features = _area_shape_features(tpl_mask, tpl_patch_anchor)
        if cur_features is None or tpl_features is None:
            return 0.0
        return _area_shape_score(cur_features, tpl_features)

    return _score_patch_variants(current_roi, template_roi, app_config, preprocess_config, scorer)


def _direction_signature(mask: np.ndarray, anchor: tuple[float, float]) -> tuple[float, float, float, float]:
    """Return normalized visible-map support in up, down, left, right directions."""
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0, 0.0, 0.0
    ax = int(max(0, min(w - 1, round(anchor[0] * (w - 1)))))
    ay = int(max(0, min(h - 1, round(anchor[1] * (h - 1)))))
    band = max(3, int(min(w, h) * 0.045))

    def band_mean(y0: int, y1: int, x0: int, x1: int) -> float:
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        if y1 <= y0 or x1 <= x0:
            return 0.0
        return float(binary[y0:y1, x0:x1].mean())

    # Weighted reach: nearby map body matters, but long continuation matters too.
    up_vals = []
    for y in range(ay - 1, -1, -1):
        up_vals.append(band_mean(y, y + 1, ax - band, ax + band + 1))
    down_vals = []
    for y in range(ay + 1, h):
        down_vals.append(band_mean(y, y + 1, ax - band, ax + band + 1))
    left_vals = []
    for x in range(ax - 1, -1, -1):
        left_vals.append(band_mean(ay - band, ay + band + 1, x, x + 1))
    right_vals = []
    for x in range(ax + 1, w):
        right_vals.append(band_mean(ay - band, ay + band + 1, x, x + 1))

    def reach_score(vals: list[float]) -> float:
        if not vals:
            return 0.0
        arr = np.asarray(vals, dtype=np.float32)
        weights = np.linspace(1.0, 0.35, len(arr), dtype=np.float32)
        support = float((arr * weights).sum() / max(weights.sum(), 1e-6))
        # Long continuous corridor should score higher than a short blob.
        continuous = 0
        for v in vals:
            if v >= 0.20:
                continuous += 1
            elif continuous > 0:
                break
        continuity = continuous / max(len(vals), 1)
        return max(0.0, min(1.0, 0.65 * support + 0.35 * continuity))

    sig = (reach_score(up_vals), reach_score(down_vals), reach_score(left_vals), reach_score(right_vals))
    total = sum(sig)
    if total <= 1e-6:
        return sig
    return tuple(float(v / total) for v in sig)  # type: ignore[return-value]


def entrance_direction_score(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> float:
    """Compare the main direction of travel immediately after entering 1-Door."""

    def scorer(
        cur_patch: np.ndarray,
        cur_patch_anchor: tuple[float, float],
        tpl_patch: np.ndarray,
        tpl_patch_anchor: tuple[float, float],
        _cur_conf: float,
        _tpl_conf: float,
        _source: str,
    ) -> float:
        cur_mask = _visible_area_mask(cur_patch, preprocess_config, is_template=False)
        tpl_mask = _visible_area_mask(tpl_patch, preprocess_config, is_template=True)
        cur_sig = _direction_signature(cur_mask, cur_patch_anchor)
        tpl_sig = _direction_signature(tpl_mask, tpl_patch_anchor)
        cosine = _safe_cosine(cur_sig, tpl_sig)
        cur_main = int(np.argmax(np.asarray(cur_sig)))
        tpl_main = int(np.argmax(np.asarray(tpl_sig)))
        main_bonus = 1.0 if cur_main == tpl_main else 0.55
        # Penalize ambiguous template intersections: if support is spread across many
        # directions, it is less like a clean first-entry corridor.
        tpl_sorted = sorted(tpl_sig, reverse=True)
        tpl_clarity = tpl_sorted[0] - (tpl_sorted[1] if len(tpl_sorted) > 1 else 0.0)
        clarity = 0.65 + 0.35 * max(0.0, min(1.0, tpl_clarity * 2.5))
        door_axis_bonus = 1.0
        if tpl_main == 0:
            door_axis_bonus = 0.78 if cur_main == 1 else 0.58
        elif tpl_main == 1:
            door_axis_bonus = 0.78 if cur_main == 0 else 0.58
        elif tpl_main == 2:
            door_axis_bonus = 0.78 if cur_main == 3 else 0.58
        elif tpl_main == 3:
            door_axis_bonus = 0.78 if cur_main == 2 else 0.58
        return float(max(0.0, min(1.0, cosine * main_bonus * clarity * door_axis_bonus)))

    return _score_patch_variants(current_roi, template_roi, app_config, preprocess_config, scorer)


def _axis_corridor_pair_strength(edge_image: np.ndarray, anchor: tuple[float, float]) -> tuple[float, float, float]:
    """Return vertical-pair, horizontal-pair, diagonal-ratio near an anchor."""
    h, w = edge_image.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0, 0.0
    ax = anchor[0] * w
    ay = anchor[1] * h
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=16,
        minLineLength=max(20, int(min(h, w) * 0.11)),
        maxLineGap=max(6, int(min(h, w) * 0.035)),
    )
    if lines is None:
        return 0.0, 0.0, 0.0
    vertical: list[tuple[float, float, float, float, float, float]] = []
    horizontal: list[tuple[float, float, float, float, float, float]] = []
    diagonal_weight = 0.0
    total_weight = 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < max(20, min(h, w) * 0.11):
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        total_weight += length
        if abs(angle - 90.0) <= 10.0:
            pos = (x1 + x2) / 2.0
            start = min(y1, y2)
            end = max(y1, y2)
            near = float(np.exp(-abs(pos - ax) / max(w * 0.42, 1.0)))
            span_from_anchor = max(0.0, min(float(end), float(h)) - max(float(start), ay - 12.0))
            vertical.append((float(pos), float(start), float(end), length, near, span_from_anchor))
        elif angle <= 10.0 or angle >= 170.0:
            pos = (y1 + y2) / 2.0
            start = min(x1, x2)
            end = max(x1, x2)
            near = float(np.exp(-abs(pos - ay) / max(h * 0.42, 1.0)))
            span_from_anchor = max(0.0, min(float(end), float(w)) - max(float(start), ax - 12.0))
            horizontal.append((float(pos), float(start), float(end), length, near, span_from_anchor))
        else:
            diagonal_weight += length

    def pair_strength(lines_in: list[tuple[float, float, float, float, float, float]], vertical_axis: bool) -> float:
        best = 0.0
        size_main = h if vertical_axis else w
        size_cross = w if vertical_axis else h
        anchor_cross = ax if vertical_axis else ay
        for i in range(len(lines_in)):
            for j in range(i + 1, len(lines_in)):
                p1, a1, b1, _l1, n1, span1 = lines_in[i]
                p2, a2, b2, _l2, n2, span2 = lines_in[j]
                gap = abs(p1 - p2)
                overlap = max(0.0, min(b1, b2) - max(a1, a2))
                if gap < max(10.0, size_cross * 0.05) or gap > size_cross * 0.55:
                    continue
                if overlap < size_main * 0.16:
                    continue
                center = (p1 + p2) / 2.0
                center_score = float(np.exp(-abs(center - anchor_cross) / max(size_cross * 0.38, 1.0)))
                gap_score = float(np.exp(-abs(gap - size_cross * 0.34) / max(size_cross * 0.28, 1.0)))
                span_score = min(1.0, overlap / max(size_main * 0.58, 1.0))
                anchor_span_score = min(1.0, (span1 + span2) / max(size_main * 0.75, 1.0))
                score = center_score * gap_score * span_score * anchor_span_score * ((n1 + n2) / 2.0)
                best = max(best, score)
        return float(max(0.0, min(1.0, best)))

    diag_ratio = diagonal_weight / max(total_weight, 1.0)
    return pair_strength(vertical, True), pair_strength(horizontal, False), float(max(0.0, min(1.0, diag_ratio)))


def entrance_corridor_pair_score(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> float:
    """Compare the paired-wall corridor immediately after the entrance icon."""
    cur_anchor, _ = locate_door_anchor(current_roi, preprocess_config)
    if cur_anchor is None:
        cur_anchor = (0.5, 0.20)
    tpl_anchor = _template_side_door_icon_anchor(template_roi, preprocess_config)
    if tpl_anchor is None:
        tpl_anchor, _ = locate_door_anchor(template_roi, preprocess_config)
    if tpl_anchor is None:
        return 0.0
    patch_fraction = float(app_config.get("entrance_patch_fraction", 0.52))
    current_patch, cur_patch_anchor = _crop_anchor_patch_with_anchor(current_roi, cur_anchor, patch_fraction, prefer_below=True)
    template_patch, tpl_patch_anchor = _crop_anchor_patch_with_anchor(template_roi, tpl_anchor, patch_fraction, prefer_below=True)
    cur_edge = preprocess_image(current_patch, preprocess_config)
    tpl_edge = preprocess_image(template_patch, preprocess_config)
    cur_v, cur_h, cur_diag = _axis_corridor_pair_strength(cur_edge, cur_patch_anchor)
    tpl_v, tpl_h, tpl_diag = _axis_corridor_pair_strength(tpl_edge, tpl_patch_anchor)
    cur_vec = (cur_v, cur_h)
    tpl_vec = (tpl_v, tpl_h)
    axis_score = _safe_cosine(cur_vec, tpl_vec)
    strength_score = float(np.exp(-(abs(cur_v - tpl_v) + abs(cur_h - tpl_h)) / 0.55))
    diag_score = float(np.exp(-abs(cur_diag - tpl_diag) / 0.35))
    # Current sample at 1-Door is often a clean vertical paired corridor. If a
    # template has mostly diagonal/branching geometry near the door, it should
    # lose even if broad direction support is similar.
    return float(max(0.0, min(1.0, 0.55 * axis_score + 0.30 * strength_score + 0.15 * diag_score)))


def _anchor_aligned_subset_edge_score(
    current_edge: np.ndarray,
    template_edge: np.ndarray,
    current_anchor: tuple[float, float],
    template_anchor: tuple[float, float],
) -> float:
    """One-way score: are current visible edges present near the anchored template?"""
    if current_edge.size == 0 or template_edge.size == 0:
        return 0.0
    cur = (current_edge > 0).astype(np.uint8)
    tpl = (template_edge > 0).astype(np.uint8)
    if int(cur.sum()) < 8 or int(tpl.sum()) < 8:
        return 0.0
    h, w = cur.shape[:2]
    if tpl.shape[:2] != (h, w):
        tpl = cv2.resize(tpl * 255, (w, h), interpolation=cv2.INTER_AREA)
        tpl = (tpl > 0).astype(np.uint8)

    cur_ax = current_anchor[0] * w
    cur_ay = current_anchor[1] * h
    tpl_ax = template_anchor[0] * w
    tpl_ay = template_anchor[1] * h
    dx = int(round(cur_ax - tpl_ax))
    dy = int(round(cur_ay - tpl_ay))

    shifted = np.zeros_like(tpl)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    copy_w = min(w - src_x0, w - dst_x0)
    copy_h = min(h - src_y0, h - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return 0.0
    shifted[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = tpl[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]

    dt = cv2.distanceTransform(1 - shifted, cv2.DIST_L2, 3)
    cur_points = cur > 0
    distances = dt[cur_points]
    if distances.size == 0:
        return 0.0
    mean_d = float(np.mean(distances))
    close_ratio = float(np.mean(distances <= 5.0))
    diag = float(np.hypot(h, w))
    distance_score = float(np.exp(-mean_d / max(diag * 0.025, 1.0)))

    # A very dense current edge map caused by player/marker artifacts should not
    # dominate; only score the subset quality.
    return float(max(0.0, min(1.0, 0.65 * distance_score + 0.35 * close_ratio)))


def anchored_subset_score(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> float:
    """Anchor side-door icons, then check whether current visible edges are a subset of template edges.

    This is useful when the template contains extra structure (e.g. treasure room)
    that is still hidden by fog in the current screenshot.
    """
    cur_anchor, _ = locate_door_anchor(current_roi, preprocess_config)
    if cur_anchor is None:
        return 0.0
    tpl_anchor = _template_side_door_icon_anchor(template_roi, preprocess_config)
    if tpl_anchor is None:
        tpl_anchor, _ = locate_door_anchor(template_roi, preprocess_config)
    if tpl_anchor is None:
        return 0.0

    patch_fraction = float(app_config.get("anchored_subset_patch_fraction", 0.86))
    current_patch, cur_patch_anchor = _crop_anchor_patch_with_anchor(current_roi, cur_anchor, patch_fraction, prefer_below=True)
    template_patch, tpl_patch_anchor = _crop_anchor_patch_with_anchor(template_roi, tpl_anchor, patch_fraction, prefer_below=True)
    cur_edge = preprocess_image(current_patch, preprocess_config)
    tpl_edge = preprocess_image(template_patch, preprocess_config)

    # Try small scale changes because screenshot/template map scales differ.
    scales = app_config.get("anchored_subset_scales", [0.85, 0.95, 1.0, 1.05, 1.15])
    best = 0.0
    for scale in scales:
        scale = float(scale)
        if scale <= 0:
            continue
        if abs(scale - 1.0) < 1e-6:
            score = _anchor_aligned_subset_edge_score(cur_edge, tpl_edge, cur_patch_anchor, tpl_patch_anchor)
        else:
            resized_tpl = cv2.resize(tpl_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            canvas = np.zeros_like(cur_edge)
            rh, rw = resized_tpl.shape[:2]
            ch, cw = canvas.shape[:2]
            # Place resized template so its anchor lands on the current anchor.
            cur_ax = int(round(cur_patch_anchor[0] * cw))
            cur_ay = int(round(cur_patch_anchor[1] * ch))
            tpl_ax = int(round(tpl_patch_anchor[0] * rw))
            tpl_ay = int(round(tpl_patch_anchor[1] * rh))
            dst_x = cur_ax - tpl_ax
            dst_y = cur_ay - tpl_ay
            src_x0 = max(0, -dst_x)
            src_y0 = max(0, -dst_y)
            dst_x0 = max(0, dst_x)
            dst_y0 = max(0, dst_y)
            copy_w = min(rw - src_x0, cw - dst_x0)
            copy_h = min(rh - src_y0, ch - dst_y0)
            if copy_w <= 0 or copy_h <= 0:
                continue
            canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = resized_tpl[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
            score = _anchor_aligned_subset_edge_score(cur_edge, canvas, cur_patch_anchor, cur_patch_anchor)
        best = max(best, score)
    return float(best)


@dataclass(frozen=True)
class EntryCorridorFeatures:
    orientation: str
    aspect: float
    door_x: float
    door_y: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    fill: float
    end_expansion: float
    confidence: float


def _largest_structure_bbox(edge_image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    mask = (edge_image > 0).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    h, w = mask.shape[:2]
    img_area = h * w
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(25, img_area * 0.001) or area > img_area * 0.85:
            continue
        candidates.append((int(area), int(x), int(y), int(bw), int(bh)))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    return x, y, bw, bh


def extract_entry_corridor_features(edge_image: np.ndarray) -> Optional[EntryCorridorFeatures]:
    """Extract a compact topology descriptor for the first corridor after the side door.

    This intentionally mimics the human judgment used here:
    - Is the first visible corridor vertical or horizontal?
    - Is the door icon near one end of that corridor?
    - Is the corridor narrow/straight?
    - Does the far end begin to expand into another room/connection?
    """
    if edge_image.size == 0:
        return None
    h, w = edge_image.shape[:2]
    bbox = _largest_structure_bbox(edge_image)
    if bbox is None:
        return None
    x, y, bw, bh = bbox
    if bw < 8 or bh < 8:
        return None
    aspect = bh / max(bw, 1)
    orientation = "vertical" if aspect >= 1.25 else "horizontal" if aspect <= 0.8 else "block"

    # Estimate door/icon location as the densest bright-edge patch near the border
    # of the structure. For current/template BW maps this tends to land on the
    # side-door icon rather than random wall segments.
    roi = (edge_image[y : y + bh, x : x + bw] > 0).astype(np.uint8)
    grid_y, grid_x = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    if int(roi.sum()) > 0:
        # Prefer edge pixels near the outer ends of the main corridor.
        if orientation == "vertical":
            top_weight = 1.0 - grid_y / max(roi.shape[0] - 1, 1)
            bottom_weight = grid_y / max(roi.shape[0] - 1, 1)
            weights = np.maximum(top_weight, bottom_weight) * roi
        elif orientation == "horizontal":
            left_weight = 1.0 - grid_x / max(roi.shape[1] - 1, 1)
            right_weight = grid_x / max(roi.shape[1] - 1, 1)
            weights = np.maximum(left_weight, right_weight) * roi
        else:
            border = np.minimum.reduce([grid_x, grid_y, roi.shape[1] - 1 - grid_x, roi.shape[0] - 1 - grid_y])
            weights = (1.0 - border / max(min(roi.shape[:2]) / 2, 1)) * roi
        if float(weights.sum()) > 0:
            door_local_x = float((weights * grid_x).sum() / weights.sum())
            door_local_y = float((weights * grid_y).sum() / weights.sum())
        else:
            door_local_x, door_local_y = roi.shape[1] / 2, roi.shape[0] / 2
    else:
        door_local_x, door_local_y = roi.shape[1] / 2, roi.shape[0] / 2

    door_x = (x + door_local_x) / max(w, 1)
    door_y = (y + door_local_y) / max(h, 1)
    fill = float(roi.sum()) / max(float(roi.size), 1.0)

    # End expansion: compare edge density near the far end versus middle. A
    # doorway that begins to open into a room tends to have higher far-end spread.
    if orientation == "vertical":
        top_density = float(roi[: max(1, roi.shape[0] // 4), :].sum()) / max(1, (roi.shape[0] // 4) * roi.shape[1])
        bottom_density = float(roi[-max(1, roi.shape[0] // 4) :, :].sum()) / max(1, (roi.shape[0] // 4) * roi.shape[1])
        end_expansion = max(top_density, bottom_density)
    elif orientation == "horizontal":
        left_density = float(roi[:, : max(1, roi.shape[1] // 4)].sum()) / max(1, roi.shape[0] * (roi.shape[1] // 4))
        right_density = float(roi[:, -max(1, roi.shape[1] // 4) :].sum()) / max(1, roi.shape[0] * (roi.shape[1] // 4))
        end_expansion = max(left_density, right_density)
    else:
        end_expansion = fill

    longness = max(bh / max(bw, 1), bw / max(bh, 1))
    confidence = min(1.0, 0.35 + 0.25 * min(longness / 3.0, 1.0) + 0.25 * min((bw * bh) / max(w * h * 0.25, 1), 1.0) + 0.15 * min(fill * 8, 1.0))
    return EntryCorridorFeatures(
        orientation=orientation,
        aspect=float(aspect),
        door_x=float(door_x),
        door_y=float(door_y),
        bbox_x=float(x / max(w, 1)),
        bbox_y=float(y / max(h, 1)),
        bbox_w=float(bw / max(w, 1)),
        bbox_h=float(bh / max(h, 1)),
        fill=float(fill),
        end_expansion=float(end_expansion),
        confidence=float(confidence),
    )


def entry_corridor_score(
    current: np.ndarray,
    template: np.ndarray,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> float:
    def scorer(
        cur_patch: np.ndarray,
        cur_patch_anchor: tuple[float, float],
        tpl_patch: np.ndarray,
        tpl_patch_anchor: tuple[float, float],
        _cur_conf: float,
        _tpl_conf: float,
        _source: str,
    ) -> float:
        cur = extract_entry_corridor_features(cur_patch)
        tpl = extract_entry_corridor_features(tpl_patch)
        if cur is None or tpl is None:
            return 0.0

        orientation_score = 1.0 if cur.orientation == tpl.orientation else 0.15 if "block" in {cur.orientation, tpl.orientation} else 0.0
        aspect_score = float(np.exp(-abs(np.log((cur.aspect + 1e-6) / (tpl.aspect + 1e-6))) / 0.9))
        bbox_score = float(np.exp(-abs(cur.bbox_w - tpl.bbox_w) / 0.35 - abs(cur.bbox_h - tpl.bbox_h) / 0.35))

        # Door relative position matters mostly along the main corridor axis.
        if cur.orientation == "vertical" and tpl.orientation == "vertical":
            door_axis_score = float(np.exp(-abs(cur.door_y - tpl.door_y) / 0.35))
        elif cur.orientation == "horizontal" and tpl.orientation == "horizontal":
            door_axis_score = float(np.exp(-abs(cur.door_x - tpl.door_x) / 0.35))
        else:
            door_axis_score = float(np.exp(-(abs(cur.door_x - tpl.door_x) + abs(cur.door_y - tpl.door_y)) / 0.6))

        expansion_score = float(np.exp(-abs(cur.end_expansion - tpl.end_expansion) / 0.35))
        axis_flip_penalty = 1.0
        if cur.orientation == tpl.orientation and cur.orientation in {"vertical", "horizontal"}:
            if cur.orientation == "vertical":
                axis_flip_penalty = 1.0 - min(0.30, abs(cur.door_y - tpl.door_y) * 0.35)
            else:
                axis_flip_penalty = 1.0 - min(0.30, abs(cur.door_x - tpl.door_x) * 0.35)
        score = (
            0.38 * orientation_score
            + 0.22 * aspect_score
            + 0.16 * bbox_score
            + 0.16 * door_axis_score
            + 0.08 * expansion_score
        )
        confidence = min(1.0, 0.70 + 0.18 * _cur_conf + 0.12 * _tpl_conf)
        return max(0.0, min(1.0, score * confidence * axis_flip_penalty))

    return _score_patch_variants(current, template, app_config, preprocess_config, scorer)


def door_anchor_score(
    current_roi: np.ndarray,
    template_roi: np.ndarray,
    preprocess_config: PreprocessConfig,
) -> tuple[float, tuple[float, float] | None, tuple[float, float] | None]:
    cur_anchor, cur_conf = locate_door_anchor(current_roi, preprocess_config)
    tpl_anchor, tpl_conf = locate_door_anchor(template_roi, preprocess_config)
    if cur_anchor is None or tpl_anchor is None:
        return 0.0, cur_anchor, tpl_anchor
    dx = cur_anchor[0] - tpl_anchor[0]
    dy = cur_anchor[1] - tpl_anchor[1]
    dist = float(np.hypot(dx, dy))
    # Anchor position is weak evidence because ROI framing can differ; keep the
    # falloff gentle and multiply by detector confidence.
    pos_score = float(np.exp(-(dist / 0.38) ** 2))
    return max(0.0, min(1.0, pos_score * cur_conf * tpl_conf)), cur_anchor, tpl_anchor


def combine_scores(
    color_score: float,
    ssim_score_value: float,
    deep_score: float,
    template_score: float,
    feature_score: float,
    structure_score: float,
    partial_score: float,
    entrance_score: float,
    entrance_shape: float,
    entrance_direction: float,
    entrance_pair: float,
    anchored_subset: float,
    corridor_score: float,
    entry_score: float,
    door_score: float,
    app_config: Mapping[str, Any],
) -> float:
    colw = float(app_config.get("color_weight", 0.0)) if bool(app_config.get("enable_color_match", True)) else 0.0
    ssimw = float(app_config.get("ssim_weight", 0.0)) if bool(app_config.get("enable_ssim_match", True)) else 0.0
    deepw = float(app_config.get("deep_match_weight", 0.0)) if bool(app_config.get("enable_deep_match", False)) else 0.0
    tw = float(app_config.get("template_weight", 0.6))
    fw = float(app_config.get("feature_weight", 0.4))
    sw = float(app_config.get("structure_weight", 0.0)) if bool(app_config.get("enable_structure_match", True)) else 0.0
    pw = float(app_config.get("partial_structure_weight", 0.0)) if bool(app_config.get("enable_partial_structure_match", True)) else 0.0
    aw = float(app_config.get("entrance_patch_weight", 0.0)) if bool(app_config.get("enable_entrance_patch_match", True)) else 0.0
    shw = float(app_config.get("entrance_shape_weight", 0.0)) if bool(app_config.get("enable_entrance_shape_match", True)) else 0.0
    diw = float(app_config.get("entrance_direction_weight", 0.0)) if bool(app_config.get("enable_entrance_direction_match", True)) else 0.0
    cpw = float(app_config.get("entrance_corridor_pair_weight", 0.0)) if bool(app_config.get("enable_entrance_corridor_pair_match", True)) else 0.0
    asw = float(app_config.get("anchored_subset_weight", 0.0)) if bool(app_config.get("enable_anchored_subset_match", True)) else 0.0
    cw = float(app_config.get("corridor_component_weight", 0.0)) if bool(app_config.get("enable_corridor_component_match", True)) else 0.0
    ew = float(app_config.get("entry_corridor_weight", 0.0)) if bool(app_config.get("enable_entry_corridor_match", True)) else 0.0
    dw = float(app_config.get("door_anchor_weight", 0.0)) if bool(app_config.get("enable_door_anchor", True)) else 0.0
    total = colw + ssimw + deepw + tw + fw + sw + pw + aw + shw + diw + cpw + asw + cw + ew + dw
    if total <= 0:
        colw, ssimw, deepw, tw, fw, sw, pw, aw, shw, diw, cpw, asw, cw, ew, dw, total = 0.40, 0.18, 0.0, 0.12, 0.04, 0.02, 0.02, 0.08, 0.06, 0.04, 0.08, 0.08, 0.02, 0.02, 0.02, 1.0
    return (
        colw * color_score
        + ssimw * ssim_score_value
        + deepw * deep_score
        + tw * template_score
        + fw * feature_score
        + sw * structure_score
        + pw * partial_score
        + aw * entrance_score
        + shw * entrance_shape
        + diw * entrance_direction
        + cpw * entrance_pair
        + asw * anchored_subset
        + cw * corridor_score
        + ew * entry_score
        + dw * door_score
    ) / total


def combine_candidate_scores(
    color_score: float,
    ssim_score_value: float,
    template_score: float,
    feature_score: float,
    app_config: Mapping[str, Any],
) -> float:
    """First-stage recall score: keep the real answer in TopN."""
    colw = float(app_config.get("candidate_color_weight", 0.45)) if bool(app_config.get("enable_color_match", True)) else 0.0
    ssimw = float(app_config.get("candidate_ssim_weight", 0.30)) if bool(app_config.get("enable_ssim_match", True)) else 0.0
    tw = float(app_config.get("candidate_template_weight", 0.20))
    fw = float(app_config.get("candidate_feature_weight", 0.05))
    total = colw + ssimw + tw + fw
    if total <= 0:
        return 0.0
    return float((colw * color_score + ssimw * ssim_score_value + tw * template_score + fw * feature_score) / total)


def match_current_roi(
    current_roi: np.ndarray,
    db: MapDatabase,
    app_config: Mapping[str, Any],
    preprocess_config: PreprocessConfig,
) -> MatchSummary:
    auto_crop_enabled = bool(app_config.get("enable_auto_crop_visible_structure", True))
    cropped_roi: Optional[np.ndarray] = None
    crop_bbox: Optional[tuple[int, int, int, int]] = None
    if auto_crop_enabled:
        cropped_roi, crop_bbox = crop_visible_structure(
            current_roi,
            preprocess_config,
            int(app_config.get("auto_crop_padding", 24)),
        )
    effective_current = cropped_roi if cropped_roi is not None and crop_bbox is not None else current_roi
    current_pre = preprocess_image(effective_current, preprocess_config)
    template_pre_by_map: dict[str, np.ndarray] = {}
    result_by_map: dict[str, MatchResult] = {}
    for template in db.templates.values():
        effective_template = template.template
        if auto_crop_enabled:
            cropped_template, template_bbox = crop_visible_structure(
                template.template,
                preprocess_config,
                int(app_config.get("auto_crop_padding", 24)),
            )
            if template_bbox is not None:
                effective_template = cropped_template
        template_pre = preprocess_image(effective_template, preprocess_config)
        template_pre_by_map[template.map_id] = template_pre

        color_score, color_scale = color_pixel_score(effective_current, template.template, app_config) if bool(app_config.get("enable_color_match", True)) else (0.0, 1.0)
        ss_score = ssim_score(effective_current, template.template, app_config) if bool(app_config.get("enable_ssim_match", True)) else 0.0
        t_score, best_scale = template_match_score(current_pre, template_pre, app_config)
        f_score, good, inliers, inlier_ratio = feature_match_score(current_pre, template, db.orb, app_config)
        candidate = combine_candidate_scores(color_score, ss_score, t_score, f_score, app_config)
        result_by_map[template.map_id] = (
            MatchResult(
                map_id=template.map_id,
                map_name=template.name,
                color_score=color_score,
                ssim_score=ss_score,
                deep_score=0.0,
                candidate_score=candidate,
                template_score=t_score,
                feature_score=f_score,
                structure_score=0.0,
                partial_structure_score=0.0,
                entrance_patch_score=0.0,
                entrance_shape_score=0.0,
                entrance_direction_score=0.0,
                entrance_corridor_pair_score=0.0,
                anchored_subset_score=0.0,
                corridor_component_score=0.0,
                entry_corridor_score=0.0,
                door_anchor_score=0.0,
                final_score=candidate,
                good_matches=good,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                best_scale=color_scale if color_score >= t_score else best_scale,
            )
        )

    top_n = int(app_config.get("candidate_top_n", 5))
    candidate_results = sorted(result_by_map.values(), key=lambda r: r.candidate_score, reverse=True)[: max(1, top_n)]
    rerank_ids = {r.map_id for r in candidate_results}
    non_candidate_penalty = float(app_config.get("non_candidate_final_penalty", 0.65))
    for result in result_by_map.values():
        if result.map_id not in rerank_ids:
            result.final_score = result.candidate_score * non_candidate_penalty
    for template in db.templates.values():
        result = result_by_map[template.map_id]
        if result.map_id not in rerank_ids:
            continue
        template_pre = template_pre_by_map[template.map_id]
        s_score = edge_distance_score(current_pre, template_pre) if bool(app_config.get("enable_structure_match", True)) else 0.0
        p_score = partial_structure_score(current_pre, template_pre, app_config) if bool(app_config.get("enable_partial_structure_match", True)) else 0.0
        a_score = entrance_patch_score(effective_current, template.template, app_config, preprocess_config) if bool(app_config.get("enable_entrance_patch_match", True)) else 0.0
        shape_score = entrance_shape_score(effective_current, template.template, app_config, preprocess_config) if bool(app_config.get("enable_entrance_shape_match", True)) else 0.0
        direction_score = entrance_direction_score(effective_current, template.template, app_config, preprocess_config) if bool(app_config.get("enable_entrance_direction_match", True)) else 0.0
        pair_score = entrance_corridor_pair_score(effective_current, template.template, app_config, preprocess_config) if bool(app_config.get("enable_entrance_corridor_pair_match", True)) else 0.0
        anchored_score = anchored_subset_score(effective_current, template.template, app_config, preprocess_config) if bool(app_config.get("enable_anchored_subset_match", True)) else 0.0
        c_score = corridor_component_score(current_pre, template_pre) if bool(app_config.get("enable_corridor_component_match", True)) else 0.0
        e_score = entry_corridor_score(current_pre, template_pre, app_config, preprocess_config) if bool(app_config.get("enable_entry_corridor_match", True)) else 0.0
        d_score, _, _ = door_anchor_score(effective_current, template.template, preprocess_config) if bool(app_config.get("enable_door_anchor", True)) else (0.0, None, None)
        deep_score = 0.0
        deep_backend = ""
        if bool(app_config.get("enable_deep_match", False)):
            deep_score, deep_backend = deep_embedding_score(effective_current, template.template, app_config)
        final = combine_scores(
            result.color_score,
            result.ssim_score,
            deep_score,
            result.template_score,
            result.feature_score,
            s_score,
            p_score,
            a_score,
            shape_score,
            direction_score,
            pair_score,
            anchored_score,
            c_score,
            e_score,
            d_score,
            app_config,
        )
        result.deep_score = deep_score
        result.deep_backend = deep_backend
        result.structure_score = s_score
        result.partial_structure_score = p_score
        result.entrance_patch_score = a_score
        result.entrance_shape_score = shape_score
        result.entrance_direction_score = direction_score
        result.entrance_corridor_pair_score = pair_score
        result.anchored_subset_score = anchored_score
        result.corridor_component_score = c_score
        result.entry_corridor_score = e_score
        result.door_anchor_score = d_score
        result.final_score = final

    results = list(result_by_map.values())
    results.sort(key=lambda r: r.final_score, reverse=True)
    top3 = results[: int(app_config.get("display_top_n", 5))]
    best = top3[0] if top3 else None
    threshold = float(app_config.get("match_threshold", 0.65))
    gap = float(app_config.get("uncertain_gap", 0.08))
    reliable = False
    status = "没有可用匹配结果"

    if best is not None:
        second_score = top3[1].final_score if len(top3) > 1 else 0.0
        delta = best.final_score - second_score
        if best.final_score < threshold:
            status = f"无法可靠识别：最高分 {best.final_score:.3f} 低于阈值 {threshold:.3f}"
        elif delta < gap:
            status = f"不确定：第一名与第二名差距 {delta:.3f} 小于 {gap:.3f}"
        else:
            reliable = True
            status = f"可靠识别：{best.map_id} / {best.map_name}"
        best.is_reliable = reliable
        best.reason = status

    best_pre = template_pre_by_map.get(best.map_id) if best is not None else None
    return MatchSummary(
        results=results,
        top3=top3,
        best=best,
        is_reliable=reliable,
        status=status,
        current_preprocessed=current_pre,
        best_template_preprocessed=best_pre,
        current_cropped_roi=cropped_roi if crop_bbox is not None else None,
        current_crop_bbox=crop_bbox,
        top_candidates=candidate_results,
    )

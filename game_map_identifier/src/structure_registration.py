from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .utils import imread_unicode


@dataclass(frozen=True)
class StructureTransform:
    map_id: str
    scale: float
    top_left: tuple[float, float]
    template_crop_offset: tuple[int, int]
    unknown_crop_offset: tuple[int, int]
    score: float
    is_ambiguous: bool = False
    second_score: float = 0.0
    affine: Optional[list[list[float]]] = None

    def map_unknown_point(self, point: tuple[float, float], *, use_affine: bool = False) -> tuple[float, float]:
        ux = point[0] - self.unknown_crop_offset[0]
        uy = point[1] - self.unknown_crop_offset[1]
        if use_affine and self.affine is not None:
            matrix = np.array(self.affine, dtype=np.float32)
            mapped = cv2.transform(np.array([[[ux, uy]]], dtype=np.float32), matrix)[0, 0]
            return (
                float(self.template_crop_offset[0] + mapped[0]),
                float(self.template_crop_offset[1] + mapped[1]),
            )
        tx = self.template_crop_offset[0] + self.top_left[0] + ux * self.scale
        ty = self.template_crop_offset[1] + self.top_left[1] + uy * self.scale
        return float(tx), float(ty)


@dataclass(frozen=True)
class StructureCandidate:
    score: float
    scale: float
    top_left: tuple[int, int]
    coarse_score: float
    color_score: float
    edge_score: float
    skeleton_score: float


@dataclass(frozen=True)
class StructureRegistrationResult:
    transform: StructureTransform
    full_map: np.ndarray
    template_crop: np.ndarray
    unknown_crop: np.ndarray
    overlay: np.ndarray
    unknown_edges: np.ndarray
    template_edges: np.ndarray
    unknown_skeleton: np.ndarray
    template_skeleton: np.ndarray
    candidates: list[StructureCandidate]


def read_image(path: Path) -> np.ndarray:
    image = imread_unicode(path)
    if image is None:
        raise RuntimeError(f"无法读取结构配准模板：{path}")
    return image


def parse_scales(values: object, default: Iterable[float]) -> list[float]:
    if isinstance(values, str):
        parsed = [float(item.strip()) for item in values.split(",") if item.strip()]
    elif isinstance(values, list):
        parsed = [float(item) for item in values]
    else:
        parsed = list(default)
    return [scale for scale in parsed if scale > 0]


def resolve_template_path(templates_dir: Path, map_id: str) -> Path:
    candidates = [
        templates_dir / f"{map_id}.png",
        templates_dir / f"{map_id}.jpg",
        templates_dir / f"{map_id}.jpeg",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(f"缺少结构配准模板：{map_id}，目录：{templates_dir}")


def crop_to_content(image: np.ndarray, margin: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 245)
    if len(xs) == 0 or len(ys) == 0:
        return image.copy(), (0, 0)
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(image.shape[1], int(xs.max()) + margin + 1)
    y2 = min(image.shape[0], int(ys.max()) + margin + 1)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def crop_unknown_to_visible_structure(image: np.ndarray, margin: int = 20) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop the explored/visible map structure from a fog-covered game screenshot.

    The unknown full-view screenshot often contains a large dark fog area plus
    fixed UI text/buttons. Matching the whole ROI lets those non-map regions
    dominate the search. This keeps only connected edge components that look
    like revealed map geometry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    valid = np.zeros_like(gray, dtype=np.uint8)
    valid[int(height * 0.08) : int(height * 0.88), int(width * 0.02) : int(width * 0.92)] = 255
    working = _remove_yellow_marker(image)
    blur = cv2.GaussianBlur(cv2.cvtColor(working, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(blur, 35, 120)
    edges = cv2.bitwise_and(edges, valid)
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dilated, 8)
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 300:
            continue
        aspect = w / max(h, 1)
        if y > height * 0.83:
            continue
        if y < height * 0.08:
            continue
        if aspect > 8.0 and h < 80:
            continue
        components.append((x, y, w, h, area))
    if not components:
        return crop_to_content(image, margin=margin)
    components.sort(key=lambda item: item[4], reverse=True)
    selected = components[:4]
    xs: list[int] = []
    ys: list[int] = []
    for x, y, w, h, _area in selected:
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(width, max(xs) + margin)
    y2 = min(height, max(ys) + margin)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def _remove_yellow_marker(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 60, 60], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))
    if cv2.countNonZero(mask) == 0:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def skeletonize(binary: np.ndarray) -> np.ndarray:
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def preprocess(image: np.ndarray, *, remove_yellow: bool = True) -> dict[str, np.ndarray]:
    working = _remove_yellow_marker(image) if remove_yellow else image
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    color = color_feature(working)
    color_mask = visible_color_mask(working)
    return {"edges": edges, "skeleton": skeletonize(edges), "color": color, "color_mask": color_mask}


def color_feature(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 0.55
    return lab


def visible_color_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 18) & (value > 45)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return mask


def overlap_score(base: np.ndarray, patch: np.ndarray, x: int, y: int) -> float:
    bh, bw = base.shape[:2]
    ph, pw = patch.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + pw)
    y2 = min(bh, y + ph)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    a = base[y1:y2, x1:x2]
    b = patch[sy1:sy2, sx1:sx2]
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum()) + 1e-6
    return inter / union


def place_scaled(image: np.ndarray, base_shape: tuple[int, ...], scale: float, top_left: tuple[int, int]) -> np.ndarray:
    scaled = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros(base_shape, dtype=image.dtype)
    x, y = top_left
    h, w = scaled.shape[:2]
    bh, bw = base_shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + w)
    y2 = min(bh, y + h)
    if x2 <= x1 or y2 <= y1:
        return canvas
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    canvas[y1:y2, x1:x2] = scaled[sy1:sy2, sx1:sx2]
    return canvas


def edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float32) / 255.0
    b_f = b.astype(np.float32) / 255.0
    inter = float(np.minimum(a_f, b_f).sum())
    denom = float(np.maximum(a_f, b_f).sum()) + 1e-6
    return inter / denom


def skeleton_similarity(template_skel: np.ndarray, placed_unknown_skel: np.ndarray) -> float:
    points = np.column_stack(np.where(placed_unknown_skel > 0))
    if len(points) == 0:
        return 0.0
    dist = cv2.distanceTransform(cv2.bitwise_not(template_skel), cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    vals = dist[points[:, 0], points[:, 1]]
    return float(np.exp(-float(vals.mean()) * 4.0))


def top_coarse_candidates(
    unknown_edges: np.ndarray,
    template_edges: np.ndarray,
    scales: Iterable[float],
    top_k: int = 12,
) -> list[tuple[float, float, tuple[int, int]]]:
    candidates: list[tuple[float, float, tuple[int, int]]] = []
    for scale in scales:
        scaled = cv2.resize(
            unknown_edges,
            (max(1, int(round(unknown_edges.shape[1] * scale))), max(1, int(round(unknown_edges.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        ph, pw = scaled.shape[:2]
        pad_x = max(8, pw // 2)
        pad_y = max(8, ph // 2)
        padded = cv2.copyMakeBorder(template_edges, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        if padded.shape[0] < ph or padded.shape[1] < pw:
            continue
        response = cv2.matchTemplate(padded, scaled, cv2.TM_CCOEFF_NORMED)
        local = response.copy()
        per_scale: list[tuple[float, float, tuple[int, int]]] = []
        for _ in range(max(top_k * 2, 8)):
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(local)
            if max_val < 0.05:
                break
            x = int(max_loc[0] - pad_x)
            y = int(max_loc[1] - pad_y)
            overlap = overlap_score(template_edges, scaled, x, y)
            score = 0.55 * float(max_val) + 0.45 * overlap
            per_scale.append((score, float(scale), (x, y)))
            suppress = max(16, min(ph, pw) // 5)
            lx1 = max(0, max_loc[0] - suppress)
            ly1 = max(0, max_loc[1] - suppress)
            lx2 = min(local.shape[1], max_loc[0] + suppress + 1)
            ly2 = min(local.shape[0], max_loc[1] + suppress + 1)
            local[ly1:ly2, lx1:lx2] = -1.0
        candidates.extend(per_scale[:top_k])
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique: list[tuple[float, float, tuple[int, int]]] = []
    for candidate in candidates:
        _score, scale, top_left = candidate
        if any(abs(scale - old[1]) < 1e-6 and abs(top_left[0] - old[2][0]) + abs(top_left[1] - old[2][1]) < 24 for old in unique):
            continue
        unique.append(candidate)
        if len(unique) >= top_k:
            break
    return unique


def top_color_candidates(
    unknown_color: np.ndarray,
    unknown_mask: np.ndarray,
    template_color: np.ndarray,
    scales: Iterable[float],
    top_k: int = 12,
) -> list[tuple[float, float, tuple[int, int]]]:
    candidates: list[tuple[float, float, tuple[int, int]]] = []
    template_f = template_color.astype(np.float32)
    for scale in scales:
        scaled_color = cv2.resize(
            unknown_color,
            (max(1, int(round(unknown_color.shape[1] * scale))), max(1, int(round(unknown_color.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        scaled_mask = cv2.resize(
            unknown_mask,
            (scaled_color.shape[1], scaled_color.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if cv2.countNonZero(scaled_mask) < 80:
            continue
        ph, pw = scaled_color.shape[:2]
        pad_x = max(8, pw // 2)
        pad_y = max(8, ph // 2)
        padded = cv2.copyMakeBorder(template_f, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        if padded.shape[0] < ph or padded.shape[1] < pw:
            continue
        response = cv2.matchTemplate(padded, scaled_color, cv2.TM_CCORR_NORMED, mask=scaled_mask)
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        local = response.copy()
        for _ in range(max(top_k * 2, 8)):
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(local)
            if max_val < 0.10:
                break
            x = int(max_loc[0] - pad_x)
            y = int(max_loc[1] - pad_y)
            candidates.append((float(max_val), float(scale), (x, y)))
            suppress = max(16, min(ph, pw) // 5)
            lx1 = max(0, max_loc[0] - suppress)
            ly1 = max(0, max_loc[1] - suppress)
            lx2 = min(local.shape[1], max_loc[0] + suppress + 1)
            ly2 = min(local.shape[0], max_loc[1] + suppress + 1)
            local[ly1:ly2, lx1:lx2] = -1.0
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique: list[tuple[float, float, tuple[int, int]]] = []
    for candidate in candidates:
        _score, scale, top_left = candidate
        if any(abs(scale - old[1]) < 1e-6 and abs(top_left[0] - old[2][0]) + abs(top_left[1] - old[2][1]) < 24 for old in unique):
            continue
        unique.append(candidate)
        if len(unique) >= top_k:
            break
    return unique


def merge_candidate_seeds(*groups: list[tuple[float, float, tuple[int, int]]], top_k: int = 30) -> list[tuple[float, float, tuple[int, int]]]:
    seeds: list[tuple[float, float, tuple[int, int]]] = []
    for group in groups:
        seeds.extend(group)
    seeds.sort(key=lambda item: item[0], reverse=True)
    unique: list[tuple[float, float, tuple[int, int]]] = []
    for seed in seeds:
        _score, scale, top_left = seed
        if any(abs(scale - old[1]) < 1e-6 and abs(top_left[0] - old[2][0]) + abs(top_left[1] - old[2][1]) < 18 for old in unique):
            continue
        unique.append(seed)
        if len(unique) >= top_k:
            break
    return unique


def competing_second_score(candidates: list[StructureCandidate], best: StructureCandidate, min_distance: float = 55.0) -> float:
    for candidate in candidates[1:]:
        distance = float(np.hypot(candidate.top_left[0] - best.top_left[0], candidate.top_left[1] - best.top_left[1]))
        if distance >= min_distance:
            return candidate.score
    return candidates[1].score if len(candidates) > 1 else 0.0


def crop_overlap(base: np.ndarray, patch: np.ndarray, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    bh, bw = base.shape[:2]
    ph, pw = patch.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + pw)
    y2 = min(bh, y + ph)
    if x2 <= x1 or y2 <= y1:
        return base[:0, :0], patch[:0, :0]
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    return base[y1:y2, x1:x2], patch[sy1:sy2, sx1:sx2]


def color_similarity(
    template_color: np.ndarray,
    unknown_color: np.ndarray,
    unknown_mask: np.ndarray,
    scale: float,
    top_left: tuple[int, int],
) -> float:
    scaled_color = cv2.resize(
        unknown_color,
        (max(1, int(round(unknown_color.shape[1] * scale))), max(1, int(round(unknown_color.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    scaled_mask = cv2.resize(
        unknown_mask,
        (scaled_color.shape[1], scaled_color.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    tpl_roi, unk_roi = crop_overlap(template_color, scaled_color, top_left[0], top_left[1])
    _tpl_mask_roi, mask_roi = crop_overlap(np.zeros(template_color.shape[:2], dtype=np.uint8), scaled_mask, top_left[0], top_left[1])
    if tpl_roi.size == 0 or mask_roi.size == 0:
        return 0.0
    valid = mask_roi > 0
    if int(valid.sum()) < 80:
        return 0.0
    diff = np.linalg.norm(tpl_roi[valid] - unk_roi[valid], axis=1)
    mean_diff = float(np.mean(np.clip(diff, 0.0, 120.0)))
    return float(np.exp(-mean_diff / 34.0))


def rerank_candidates(
    coarse_candidates: list[tuple[float, float, tuple[int, int]]],
    unknown_features: dict[str, np.ndarray],
    template_features: dict[str, np.ndarray],
    *,
    color_weight: float = 0.25,
    edge_weight: float = 0.35,
    skeleton_weight: float = 0.30,
    coarse_weight: float = 0.10,
) -> list[StructureCandidate]:
    ranked: list[StructureCandidate] = []
    total = max(1e-6, color_weight + edge_weight + skeleton_weight + coarse_weight)
    for coarse, scale, top_left in coarse_candidates:
        placed_edges = place_scaled(unknown_features["edges"], template_features["edges"].shape, scale, top_left)
        placed_skel = place_scaled(unknown_features["skeleton"], template_features["skeleton"].shape, scale, top_left)
        edge_score = edge_similarity(template_features["edges"], placed_edges)
        skel_score = skeleton_similarity(template_features["skeleton"], placed_skel)
        color_score = color_similarity(
            template_features["color"],
            unknown_features["color"],
            unknown_features["color_mask"],
            scale,
            top_left,
        )
        score = (
            color_weight * color_score
            + edge_weight * edge_score
            + skeleton_weight * skel_score
            + coarse_weight * coarse
        ) / total
        ranked.append(
            StructureCandidate(
                score=float(score),
                scale=float(scale),
                top_left=(int(top_left[0]), int(top_left[1])),
                coarse_score=float(coarse),
                color_score=float(color_score),
                edge_score=float(edge_score),
                skeleton_score=float(skel_score),
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def refine_affine(unknown_img: np.ndarray, template_img: np.ndarray, scale: float) -> Optional[list[list[float]]]:
    unknown_gray = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    unknown_gray = cv2.resize(
        unknown_gray,
        (max(1, int(round(unknown_gray.shape[1] * scale))), max(1, int(round(unknown_gray.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(unknown_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)
    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 6:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    affine, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if affine is None:
        return None
    return affine.tolist()


def make_overlay(full_map: np.ndarray, unknown_crop: np.ndarray, transform: StructureTransform) -> np.ndarray:
    scaled = cv2.resize(
        unknown_crop,
        (max(1, int(round(unknown_crop.shape[1] * transform.scale))), max(1, int(round(unknown_crop.shape[0] * transform.scale)))),
        interpolation=cv2.INTER_AREA,
    )
    out = full_map.copy()
    x = int(round(transform.template_crop_offset[0] + transform.top_left[0]))
    y = int(round(transform.template_crop_offset[1] + transform.top_left[1]))
    h, w = scaled.shape[:2]
    bh, bw = out.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + w)
    y2 = min(bh, y + h)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    roi = out[y1:y2, x1:x2]
    patch = scaled[sy1:sy2, sx1:sx2]
    cv2.addWeighted(patch, 0.5, roi, 0.5, 0, roi)
    return out


def align_structure_to_known(
    *,
    map_id: str,
    unknown_image: np.ndarray,
    template_path: Path,
    scales: Iterable[float],
    enable_affine_refine: bool = True,
) -> StructureRegistrationResult:
    full_map = read_image(template_path)
    template_crop, template_offset = crop_to_content(full_map)
    unknown_crop, unknown_offset = crop_unknown_to_visible_structure(unknown_image)
    template_features = preprocess(template_crop, remove_yellow=False)
    unknown_features = preprocess(unknown_crop, remove_yellow=True)
    edge_candidates = top_coarse_candidates(unknown_features["edges"], template_features["edges"], scales, top_k=14)
    color_candidates = top_color_candidates(
        unknown_features["color"],
        unknown_features["color_mask"],
        template_features["color"],
        scales,
        top_k=18,
    )
    candidate_seeds = merge_candidate_seeds(edge_candidates, color_candidates, top_k=32)
    ranked = rerank_candidates(candidate_seeds, unknown_features, template_features)
    if not ranked:
        raise RuntimeError("结构配准失败：没有找到候选位置")
    best = ranked[0]
    second_score = competing_second_score(ranked, best)
    is_ambiguous = len(ranked) > 1 and (best.score - second_score) < 0.045
    scale = best.scale
    top_left = best.top_left
    score = best.score
    affine = refine_affine(unknown_crop, template_crop, scale) if enable_affine_refine else None
    if affine is not None:
        score += 0.03
    transform = StructureTransform(
        map_id=map_id,
        scale=scale,
        top_left=(float(top_left[0]), float(top_left[1])),
        template_crop_offset=template_offset,
        unknown_crop_offset=unknown_offset,
        score=float(score),
        is_ambiguous=is_ambiguous,
        second_score=float(second_score),
        affine=affine,
    )
    return StructureRegistrationResult(
        transform=transform,
        full_map=full_map,
        template_crop=template_crop,
        unknown_crop=unknown_crop,
        overlay=make_overlay(full_map, unknown_crop, transform),
        unknown_edges=unknown_features["edges"],
        template_edges=template_features["edges"],
        unknown_skeleton=unknown_features["skeleton"],
        template_skeleton=template_features["skeleton"],
        candidates=ranked,
    )

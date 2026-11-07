from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: Optional[Tuple[int, int]] = (220, 220)
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    enable_overlay_mask: bool = True
    overlay_mask_dilate: int = 3
    overlay_inpaint: bool = True
    yellow_hue_min: int = 15
    yellow_hue_max: int = 45
    overlay_saturation_min: int = 70
    overlay_value_min: int = 110
    enable_canny: bool = True
    canny_threshold1: int = 60
    canny_threshold2: int = 160
    enable_morph_close: bool = True
    morph_kernel_size: int = 3
    dark_mask_threshold: int = 12
    enable_binary: bool = False
    binary_threshold: int = 120


def preprocess_config_from_dict(data: Mapping[str, Any] | None) -> PreprocessConfig:
    data = data or {}
    target = data.get("target_size", [220, 220])
    target_size: Optional[Tuple[int, int]]
    if target is None:
        target_size = None
    else:
        target_size = (int(target[0]), int(target[1]))
    tile = data.get("clahe_tile_grid_size", [8, 8])
    return PreprocessConfig(
        target_size=target_size,
        enable_clahe=bool(data.get("enable_clahe", True)),
        clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=(int(tile[0]), int(tile[1])),
        enable_overlay_mask=bool(data.get("enable_overlay_mask", True)),
        overlay_mask_dilate=int(data.get("overlay_mask_dilate", 3)),
        overlay_inpaint=bool(data.get("overlay_inpaint", True)),
        yellow_hue_min=int(data.get("yellow_hue_min", 15)),
        yellow_hue_max=int(data.get("yellow_hue_max", 45)),
        overlay_saturation_min=int(data.get("overlay_saturation_min", 70)),
        overlay_value_min=int(data.get("overlay_value_min", 110)),
        enable_canny=bool(data.get("enable_canny", True)),
        canny_threshold1=int(data.get("canny_threshold1", 60)),
        canny_threshold2=int(data.get("canny_threshold2", 160)),
        enable_morph_close=bool(data.get("enable_morph_close", True)),
        morph_kernel_size=int(data.get("morph_kernel_size", 3)),
        dark_mask_threshold=int(data.get("dark_mask_threshold", 12)),
        enable_binary=bool(data.get("enable_binary", False)),
        binary_threshold=int(data.get("binary_threshold", 120)),
    )


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_normalized(image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if target_size is None:
        return image.copy()
    width, height = target_size
    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def suppress_dark_regions(gray: np.ndarray, threshold: int) -> np.ndarray:
    if threshold <= 0:
        return gray
    out = gray.copy()
    out[out < threshold] = 0
    return out


def apply_clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid_size,
    )
    return clahe.apply(gray)


def overlay_mask(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Return a mask for bright saturated annotations, especially yellow route/player marks."""
    if image.ndim == 2:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    yellow = (
        (h >= cfg.yellow_hue_min)
        & (h <= cfg.yellow_hue_max)
        & (s >= cfg.overlay_saturation_min)
        & (v >= cfg.overlay_value_min)
    )
    # Cyan, green, pink route boxes in template images are also high-saturation
    # overlays. Keep this conservative so gray/blue map walls survive.
    high_sat = (s >= max(110, cfg.overlay_saturation_min + 25)) & (v >= max(130, cfg.overlay_value_min))
    mask = np.where(yellow | high_sat, 255, 0).astype(np.uint8)
    if cfg.overlay_mask_dilate > 0:
        k = 2 * cfg.overlay_mask_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def remove_overlay_annotations(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Suppress colored route text, player markers, and annotation boxes before structure matching."""
    if image.ndim == 2 or not cfg.enable_overlay_mask:
        return image.copy()
    mask = overlay_mask(image, cfg)
    if not np.any(mask):
        return image.copy()
    if cfg.overlay_inpaint:
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    out = image.copy()
    # A neutral dark gray is safer than black because Canny would otherwise add
    # artificial hard borders around removed labels.
    out[mask > 0] = (35, 35, 35)
    return out


def normalize_for_structure(image: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Return a cleaned, resized grayscale map image before edge extraction."""
    cfg = cfg or PreprocessConfig()
    cleaned = remove_overlay_annotations(image, cfg)
    gray = to_gray(cleaned)
    gray = resize_normalized(gray, cfg.target_size)
    gray = suppress_dark_regions(gray, cfg.dark_mask_threshold)
    if cfg.enable_clahe:
        gray = apply_clahe(gray, cfg)
    return gray


def preprocess_image(image: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Return a single-channel processed image optimized for structural matching."""
    cfg = cfg or PreprocessConfig()
    gray = normalize_for_structure(image, cfg)

    if cfg.enable_binary:
        _, gray = cv2.threshold(gray, cfg.binary_threshold, 255, cv2.THRESH_BINARY)

    if cfg.enable_canny:
        gray = cv2.Canny(gray, cfg.canny_threshold1, cfg.canny_threshold2)
        if cfg.enable_morph_close and cfg.morph_kernel_size > 1:
            k = cfg.morph_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray


def locate_door_anchor(image: np.ndarray, cfg: PreprocessConfig | None = None) -> tuple[Optional[tuple[float, float]], float]:
    """Estimate a normalized door-icon anchor in an ROI.

    The icon is usually a compact bright/white mark near one side of the visible
    entrance corridor. This is deliberately a weak signal: if no plausible icon is
    found, confidence is 0 and callers should rely on structure matching.
    """
    cfg = cfg or PreprocessConfig()

    # First try the dedicated side-door glyph detector. It uses the real door
    # icon templates from assets/icons and is much less likely to confuse the
    # yellow player marker with the side door.
    try:
        from .door_icon_detector import detect_side_door_icon

        detection = detect_side_door_icon(image, cfg, kind="game")
        # The glyph matcher is useful when very confident, but at lower scores
        # it may match wall corners. Fall back to the color-aware component
        # detector below in those cases.
        if detection is not None and detection.confidence >= 0.70:
            return detection.anchor, detection.confidence
    except Exception:
        # Keep preprocessing usable even if icon assets are missing; fall back to
        # the heuristic detector below.
        pass

    # Prefer a color-aware detector on gameplay/template RGB images. The side
    # door icon is a compact low-saturation white/gray mark, while the player's
    # marker is yellow/high-saturation and must not become the anchor.
    if image.ndim == 3 and image.size > 0:
        raw_h, raw_w = image.shape[:2]
        if raw_h >= 16 and raw_w >= 16:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)
            raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            yellow = (
                (hue >= cfg.yellow_hue_min)
                & (hue <= cfg.yellow_hue_max)
                & (sat >= cfg.overlay_saturation_min)
                & (val >= cfg.overlay_value_min)
            ).astype(np.uint8) * 255
            if cfg.overlay_mask_dilate > 0:
                k = 2 * cfg.overlay_mask_dilate + 1
                yellow = cv2.dilate(yellow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)

            # Door icons survive at higher value thresholds and have much lower
            # saturation than route/player overlays. Use two thresholds so dim
            # anti-aliased game icons are still detected.
            best_center_color: Optional[tuple[float, float]] = None
            best_score_color = 0.0
            for value_threshold in (155, 140, 125, 110):
                white_gray = np.where(
                    (raw_gray >= value_threshold)
                    & (sat <= 95)
                    & (yellow == 0),
                    255,
                    0,
                ).astype(np.uint8)
                white_gray = cv2.morphologyEx(
                    white_gray,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                    iterations=1,
                )
                n, labels, stats, centroids = cv2.connectedComponentsWithStats(white_gray, 8)
                img_area = float(raw_w * raw_h)
                for i in range(1, n):
                    x, y, bw, bh, area = stats[i]
                    if area < 12 or area > img_area * 0.025:
                        continue
                    if bw < 3 or bh < 3:
                        continue
                    if bw > raw_w * 0.22 or bh > raw_h * 0.25:
                        continue
                    aspect = bw / max(bh, 1)
                    if aspect < 0.18 or aspect > 4.5:
                        continue
                    component = labels == i
                    mean_sat = float(sat[component].mean()) if np.any(component) else 255.0
                    mean_val = float(raw_gray[component].mean()) if np.any(component) else 0.0
                    density = area / max(bw * bh, 1)
                    compact = min(1.0, max(0.0, 1.0 - abs(np.log(max(aspect, 1e-3))) / 2.5))
                    cx, cy = centroids[i]
                    border_dist = min(cx, cy, raw_w - cx, raw_h - cy) / max(min(raw_w, raw_h), 1)
                    border_bonus = 1.0 - min(1.0, border_dist * 2.6)

                    # Door icons often contain a vertical bar plus arrow, so the
                    # component is not a huge filled blob. Penalize saturated or
                    # very low-value candidates, which catches player/route UI.
                    saturation_score = 1.0 - min(1.0, mean_sat / 105.0)
                    value_score = min(1.0, max(0.0, (mean_val - 95.0) / 90.0))
                    size_score = min(1.0, area / max(img_area * 0.0018, 1.0))
                    score = (
                        0.24 * density
                        + 0.20 * compact
                        + 0.18 * border_bonus
                        + 0.18 * saturation_score
                        + 0.12 * value_score
                        + 0.08 * size_score
                    )
                    if score > best_score_color:
                        best_score_color = float(score)
                        best_center_color = (float(cx / raw_w), float(cy / raw_h))
                if best_center_color is not None and best_score_color >= 0.34:
                    return best_center_color, min(1.0, best_score_color)

    gray = normalize_for_structure(image, cfg)
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return None, 0.0

    threshold = max(105, int(np.percentile(gray, 88)))
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_center: Optional[tuple[float, float]] = None
    best_score = 0.0
    img_area = float(w * h)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 10 or area > img_area * 0.08:
            continue
        if bw > w * 0.35 or bh > h * 0.35:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue
        density = area / max(bw * bh, 1)
        compact = min(1.0, max(0.0, 1.0 - abs(np.log(max(aspect, 1e-3))) / 2.2))
        cx, cy = centroids[i]
        border_dist = min(cx, cy, w - cx, h - cy) / max(min(w, h), 1)
        border_bonus = 1.0 - min(1.0, border_dist * 3.0)
        score = 0.45 * density + 0.30 * compact + 0.25 * border_bonus
        if score > best_score:
            best_score = float(score)
            best_center = (float(cx / w), float(cy / h))

    if best_center is None or best_score < 0.25:
        return None, 0.0
    return best_center, min(1.0, best_score)


def visible_structure_bbox(
    image: np.ndarray,
    cfg: PreprocessConfig | None = None,
    padding: int = 18,
) -> Optional[tuple[int, int, int, int]]:
    """Find the tight bounding box of visible map structure inside a foggy ROI.

    This is designed for cases where the user-selected current_map_feature_roi is
    intentionally generous and contains lots of fog/empty background. We detect
    bright map-wall and icon structure, then crop around it before normalization.
    """
    cfg = cfg or PreprocessConfig()
    gray = normalize_for_structure(image, cfg)
    if gray.size == 0:
        return None

    # Work before final target normalization if possible. normalize_for_structure
    # resizes, so instead build a mask directly on the original cleaned image.
    cleaned = remove_overlay_annotations(image, cfg)
    raw_gray = to_gray(cleaned)
    raw_gray = suppress_dark_regions(raw_gray, cfg.dark_mask_threshold)
    if cfg.enable_clahe:
        raw_gray = apply_clahe(raw_gray, cfg)

    edges = cv2.Canny(raw_gray, max(30, cfg.canny_threshold1 // 2), max(90, cfg.canny_threshold2))
    bright_threshold = max(65, int(np.percentile(raw_gray, 82)))
    bright = np.where(raw_gray >= bright_threshold, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(edges, bright)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    h, w = raw_gray.shape[:2]
    img_area = h * w
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(30, img_area * 0.0008):
            continue
        if area > img_area * 0.9:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh), int(area)))
    if not boxes:
        return None

    # Keep components near the largest structure. This avoids UI speckles if they
    # sneak into the ROI, while preserving the door icon and adjacent corridor.
    boxes.sort(key=lambda b: b[4], reverse=True)
    keep = boxes[: min(6, len(boxes))]
    x1 = min(b[0] for b in keep)
    y1 = min(b[1] for b in keep)
    x2 = max(b[0] + b[2] for b in keep)
    y2 = max(b[1] + b[3] for b in keep)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2 - x1, y2 - y1


def crop_visible_structure(
    image: np.ndarray,
    cfg: PreprocessConfig | None = None,
    padding: int = 18,
) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    bbox = visible_structure_bbox(image, cfg, padding)
    if bbox is None:
        return image.copy(), None
    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy(), bbox

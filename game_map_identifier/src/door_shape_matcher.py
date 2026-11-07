from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .door_icon_detector import detect_side_door_icon
from .preprocess import PreprocessConfig, locate_door_anchor, overlay_mask, remove_overlay_annotations, to_gray, visible_structure_bbox


@dataclass(frozen=True)
class DoorShapeScore:
    score: float
    direction_score: float
    patch_score: float
    edge_score: float
    profile_score: float
    branch_score: float
    current_direction: str
    template_direction: str


@dataclass(frozen=True)
class DoorShapeDescriptor:
    gray: np.ndarray
    mask: np.ndarray
    edges: np.ndarray
    row_profile: np.ndarray
    col_profile: np.ndarray
    branch_signature: np.ndarray
    southwest_patch: np.ndarray
    southwest_structure: float
    direction: str
    confidence: float


def _visible_structure_mask(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if image.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    yellow = overlay_mask(bgr, cfg)
    gray = to_gray(bgr)
    gray[yellow > 0] = 0
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # The explored corridor/floor is consistently brighter than surrounding fog.
    p70 = float(np.percentile(gray[gray > 0], 70)) if np.any(gray > 0) else 70.0
    threshold = max(42.0, min(95.0, p70 * 0.82))
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    mask[yellow > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return mask


def _estimate_direction(mask: np.ndarray, cx: int, cy: int, icon_size: int) -> str:
    h, w = mask.shape[:2]
    r = max(18, int(icon_size * 1.2))
    gap = max(8, int(icon_size * 0.45))
    boxes = {
        "right": (min(w, cx + gap), max(0, cy - r), min(w, cx + 7 * r), min(h, cy + r)),
        "left": (max(0, cx - 7 * r), max(0, cy - r), max(0, cx - gap), min(h, cy + r)),
        "down": (max(0, cx - r), min(h, cy + gap), min(w, cx + r), min(h, cy + 7 * r)),
        "up": (max(0, cx - r), max(0, cy - 7 * r), min(w, cx + r), max(0, cy - gap)),
    }
    scores: dict[str, float] = {}
    for key, (x0, y0, x1, y1) in boxes.items():
        patch = mask[y0:y1, x0:x1]
        scores[key] = float((patch > 0).sum()) / max(float(patch.size), 1.0)
    return max(scores, key=scores.get)


def _estimate_direction_from_layout(
    image: np.ndarray,
    cfg: PreprocessConfig,
    cx: int,
    cy: int,
    fallback_mask: np.ndarray,
    icon_size: int,
) -> str:
    bbox = visible_structure_bbox(image, cfg, padding=8)
    if bbox is None:
        return _estimate_direction(fallback_mask, cx, cy, icon_size)
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return _estimate_direction(fallback_mask, cx, cy, icon_size)
    rel_x = (cx - x) / max(w, 1)
    rel_y = (cy - y) / max(h, 1)
    if w >= h * 0.80:
        if rel_x <= 0.46:
            return "right"
        if rel_x >= 0.54:
            return "left"
    if h >= w * 0.80:
        if rel_y <= 0.46:
            return "down"
        if rel_y >= 0.54:
            return "up"
    return _estimate_direction(fallback_mask, cx, cy, icon_size)


def _locate_real_door(image: np.ndarray, cfg: PreprocessConfig) -> tuple[Optional[tuple[float, float]], float, int]:
    detection = detect_side_door_icon(
        image,
        cfg,
        kind="game",
        scales=(0.20, 0.25, 0.30, 0.35, 0.45, 0.55),
    )
    if detection is not None and detection.confidence >= 0.38:
        return detection.anchor, float(detection.confidence), max(18, int(max(detection.bbox[2], detection.bbox[3])))

    anchor, confidence = locate_door_anchor(image, cfg)
    if anchor is None:
        return None, 0.0, 0
    h, w = image.shape[:2]
    return anchor, float(confidence), max(18, int(min(h, w) * 0.055))


def _oriented_patch(image: np.ndarray, cfg: PreprocessConfig) -> Optional[tuple[np.ndarray, np.ndarray, str, float]]:
    anchor, confidence, icon = _locate_real_door(image, cfg)
    if anchor is None or confidence < 0.20:
        return None
    h, w = image.shape[:2]
    cx = int(round(anchor[0] * w))
    cy = int(round(anchor[1] * h))
    mask = _visible_structure_mask(image, cfg)
    direction = _estimate_direction_from_layout(image, cfg, cx, cy, mask, icon)

    # Crop a door-centered window that includes a little space behind the door
    # and a longer look-ahead through the entrance corridor.
    back = int(icon * 1.4)
    ahead = int(icon * 8.5)
    half = int(icon * 3.0)
    if direction == "right":
        x0, x1 = cx - back, cx + ahead
        y0, y1 = cy - half, cy + half
        rotate = None
    elif direction == "left":
        x0, x1 = cx - ahead, cx + back
        y0, y1 = cy - half, cy + half
        rotate = "flip_h"
    elif direction == "down":
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - back, cy + ahead
        rotate = "ccw"
    else:
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - ahead, cy + back
        rotate = "cw"

    pad = max(icon * 4, 80)
    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(18, 26, 34))
    padded_mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x0 += pad
    x1 += pad
    y0 += pad
    y1 += pad
    patch = padded_img[y0:y1, x0:x1]
    patch_mask = padded_mask[y0:y1, x0:x1]
    if patch.size == 0 or patch_mask.size == 0:
        return None

    if rotate == "flip_h":
        patch = cv2.flip(patch, 1)
        patch_mask = cv2.flip(patch_mask, 1)
    elif rotate == "ccw":
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        patch_mask = cv2.rotate(patch_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == "cw":
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        patch_mask = cv2.rotate(patch_mask, cv2.ROTATE_90_CLOCKWISE)

    target = (260, 130)
    patch = cv2.resize(patch, target, interpolation=cv2.INTER_AREA)
    patch_mask = cv2.resize(patch_mask, target, interpolation=cv2.INTER_NEAREST)
    return patch, patch_mask, direction, float(confidence)


def _southwest_descriptor(patch: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    cleaned = remove_overlay_annotations(patch, cfg)
    gray = to_gray(cleaned)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    # After orientation normalization, the real door is near the left side and
    # the player's corridor extends right. The map_06 discriminator is the small
    # opposite structure below-left / southwest of that door area.
    x0 = int(w * 0.02)
    x1 = int(w * 0.34)
    y0 = int(h * 0.50)
    y1 = int(h * 0.96)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros((48, 80), dtype=np.float32), 0.0
    norm = cv2.equalizeHist(roi)
    resized = cv2.resize(norm, (80, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    resized = (resized - float(resized.mean())) / max(float(resized.std()), 1e-4)
    edges = cv2.Canny(norm, 42, 126)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))
    edge_density = float((edges > 0).mean())
    line_density = float(((vertical > 0) | (horizontal > 0)).mean())
    bright_spread = float((norm > max(70, int(np.percentile(norm, 72)))).mean())
    structure = max(0.0, min(1.0, 7.0 * edge_density + 9.0 * line_density + 0.35 * bright_spread))
    return resized, structure


def extract_door_shape_descriptor(image: np.ndarray, cfg: PreprocessConfig) -> Optional[DoorShapeDescriptor]:
    oriented = _oriented_patch(image, cfg)
    if oriented is None:
        return None
    patch, mask, direction, confidence = oriented
    cleaned = remove_overlay_annotations(patch, cfg)
    gray = to_gray(cleaned)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    gray_small = cv2.resize(gray, (130, 65), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray_small = (gray_small - float(gray_small.mean())) / max(float(gray_small.std()), 1e-4)

    edge_u8 = cv2.Canny(gray, 45, 135)
    edge_u8 = cv2.morphologyEx(edge_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    edges = (edge_u8 > 0).astype(np.uint8)

    mask = (mask > 0).astype(np.uint8)
    if int(edges.sum()) < 30:
        return None
    row_profile = edges.mean(axis=1).astype(np.float32)
    col_profile = edges.mean(axis=0).astype(np.float32)

    h, w = edges.shape[:2]
    zones: list[float] = []
    for y_part in np.array_split(edges, 4, axis=0):
        for cell in np.array_split(y_part, 4, axis=1):
            zones.append(float(cell.mean()))

    # Extra hand-shaped cues for the exact reasoning here: a clean horizontal
    # entrance corridor has little branch energy above/below the center band,
    # while the far end tends to show a vertical wall/fog edge.
    top = edges[: h // 3, w // 4 :]
    mid = edges[h // 3 : 2 * h // 3, w // 4 :]
    bottom = edges[2 * h // 3 :, w // 4 :]
    right_end = edges[:, -max(8, w // 7) :]
    left_door = edges[:, : max(8, w // 6)]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
    vertical_lines = cv2.morphologyEx(edge_u8, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(edge_u8, cv2.MORPH_OPEN, horizontal_kernel)
    zones.extend(
        [
            float(top.mean()),
            float(mid.mean()),
            float(bottom.mean()),
            float(right_end.mean()),
            float(left_door.mean()),
            float(vertical_lines[:, -max(8, w // 5) :].mean() / 255.0),
            float(horizontal_lines[h // 3 : 2 * h // 3, :].mean() / 255.0),
        ]
    )
    branch_signature = np.asarray(zones, dtype=np.float32)
    southwest_patch, southwest_structure = _southwest_descriptor(patch, cfg)
    return DoorShapeDescriptor(
        gray=gray_small,
        mask=mask,
        edges=edges,
        row_profile=row_profile,
        col_profile=col_profile,
        branch_signature=branch_signature,
        southwest_patch=southwest_patch,
        southwest_structure=southwest_structure,
        direction=direction,
        confidence=confidence,
    )


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


def _edge_distance_score(a: np.ndarray, b: np.ndarray) -> float:
    if int(a.sum()) < 8 or int(b.sum()) < 8:
        return 0.0
    dt_b = cv2.distanceTransform(1 - b.astype(np.uint8), cv2.DIST_L2, 3)
    dt_a = cv2.distanceTransform(1 - a.astype(np.uint8), cv2.DIST_L2, 3)
    d1 = float(np.mean(dt_b[a > 0]))
    d2 = float(np.mean(dt_a[b > 0]))
    norm = max(float(np.hypot(*a.shape[:2])) * 0.035, 1.0)
    return float(np.exp(-((d1 + d2) / 2.0) / norm))


def _profile_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b.reshape(1, -1), (a.size, 1), interpolation=cv2.INTER_AREA).ravel()
    denom = max(float(np.max(a)), float(np.max(b)), 1e-6)
    l1 = float(np.mean(np.abs(a - b))) / denom
    return max(0.0, min(1.0, 0.45 * _cosine01(a, b) + 0.55 * float(np.exp(-l1 / 0.32))))


def _signature_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    l1 = float(np.mean(np.abs(a - b)))
    cosine = _cosine01(a, b)
    return max(0.0, min(1.0, 0.40 * cosine + 0.60 * float(np.exp(-l1 / 0.025))))


def _gray_patch_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    corr = float(np.mean(a * b) / max(float(a.std() * b.std()), 1e-6))
    corr_score = max(0.0, min(1.0, (corr + 1.0) * 0.5))
    diff_score = float(np.exp(-float(np.mean(np.abs(a - b))) / 0.95))
    return max(0.0, min(1.0, 0.62 * corr_score + 0.38 * diff_score))


def _southwest_score(cur: DoorShapeDescriptor, tpl: DoorShapeDescriptor) -> float:
    patch = _gray_patch_score(cur.southwest_patch, tpl.southwest_patch)
    presence = float(np.exp(-abs(cur.southwest_structure - tpl.southwest_structure) / 0.18))
    return max(0.0, min(1.0, 0.54 * patch + 0.46 * presence))


def door_shape_score(current: np.ndarray, template: np.ndarray, cfg: PreprocessConfig) -> DoorShapeScore:
    cur = extract_door_shape_descriptor(current, cfg)
    tpl = extract_door_shape_descriptor(template, cfg)
    if cur is None or tpl is None:
        return DoorShapeScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cur.direction if cur else "", tpl.direction if tpl else "")

    patch = _gray_patch_score(cur.gray, tpl.gray)
    edge = _edge_distance_score(cur.edges, tpl.edges)
    profile = 0.5 * _profile_score(cur.row_profile, tpl.row_profile) + 0.5 * _profile_score(cur.col_profile, tpl.col_profile)
    branch = _signature_score(cur.branch_signature, tpl.branch_signature)
    southwest = _southwest_score(cur, tpl)
    direction = 1.0 if cur.direction == tpl.direction else 0.15
    confidence = min(1.0, 0.55 + 0.25 * cur.confidence + 0.20 * tpl.confidence)
    score = (0.24 * patch + 0.20 * edge + 0.14 * profile + 0.16 * branch + 0.18 * southwest + 0.08 * direction) * confidence
    return DoorShapeScore(
        score=max(0.0, min(1.0, float(score))),
        direction_score=direction,
        patch_score=float(patch),
        edge_score=float(edge),
        profile_score=float(profile),
        branch_score=float(southwest),
        current_direction=cur.direction,
        template_direction=tpl.direction,
    )

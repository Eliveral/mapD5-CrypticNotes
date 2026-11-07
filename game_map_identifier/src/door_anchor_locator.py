from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .door_icon_detector import detect_side_door_icon
from .preprocess import PreprocessConfig, locate_door_anchor
from .utils import ROI


Point = Tuple[float, float]


@dataclass(frozen=True)
class SideDoorAnchor:
    local_point: Point
    absolute_point: Point
    normalized_point: Point
    confidence: float
    method: str
    bbox: Optional[ROI] = None
    source_roi: Optional[ROI] = None


def _point_to_bbox(point: Point, image_shape: tuple[int, ...], radius: int = 8) -> ROI:
    h, w = image_shape[:2]
    cx, cy = point
    x1 = max(0, int(round(cx)) - radius)
    y1 = max(0, int(round(cy)) - radius)
    x2 = min(w, int(round(cx)) + radius)
    y2 = min(h, int(round(cy)) + radius)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _clip_roi(image_shape: tuple[int, ...], roi: ROI) -> ROI:
    h, w = image_shape[:2]
    x, y, rw, rh = roi
    x1 = max(0, min(w - 1, int(x)))
    y1 = max(0, min(h - 1, int(y)))
    x2 = max(x1 + 1, min(w, int(x + rw)))
    y2 = max(y1 + 1, min(h, int(y + rh)))
    return x1, y1, x2 - x1, y2 - y1


def locate_side_door_anchor(
    image: np.ndarray,
    cfg: PreprocessConfig,
    *,
    origin: tuple[int, int] = (0, 0),
    source_roi: Optional[ROI] = None,
    kind: str = "any",
) -> Optional[SideDoorAnchor]:
    """Locate the side-door anchor in one ROI/image.

    The returned absolute point is `origin + local_point`, so callers can pass a
    full-map ROI origin and get full-map coordinates directly.
    """
    if image.size == 0:
        return None

    h, w = image.shape[:2]
    ox, oy = origin

    icon = detect_side_door_icon(image, cfg, kind=kind)
    if icon is not None:
        nx, ny = icon.anchor
        local = (float(nx * w), float(ny * h))
        return SideDoorAnchor(
            local_point=local,
            absolute_point=(float(ox + local[0]), float(oy + local[1])),
            normalized_point=(float(nx), float(ny)),
            confidence=float(icon.confidence),
            method=f"icon:{icon.template_name}",
            bbox=icon.bbox,
            source_roi=source_roi,
        )

    anchor, confidence = locate_door_anchor(image, cfg)
    if anchor is None:
        return None

    nx, ny = anchor
    local = (float(nx * w), float(ny * h))
    return SideDoorAnchor(
        local_point=local,
        absolute_point=(float(ox + local[0]), float(oy + local[1])),
        normalized_point=(float(nx), float(ny)),
        confidence=float(confidence),
        method="heuristic",
        bbox=_point_to_bbox(local, image.shape),
        source_roi=source_roi,
    )


def locate_template_side_door_anchor(
    full_map_image: np.ndarray,
    anchor_roi: ROI,
    cfg: PreprocessConfig,
) -> Optional[SideDoorAnchor]:
    clipped_roi = _clip_roi(full_map_image.shape, anchor_roi)
    x, y, w, h = clipped_roi
    patch = full_map_image[y : y + h, x : x + w].copy()
    return locate_side_door_anchor(
        patch,
        cfg,
        origin=(x, y),
        source_roi=clipped_roi,
        kind="template",
    )


def locate_current_side_door_anchor(
    current_roi_image: np.ndarray,
    cfg: PreprocessConfig,
) -> Optional[SideDoorAnchor]:
    return locate_side_door_anchor(
        current_roi_image,
        cfg,
        origin=(0, 0),
        source_roi=(0, 0, current_roi_image.shape[1], current_roi_image.shape[0]),
        kind="game",
    )


def draw_side_door_anchor(
    image: np.ndarray,
    anchor: Optional[SideDoorAnchor],
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    label: str = "side door",
) -> np.ndarray:
    out = image.copy()
    if anchor is None:
        cv2.putText(out, f"{label}: not found", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    if anchor.source_roi is not None:
        rx, ry, rw, rh = anchor.source_roi
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (90, 180, 255), 2)

    ax, ay = anchor.absolute_point
    cx, cy = int(round(ax)), int(round(ay))
    cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 8, color, 2, cv2.LINE_AA)

    if anchor.bbox is not None:
        bx, by, bw, bh = anchor.bbox
        if anchor.source_roi is not None:
            bx += anchor.source_roi[0]
            by += anchor.source_roi[1]
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, 1)

    text = f"{label}: ({ax:.1f}, {ay:.1f}) conf={anchor.confidence:.2f} {anchor.method}"
    cv2.putText(out, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return out

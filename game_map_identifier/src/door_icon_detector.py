from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .preprocess import PreprocessConfig, overlay_mask, resize_normalized, to_gray
from .utils import imread_unicode


@dataclass(frozen=True)
class DoorIconDetection:
    anchor: tuple[float, float]
    confidence: float
    score: float
    bbox: tuple[int, int, int, int]
    template_name: str
    scale: float


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_icon_edges(image: np.ndarray, cfg: PreprocessConfig, *, remove_yellow: bool = True) -> np.ndarray:
    """Prepare a binary/edge image for door-icon matching.

    This intentionally keeps white/gray icon strokes and removes yellow player
    markers/routes. It is separate from general map preprocessing because the
    door icon is a small UI glyph, not a large map structure.
    """
    if image.ndim == 3 and remove_yellow:
        mask = overlay_mask(image, cfg)
        cleaned = image.copy()
        # Neutralize yellow overlays without inpainting tiny glyphs.
        cleaned[mask > 0] = (20, 20, 20)
    else:
        cleaned = image.copy()
    gray = to_gray(cleaned)
    if gray.size == 0:
        return gray
    # Local normalization helps because the gameplay icon is dimmer than the
    # template icon.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    high = max(80, int(np.percentile(gray, 88)))
    _, bright = cv2.threshold(gray, high, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 45, 135)
    combined = cv2.bitwise_or(bright, edges)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return combined


@lru_cache(maxsize=8)
def _load_icon_template(path_str: str, target_long_side: int, cfg_tuple: tuple) -> Optional[np.ndarray]:
    # cfg_tuple is present so cache invalidates if relevant preprocess settings
    # change; values are not directly used here.
    del cfg_tuple
    image = imread_unicode(Path(path_str))
    if image is None:
        return None
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = target_long_side / max(h, w)
    resized = cv2.resize(image, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    # Use a default config for icon internals; caller's cfg is represented in the
    # cache key and used for search image preparation.
    return _prepare_icon_edges(resized, PreprocessConfig(), remove_yellow=True)


def _cfg_cache_tuple(cfg: PreprocessConfig) -> tuple:
    return (
        cfg.yellow_hue_min,
        cfg.yellow_hue_max,
        cfg.overlay_saturation_min,
        cfg.overlay_value_min,
        cfg.overlay_mask_dilate,
    )


def _iter_icon_template_paths(kind: str) -> Iterable[tuple[str, Path]]:
    root = _project_root()
    icon_dir = root / "assets" / "icons"
    if kind == "template":
        names = ["template_side_door.png", "game_side_door.png"]
    elif kind == "game":
        names = ["game_side_door.png"]
    else:
        names = ["game_side_door.png", "template_side_door.png"]
    for name in names:
        path = icon_dir / name
        if path.exists():
            yield name, path


def detect_side_door_icon(
    image: np.ndarray,
    cfg: PreprocessConfig,
    *,
    kind: str = "game",
    scales: Iterable[float] = (0.55, 0.70, 0.85, 1.0, 1.15, 1.30, 1.50),
) -> Optional[DoorIconDetection]:
    """Detect the side-door icon using dedicated glyph templates.

    Returns a normalized anchor in the input image. This detector is used before
    generic connected-component heuristics because player markers and map walls
    can otherwise be mistaken for the door.
    """
    if image.size == 0:
        return None
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        return None

    search = _prepare_icon_edges(image, cfg, remove_yellow=True)
    best: Optional[DoorIconDetection] = None
    target_long_side = max(18, int(min(h, w) * 0.12))
    cfg_key = _cfg_cache_tuple(cfg)

    for template_name, path in _iter_icon_template_paths(kind):
        base_tpl = _load_icon_template(str(path), target_long_side, cfg_key)
        if base_tpl is None or base_tpl.size == 0:
            continue
        for scale in scales:
            scale = float(scale)
            if scale <= 0:
                continue
            tpl = cv2.resize(base_tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = tpl.shape[:2]
            if th < 8 or tw < 8 or th > h or tw > w:
                continue
            try:
                res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            score = float(max_val)
            if score < 0.12:
                continue
            x, y = int(max_loc[0]), int(max_loc[1])
            patch = search[y : y + th, x : x + tw]
            if patch.size == 0:
                continue
            edge_density = float((patch > 0).mean())
            # Door glyphs are compact but not empty. Very dense patches are often
            # wall intersections; very sparse patches are noise.
            density_score = float(np.exp(-abs(edge_density - 0.28) / 0.28))
            border_dist = min(x + tw / 2, y + th / 2, w - (x + tw / 2), h - (y + th / 2)) / max(min(w, h), 1)
            border_bonus = 1.0 - min(1.0, border_dist * 2.4)
            confidence = max(0.0, min(1.0, 0.72 * max(0.0, score) + 0.18 * density_score + 0.10 * border_bonus))
            detection = DoorIconDetection(
                anchor=(float((x + tw / 2) / w), float((y + th / 2) / h)),
                confidence=confidence,
                score=score,
                bbox=(x, y, tw, th),
                template_name=template_name,
                scale=scale,
            )
            if best is None or detection.confidence > best.confidence:
                best = detection

    if best is None or best.confidence < 0.18:
        return None
    return best

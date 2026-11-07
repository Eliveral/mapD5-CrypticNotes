from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

ROI = Tuple[int, int, int, int]


def setup_logging(level: int = logging.INFO) -> None:
    from .runtime_diagnostics import initialize_runtime_diagnostics

    initialize_runtime_diagnostics(level)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def validate_roi(image_shape: Tuple[int, ...], roi: ROI, label: str = "ROI") -> None:
    h, w = image_shape[:2]
    x, y, rw, rh = roi
    if x < 0 or y < 0 or rw <= 0 or rh <= 0:
        raise ValueError(f"{label} 坐标非法：{roi}")
    if x + rw > w or y + rh > h:
        raise ValueError(f"{label} 超出图像范围：{roi}，图像尺寸={w}x{h}")


def crop_roi(image: np.ndarray, roi: ROI, label: str = "ROI") -> np.ndarray:
    validate_roi(image.shape, roi, label)
    x, y, w, h = roi
    return image[y : y + h, x : x + w].copy()


def imread_unicode(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ensure_dir(path.parent)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    with path.open("wb") as f:
        f.write(encoded.tobytes())
    return True


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_roi(image: np.ndarray, roi: ROI, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    out = image.copy()
    x, y, w, h = roi
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    l = to_bgr(left)
    r = to_bgr(right)
    target_h = max(l.shape[0], r.shape[0])
    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        pad_h = target_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    return np.hstack([pad(l), pad(r)])

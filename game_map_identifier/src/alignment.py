from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from .preprocess import PreprocessConfig, remove_overlay_annotations
from .utils import imread_unicode


@dataclass(frozen=True)
class AnchorTransform:
    map_id: str
    scale: float
    known_anchor: tuple[float, float]
    unknown_anchor: tuple[float, float]
    extra_offset: tuple[float, float]
    top_left: tuple[float, float]


@dataclass(frozen=True)
class AlignmentResult:
    transform: AnchorTransform
    full_map: np.ndarray
    scaled_unknown: np.ndarray
    overlay: np.ndarray
    diff: np.ndarray
    heatmap: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件格式无效：{path}")
    return data


def resolve_config_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_dir / path


def path_from_saved(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_dir / path


def load_known_anchor(root_dir: Path, anchors_path: Path, map_id: str) -> tuple[tuple[float, float], Path]:
    data = _read_json(anchors_path)
    anchors = data.get("anchors", {})
    if not isinstance(anchors, Mapping):
        raise RuntimeError(f"锚点配置缺少 anchors：{anchors_path}")
    item = anchors.get(map_id)
    if not isinstance(item, Mapping):
        raise RuntimeError(f"缺少已知侧门锚点：{map_id}")
    point = item.get("point")
    if not isinstance(point, list) or len(point) != 2:
        raise RuntimeError(f"已知侧门锚点格式无效：{map_id}")
    full_image_value = item.get("full_image")
    if not full_image_value:
        raise RuntimeError(f"已知侧门锚点缺少 full_image：{map_id}")
    return (float(point[0]), float(point[1])), path_from_saved(root_dir, str(full_image_value))


def load_scale_config(scales_path: Path, map_id: str, default_scale: float) -> tuple[float, tuple[float, float]]:
    if not scales_path.exists():
        return float(default_scale), (0.0, 0.0)
    data = _read_json(scales_path)
    scales = data.get("scales", {})
    if not isinstance(scales, Mapping):
        return float(default_scale), (0.0, 0.0)
    item = scales.get("global")
    if not isinstance(item, Mapping):
        item = scales.get(map_id)
    if not isinstance(item, Mapping):
        return float(default_scale), (0.0, 0.0)
    scale = float(item.get("scale", default_scale))
    raw_offset = item.get("extra_offset", [0.0, 0.0])
    if isinstance(raw_offset, list) and len(raw_offset) == 2:
        extra_offset = (float(raw_offset[0]), float(raw_offset[1]))
    else:
        extra_offset = (0.0, 0.0)
    return scale, extra_offset


def alpha_overlay(base: np.ndarray, overlay: np.ndarray, top_left: tuple[int, int], alpha: float) -> np.ndarray:
    out = base.copy()
    x, y = top_left
    h, w = overlay.shape[:2]
    base_h, base_w = out.shape[:2]
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(base_w, x + w)
    dst_y2 = min(base_h, y + h)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return out
    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    roi = out[dst_y1:dst_y2, dst_x1:dst_x2]
    src = overlay[src_y1:src_y2, src_x1:src_x2]
    cv2.addWeighted(src, alpha, roi, 1.0 - alpha, 0, roi)
    return out


def _placed_canvas(base_shape: tuple[int, ...], image: np.ndarray, top_left: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros(base_shape, dtype=image.dtype)
    mask = np.zeros(base_shape[:2], dtype=np.uint8)
    x, y = top_left
    h, w = image.shape[:2]
    base_h, base_w = base_shape[:2]
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(base_w, x + w)
    dst_y2 = min(base_h, y + h)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return canvas, mask
    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    mask[dst_y1:dst_y2, dst_x1:dst_x2] = 255
    return canvas, mask


def make_alignment_diff(
    full_map: np.ndarray,
    scaled_unknown: np.ndarray,
    top_left: tuple[int, int],
    preprocess_config: Optional[PreprocessConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    placed, mask = _placed_canvas(full_map.shape, scaled_unknown, top_left)
    cfg = preprocess_config or PreprocessConfig(target_size=None)
    full_clean = remove_overlay_annotations(full_map, cfg)
    placed_clean = remove_overlay_annotations(placed, cfg)
    full_gray = cv2.cvtColor(full_clean, cv2.COLOR_BGR2GRAY)
    placed_gray = cv2.cvtColor(placed_clean, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(full_gray, placed_gray)
    diff_gray[mask == 0] = 0
    diff_gray = cv2.GaussianBlur(diff_gray, (3, 3), 0)
    _ret, strong = cv2.threshold(diff_gray, 32, 255, cv2.THRESH_BINARY)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    heatmap[mask == 0] = (0, 0, 0)
    diff_vis = full_map.copy()
    diff_vis[strong > 0] = (0, 0, 255)
    diff_vis = cv2.addWeighted(full_map, 0.72, diff_vis, 0.28, 0)
    return diff_vis, heatmap


def align_unknown_to_known(
    *,
    root_dir: Path,
    map_id: str,
    unknown_image: np.ndarray,
    unknown_anchor: tuple[float, float],
    known_anchors_path: Path,
    scales_path: Path,
    preprocess_config: Optional[PreprocessConfig] = None,
    default_scale: float = 1.0,
    overlay_alpha: float = 0.45,
    use_extra_offset: bool = False,
) -> AlignmentResult:
    known_anchor, full_map_path = load_known_anchor(root_dir, known_anchors_path, map_id)
    full_map = imread_unicode(full_map_path)
    if full_map is None:
        raise RuntimeError(f"无法读取已知地图：{full_map_path}")
    scale, configured_extra_offset = load_scale_config(scales_path, map_id, default_scale)
    extra_offset = configured_extra_offset if use_extra_offset else (0.0, 0.0)
    scaled_unknown = cv2.resize(
        unknown_image,
        (max(1, int(round(unknown_image.shape[1] * scale))), max(1, int(round(unknown_image.shape[0] * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    top_left_float = (
        known_anchor[0] - unknown_anchor[0] * scale + extra_offset[0],
        known_anchor[1] - unknown_anchor[1] * scale + extra_offset[1],
    )
    top_left = (int(round(top_left_float[0])), int(round(top_left_float[1])))
    overlay = alpha_overlay(full_map, scaled_unknown, top_left, overlay_alpha)
    diff, heatmap = make_alignment_diff(full_map, scaled_unknown, top_left, preprocess_config)
    transform = AnchorTransform(
        map_id=map_id,
        scale=scale,
        known_anchor=known_anchor,
        unknown_anchor=unknown_anchor,
        extra_offset=extra_offset,
        top_left=top_left_float,
    )
    return AlignmentResult(
        transform=transform,
        full_map=full_map,
        scaled_unknown=scaled_unknown,
        overlay=overlay,
        diff=diff,
        heatmap=heatmap,
    )

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .capture import (
    CaptureInfo,
    capture_fixed_roi,
    grab_screen_roi,
    resolve_game_capture_target,
    scale_reference_local_roi,
    scale_reference_roi,
)
from .config import roi_from_app
from .utils import crop_roi


REFERENCE_LOCATION_CAPTURE_SIZE = (2560, 1440)


def _reference_size(app_config: Mapping[str, Any]) -> tuple[int, int]:
    value = app_config.get(
        "alignment_reference_capture_size",
        app_config.get("alignment_reference_game_window_size", REFERENCE_LOCATION_CAPTURE_SIZE),
    )
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("alignment_reference_capture_size must contain width and height")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid alignment reference size: {value}")
    return width, height


def scale_location_reference_roi(
    roi: tuple[int, int, int, int],
    *,
    reference_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    return scale_reference_roi(roi, reference_size=reference_size, target_size=target_size)


def scale_location_local_roi(
    roi: tuple[int, int, int, int],
    *,
    reference_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    return scale_reference_local_roi(roi, reference_size=reference_size, target_size=target_size)


def capture_location_roi(
    app_config: Mapping[str, Any],
    *,
    capture_key: str,
    map_roi_key: str,
    message: str,
) -> tuple[CaptureInfo, np.ndarray, tuple[int, int, int, int]]:
    capture_roi = roi_from_app(app_config, capture_key)
    map_roi = roi_from_app(app_config, map_roi_key)
    if not bool(app_config.get("alignment_fullscreen_transform_enabled", False)):
        capture = capture_fixed_roi(app_config, capture_key, message=message)
        return capture, crop_roi(capture.image, map_roi, map_roi_key), map_roi

    reference_size = _reference_size(app_config)
    target = resolve_game_capture_target(app_config, reference_size=reference_size)
    target_w, target_h = target.size
    scaled_capture_roi = scale_location_reference_roi(
        capture_roi,
        reference_size=reference_size,
        target_size=(target_w, target_h),
    )
    scaled_map_roi = scale_location_local_roi(
        map_roi,
        reference_size=reference_size,
        target_size=(target_w, target_h),
    )
    x, y, width, height = scaled_capture_roi
    absolute_capture_roi = (target.left + x, target.top + y, width, height)
    image = grab_screen_roi(absolute_capture_roi)
    capture = CaptureInfo(
        image=image,
        roi=absolute_capture_roi,
        mode=f"scaled_location_roi:{target.mode}:{capture_key}",
        window_title=target.window_title,
        message=(
            f"{message}; {target.message}; scaled from {reference_size[0]}x{reference_size[1]} "
            f"to {target_w}x{target_h}."
        ),
        target_size=target.size,
        reference_size=reference_size,
    )
    map_x, map_y, map_width, map_height = scaled_map_roi
    overflow_x = map_x + map_width - image.shape[1]
    overflow_y = map_y + map_height - image.shape[0]
    if 0 < overflow_x <= 2:
        map_width -= overflow_x
    if 0 < overflow_y <= 2:
        map_height -= overflow_y
    adjusted_map_roi = (map_x, map_y, map_width, map_height)
    return capture, crop_roi(image, adjusted_map_roi, map_roi_key), adjusted_map_roi

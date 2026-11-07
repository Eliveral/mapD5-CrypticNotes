from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.capture import (
    CaptureInfo,
    grab_screen_roi,
    resolve_game_capture_target,
    scale_reference_roi_via_canvas,
)
from src.utils import crop_roi


# This profile is the sole source of CNN training samples.
FIXED_CNN_SAMPLE_SCREEN_ROI = (1037, 430, 963, 714)
FIXED_CNN_SAMPLE_FEATURE_ROI = (0, 0, 963, 714)
FIXED_CNN_SAMPLE_ROI_VERSION = "fixed_cnn_roi_963x714"
REFERENCE_GAME_WINDOW_SIZE = (2560, 1351)
REFERENCE_GAME_UI_CANVAS_SIZE = (2560, 1440)

FIXED_CNN_SAMPLE_APP_CONFIG: dict[str, Any] = {
    "capture_mode": "fixed_screen_roi",
    "screen_capture_roi": list(FIXED_CNN_SAMPLE_SCREEN_ROI),
    "current_map_feature_roi": list(FIXED_CNN_SAMPLE_FEATURE_ROI),
    "current_map_feature_roi_version": FIXED_CNN_SAMPLE_ROI_VERSION,
}


def capture_fixed_cnn_sample_roi() -> tuple[CaptureInfo, np.ndarray]:
    image = grab_screen_roi(FIXED_CNN_SAMPLE_SCREEN_ROI)
    capture = CaptureInfo(
        image=image,
        roi=FIXED_CNN_SAMPLE_SCREEN_ROI,
        mode="fixed_cnn_sample_roi",
        message="Captured fixed CNN sample ROI.",
    )
    roi = crop_roi(image, FIXED_CNN_SAMPLE_FEATURE_ROI, "fixed_cnn_sample_roi")
    return capture, roi


def scaled_cnn_recognition_roi(target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Fit the legacy CNN sample ROI into the player's centered 16:9 UI canvas."""
    target_w, target_h = (int(v) for v in target_size)
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"invalid target game window size: {target_size}")
    reference_w, reference_h = REFERENCE_GAME_WINDOW_SIZE
    if abs(target_w / target_h - reference_w / reference_h) <= 0.01:
        # Preserve the developer capture window that produced the CNN samples.
        x, y, width, height = FIXED_CNN_SAMPLE_SCREEN_ROI
        return (
            round(x * target_w / reference_w),
            round(y * target_h / reference_h),
            round(width * target_w / reference_w),
            round(height * target_h / reference_h),
        )
    return scale_reference_roi_via_canvas(
        FIXED_CNN_SAMPLE_SCREEN_ROI,
        roi_reference_size=REFERENCE_GAME_WINDOW_SIZE,
        canvas_reference_size=REFERENCE_GAME_UI_CANVAS_SIZE,
        target_size=(target_w, target_h),
    )


def capture_scaled_cnn_recognition_roi(
    app_config: Mapping[str, Any] | None = None,
) -> tuple[CaptureInfo, np.ndarray]:
    """Capture the CNN ROI relative to a game client, with fullscreen fallback."""
    target = resolve_game_capture_target(
        app_config or {"game_window_capture_mode": "fullscreen"},
        reference_size=REFERENCE_GAME_WINDOW_SIZE,
    )
    local_roi = scaled_cnn_recognition_roi(target.size)
    x, y, width, height = local_roi
    absolute_roi = (target.left + x, target.top + y, width, height)
    image = grab_screen_roi(absolute_roi)
    capture = CaptureInfo(
        image=image,
        roi=absolute_roi,
        mode=f"scaled_cnn_{target.mode}",
        window_title=target.window_title,
        message=(
            f"{target.message}; captured CNN ROI {absolute_roi} scaled from "
            f"{REFERENCE_GAME_WINDOW_SIZE[0]}x{REFERENCE_GAME_WINDOW_SIZE[1]} "
            f"to {target.width}x{target.height}."
        ),
        target_size=target.size,
        reference_size=REFERENCE_GAME_WINDOW_SIZE,
    )
    return capture, image

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import cv2
import mss
import numpy as np

from .config import roi_from_app

ROI = Tuple[int, int, int, int]


class CaptureError(RuntimeError):
    """Raised when screen capture cannot be completed."""


@dataclass(frozen=True)
class CaptureTarget:
    """The coordinate space used to scale a reference-game ROI."""

    left: int
    top: int
    width: int
    height: int
    mode: str
    window_title: str = ""
    message: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class CaptureInfo:
    image: np.ndarray
    roi: ROI
    mode: str
    window_title: str = ""
    message: str = ""
    target_size: Optional[tuple[int, int]] = None
    reference_size: Optional[tuple[int, int]] = None


def primary_screen_bounds() -> ROI:
    """Return the primary monitor bounds used by fullscreen game capture."""
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        return (
            int(monitor["left"]),
            int(monitor["top"]),
            int(monitor["width"]),
            int(monitor["height"]),
        )


def _virtual_screen_bounds() -> ROI:
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        return (
            int(monitor["left"]),
            int(monitor["top"]),
            int(monitor["width"]),
            int(monitor["height"]),
        )


def validate_screen_roi(roi: ROI) -> None:
    screen_left, screen_top, screen_w, screen_h = _virtual_screen_bounds()
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise CaptureError(f"invalid capture ROI: {roi}")
    if x < screen_left or y < screen_top or x + w > screen_left + screen_w or y + h > screen_top + screen_h:
        raise CaptureError(
            f"capture ROI {roi} is outside the virtual desktop "
            f"{screen_left},{screen_top},{screen_w},{screen_h}"
        )


def grab_screen_roi(roi: ROI) -> np.ndarray:
    validate_screen_roi(roi)
    x, y, w, h = roi
    with mss.mss() as sct:
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
        frame = np.array(shot, dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def reference_canvas_transform(
    *,
    reference_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float]:
    """Return the uniform scale and centered offset of a reference UI canvas."""
    reference_w, reference_h = (int(value) for value in reference_size)
    target_w, target_h = (int(value) for value in target_size)
    if reference_w <= 0 or reference_h <= 0 or target_w <= 0 or target_h <= 0:
        raise ValueError(f"invalid reference/target sizes: {reference_size} -> {target_size}")
    scale = min(target_w / reference_w, target_h / reference_h)
    offset_x = (target_w - reference_w * scale) / 2.0
    offset_y = (target_h - reference_h * scale) / 2.0
    return scale, offset_x, offset_y


def _scale_roi_with_transform(
    roi: ROI,
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> ROI:
    x, y, width, height = roi
    left = round(offset_x + x * scale)
    top = round(offset_y + y * scale)
    right = round(offset_x + (x + width) * scale)
    bottom = round(offset_y + (y + height) * scale)
    return left, top, right - left, bottom - top


def scale_reference_roi(roi: ROI, *, reference_size: tuple[int, int], target_size: tuple[int, int]) -> ROI:
    """Fit a full-window reference ROI into the target's centered game UI canvas."""
    scale, offset_x, offset_y = reference_canvas_transform(
        reference_size=reference_size,
        target_size=target_size,
    )
    return _scale_roi_with_transform(
        roi,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def scale_reference_local_roi(
    roi: ROI,
    *,
    reference_size: tuple[int, int],
    target_size: tuple[int, int],
) -> ROI:
    """Scale a ROI local to an already transformed reference capture, without canvas offset."""
    scale, _offset_x, _offset_y = reference_canvas_transform(
        reference_size=reference_size,
        target_size=target_size,
    )
    return _scale_roi_with_transform(roi, scale=scale, offset_x=0.0, offset_y=0.0)


def scale_reference_roi_via_canvas(
    roi: ROI,
    *,
    roi_reference_size: tuple[int, int],
    canvas_reference_size: tuple[int, int],
    target_size: tuple[int, int],
) -> ROI:
    """Normalize a legacy ROI coordinate space, then fit it into a centered UI canvas."""
    roi_reference_w, roi_reference_h = (int(value) for value in roi_reference_size)
    canvas_w, canvas_h = (int(value) for value in canvas_reference_size)
    if min(roi_reference_w, roi_reference_h, canvas_w, canvas_h) <= 0:
        raise ValueError(
            f"invalid ROI/canvas reference sizes: {roi_reference_size} -> {canvas_reference_size}"
        )
    scale, offset_x, offset_y = reference_canvas_transform(
        reference_size=canvas_reference_size,
        target_size=target_size,
    )
    x, y, width, height = roi
    normalized_roi = (
        x * canvas_w / roi_reference_w,
        y * canvas_h / roi_reference_h,
        width * canvas_w / roi_reference_w,
        height * canvas_h / roi_reference_h,
    )
    left = round(offset_x + normalized_roi[0] * scale)
    top = round(offset_y + normalized_roi[1] * scale)
    right = round(offset_x + (normalized_roi[0] + normalized_roi[2]) * scale)
    bottom = round(offset_y + (normalized_roi[1] + normalized_roi[3]) * scale)
    return left, top, right - left, bottom - top


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _window_settings(app_config: Mapping[str, Any]) -> tuple[str, list[str], list[str], tuple[int, int]]:
    mode = str(app_config.get("game_window_capture_mode", "auto")).strip().lower()
    if mode not in {"auto", "windowed", "fullscreen"}:
        raise CaptureError(f"unknown game_window_capture_mode: {mode}")

    titles = _as_strings(app_config.get("game_window_title"))
    titles.extend(_as_strings(app_config.get("game_window_title_candidates")))
    # Keep built-in aliases even when a packaged config contains an old or
    # mojibake Chinese title. Windowed capture must not silently fall back to
    # fixed desktop coordinates just because one config entry is malformed.
    titles.extend(["第五人格", "identity v", "identityv"])
    titles = list(dict.fromkeys(item.strip().lower() for item in titles if item.strip()))
    process_names = [item.lower() for item in _as_strings(app_config.get("game_window_process_names"))]
    minimum = app_config.get("game_window_min_client_size", (800, 450))
    if not isinstance(minimum, (list, tuple)) or len(minimum) != 2:
        minimum = (800, 450)
    return mode, titles, process_names, (int(minimum[0]), int(minimum[1]))


def _client_rect_for_window(hwnd: int) -> Optional[ROI]:
    if os.name != "nt":  # pragma: no cover - game client is Windows-only.
        return None
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        return None
    point = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(point)):
        return None
    width, height = int(rect.right - rect.left), int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        return None
    return int(point.x), int(point.y), width, height


def _process_basename(pid: int) -> str:
    if os.name != "nt":  # pragma: no cover - game client is Windows-only.
        return ""
    kernel32 = ctypes.windll.kernel32
    process = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return ""
        return os.path.basename(buffer.value).lower()
    finally:
        kernel32.CloseHandle(process)


def _find_game_client_window(app_config: Mapping[str, Any]) -> Optional[CaptureTarget]:
    if os.name != "nt":  # pragma: no cover - game client is Windows-only.
        return None
    _mode, title_candidates, process_candidates, minimum = _window_settings(app_config)
    user32 = ctypes.windll.user32
    results: list[CaptureTarget] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def enumerate_window(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        title_length = user32.GetWindowTextLengthW(hwnd)
        if title_length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, title_length + 1)
        title = title_buffer.value.strip()
        client = _client_rect_for_window(int(hwnd))
        if not title or client is None or client[2] < minimum[0] or client[3] < minimum[1]:
            return True
        title_match = any(candidate in title.lower() for candidate in title_candidates)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_match = bool(process_candidates) and _process_basename(int(pid.value)) in process_candidates
        if not title_match and not process_match:
            return True
        results.append(CaptureTarget(*client, mode="game_client", window_title=title))
        return True

    user32.EnumWindows(enumerate_window, 0)
    if not results:
        return None
    return max(results, key=lambda target: target.width * target.height)


def resolve_game_capture_target(
    app_config: Mapping[str, Any], *, reference_size: tuple[int, int]
) -> CaptureTarget:
    """Prefer a visible game client; automatically fall back to the primary monitor."""
    mode, _titles, _processes, _minimum = _window_settings(app_config)
    if mode != "fullscreen":
        target = _find_game_client_window(app_config)
        if target is not None:
            return CaptureTarget(
                target.left,
                target.top,
                target.width,
                target.height,
                target.mode,
                target.window_title,
                message=(
                    f"game client detected: {target.window_title} "
                    f"({target.width}x{target.height}); reference {reference_size[0]}x{reference_size[1]}"
                ),
            )
        if mode == "windowed":
            raise CaptureError("game client window was not found; check title/process settings or use fullscreen mode")

    left, top, width, height = primary_screen_bounds()
    fallback_note = "fullscreen capture selected" if mode == "fullscreen" else "game client not found; using primary monitor"
    return CaptureTarget(left, top, width, height, "primary_screen", message=fallback_note)


def capture_fixed_roi(app_config: Mapping[str, Any], roi_key: str, message: str = "") -> CaptureInfo:
    roi = roi_from_app(app_config, roi_key)
    return CaptureInfo(
        image=grab_screen_roi(roi),
        roi=roi,
        mode=f"fixed_screen_roi:{roi_key}",
        message=message or f"fixed screen capture: {roi_key}",
    )


def find_window(title_keyword: str):
    """Compatibility wrapper for older tools using the original title option."""
    target = _find_game_client_window({"game_window_capture_mode": "windowed", "game_window_title": title_keyword})
    if target is None:
        raise CaptureError(f"could not find a visible game window matching: {title_keyword}")
    return target


def capture_from_config(app_config: Mapping[str, Any], allow_fallback: bool = True) -> CaptureInfo:
    mode = str(app_config.get("capture_mode", "fixed_screen_roi"))
    fixed_roi = roi_from_app(app_config, "screen_capture_roi")
    if mode == "fixed_screen_roi":
        return CaptureInfo(image=grab_screen_roi(fixed_roi), roi=fixed_roi, mode=mode, message="fixed screen capture")
    if mode != "window_relative":
        raise CaptureError(f"unknown capture mode: {mode}")

    try:
        reference_size = app_config.get("window_relative_reference_size", (2560, 1440))
        reference_size = int(reference_size[0]), int(reference_size[1])
        target = resolve_game_capture_target(app_config, reference_size=reference_size)
        rel = scale_reference_roi(
            roi_from_app(app_config, "window_relative_map_roi"),
            reference_size=reference_size,
            target_size=target.size,
        )
        absolute_roi = (target.left + rel[0], target.top + rel[1], rel[2], rel[3])
        return CaptureInfo(
            image=grab_screen_roi(absolute_roi), roi=absolute_roi, mode=f"window_relative:{target.mode}",
            window_title=target.window_title, message=target.message, target_size=target.size,
            reference_size=reference_size,
        )
    except CaptureError:
        if not allow_fallback:
            raise
        return CaptureInfo(image=grab_screen_roi(fixed_roi), roi=fixed_roi, mode="fixed_screen_roi", message="window lookup failed; fixed ROI used")

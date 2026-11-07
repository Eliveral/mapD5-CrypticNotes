from __future__ import annotations

import logging
import ctypes
import importlib.util
import json
from multiprocessing import shared_memory
import os
import queue
import re
import socket
import subprocess
import threading
from ctypes import wintypes
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Any, Callable, Mapping, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk

from .alignment import align_unknown_to_known, load_known_anchor, resolve_config_path
from .capture import (
    CaptureError,
    CaptureInfo,
    capture_from_config,
    primary_screen_bounds,
    reference_canvas_transform,
)
from .cnn_sample_capture import capture_scaled_cnn_recognition_roi
from .cnn_recognition import CNNRecognitionService
from .config import ProjectConfig, project_root, roi_from_app, save_app_config
from .location_capture import capture_location_roi
from .scale_store import (
    SavedScale,
    add_confirmed_scale,
    delete_saved_scale,
    scale_client_resolution,
    scale_profile_app_config,
    scale_resolution_display,
    saved_scale_records,
    update_saved_scale,
)
from .map_database import MapDatabase
from .matcher import MatchSummary
from .matcher import MatchResult
from .preprocess import PreprocessConfig
from .runtime_diagnostics import diagnostic_event, diagnostic_path
from .recognition_profiles import RecognitionProfile, get_recognition_profile
from .sample_database import ScreenshotSampleDatabase
from .structure_registration import StructureRegistrationResult, align_structure_to_known, parse_scales, resolve_template_path
from .utils import crop_roi, ensure_dir, imread_unicode, imwrite_unicode, make_side_by_side, timestamp

LOGGER = logging.getLogger(__name__)


class SemanticCVSearchCancelled(RuntimeError):
    """Raised when the player explicitly stops an active scale search."""

HOTKEY_DEFAULTS = {
    "recognize": "F8",
    "confirm_structure": "F4",
    "switch_layout": "F1",
    "locate_current_floor": "key:71:G",
}
HOTKEY_LABELS = {
    "recognize": "开始识别",
    "confirm_structure": "确认当前侧门结构",
    "switch_layout": "切换同结构布局",
    "locate_current_floor": "自动楼层定位",
}


def detect_open_map_floor(
    image: np.ndarray,
    *,
    minimum_margin: float = 12.0,
    search_roi_normalized: object = None,
    available_floors: tuple[str, ...] = ("1st", "2nd"),
    debug_details: Optional[dict[str, Any]] = None,
) -> tuple[str, float, float, float]:
    """Detect the selected floor from the top-right map-tab backplates."""
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        raise ValueError("invalid map ROI for floor detection")
    height, width = image.shape[:2]
    if width < 200 or height < 120:
        raise ValueError(f"map ROI is too small for floor detection: {width}x{height}")

    scale_x = width / 1474.0
    scale_y = height / 1332.0
    search_x0, search_y0, search_x1, search_y1 = floor_detection_search_roi(
        (width, height), search_roi_normalized
    )
    search_width = search_x1 - search_x0
    search_image = image[search_y0:search_y1, search_x0:search_x1]
    search_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else search_image
    requested_floors = tuple(floor for floor in available_floors if floor in {"b1", "1st", "2nd"})
    if not requested_floors:
        requested_floors = ("1st", "2nd")

    def find_digit_pair(brightness_threshold: int) -> Optional[tuple[float, float, float]]:
        digit_mask = cv2.inRange(search_gray, brightness_threshold, 255)
        _, _, stats, _ = cv2.connectedComponentsWithStats(digit_mask, connectivity=8)
        first_digits: list[tuple[float, float, int]] = []
        second_digits: list[tuple[float, float, int]] = []
        for x, y, component_width, component_height, area in stats[1:]:
            x += search_x0
            y += search_y0
            if not (
                round(16 * scale_y) <= component_height <= round(44 * scale_y)
                and area >= round(35 * scale_x * scale_y)
            ):
                continue
            center = (x + component_width / 2.0, y + component_height / 2.0, int(area))
            if round(3 * scale_x) <= component_width <= round(11 * scale_x):
                first_digits.append(center)
            elif round(10 * scale_x) <= component_width <= round(24 * scale_x):
                second_digits.append(center)

        digit_pair: Optional[tuple[float, float, float]] = None
        pair_rank: Optional[tuple[float, int]] = None
        for first_center_x, first_center_y, first_area in first_digits:
            for second_center_x, second_center_y, second_area in second_digits:
                separation = second_center_x - first_center_x
                if not (
                    70 * scale_x <= separation <= 125 * scale_x
                    and abs(second_center_y - first_center_y) <= 8 * scale_y
                ):
                    continue
                rank = (abs(separation - 94 * scale_x), -(first_area + second_area))
                if pair_rank is None or rank < pair_rank:
                    pair_rank = rank
                    digit_pair = (
                        first_center_x,
                        second_center_x,
                        (first_center_y + second_center_y) / 2.0,
                    )
        return digit_pair

    anchor_source = "digits_140"
    digit_pair = find_digit_pair(140)
    if digit_pair is None:
        anchor_source = "digits_90"
        digit_pair = find_digit_pair(90)

    if digit_pair is not None:
        half_width = round(40 * scale_x)
        first_center_x, second_center_x, digit_center_y = digit_pair
        first_x0, first_x1 = round(first_center_x) - half_width, round(first_center_x) + half_width
        second_x0, second_x1 = round(second_center_x) - half_width, round(second_center_x) + half_width
        score_y0 = round(digit_center_y - 40 * scale_y)
        score_y1 = round(digit_center_y + 15 * scale_y)
    else:
        if "b1" in requested_floors:
            # Nightmare uses three equally spaced tabs within the wider search ROI.
            half_width = round(40 * scale_x)
            first_center_x = search_x0 + round(search_width * 0.446)
            second_center_x = search_x0 + round(search_width * 0.629)
            first_x0, first_x1 = first_center_x - half_width, first_center_x + half_width
            second_x0, second_x1 = second_center_x - half_width, second_center_x + half_width
            anchor_source = "fallback_three_tabs"
        else:
            first_x0 = search_x0 + round(search_width * 0.546)
            first_x1 = search_x0 + round(search_width * 0.700)
            second_x0 = search_x0 + round(search_width * 0.740)
            second_x1 = search_x0 + round(search_width * 0.894)
            anchor_source = "fallback_two_tabs"
        score_y0 = search_y0 + round(25 * scale_y)
        score_y1 = search_y0 + round(80 * scale_y)
    score_y0 = max(search_y0, min(search_y1 - 1, score_y0))
    score_y1 = max(score_y0 + 1, min(search_y1, score_y1))
    tab_boxes = {
        "1st": (first_x0, first_x1),
        "2nd": (second_x0, second_x1),
    }
    if "b1" in requested_floors:
        spacing = ((second_x0 + second_x1) - (first_x0 + first_x1)) / 2.0
        tab_boxes["b1"] = (round(first_x0 - spacing), round(first_x1 - spacing))

    scores: dict[str, float] = {}
    for floor in requested_floors:
        x0, x1 = tab_boxes[floor]
        x0 = max(search_x0, min(search_x1 - 1, x0))
        x1 = max(x0 + 1, min(search_x1, x1))
        tab = image[score_y0:score_y1, x0:x1]
        if tab.size == 0:
            raise ValueError(f"floor-tab region is outside the map ROI: {floor}")
        if tab.ndim == 3:
            tab = cv2.cvtColor(tab, cv2.COLOR_BGR2GRAY)
        scores[floor] = float(np.mean(tab))

    if debug_details is not None:
        debug_details.update(
            {
                "anchor_source": anchor_source,
                "search_roi": [search_x0, search_y0, search_x1 - search_x0, search_y1 - search_y0],
                "score_y_range": [score_y0, score_y1],
                "tab_boxes": {
                    floor: [int(tab_boxes[floor][0]), score_y0, int(tab_boxes[floor][1]), score_y1]
                    for floor in requested_floors
                },
                "scores": scores.copy(),
            }
        )

    first_score = scores.get("1st", 0.0)
    second_score = scores.get("2nd", 0.0)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    detected_floor, detected_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else detected_score
    confidence = detected_score - runner_up_score
    if confidence < max(0.0, float(minimum_margin)):
        raise ValueError(
            "unable to determine the selected map floor "
            f"(scores={scores}, margin={confidence:.1f})"
        )
    return detected_floor, confidence, first_score, second_score


def floor_detection_search_roi(
    image_size: tuple[int, int],
    normalized_roi: object = None,
) -> tuple[int, int, int, int]:
    """Resolve a normalized floor-tab search rectangle for the current map ROI."""
    width, height = image_size
    values = normalized_roi if isinstance(normalized_roi, (list, tuple)) else None
    if (
        values is not None
        and len(values) >= 4
        and all(isinstance(value, (int, float)) for value in values[:4])
    ):
        x, y, roi_width, roi_height = (float(value) for value in values[:4])
        if 0.0 <= x < 1.0 and 0.0 <= y < 1.0 and roi_width > 0.0 and roi_height > 0.0:
            x0 = max(0, min(width - 1, round(width * x)))
            y0 = max(0, min(height - 1, round(height * y)))
            x1 = max(x0 + 1, min(width, round(width * (x + roi_width))))
            y1 = max(y0 + 1, min(height, round(height * (y + roi_height))))
            return x0, y0, x1, y1
    return round(width * 0.65), 0, width, max(1, round(height * (90.0 / 1332.0)))


def floor_detection_search_roi_for_profile(
    app_config: Mapping[str, Any],
    profile_key: str,
) -> object:
    """Return the normalized floor-tab ROI assigned to one recognition profile.

    The captured map ROI has already been transformed from the 16:9 reference
    canvas, so this value intentionally remains local to that ROI.  It must not
    be transformed again for a 16:10 client.
    """
    normalized_profile = str(profile_key).strip().lower()
    configured = app_config.get("map_floor_detection_search_roi_by_profile")
    if isinstance(configured, Mapping):
        value = configured.get(normalized_profile)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return value
    return app_config.get("map_floor_detection_search_roi_normalized")


def save_semantic_cv_development_capture(
    call_dir: Path,
    alignment_roi: np.ndarray,
    *,
    floor_detection_strip: Optional[np.ndarray] = None,
    floor_detection_debug: Optional[dict[str, Any]] = None,
) -> None:
    """Persist a captured localization ROI before any CV search begins."""
    ensure_dir(call_dir)
    imwrite_unicode(call_dir / "input_roi.png", alignment_roi)
    if floor_detection_strip is not None:
        imwrite_unicode(call_dir / "floor_tabs.png", floor_detection_strip)
    if floor_detection_debug is not None:
        (call_dir / "floor_detection.json").write_text(
            json.dumps(floor_detection_debug, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def consume_initial_first_floor_location_attempt(
    attempted_floors: set[str],
    floor_suffix: str,
    *,
    consume: bool = True,
) -> bool:
    """Return whether this is the first first-floor location attempt in the current F8 session."""
    if not consume:
        return False
    is_initial_first_floor = floor_suffix == "1st" and "1st" not in attempted_floors
    attempted_floors.add(floor_suffix)
    return is_initial_first_floor


def crop_route_overlay_content(
    image: np.ndarray,
    *,
    black_threshold: int = 16,
    padding: int = 2,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop transparent-black route margins while retaining interpolation support."""
    if image.size == 0:
        return image, (0, 0)
    threshold = max(0, min(255, int(black_threshold)))
    content = cv2.inRange(image, (threshold + 1,) * 3, (255,) * 3)
    points = cv2.findNonZero(content)
    if points is None:
        return image[:1, :1].copy(), (0, 0)
    x, y, width, height = cv2.boundingRect(points)
    margin = max(0, int(padding))
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image.shape[1], x + width + margin)
    bottom = min(image.shape[0], y + height + margin)
    return image[top:bottom, left:right].copy(), (left, top)


def warp_route_overlay_visible_region(
    route_image: np.ndarray,
    source_to_roi: np.ndarray,
    roi_size: tuple[int, int],
    *,
    interpolation_padding: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Warp only the transformed route bounds instead of the full map ROI."""
    roi_width, roi_height = (int(roi_size[0]), int(roi_size[1]))
    if route_image.size == 0 or roi_width <= 0 or roi_height <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8), (0, 0, 0, 0)
    height, width = route_image.shape[:2]
    corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
    transformed = cv2.transform(corners, np.float32(source_to_roi)).reshape(-1, 2)
    padding = max(0, int(interpolation_padding))
    left = max(0, int(np.floor(np.min(transformed[:, 0]))) - padding)
    top = max(0, int(np.floor(np.min(transformed[:, 1]))) - padding)
    right = min(roi_width, int(np.ceil(np.max(transformed[:, 0]))) + padding)
    bottom = min(roi_height, int(np.ceil(np.max(transformed[:, 1]))) + padding)
    if right <= left or bottom <= top:
        return np.zeros((1, 1, 3), dtype=np.uint8), (0, 0, 0, 0)
    local_affine = np.float32(source_to_roi).copy()
    local_affine[:, 2] -= np.float32([left, top])
    warped = cv2.warpAffine(
        route_image,
        local_affine,
        (right - left, bottom - top),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, (left, top, right - left, bottom - top)
VARIANT_RETURN_HOTKEY = "F3"
GAME_MAP_MOUSE_BINDINGS = {
    "left": ("鼠标左键", 0x0002, 0x0004, 0),
    "right": ("鼠标右键", 0x0008, 0x0010, 0),
    "middle": ("鼠标中键", 0x0020, 0x0040, 0),
    "x1": ("鼠标侧键1", 0x0080, 0x0100, 1),
    "x2": ("鼠标侧键2", 0x0080, 0x0100, 2),
}
GAME_MAP_MOUSE_VIRTUAL_KEYS = {
    "left": 0x01,
    "right": 0x02,
    "middle": 0x04,
    "x1": 0x05,
    "x2": 0x06,
}


def game_map_toggle_display_name(binding: object) -> str:
    value = str(binding).strip()
    mouse = GAME_MAP_MOUSE_BINDINGS.get(value.lower())
    if mouse is not None:
        return mouse[0]
    match = re.fullmatch(r"key:(\d{1,3}):(.*)", value)
    if match is not None and 1 <= int(match.group(1)) <= 255:
        label = match.group(2).strip()
        return label or f"键盘键 VK{match.group(1)}"
    return "鼠标右键"


def hotkey_binding_identity(binding: object) -> Optional[tuple[str, object]]:
    value = str(binding).strip()
    function_key = re.fullmatch(r"F(\d{1,2})", value.upper())
    if function_key is not None:
        number = int(function_key.group(1))
        if 1 <= number <= 24:
            return ("key", 0x70 + number - 1)
    keyboard_key = re.fullmatch(r"key:(\d{1,3}):.*", value)
    if keyboard_key is not None:
        virtual_key = int(keyboard_key.group(1))
        if 1 <= virtual_key <= 255:
            return ("key", virtual_key)
    mouse_key = re.fullmatch(r"mouse:(left|right|middle|x1|x2)", value.lower())
    if mouse_key is not None:
        return ("mouse", mouse_key.group(1))
    return None


def game_map_toggle_binding_identity(binding: object) -> Optional[tuple[str, object]]:
    value = str(binding).strip()
    if value.lower() in GAME_MAP_MOUSE_BINDINGS:
        return ("mouse", value.lower())
    return hotkey_binding_identity(value)


def hotkey_binding_display_name(binding: object) -> str:
    value = str(binding).strip()
    identity = hotkey_binding_identity(value)
    if identity is None:
        return value or "未设置"
    kind, code = identity
    if kind == "mouse":
        return GAME_MAP_MOUSE_BINDINGS[str(code)][0]
    function_key = re.fullmatch(r"F(\d{1,2})", value.upper())
    if function_key is not None:
        return value.upper()
    match = re.fullmatch(r"key:\d{1,3}:(.*)", value)
    if match is not None and match.group(1).strip():
        return match.group(1).strip()
    return f"键盘键 VK{code}"


def configured_hotkey_display_name(owner: object, action: str) -> str:
    hotkeys = getattr(owner, "_hotkeys", {})
    binding = hotkeys.get(action, HOTKEY_DEFAULTS.get(action, "")) if isinstance(hotkeys, dict) else HOTKEY_DEFAULTS.get(action, "")
    return hotkey_binding_display_name(binding)


class ImagePanel(ttk.Frame):
    def __init__(
        self,
        master,
        title: str,
        width: int = 300,
        height: int = 220,
        on_image=None,
    ) -> None:
        super().__init__(master, style="Main.Panel.TFrame")
        self.width = width
        self.height = height
        self._on_image = on_image
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._display_info: Optional[tuple[int, int, int, int, int, int]] = None
        ttk.Label(self, text=title, style="Main.Panel.TLabel").pack(anchor="w")
        self.label = ttk.Label(self, relief="solid", anchor="center", style="Main.Preview.TLabel")
        self.label.pack(fill="both", expand=True)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self.label.configure(image="", text="暂无图像")
            self._photo = None
            self._display_info = None
            return
        if self._on_image is not None:
            self._on_image()
        src_h, src_w = image.shape[:2]
        if image.ndim == 2:
            pil = Image.fromarray(image)
        else:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.width, self.height), (32, 32, 32))
        x = (self.width - pil.width) // 2
        y = (self.height - pil.height) // 2
        canvas.paste(pil.convert("RGB"), (x, y))
        self._display_info = (x, y, pil.width, pil.height, src_w, src_h)
        self._photo = ImageTk.PhotoImage(canvas)
        self.label.configure(image=self._photo, text="")

    def widget_to_image_point(self, event_x: int, event_y: int) -> Optional[tuple[float, float]]:
        if self._display_info is None:
            return None
        x, y, display_w, display_h, src_w, src_h = self._display_info
        if event_x < x or event_y < y or event_x >= x + display_w or event_y >= y + display_h:
            return None
        return (
            float((event_x - x) * src_w / max(display_w, 1)),
            float((event_y - y) * src_h / max(display_h, 1)),
        )


class HiddenPreviewSink:
    """Retain image-update call sites after the main-window previews were removed."""

    def set_image(self, _image: Optional[np.ndarray]) -> None:
        return

class CheatSheetWindow:
    CONTROL_ROW_HEIGHT = 34
    TITLE_ROW_HEIGHT = 34

    def __init__(self, master: tk.Tk, project_config: ProjectConfig) -> None:
        self.master = master
        self.project_config = project_config
        self._last_content: Optional[tuple[str, tuple]] = None
        self._temporary_size: Optional[tuple[int, int]] = None
        self._adaptive_base_size: Optional[tuple[int, int]] = None
        self._adaptive_reference_scale: Optional[float] = None
        self.window = tk.Toplevel(master)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(bg="#05070a")
        self.window.attributes("-topmost", True)
        self.alpha = float(project_config.app.get("cheatsheet_alpha", 0.82))
        self.window.attributes("-alpha", self.alpha)
        self.photo: Optional[ImageTk.PhotoImage] = None
        self._image_display_info: Optional[tuple[int, int, int, int, int, int]] = None
        self._click_handler = None
        self._sheet_click_handler = None
        self._bottom_action_height = 0
        self._bottom_action_region: Optional[tuple[int, int, int, int]] = None
        self._hud_click_passthrough_enabled = True
        self._hud_interactive_regions: list[tuple[int, int, int, int]] = []
        self._hud_hit_test_procs: dict[int, object] = {}
        self._hud_interaction_windows: list[tk.Toplevel] = []
        default_height = int(project_config.app.get("cheatsheet_height", 480))
        default_y = max(0, self.master.winfo_screenheight() // 2 - default_height // 2)
        self.x = int(project_config.app.get("cheatsheet_x", 0))
        self.y = int(project_config.app.get("cheatsheet_y", default_y))
        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self.is_collapsed = False

        self.body = tk.Label(self.window, bg="#05070a", bd=0, highlightthickness=0)
        self.body.pack(fill="both", expand=True)
        self.window.bind("<ButtonPress-1>", self._handle_body_click)
        self.body.bind("<ButtonPress-1>", self._handle_body_click)
        self.anchor = tk.Toplevel(master)
        self.anchor.withdraw()
        self.anchor.overrideredirect(True)
        self.anchor.attributes("-topmost", True)
        self.anchor.attributes("-alpha", 0.78)
        self.anchor.configure(bg="#111827")
        self.anchor_label = tk.Label(
            self.anchor,
            text="移动小抄",
            bg="#111827",
            fg="#f9fafb",
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            pady=4,
            cursor="fleur",
        )
        self.anchor_label.pack(side="left")
        self.toggle_button = tk.Button(
            self.anchor,
            text="最小化",
            bg="#111827",
            fg="#f9fafb",
            activebackground="#1f2937",
            activeforeground="#f9fafb",
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            command=self.toggle_collapsed,
        )
        self.toggle_button.pack(side="left")
        self.anchor_label.bind("<ButtonPress-1>", self._start_drag)
        self.anchor_label.bind("<B1-Motion>", self._drag)

    def _geometry(self) -> tuple[int, int]:
        screen_w = max(1, self.master.winfo_screenwidth())
        screen_h = max(1, self.master.winfo_screenheight())
        if self._temporary_size is not None:
            width, height = self._temporary_size
        else:
            width = int(self.project_config.app.get("cheatsheet_width", screen_w // 2))
            height = int(self.project_config.app.get("cheatsheet_height", screen_h // 2))
        width = max(320, min(width, screen_w))
        height = max(240, min(height, screen_h))
        height = min(
            screen_h,
            height + self.CONTROL_ROW_HEIGHT + max(0, int(self._bottom_action_height)),
        )
        self.x = max(0, min(int(self.x), max(0, screen_w - 1)))
        self.y = max(0, min(int(self.y), max(0, screen_h - 1)))
        self.window.geometry(f"{width}x{height}+{self.x}+{self.y}")
        self.anchor.geometry(f"+{self.x}+{self.y}")
        return width, height

    def clear_temporary_size(self) -> None:
        self.clear_adaptive_size()

    def clear_adaptive_size(self) -> None:
        self._temporary_size = None
        self._adaptive_base_size = None
        self._adaptive_reference_scale = None

    def set_adaptive_size(self, width: int, height: int, reference_scale: float) -> None:
        self._adaptive_base_size = (max(320, int(width)), max(240, int(height)))
        self._adaptive_reference_scale = max(0.01, float(reference_scale))
        self._temporary_size = self._adaptive_base_size

    def adaptive_size_for_scale(self, scale: float) -> Optional[tuple[int, int]]:
        if self._adaptive_base_size is None or self._adaptive_reference_scale is None:
            return None
        ratio = max(0.1, float(scale)) / self._adaptive_reference_scale
        return (
            max(320, int(round(self._adaptive_base_size[0] * ratio))),
            max(240, int(round(self._adaptive_base_size[1] * ratio))),
        )

    def _configure_image_adaptive_size(
        self,
        image: Optional[np.ndarray],
        *,
        reserved_height: int = 34,
    ) -> None:
        """Fit route sheets to their content without cropping the route image."""
        if image is None:
            self.clear_adaptive_size()
            return
        route_version = str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower()
        display_mode = str(self.project_config.app.get("lianghapi_cheatsheet_mode", "single")).strip().lower()
        profile_key = str(self.project_config.app.get("recognition_profile", "difficulty")).strip().lower()
        nightmare_mode = str(self.project_config.app.get("nightmare_cheatsheet_mode", "single")).strip().lower()
        nightmare_versioned_route = profile_key == "nightmare" and nightmare_route_map_enabled(self.project_config.app)
        difficulty_profile = profile_key != "nightmare"
        single_legacy_sheet = difficulty_profile and route_version == "lianghapi" and display_mode == "single"
        single_nightmare_sheet = profile_key == "nightmare" and not nightmare_versioned_route and nightmare_mode == "single"
        triple_nightmare_sheet = profile_key == "nightmare" and not nightmare_versioned_route and nightmare_mode == "triple"
        single_nightmare_lianghapi_sheet = (
            profile_key == "nightmare"
            and nightmare_lianghapi_single_cheatsheet_enabled(self.project_config.app)
        )
        sizing_image = self._trim_dark_border(image) if (
            single_legacy_sheet or single_nightmare_sheet or single_nightmare_lianghapi_sheet
        ) else image
        image_h, image_w = sizing_image.shape[:2]
        image_ratio = image_h / max(image_w, 1)
        dual_legacy_sheet = (
            difficulty_profile
            and
            route_version == "lianghapi"
            and display_mode == "dual"
            and image_ratio >= 1.2
        )
        if (
            route_version != "zhanshi"
            and not nightmare_versioned_route
            and not single_legacy_sheet
            and not dual_legacy_sheet
            and not single_nightmare_sheet
            and not triple_nightmare_sheet
        ):
            self.clear_adaptive_size()
            return
        screen_w = max(320, int(self.master.winfo_screenwidth()))
        screen_h = max(240, int(self.master.winfo_screenheight()))
        configured_base_width = int(self.project_config.app.get("cheatsheet_width", 760))
        base_width = min(configured_base_width, int(screen_w * 0.72))
        max_height = int(screen_h * (0.92 if triple_nightmare_sheet else 0.82))
        if single_legacy_sheet or single_nightmare_sheet or single_nightmare_lianghapi_sheet:
            chrome_height = max(0, int(reserved_height))
            target_height = min(int(round(base_width * image_ratio)) + chrome_height, max_height)
            available_image_height = max(1, target_height - chrome_height)
            target_width = max(
                320,
                min(base_width, int(round(available_image_height / max(image_ratio, 0.01)))),
            )
        else:
            target_height = min(int(round(base_width * image_ratio)), max_height)
            target_width = max(320, min(base_width, int(round(target_height / max(image_ratio, 0.01)))))
        configured_width = max(320, int(self.project_config.app.get("cheatsheet_width", 760)))
        self.set_adaptive_size(target_width, max(240, target_height), configured_width / 760.0)

    def clear_bottom_action(self) -> None:
        """Remove the transient overlay action row and restore the normal sheet size."""
        if self._bottom_action_height <= 0:
            return
        self._bottom_action_height = 0
        self._bottom_action_region = None
        self._sheet_click_handler = None
        self._hud_interactive_regions = []
        self._destroy_hud_interaction_windows()
        if self._last_content is None:
            return
        kind, payload = self._last_content
        if kind != "image_with_bottom_action":
            return
        image, title, footer, compact_footer, _button_text, _on_click = payload
        self._last_content = (
            ("image_with_message", (image, title, footer, compact_footer)) if footer else ("image", (image, title))
        )
        self._rerender_last_content()

    def set_alpha(self, value: float) -> None:
        self.alpha = max(0.25, min(1.0, float(value)))
        self.project_config.app["cheatsheet_alpha"] = self.alpha
        self.window.attributes("-alpha", self.alpha)
        self.window.update_idletasks()

    def set_size(self, width: int, height: int) -> None:
        screen_w = max(1, self.master.winfo_screenwidth())
        screen_h = max(1, self.master.winfo_screenheight())
        self.project_config.app["cheatsheet_width"] = max(320, min(int(width), screen_w))
        self.project_config.app["cheatsheet_height"] = max(240, min(int(height), screen_h))
        if self._adaptive_base_size is not None:
            self._temporary_size = (self.project_config.app["cheatsheet_width"], self.project_config.app["cheatsheet_height"])
        self._geometry()
        self._rerender_last_content()

    def set_position(self, x: int, y: int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.project_config.app["cheatsheet_x"] = self.x
        self.project_config.app["cheatsheet_y"] = self.y
        self._geometry()
        sync_windows = getattr(self, "_sync_hud_interaction_windows", None)
        if callable(sync_windows):
            sync_windows()

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_offset_x = int(event.x_root) - self.x
        self._drag_offset_y = int(event.y_root) - self.y

    def _drag(self, event: tk.Event) -> None:
        self.set_position(int(event.x_root) - self._drag_offset_x, int(event.y_root) - self._drag_offset_y)

    def _ignore_body_click(self, _event: tk.Event) -> str:
        self.window.attributes("-topmost", True)
        self.anchor.attributes("-topmost", True)
        try:
            self.master.focus_force()
        except tk.TclError:
            pass
        return "break"

    def set_click_handler(self, handler) -> None:
        self._click_handler = handler
        self._hud_interactive_regions = []
        self._destroy_hud_interaction_windows()
        self._set_hud_click_passthrough(False)

    def clear_click_handler(self) -> None:
        self._click_handler = None
        self._sheet_click_handler = None
        self._hud_interactive_regions = []
        self._destroy_hud_interaction_windows()
        self._set_hud_click_passthrough(True)

    def _set_hud_click_passthrough(self, enabled: bool) -> None:
        self._hud_click_passthrough_enabled = bool(enabled)
        # Window styles are deliberately applied only after Tk has rendered
        # new content in _show_no_activate().  Updating them before rendering
        # can leave a layered Tk toplevel blank on some Windows builds.

    def _set_hud_interactive_window_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> None:
        self._hud_interactive_regions = [
            (int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2 in regions
            if x2 >= x1 and y2 >= y1
        ]
        sync_windows = getattr(self, "_sync_hud_interaction_windows", None)
        if callable(sync_windows):
            sync_windows()

    def _destroy_hud_interaction_windows(self) -> None:
        windows, self._hud_interaction_windows = self._hud_interaction_windows, []
        for window in windows:
            try:
                window.destroy()
            except tk.TclError:
                pass

    def _dispatch_hud_catcher_click(self, event) -> bool:
        """Translate a catcher click from screen coordinates to HUD coordinates."""
        try:
            origin_x = int(self.window.winfo_rootx())
            origin_y = int(self.window.winfo_rooty())
        except tk.TclError:
            origin_x, origin_y = int(self.x), int(self.y)
        return self._dispatch_body_click(
            int(event.x_root) - origin_x,
            int(event.y_root) - origin_y,
        )

    def _sync_hud_interaction_windows(self) -> None:
        """Keep only the button rectangles interactive while the HUD itself stays transparent."""
        self._destroy_hud_interaction_windows()
        if not self._hud_interactive_regions or not self.window.winfo_exists():
            return
        try:
            if str(self.window.state()) == "withdrawn" or self.is_collapsed:
                return
        except tk.TclError:
            return

        for x1, y1, x2, y2 in self._hud_interactive_regions:
            width = max(1, int(x2) - int(x1) + 1)
            height = max(1, int(y2) - int(y1) + 1)
            catcher = tk.Toplevel(self.master)
            catcher.overrideredirect(True)
            catcher.configure(bg="#000000")
            catcher.attributes("-topmost", True)
            # A nearly invisible independent window is required here.  Returning
            # HTTRANSPARENT from the Tk HUD does not reliably forward clicks to a
            # game owned by another process/thread.
            catcher.attributes("-alpha", 0.01)
            # Use the actual native HUD origin. The configured x/y can differ
            # after dragging or Windows DPI repositioning.
            try:
                origin_x = int(self.window.winfo_rootx())
                origin_y = int(self.window.winfo_rooty())
            except tk.TclError:
                origin_x, origin_y = int(self.x), int(self.y)
            catcher.geometry(f"{width}x{height}+{origin_x + int(x1)}+{origin_y + int(y1)}")
            catcher.bind("<ButtonPress-1>", self._dispatch_hud_catcher_click)
            catcher.update_idletasks()
            if sys.platform.startswith("win"):
                try:
                    user32, get_window_long, set_window_long, _call_window_proc = self._win32_hud_window_functions()
                    gwl_exstyle = -20
                    ws_ex_toolwindow = 0x00000080
                    ws_ex_noactivate = 0x08000000
                    ga_root = 2
                    hwnd = int(catcher.winfo_id())
                    root_hwnd = int(user32.GetAncestor(wintypes.HWND(hwnd), ga_root) or hwnd)
                    for target_hwnd in {hwnd, root_hwnd}:
                        style = int(get_window_long(wintypes.HWND(target_hwnd), gwl_exstyle))
                        set_window_long(
                            wintypes.HWND(target_hwnd),
                            gwl_exstyle,
                            ctypes.c_ssize_t(style | ws_ex_toolwindow | ws_ex_noactivate),
                        )
                        # Tk's -topmost flag is not sufficient when the route
                        # overlay is raised afterwards. Explicitly put each
                        # catcher at the top of the topmost window band while
                        # preserving its no-activate behavior.
                        user32.SetWindowPos(
                            wintypes.HWND(target_hwnd),
                            wintypes.HWND(-1),
                            0,
                            0,
                            0,
                            0,
                            0x0001 | 0x0002 | 0x0010,
                        )
                except Exception:
                    LOGGER.debug("failed to apply cheatsheet button catcher style", exc_info=True)
            catcher.lift()
            self._hud_interaction_windows.append(catcher)

    def _set_hud_interactive_image_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> None:
        """Translate source-image controls to their displayed HUD coordinates."""
        display = self._image_display_info
        if display is None:
            self._set_hud_interactive_window_regions([])
            return
        display_x, display_y, display_w, display_h, source_w, source_h = display
        if display_w <= 0 or display_h <= 0 or source_w <= 0 or source_h <= 0:
            self._set_hud_interactive_window_regions([])
            return
        converted = []
        for x1, y1, x2, y2 in regions:
            converted.append(
                (
                    display_x + int(np.floor(x1 * display_w / source_w)),
                    display_y + int(np.floor(y1 * display_h / source_h)),
                    display_x + int(np.ceil(x2 * display_w / source_w)),
                    display_y + int(np.ceil(y2 * display_h / source_h)),
                )
            )
        self._set_hud_interactive_window_regions(converted)

    def _hud_hit_test_is_interactive(self, screen_x: int, screen_y: int) -> bool:
        window_x = int(self.x)
        window_y = int(self.y)
        local_x = int(screen_x) - window_x
        local_y = int(screen_y) - window_y
        return any(
            x1 <= local_x <= x2 and y1 <= local_y <= y2
            for x1, y1, x2, y2 in self._hud_interactive_regions
        )

    @staticmethod
    def _win32_hud_window_functions():
        """Return pointer-safe Win32 APIs used by the HUD on 64-bit Windows."""
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        get_window_long = user32.GetWindowLongPtrW
        set_window_long = user32.SetWindowLongPtrW
        call_window_proc = user32.CallWindowProcW
        get_window_long.argtypes = (wintypes.HWND, ctypes.c_int)
        get_window_long.restype = ctypes.c_ssize_t
        set_window_long.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
        set_window_long.restype = ctypes.c_ssize_t
        call_window_proc.argtypes = (
            ctypes.c_void_p,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        call_window_proc.restype = ctypes.c_ssize_t
        user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
        user32.GetAncestor.restype = wintypes.HWND
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        return user32, get_window_long, set_window_long, call_window_proc

    def _display_point_to_image_point(self, event_x: int, event_y: int) -> Optional[tuple[float, float]]:
        if self._image_display_info is None:
            return None
        x, y, display_w, display_h, src_w, src_h = self._image_display_info
        if event_x < x or event_y < y or event_x >= x + display_w or event_y >= y + display_h:
            return None
        return (
            float((event_x - x) * src_w / max(display_w, 1)),
            float((event_y - y) * src_h / max(display_h, 1)),
        )

    def _dispatch_body_click(self, event_x: int, event_y: int) -> str:
        self.window.attributes("-topmost", True)
        self.anchor.attributes("-topmost", True)
        sheet_click_handler = getattr(self, "_sheet_click_handler", None)
        if sheet_click_handler is not None:
            try:
                if bool(sheet_click_handler(event_x, event_y)):
                    return "break"
            except Exception:
                LOGGER.exception("cheatsheet sheet click handler failed")
        point = self._display_point_to_image_point(event_x, event_y)
        if point is not None and self._click_handler is not None:
            try:
                handled = bool(self._click_handler(point))
                if handled:
                    return "break"
            except Exception:
                LOGGER.exception("cheatsheet click handler failed")
        if self._click_handler is not None:
            return "break"
        try:
            self.master.focus_force()
        except tk.TclError:
            pass
        return "break"

    def _handle_body_click(self, event: tk.Event) -> str:
        try:
            event_x = int(event.x_root) - int(self.window.winfo_rootx())
            event_y = int(event.y_root) - int(self.window.winfo_rooty())
        except (AttributeError, tk.TclError, TypeError, ValueError):
            event_x = int(event.x)
            event_y = int(event.y)
        return CheatSheetWindow._dispatch_body_click(self, event_x, event_y)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self.is_collapsed)

    def expand_on_next_show(self) -> None:
        """Clear the collapsed state without briefly repainting stale content."""
        self.is_collapsed = False
        self.toggle_button.configure(text="最小化")

    def set_collapsed(self, collapsed: bool) -> None:
        self.is_collapsed = collapsed
        if collapsed:
            self._destroy_hud_interaction_windows()
            self.window.withdraw()
            self.toggle_button.configure(text="展开")
        else:
            self.window.deiconify()
            self.window.lift()
            self._sync_hud_interaction_windows()
            self.toggle_button.configure(text="最小化")
        self.anchor.deiconify()
        self.anchor.lift()
        self.anchor.attributes("-topmost", True)

    def show_map(self, map_id: str, confidence: float) -> None:
        width, height = self._geometry()
        cfg = self.project_config.maps.get(map_id)
        if cfg is None:
            return
        image = imread_unicode(cfg.full_image)
        if image is None:
            self.show_message(f"无法读取完整地图：{map_id}", title="小抄界面")
            return
        self.show_image(image, title=f"{map_id} / {cfg.name}    CNN {confidence:.3f}")

    def show_image(self, image: np.ndarray, title: str, footer: str = "", compact_footer: bool = False) -> None:
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        footer_height = 30 if footer and compact_footer else (84 if footer else 0)
        self._configure_image_adaptive_size(image, reserved_height=34 + footer_height)
        self.clear_click_handler()
        if footer:
            self._last_content = ("image_with_message", (image.copy(), title, footer, compact_footer))
            self._render_image_with_message(image, title, footer, compact_footer=compact_footer)
            return
        self._last_content = ("image", (image.copy(), title))
        self._render_image(image, title)

    def show_image_with_bottom_action(
        self,
        image: np.ndarray,
        title: str,
        *,
        button_text: str,
        on_click,
        footer: str = "",
        compact_footer: bool = False,
    ) -> None:
        """Show a route sheet with a separate bottom row for one overlay action."""
        self.clear_click_handler()
        footer_height = 30 if footer and compact_footer else (84 if footer else 0)
        self._configure_image_adaptive_size(image, reserved_height=34 + footer_height)
        normal_height = int(self.project_config.app.get("cheatsheet_height", 480))
        self._bottom_action_height = max(44, min(60, int(round(normal_height * 0.16))))
        self._last_content = (
            "image_with_bottom_action",
            (image.copy(), title, footer, compact_footer, button_text, on_click),
        )

        def clicked(event_x: int, event_y: int) -> bool:
            region = self._bottom_action_region
            if region is None:
                return False
            x1, y1, x2, y2 = region
            if not (x1 <= event_x <= x2 and y1 <= event_y <= y2):
                return False
            self.clear_click_handler()
            self.master.after_idle(on_click)
            return True

        self._sheet_click_handler = clicked
        self._set_hud_click_passthrough(False)
        self._render_image_with_bottom_action(image, title, footer, compact_footer, button_text)
        register_regions = getattr(self, "_set_hud_interactive_window_regions", None)
        if self._bottom_action_region is not None and callable(register_regions):
            register_regions([self._bottom_action_region])
        self._force_interactive_input_mode()

    def show_interactive_image(self, image: np.ndarray, title: str) -> None:
        """Render an image without clearing its registered image click handler."""
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        self._last_content = ("interactive_image", (image.copy(), title))
        self._render_image(image, title)
        self._force_interactive_input_mode()

    def show_confirmation(
        self,
        message: str,
        *,
        title: str,
        yes_text: str,
        no_text: str,
        on_yes,
        on_no,
    ) -> None:
        """Show two large clickable choices inside the always-on-top sheet."""
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        canvas_w, canvas_h = 900, 520
        canvas = Image.new("RGB", (canvas_w, canvas_h), (5, 7, 10))
        draw = ImageDraw.Draw(canvas)
        self._draw_fitted_text(
            draw,
            message,
            (55, 45, canvas_w - 55, 300),
            max_size=30,
            min_size=14,
            bold=True,
            spacing=12,
        )
        yes_region = (80, 350, 420, 465)
        no_region = (480, 350, 820, 465)
        draw.rounded_rectangle(yes_region, radius=18, fill=(22, 101, 52), outline=(74, 222, 128), width=4)
        draw.rounded_rectangle(no_region, radius=18, fill=(127, 29, 29), outline=(248, 113, 113), width=4)
        self._draw_fitted_text(draw, yes_text, yes_region, max_size=34, min_size=16, bold=True, centered=True)
        self._draw_fitted_text(draw, no_text, no_region, max_size=34, min_size=16, bold=True, centered=True)
        image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

        def clicked(point: tuple[float, float]) -> bool:
            x, y = point
            callback = None
            if yes_region[0] <= x <= yes_region[2] and yes_region[1] <= y <= yes_region[3]:
                callback = on_yes
            elif no_region[0] <= x <= no_region[2] and no_region[1] <= y <= no_region[3]:
                callback = on_no
            if callback is None:
                return False
            self.clear_click_handler()
            self.master.after_idle(callback)
            return True

        self.set_click_handler(clicked)
        self._last_content = ("interactive_image", (image.copy(), title))
        self._render_image(image, title)
        register_regions = getattr(self, "_set_hud_interactive_image_regions", None)
        if callable(register_regions):
            register_regions([yes_region, no_region])
        self._force_interactive_input_mode()

    def show_action_button(
        self,
        message: str,
        *,
        title: str,
        button_text: str,
        on_click,
    ) -> None:
        """Show a message with one large clickable action button."""
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        canvas_w, canvas_h = 900, 520
        canvas = Image.new("RGB", (canvas_w, canvas_h), (5, 7, 10))
        draw = ImageDraw.Draw(canvas)
        self._draw_fitted_text(
            draw,
            message,
            (55, 45, canvas_w - 55, 300),
            max_size=30,
            min_size=14,
            bold=True,
            spacing=12,
        )
        button_region = (240, 350, 660, 465)
        draw.rounded_rectangle(
            button_region,
            radius=18,
            fill=(30, 64, 112),
            outline=(96, 165, 250),
            width=4,
        )
        self._draw_fitted_text(
            draw,
            button_text,
            button_region,
            max_size=34,
            min_size=16,
            bold=True,
            centered=True,
        )
        image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

        def clicked(point: tuple[float, float]) -> bool:
            x, y = point
            if not (
                button_region[0] <= x <= button_region[2]
                and button_region[1] <= y <= button_region[3]
            ):
                return False
            self.clear_click_handler()
            self.master.after_idle(on_click)
            return True

        self.set_click_handler(clicked)
        self._last_content = ("interactive_image", (image.copy(), title))
        self._render_image(image, title)
        register_regions = getattr(self, "_set_hud_interactive_image_regions", None)
        if callable(register_regions):
            register_regions([button_region])
        self._force_interactive_input_mode()

    def show_image_confirmation(
        self,
        image: np.ndarray,
        message: str,
        *,
        title: str,
        yes_text: str,
        no_text: str,
        on_yes,
        on_no,
    ) -> None:
        """Show a reference image followed by explicit yes/no choices."""
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        canvas_w, canvas_h = 900, 700
        canvas = Image.new("RGB", (canvas_w, canvas_h), (5, 7, 10))
        draw = ImageDraw.Draw(canvas)
        source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        source.thumbnail((810, 390), Image.Resampling.LANCZOS)
        image_x = (canvas_w - source.width) // 2
        image_y = 30 + (390 - source.height) // 2
        canvas.paste(source, (image_x, image_y))
        self._draw_fitted_text(
            draw,
            message,
            (55, 430, canvas_w - 55, 525),
            max_size=28,
            min_size=14,
            bold=True,
            spacing=10,
        )
        yes_region = (80, 555, 420, 665)
        no_region = (480, 555, 820, 665)
        draw.rounded_rectangle(yes_region, radius=18, fill=(22, 101, 52), outline=(74, 222, 128), width=4)
        draw.rounded_rectangle(no_region, radius=18, fill=(127, 29, 29), outline=(248, 113, 113), width=4)
        self._draw_fitted_text(draw, yes_text, yes_region, max_size=32, min_size=16, bold=True, centered=True)
        self._draw_fitted_text(draw, no_text, no_region, max_size=32, min_size=16, bold=True, centered=True)
        rendered = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

        def clicked(point: tuple[float, float]) -> bool:
            x, y = point
            callback = None
            if yes_region[0] <= x <= yes_region[2] and yes_region[1] <= y <= yes_region[3]:
                callback = on_yes
            elif no_region[0] <= x <= no_region[2] and no_region[1] <= y <= no_region[3]:
                callback = on_no
            if callback is None:
                return False
            self.clear_click_handler()
            self.master.after_idle(callback)
            return True

        self.set_click_handler(clicked)
        self._last_content = ("interactive_image", (rendered.copy(), title))
        self._render_image(rendered, title)
        register_regions = getattr(self, "_set_hud_interactive_image_regions", None)
        if callable(register_regions):
            register_regions([yes_region, no_region])
        self._force_interactive_input_mode()

    def show_image_confirmation_with_choices(
        self,
        image: np.ndarray,
        message: str,
        choices_image: np.ndarray,
        choice_regions: list[tuple[str, tuple[int, int, int, int]]],
        *,
        title: str,
        yes_text: str,
        no_text: str,
        on_yes,
        on_no,
        on_choice,
    ) -> None:
        """Show confirmation controls followed by clickable ambiguous structures."""
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        canvas_w = 900
        choice_source = Image.fromarray(cv2.cvtColor(choices_image, cv2.COLOR_BGR2RGB))
        choice_scale = min(1.0, 810 / max(1, choice_source.width))
        choice_w = max(1, int(round(choice_source.width * choice_scale)))
        choice_h = max(1, int(round(choice_source.height * choice_scale)))
        if (choice_w, choice_h) != choice_source.size:
            choice_source = choice_source.resize((choice_w, choice_h), Image.Resampling.LANCZOS)
        choices_y = 720
        canvas_h = choices_y + choice_h + 30
        canvas = Image.new("RGB", (canvas_w, canvas_h), (5, 7, 10))
        draw = ImageDraw.Draw(canvas)

        source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        source.thumbnail((810, 360), Image.Resampling.LANCZOS)
        image_x = (canvas_w - source.width) // 2
        image_y = 25 + (360 - source.height) // 2
        canvas.paste(source, (image_x, image_y))
        self._draw_fitted_text(
            draw,
            message,
            (55, 400, canvas_w - 55, 495),
            max_size=28,
            min_size=14,
            bold=True,
            spacing=10,
        )
        yes_region = (80, 520, 420, 625)
        no_region = (480, 520, 820, 625)
        draw.rounded_rectangle(yes_region, radius=18, fill=(22, 101, 52), outline=(74, 222, 128), width=4)
        draw.rounded_rectangle(no_region, radius=18, fill=(127, 29, 29), outline=(248, 113, 113), width=4)
        self._draw_fitted_text(draw, yes_text, yes_region, max_size=32, min_size=16, bold=True, centered=True)
        self._draw_fitted_text(draw, no_text, no_region, max_size=32, min_size=16, bold=True, centered=True)
        self._draw_fitted_text(
            draw,
            "或点击下方侧门结构图手动选择",
            (55, 650, canvas_w - 55, 700),
            max_size=24,
            min_size=14,
            bold=True,
        )
        choices_x = (canvas_w - choice_w) // 2
        canvas.paste(choice_source, (choices_x, choices_y))
        rendered_regions = [
            (
                map_id,
                (
                    int(round(choices_x + x1 * choice_scale)),
                    int(round(choices_y + y1 * choice_scale)),
                    int(round(choices_x + x2 * choice_scale)),
                    int(round(choices_y + y2 * choice_scale)),
                ),
            )
            for map_id, (x1, y1, x2, y2) in choice_regions
        ]
        rendered = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

        def clicked(point: tuple[float, float]) -> bool:
            x, y = point
            callback = None
            if yes_region[0] <= x <= yes_region[2] and yes_region[1] <= y <= yes_region[3]:
                callback = on_yes
            elif no_region[0] <= x <= no_region[2] and no_region[1] <= y <= no_region[3]:
                callback = on_no
            else:
                for map_id, (x1, y1, x2, y2) in rendered_regions:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        callback = lambda value=map_id: on_choice(value)
                        break
            if callback is None:
                return False
            self.clear_click_handler()
            self.master.after_idle(callback)
            return True

        screen_h = max(1, self.master.winfo_screenheight())
        normal_width = int(self.project_config.app.get("cheatsheet_width", self.master.winfo_screenwidth() // 2))
        normal_height = int(self.project_config.app.get("cheatsheet_height", screen_h // 2))
        content_height = 34 + int(round(canvas_h * normal_width / canvas_w))
        height_key = (
            "cheatsheet_ambiguous_two_choice_height"
            if len(choice_regions) <= 2
            else "cheatsheet_ambiguous_multi_choice_height"
        )
        try:
            configured_height = int(self.project_config.app.get(height_key, content_height))
        except (TypeError, ValueError):
            configured_height = content_height
        temporary_height = min(int(screen_h * 0.90), max(normal_height, configured_height))
        self._temporary_size = (normal_width, temporary_height)
        self.set_click_handler(clicked)
        self._last_content = ("interactive_image", (rendered.copy(), title))
        self._render_image(rendered, title)
        register_regions = getattr(self, "_set_hud_interactive_image_regions", None)
        if callable(register_regions):
            register_regions([yes_region, no_region, *(region for _map_id, region in rendered_regions)])
        self._force_interactive_input_mode()
        # A confirmation sheet may be created immediately after another HUD
        # state. Re-sync once after the final render so all choice catchers are
        # present at their current native coordinates.
        sync_windows = getattr(self, "_sync_hud_interaction_windows", None)
        if callable(sync_windows):
            sync_windows()

    def restore_content(self, content: Optional[tuple[str, tuple]]) -> None:
        if content is None:
            return
        kind, payload = content
        if kind == "image_with_bottom_action":
            image, title, footer, compact_footer, button_text, on_click = payload
            self.show_image_with_bottom_action(
                image,
                title,
                footer=footer,
                compact_footer=compact_footer,
                button_text=button_text,
                on_click=on_click,
            )
            return
        self.clear_click_handler()
        self._last_content = content
        self._rerender_last_content()

    def _force_interactive_input_mode(self) -> None:
        """Keep the HUD transparent and expose only its independent button catchers."""
        self._hud_click_passthrough_enabled = False
        if not sys.platform.startswith("win"):
            self._sync_hud_interaction_windows()
            return
        try:
            user32, get_window_long, set_window_long, _call_window_proc = self._win32_hud_window_functions()
            gwl_exstyle = -20
            ws_ex_transparent = 0x00000020
            ga_root = 2
            swp_framechanged = 0x0020
            swp_no_move = 0x0002
            swp_no_size = 0x0001
            swp_no_activate = 0x0010
            hwnd = int(self.window.winfo_id())
            root_hwnd = int(user32.GetAncestor(wintypes.HWND(hwnd), ga_root) or hwnd)
            for target_hwnd in {hwnd, root_hwnd}:
                style = int(get_window_long(wintypes.HWND(target_hwnd), gwl_exstyle))
                set_window_long(
                    wintypes.HWND(target_hwnd),
                    gwl_exstyle,
                    ctypes.c_ssize_t(style | ws_ex_transparent),
                )
                user32.SetWindowPos(
                    wintypes.HWND(target_hwnd),
                    wintypes.HWND(0),
                    0,
                    0,
                    0,
                    0,
                    swp_framechanged | swp_no_move | swp_no_size | swp_no_activate,
                )
            self.window.update_idletasks()
            self._install_hud_hit_test_passthrough()
            self._sync_hud_interaction_windows()
        except Exception:
            LOGGER.debug("failed to enable interactive cheatsheet input", exc_info=True)

    def _render_image(self, image: np.ndarray, title: str) -> None:
        width, height = self._geometry()
        full = Image.new("RGB", (width, height), (5, 7, 10))
        draw = ImageDraw.Draw(full)
        control_h = self.CONTROL_ROW_HEIGHT
        title_h = self.TITLE_ROW_HEIGHT
        title_top = control_h
        draw.rectangle((0, 0, width, title_top + title_h), fill=(10, 14, 20))
        self._draw_fitted_text(
            draw,
            title,
            (10, title_top + 2, width - 10, title_top + title_h - 2),
            max_size=16,
            min_size=9,
            bold=True,
        )
        pil = self._image_to_pil(image, trim_dark_border=self._click_handler is None)
        image_h = max(1, height - title_top - title_h)
        pil.thumbnail((width, image_h), Image.Resampling.LANCZOS)
        x = (width - pil.width) // 2
        y = title_top + title_h + (image_h - pil.height) // 2
        full.paste(pil, (x, y))
        self._image_display_info = (x, y, pil.width, pil.height, image.shape[1], image.shape[0])
        self.photo = ImageTk.PhotoImage(full)
        self.body.configure(image=self.photo)
        self._show_no_activate(width, height)
        self.window.update_idletasks()

    def show_image_with_message(self, image: np.ndarray, title: str, message: str) -> None:
        self._last_content = ("image_with_message", (image.copy(), title, message, False))
        self._render_image_with_message(image, title, message, compact_footer=False)

    def _render_image_with_message(self, image: np.ndarray, title: str, message: str, *, compact_footer: bool = False) -> None:
        width, height = self._geometry()
        full = Image.new("RGB", (width, height), (5, 7, 10))
        draw = ImageDraw.Draw(full)
        control_h = self.CONTROL_ROW_HEIGHT
        title_h = self.TITLE_ROW_HEIGHT
        title_top = control_h
        footer_h = 30 if compact_footer else min(
            max(84, int((height - control_h) * 0.30)),
            max(60, height - control_h - title_h - 70),
        )
        draw.rectangle((0, 0, width, title_top + title_h), fill=(10, 14, 20))
        self._draw_fitted_text(
            draw,
            title,
            (10, title_top + 2, width - 10, title_top + title_h - 2),
            max_size=16,
            min_size=9,
            bold=True,
        )
        pil = self._image_to_pil(image, trim_dark_border=self._click_handler is None)
        image_h = max(1, height - title_top - title_h - footer_h)
        pil.thumbnail((width, image_h), Image.Resampling.LANCZOS)
        x = (width - pil.width) // 2
        y = title_top + title_h + (image_h - pil.height) // 2
        full.paste(pil, (x, y))
        self._image_display_info = (x, y, pil.width, pil.height, image.shape[1], image.shape[0])
        draw.rectangle((0, height - footer_h, width, height), fill=(10, 14, 20))
        if compact_footer:
            self._draw_fitted_text(
                draw,
                message,
                (10, height - footer_h + 4, width - 10, height - 4),
                max_size=16,
                min_size=6,
                bold=True,
                spacing=0,
            )
        else:
            self._draw_fitted_text(
                draw,
                message,
                (16, height - footer_h + 8, width - 16, height - 8),
                max_size=18,
                min_size=6,
                bold=True,
                spacing=5,
            )
        self.photo = ImageTk.PhotoImage(full)
        self.body.configure(image=self.photo)
        self._show_no_activate(width, height)
        self.window.update_idletasks()

    def _render_image_with_bottom_action(
        self,
        image: np.ndarray,
        title: str,
        footer: str,
        compact_footer: bool,
        button_text: str,
    ) -> None:
        """Render the normal image area plus an appended, non-overlapping action row."""
        width, height = self._geometry()
        full = Image.new("RGB", (width, height), (5, 7, 10))
        draw = ImageDraw.Draw(full)
        control_h = self.CONTROL_ROW_HEIGHT
        title_h = self.TITLE_ROW_HEIGHT
        title_top = control_h
        action_h = max(40, int(self._bottom_action_height))
        footer_h = 30 if footer and compact_footer else (
            min(
                max(84, int((height - control_h) * 0.30)),
                max(60, height - control_h - title_h - action_h - 70),
            )
            if footer
            else 0
        )
        draw.rectangle((0, 0, width, title_top + title_h), fill=(10, 14, 20))
        self._draw_fitted_text(
            draw,
            title,
            (10, title_top + 2, width - 10, title_top + title_h - 2),
            max_size=16,
            min_size=9,
            bold=True,
        )

        pil = self._image_to_pil(image, trim_dark_border=True)
        image_h = max(1, height - title_top - title_h - footer_h - action_h)
        pil.thumbnail((width, image_h), Image.Resampling.LANCZOS)
        x = (width - pil.width) // 2
        y = title_top + title_h + (image_h - pil.height) // 2
        full.paste(pil, (x, y))
        self._image_display_info = (x, y, pil.width, pil.height, image.shape[1], image.shape[0])

        action_top = height - action_h
        if footer_h:
            footer_top = action_top - footer_h
            draw.rectangle((0, footer_top, width, action_top), fill=(10, 14, 20))
            self._draw_fitted_text(
                draw,
                footer,
                (10, footer_top + 4, width - 10, action_top - 4),
                max_size=16 if compact_footer else 18,
                min_size=6,
                bold=True,
                spacing=0 if compact_footer else 5,
            )
        draw.rectangle((0, action_top, width, height), fill=(10, 14, 20))
        margin_x = max(14, int(width * 0.08))
        button_region = (margin_x, action_top + 6, width - margin_x, height - 6)
        draw.rounded_rectangle(button_region, radius=9, fill=(20, 66, 106), outline=(101, 189, 255), width=2)
        self._draw_fitted_text(
            draw,
            button_text,
            button_region,
            max_size=18,
            min_size=10,
            bold=True,
            centered=True,
        )
        self._bottom_action_region = button_region
        self.photo = ImageTk.PhotoImage(full)
        self.body.configure(image=self.photo)
        self._show_no_activate(width, height)
        self.window.update_idletasks()

    def show_message(self, message: str, title: str = "识别提示") -> None:
        clear_bottom_action = getattr(self, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        self.clear_click_handler()
        self._last_content = ("message", (message, title))
        self._render_message(message, title)

    def _render_message(self, message: str, title: str = "识别提示") -> None:
        width, height = self._geometry()
        full = Image.new("RGB", (width, height), (5, 7, 10))
        draw = ImageDraw.Draw(full)
        control_h = self.CONTROL_ROW_HEIGHT
        title_h = self.TITLE_ROW_HEIGHT
        title_top = control_h
        draw.rectangle((0, 0, width, title_top + title_h), fill=(10, 14, 20))
        self._draw_fitted_text(
            draw,
            title,
            (10, title_top + 2, width - 10, title_top + title_h - 2),
            max_size=16,
            min_size=9,
            bold=True,
        )
        self._draw_fitted_text(
            draw,
            message,
            (24, title_top + title_h + 18, width - 24, height - 18),
            max_size=20,
            min_size=6,
            bold=True,
            spacing=8,
        )
        self.photo = ImageTk.PhotoImage(full)
        self._image_display_info = None
        self.body.configure(image=self.photo)
        self._show_no_activate(width, height)
        self.window.update_idletasks()

    def _rerender_last_content(self) -> None:
        if self._last_content is None:
            return
        kind, payload = self._last_content
        if kind == "image":
            image, title = payload
            self._render_image(image, title)
        elif kind == "image_with_message":
            image, title, message, compact_footer = payload
            self._render_image_with_message(image, title, message, compact_footer=compact_footer)
        elif kind == "interactive_image":
            image, title = payload
            self._render_image(image, title)
        elif kind == "image_with_bottom_action":
            image, title, footer, compact_footer, button_text, _on_click = payload
            self._render_image_with_bottom_action(image, title, footer, compact_footer, button_text)
        elif kind == "message":
            message, title = payload
            self._render_message(message, title)

    def _font(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        name = "msyhbd.ttc" if bold else "msyh.ttc"
        for candidate in (Path("C:/Windows/Fonts") / name, Path(name)):
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _image_to_pil(self, image: np.ndarray, *, trim_dark_border: bool = True) -> Image.Image:
        if trim_dark_border and bool(self.project_config.app.get("cheatsheet_trim_dark_border", True)):
            image = self._trim_dark_border(image)
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def _trim_dark_border(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            return image
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = int(self.project_config.app.get("cheatsheet_trim_dark_threshold", 12))
        mask = gray > threshold
        if int(mask.sum()) < 64:
            return image
        ys, xs = np.where(mask)
        h, w = gray.shape[:2]
        margin = int(self.project_config.app.get("cheatsheet_trim_dark_margin", 8))
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(w, int(xs.max()) + margin + 1)
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(h, int(ys.max()) + margin + 1)
        if (x1 - x0) < int(w * 0.55) or (y1 - y0) < int(h * 0.55):
            return image
        if x0 <= 0 and y0 <= 0 and x1 >= w and y1 >= h:
            return image
        return image[y0:y1, x0:x1].copy()

    def _draw_fitted_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box: tuple[int, int, int, int],
        *,
        max_size: int,
        min_size: int,
        bold: bool,
        spacing: int = 4,
        fill: tuple[int, int, int] = (249, 250, 251),
        centered: bool = False,
    ) -> None:
        x0, y0, x1, y1 = box
        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)
        fitted_font = self._font(min_size, bold=bold)
        fitted_lines = self._wrap_text(draw, text, fitted_font, box_w)
        fitted_spacing = max(1, spacing)
        for size in range(max_size, min_size - 1, -1):
            font = self._font(size, bold=bold)
            lines = self._wrap_text(draw, text, font, box_w)
            line_h = self._line_height(draw, font)
            total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing
            if total_h <= box_h:
                fitted_font = font
                fitted_lines = lines
                fitted_spacing = spacing
                break
        rendered_text = "\n".join(fitted_lines)
        if centered:
            bbox = draw.multiline_textbbox(
                (0, 0),
                rendered_text,
                font=fitted_font,
                spacing=fitted_spacing,
                align="center",
            )
            text_x = x0 + (box_w - (bbox[2] - bbox[0])) / 2 - bbox[0]
            text_y = y0 + (box_h - (bbox[3] - bbox[1])) / 2 - bbox[1]
            draw.multiline_text(
                (text_x, text_y),
                rendered_text,
                fill=fill,
                font=fitted_font,
                spacing=fitted_spacing,
                align="center",
            )
            return
        draw.multiline_text((x0, y0), rendered_text, fill=fill, font=fitted_font, spacing=fitted_spacing)

    def _wrap_text(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        lines: list[str] = []
        for raw_line in str(text).splitlines() or [""]:
            current = ""
            for char in raw_line:
                candidate = current + char
                if current and self._text_width(draw, candidate, font) > max_width:
                    lines.append(current)
                    current = char
                else:
                    current = candidate
            lines.append(current)
        return lines

    def _text_width(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0])

    def _line_height(self, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), "汉Ay", font=font)
        return max(1, int(bbox[3] - bbox[1]))

    def hide(self) -> None:
        self._destroy_hud_interaction_windows()
        self.window.withdraw()
        self.anchor.withdraw()

    def _install_hud_hit_test_passthrough(self) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            user32, get_window_long, set_window_long, call_window_proc = self._win32_hud_window_functions()
            wndproc_type = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.c_ssize_t,
            )
            gwl_wndproc = -4
            wm_nchittest = 0x0084
            ht_transparent = -1
            ht_client = 1
            ga_root = 2
            handles: set[int] = set()
            for widget in (self.window, self.body):
                hwnd = int(widget.winfo_id())
                handles.add(hwnd)
                try:
                    root_hwnd = int(user32.GetAncestor(wintypes.HWND(hwnd), ga_root) or hwnd)
                    handles.add(root_hwnd)
                except Exception:
                    pass
            for hwnd in handles:
                if hwnd in self._hud_hit_test_procs:
                    continue
                original_proc = get_window_long(wintypes.HWND(hwnd), gwl_wndproc)

                @wndproc_type
                def hit_test_proc(window_handle, message, wparam, lparam, original=original_proc):
                    if message == wm_nchittest:
                        packed_point = int(lparam)
                        screen_x = ctypes.c_short(packed_point & 0xFFFF).value
                        screen_y = ctypes.c_short((packed_point >> 16) & 0xFFFF).value
                        return (
                            ht_client
                            if self._hud_hit_test_is_interactive(screen_x, screen_y)
                            else ht_transparent
                        )
                    return call_window_proc(
                        ctypes.c_void_p(original),
                        wintypes.HWND(window_handle),
                        message,
                        wparam,
                        lparam,
                    )

                callback_address = ctypes.cast(hit_test_proc, ctypes.c_void_p).value
                set_window_long(
                    wintypes.HWND(hwnd),
                    gwl_wndproc,
                    ctypes.c_ssize_t(int(callback_address or 0)),
                )
                self._hud_hit_test_procs[hwnd] = hit_test_proc
        except Exception:
            LOGGER.debug("failed to install cheatsheet hit-test passthrough", exc_info=True)

    def _show_no_activate(self, width: int, height: int) -> None:
        if not self.is_collapsed:
            self.window.deiconify()
        self.anchor.deiconify()
        if not self.is_collapsed:
            self.window.lift()
        self.anchor.lift()
        if not self.is_collapsed:
            self.window.attributes("-topmost", True)
        self.anchor.attributes("-topmost", True)
        if sys.platform.startswith("win"):
            try:
                hwnd = int(self.window.winfo_id())
                user32, get_window_long, set_window_long, _call_window_proc = self._win32_hud_window_functions()
                gwl_exstyle = -20
                ws_ex_toolwindow = 0x00000080
                ws_ex_transparent = 0x00000020
                hwnd_topmost = -1
                swp_showwindow = 0x0040
                swp_framechanged = 0x0020
                if not self.is_collapsed:
                    ga_root = 2
                    try:
                        root_hwnd = int(user32.GetAncestor(wintypes.HWND(hwnd), ga_root) or hwnd)
                    except Exception:
                        root_hwnd = hwnd
                    for target_hwnd in {hwnd, root_hwnd}:
                        style = int(get_window_long(wintypes.HWND(target_hwnd), gwl_exstyle))
                        style |= ws_ex_toolwindow
                        # The rendered HUD always passes through. Interactive
                        # rectangles are handled by separate no-activate windows.
                        style |= ws_ex_transparent
                        set_window_long(
                            wintypes.HWND(target_hwnd),
                            gwl_exstyle,
                            ctypes.c_ssize_t(style),
                        )
                    self.window.attributes("-alpha", self.alpha)
                    user32.SetWindowPos(
                        wintypes.HWND(root_hwnd),
                        wintypes.HWND(hwnd_topmost),
                        self.x,
                        self.y,
                        width,
                        height,
                        swp_showwindow | swp_framechanged,
                    )
                    self.window.update_idletasks()
                    self._install_hud_hit_test_passthrough()
                    self._sync_hud_interaction_windows()
                anchor_hwnd = int(self.anchor.winfo_id())
                anchor_style = int(get_window_long(wintypes.HWND(anchor_hwnd), gwl_exstyle))
                set_window_long(
                    wintypes.HWND(anchor_hwnd),
                    gwl_exstyle,
                    ctypes.c_ssize_t(anchor_style | ws_ex_toolwindow),
                )
            except Exception:
                LOGGER.debug("failed to apply no-activate style to cheatsheet", exc_info=True)


def make_top_candidate_contact_sheet(
    current_roi: np.ndarray,
    summary: MatchSummary,
    map_db: MapDatabase,
    sample_db: Optional[ScreenshotSampleDatabase] = None,
) -> np.ndarray:
    target_w, target_h = 180, 160
    header_h = 28

    def fit_tile(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = np.full((target_h + header_h, target_w, 3), 32, dtype=np.uint8)
        h, w = image.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        y = header_h + (target_h - resized.shape[0]) // 2
        x = (target_w - resized.shape[1]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return tile

    def label(tile: np.ndarray, text: str) -> np.ndarray:
        out = tile.copy()
        cv2.putText(out, text[:26], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        return out

    tiles = [label(fit_tile(current_roi), "current ROI")]
    for idx, result in enumerate(summary.top3, 1):
        image = None
        if result.source in {"sample", "sample_map"} and sample_db is not None:
            image = sample_db.image_by_sample_id(result.sample_id)
        elif result.map_id in map_db.templates:
            image = map_db.templates[result.map_id].template
        if image is None:
            continue
        title = f"{idx} {result.map_id} f={result.final_score:.3f}"
        tiles.append(label(fit_tile(image), title))
    return np.hstack(tiles)


def base_map_id_from_alignment(map_id: str) -> str:
    for suffix in ("_b1", "_1st", "_2nd"):
        if map_id.endswith(suffix):
            return map_id[: -len(suffix)]
    return map_id


def floor_template_dir(app: dict, floor_suffix: str) -> Path:
    if floor_suffix == "2nd":
        raw_dir = app.get(
            "second_floor_template_maps_dir",
            app.get("second_floor_structure_templates_dir", r"C:\Users\LD\Desktop\摸金工具\地图合集\二楼"),
        )
    else:
        raw_dir = app.get("floor_template_maps_dir", r"C:\Users\LD\Desktop\摸金工具\maps")
    return resolve_config_path(project_root(), str(raw_dir))


def resolve_floor_template_path(app: dict, alignment_map_id: str, floor_suffix: str) -> Optional[Path]:
    base = base_map_id_from_alignment(alignment_map_id)
    nightmare_match = re.fullmatch(r"nightmare_(\d+)", base)
    if nightmare_match is not None and floor_suffix in {"b1", "1st", "2nd"}:
        root = resolve_config_path(
            project_root(),
            str(
                app.get(
                    "nightmare_clean_template_dir",
                    r"C:\Users\LD\Desktop\摸金工具\nightmare_maps\clean_template",
                )
            ),
        )
        number = int(nightmare_match.group(1))
        source_number = 14 if number == 13 else number
        patterns = {
            "b1": (r"^负一楼$", r"^f-?1$", r"^b1$"),
            "1st": (r"^1楼$", r"^f1$", r".*1楼$"),
            "2nd": (r"^2楼$", r"^f2$", r".*2楼$"),
        }
        for directory in (root / f"{source_number}_TBD", root / str(source_number)):
            if not directory.exists():
                continue
            candidates: list[tuple[int, str, Path]] = []
            for path in directory.iterdir():
                if not path.is_file() or path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                stem = path.stem.strip().lower().replace(" ", "")
                for rank, pattern in enumerate(patterns[floor_suffix]):
                    if re.fullmatch(pattern, stem):
                        candidates.append((rank, path.name.lower(), path))
                        break
            if candidates:
                return min(candidates)[2]
        return None
    directory = floor_template_dir(app, floor_suffix)
    expected_stem = f"{base}_{floor_suffix}"
    for ext in (".jpg", ".png", ".jpeg"):
        path = directory / f"{expected_stem}{ext}"
        if path.exists():
            return path
    # Historical source images for layouts 23-28 retain their collection suffix,
    # for example ``map_23_1st(03).jpg``. They are the same localization source.
    try:
        candidates = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".png", ".jpeg"}
            and path.stem.startswith(f"{expected_stem}(")
            and path.stem.endswith(")")
        )
    except OSError:
        candidates = []
    if candidates:
        return candidates[0]
    return None


def load_floor_template(app: dict, alignment_map_id: str, floor_suffix: str) -> Optional[np.ndarray]:
    path = resolve_floor_template_path(app, alignment_map_id, floor_suffix)
    if path is None:
        return None
    return imread_unicode(path)


def entry_template_dir(app: dict, profile_key: str = "difficulty") -> Path:
    if profile_key == "nightmare":
        raw_dir = app.get("nightmare_recognition_entry_template_dir", "assets/entry_templates/nightmare")
        return resolve_config_path(project_root(), str(raw_dir))
    raw_dir = app.get("recognition_entry_template_dir")
    if raw_dir:
        return resolve_config_path(project_root(), str(raw_dir))
    first_floor_dir = floor_template_dir(app, "1st")
    return first_floor_dir.parent / "入口"


def resolve_entry_template_path(app: dict, alignment_map_id: str) -> Optional[Path]:
    base = base_map_id_from_alignment(alignment_map_id)
    profile_key = "nightmare" if base.startswith("nightmare_") else "difficulty"
    pattern = r"nightmare_(\d+)" if profile_key == "nightmare" else r"map_(\d+)"
    match = re.fullmatch(pattern, base)
    if match is None:
        return None
    number = str(int(match.group(1)))
    directory = entry_template_dir(app, profile_key)
    if not directory.exists():
        return None
    expected_stems = {number, f"nightmare_{int(number):02d}"} if profile_key == "nightmare" else {number}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
            continue
        stem = path.stem.strip()
        if stem in expected_stems or (
            profile_key == "difficulty"
            and stem.startswith(number)
            and (len(stem) == len(number) or not stem[len(number)].isdigit())
        ):
            return path
    return None


def load_entry_template(app: dict, alignment_map_id: str) -> Optional[np.ndarray]:
    path = resolve_entry_template_path(app, alignment_map_id)
    if path is None:
        return None
    return imread_unicode(path)


def make_floor_contact_sheet(app: dict, alignment_map_id: str, *, player_map: Optional[np.ndarray] = None, active_floor: str = "1st") -> Optional[np.ndarray]:
    first = player_map if active_floor == "1st" and player_map is not None else load_floor_template(app, alignment_map_id, "1st")
    second = player_map if active_floor == "2nd" and player_map is not None else load_floor_template(app, alignment_map_id, "2nd")
    images: list[tuple[str, np.ndarray]] = []
    if first is not None:
        images.append(("1F", first))
    if second is not None:
        images.append(("2F", second))
    if not images:
        return None
    return make_labeled_floor_contact_sheet(alignment_map_id, images)


def make_labeled_floor_contact_sheet(alignment_map_id: str, images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    width = max(image.shape[1] for _label, image in images)
    header_h = 34
    total_h = sum(image.shape[0] + header_h for _label, image in images)
    out = np.full((total_h, width, 3), (5, 7, 10), dtype=np.uint8)
    y = 0
    for label, image in images:
        cv2.rectangle(out, (0, y), (width, y + header_h), (10, 14, 20), -1)
        cv2.putText(out, f"{label}  {alignment_map_id}", (12, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (249, 250, 251), 2, cv2.LINE_AA)
        x = (width - image.shape[1]) // 2
        out[y + header_h : y + header_h + image.shape[0], x : x + image.shape[1]] = image
        y += image.shape[0] + header_h
    return out


def make_route_floor_contact_sheet(alignment_map_id: str, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Stack the legacy route images in the same visual order as the floor sheets."""
    return make_labeled_floor_contact_sheet(alignment_map_id, [("1F", first), ("2F", second)])


def trim_dark_image_border(image: np.ndarray, threshold: int = 12, padding: int = 8) -> np.ndarray:
    """Crop empty dark margins while retaining a small visual gutter."""
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ys, xs = np.where(gray > max(0, int(threshold)))
    if not len(xs) or not len(ys):
        return image
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(image.shape[1], int(xs.max()) + padding + 1)
    y1 = min(image.shape[0], int(ys.max()) + padding + 1)
    return image[y0:y1, x0:x1].copy()


def make_nightmare_three_floor_sheet(
    alignment_map_id: str,
    images: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Lay out B1/1F/2F vertically with compact headers and minimal dead space."""
    prepared = [(label, trim_dark_image_border(image, padding=2)) for label, image in images if image is not None]
    if not prepared:
        raise ValueError("nightmare floor sheet requires at least one image")
    canvas_w, header_h, gap = 1000, 30, 5
    resized_images: list[tuple[str, np.ndarray]] = []
    for label, image in prepared:
        height, width = image.shape[:2]
        scale = canvas_w / max(width, 1)
        resized_images.append(
            (
                label,
                cv2.resize(
                    image,
                    (canvas_w, max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
                ),
            )
        )
    canvas_h = sum(header_h + image.shape[0] for _label, image in resized_images)
    canvas_h += max(0, len(resized_images) - 1) * gap
    output = np.full((canvas_h, canvas_w, 3), (5, 7, 10), dtype=np.uint8)
    origin_y = 0
    for label, resized in resized_images:
        cv2.rectangle(output, (0, origin_y), (canvas_w, origin_y + header_h), (10, 18, 27), -1)
        cv2.putText(
            output,
            f"{label}  {base_map_id_from_alignment(alignment_map_id)}",
            (12, origin_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 247, 250),
            2,
            cv2.LINE_AA,
        )
        y = origin_y + header_h
        output[y : y + resized.shape[0], :canvas_w] = resized
        origin_y = y + resized.shape[0] + gap
    return output


def crop_route_image_to_template_bounds(
    route_image: np.ndarray,
    template_size: tuple[int, int],
    template_to_route: np.ndarray,
    *,
    padding: int = 16,
) -> np.ndarray:
    """Crop one floor from a combined route canvas using its reviewed affine."""
    if route_image is None or route_image.size == 0:
        raise ValueError("route image is empty")
    template_w, template_h = (int(value) for value in template_size)
    matrix = np.asarray(template_to_route, dtype=np.float32)
    if template_w <= 0 or template_h <= 0 or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid route crop transform")
    corners = np.float32(
        [[0.0, 0.0], [float(template_w), 0.0], [0.0, float(template_h)], [float(template_w), float(template_h)]]
    )
    transformed = cv2.transform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
    route_h, route_w = route_image.shape[:2]
    margin = max(0, int(padding))
    x0 = max(0, int(np.floor(float(transformed[:, 0].min()))) - margin)
    y0 = max(0, int(np.floor(float(transformed[:, 1].min()))) - margin)
    x1 = min(route_w, int(np.ceil(float(transformed[:, 0].max()))) + margin + 1)
    y1 = min(route_h, int(np.ceil(float(transformed[:, 1].max()))) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("route crop is outside the combined route canvas")
    return route_image[y0:y1, x0:x1].copy()


def lianghapi_dual_cheatsheet_enabled(
    app: Mapping[str, Any],
    *,
    profile_key: Optional[str] = None,
) -> bool:
    """Whether Difficulty-mode Lianghapi routes should show both floors."""
    active_profile = str(
        profile_key if profile_key is not None else app.get("recognition_profile", "difficulty")
    ).strip().lower()
    version = str(app.get("route_map_version", "lianghapi")).strip().lower()
    display_mode = str(app.get("lianghapi_cheatsheet_mode", "single")).strip().lower()
    return active_profile != "nightmare" and version == "lianghapi" and display_mode == "dual"


def nightmare_three_cheatsheet_enabled(app: dict) -> bool:
    return (
        nightmare_route_map_version(app) == "clean_template"
        and str(app.get("nightmare_cheatsheet_mode", "single")).strip().lower() == "triple"
    )


def nightmare_lianghapi_cheatsheet_mode(app: Mapping[str, Any]) -> str:
    """Return the Nightmare-only display mode for Lianghapi route canvases."""
    value = str(app.get("nightmare_lianghapi_cheatsheet_mode", "triple")).strip().lower()
    return value if value in {"single", "triple"} else "triple"


def nightmare_lianghapi_single_cheatsheet_enabled(app: Mapping[str, Any]) -> bool:
    return (
        nightmare_route_map_version(app) == "lianghapi"
        and nightmare_lianghapi_cheatsheet_mode(app) == "single"
    )


def nightmare_route_map_version(app: Mapping[str, Any]) -> str:
    value = str(app.get("nightmare_route_map_version", "clean_template")).strip().lower()
    return value if value in {"clean_template", "zhanshi", "lianghapi"} else "clean_template"


def nightmare_zhanshi_route_variant(app: Mapping[str, Any]) -> str:
    """Return the Nightmare-only Zhanshi route set without touching Difficulty."""
    value = str(app.get("nightmare_zhanshi_route_variant", "fast")).strip().lower()
    return value if value in {"fast", "full_coffin"} else "fast"


def difficulty_zhanshi_route_variant(app: Mapping[str, Any]) -> str:
    """Return the Difficulty-only Zhanshi route set without touching Nightmare."""
    value = str(app.get("difficulty_zhanshi_route_variant", "full_coffin")).strip().lower()
    return value if value in {"fast", "full_coffin"} else "full_coffin"


def nightmare_zhanshi_route_enabled(app: Mapping[str, Any]) -> bool:
    return nightmare_route_map_version(app) == "zhanshi"


def nightmare_lianghapi_route_enabled(app: Mapping[str, Any]) -> bool:
    return nightmare_route_map_version(app) == "lianghapi"


def nightmare_route_map_enabled(app: Mapping[str, Any]) -> bool:
    return nightmare_route_map_version(app) in {"zhanshi", "lianghapi"}


def piecewise_diagonal_matrix_at_y(piecewise: Mapping[str, Any], source_y: float) -> np.ndarray:
    """Return the local template-to-route matrix for a vertical blend transform."""
    upper = np.asarray(piecewise.get("upper_matrix"), dtype=np.float32)
    lower = np.asarray(piecewise.get("lower_matrix"), dtype=np.float32)
    blend = piecewise.get("source_y_blend")
    if upper.shape != (2, 3) or lower.shape != (2, 3) or not isinstance(blend, (list, tuple)) or len(blend) < 2:
        raise ValueError("invalid piecewise route transform")
    start, end = float(blend[0]), float(blend[1])
    weight = 0.0 if source_y <= start else 1.0 if source_y >= end else (source_y - start) / max(1e-6, end - start)
    return upper * (1.0 - weight) + lower * weight


def remap_piecewise_route_to_template(
    route_image: np.ndarray,
    template_size: tuple[int, int],
    piecewise: Mapping[str, Any],
) -> np.ndarray:
    """Resample a piecewise route image into its clean-template coordinate space."""
    template_width, template_height = (int(value) for value in template_size)
    if template_width <= 0 or template_height <= 0:
        raise ValueError("invalid template size for piecewise route transform")
    top = np.asarray(piecewise.get("upper_matrix"), dtype=np.float32)
    bottom = np.asarray(piecewise.get("lower_matrix"), dtype=np.float32)
    blend = piecewise.get("source_y_blend")
    if top.shape != (2, 3) or bottom.shape != (2, 3) or not isinstance(blend, (list, tuple)) or len(blend) < 2:
        raise ValueError("invalid piecewise route transform")
    start, end = float(blend[0]), float(blend[1])
    y = np.arange(template_height, dtype=np.float32)
    weight = np.clip((y - start) / max(1e-6, end - start), 0.0, 1.0)
    scale_x = top[0, 0] * (1.0 - weight) + bottom[0, 0] * weight
    scale_y = top[1, 1] * (1.0 - weight) + bottom[1, 1] * weight
    offset_x = top[0, 2] * (1.0 - weight) + bottom[0, 2] * weight
    offset_y = top[1, 2] * (1.0 - weight) + bottom[1, 2] * weight
    x = np.arange(template_width, dtype=np.float32)[None, :]
    map_x = scale_x[:, None] * x + offset_x[:, None]
    map_y = np.broadcast_to(
        scale_y[:, None] * y[:, None] + offset_y[:, None],
        map_x.shape,
    ).copy()
    return cv2.remap(route_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def route_region_matrix_at_point(
    global_matrix: np.ndarray,
    regions: object,
    point: tuple[float, float],
) -> np.ndarray:
    """Return the local template-to-route affine selected by the configured region."""
    matrix = np.asarray(global_matrix, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid global route transform")
    if not isinstance(regions, list):
        return matrix
    px, py = (float(point[0]), float(point[1]))
    for region in reversed(regions):
        if not isinstance(region, Mapping):
            continue
        try:
            candidate = np.asarray(region.get("matrix"), dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if candidate.shape != (2, 3) or not np.isfinite(candidate).all():
            continue
        rect = region.get("template_rect")
        if isinstance(rect, (list, tuple)) and len(rect) >= 4:
            try:
                x, y, width, height = (float(value) for value in rect[:4])
            except (TypeError, ValueError):
                continue
            if x <= px <= x + width and y <= py <= y + height:
                return candidate
            continue
        polygon = region.get("template_polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        try:
            contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        except (TypeError, ValueError):
            continue
        if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
            return candidate
    return matrix


def remap_region_route_to_template(
    route_image: np.ndarray,
    template_size: tuple[int, int],
    global_matrix: np.ndarray,
    regions: object,
) -> np.ndarray:
    """Resample a route image into clean-template coordinates with local region affines."""
    template_width, template_height = (int(value) for value in template_size)
    matrix = np.asarray(global_matrix, dtype=np.float32)
    if template_width <= 0 or template_height <= 0 or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid region route transform")
    y, x = np.indices((template_height, template_width), dtype=np.float32)
    map_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    map_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            try:
                local = np.asarray(region.get("matrix"), dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if local.shape != (2, 3) or not np.isfinite(local).all():
                continue
            mask = np.zeros((template_height, template_width), dtype=np.uint8)
            rect = region.get("template_rect")
            if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                try:
                    left, top, width, height = (int(round(float(value))) for value in rect[:4])
                except (TypeError, ValueError):
                    continue
                cv2.rectangle(mask, (left, top), (left + width, top + height), 255, -1)
            else:
                polygon = region.get("template_polygon")
                if not isinstance(polygon, list) or len(polygon) < 3:
                    continue
                try:
                    contour = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
                except (TypeError, ValueError):
                    continue
                cv2.fillPoly(mask, [contour], 255)
            selected = mask > 0
            map_x[selected] = local[0, 0] * x[selected] + local[0, 1] * y[selected] + local[0, 2]
            map_y[selected] = local[1, 0] * x[selected] + local[1, 1] * y[selected] + local[1, 2]
    return cv2.remap(route_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def semantic_cv_player_marker_scale(
    app_config: Mapping[str, Any],
    image_size: tuple[int, int],
) -> float:
    """Keep player labels equally legible after maps are fit into a cheatsheet."""
    width, height = (max(1, int(value)) for value in image_size)
    try:
        reference_width = max(1.0, float(app_config.get("semantic_cv_player_marker_reference_width", 1474)))
        base_scale = max(0.5, float(app_config.get("semantic_cv_player_marker_display_scale", 2.0)))
        minimum = max(0.5, float(app_config.get("semantic_cv_player_marker_min_scale", 2.0)))
        maximum = max(minimum, float(app_config.get("semantic_cv_player_marker_max_scale", 10.0)))
    except (TypeError, ValueError):
        reference_width, base_scale, minimum, maximum = 1474.0, 2.0, 2.0, 10.0
    # Cheat sheets fit maps primarily by width. Scale the source annotation
    # by source width so its visible diameter remains stable after that fit.
    scale = base_scale * float(width) / reference_width
    return min(max(scale, minimum), maximum)


def route_to_screen_roi_affine(
    *,
    best_position: tuple[float, float],
    best_scale: float,
    roi_size: tuple[int, int],
    template_to_route: np.ndarray,
    observation_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return the affine transform which places route-map pixels in the CV ROI."""
    roi_w, roi_h = roi_size
    if roi_w <= 0 or roi_h <= 0 or best_scale <= 0:
        raise ValueError("invalid ROI size or CV scale")
    matrix = np.asarray(template_to_route, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid template-to-route affine transform")
    roi_points = np.float32([[0.0, 0.0], [float(roi_w), 0.0], [0.0, float(roi_h)]])
    template_points = (
        (roi_points - np.float32(observation_offset)) * float(best_scale)
        + np.float32(best_position)
    )
    route_points = cv2.transform(template_points.reshape(1, -1, 2), matrix).reshape(-1, 2)
    roi_to_route = cv2.getAffineTransform(roi_points, route_points.astype(np.float32))
    return cv2.invertAffineTransform(roi_to_route)


def route_to_screen_roi_from_calibrated_affines(
    *,
    roi_to_template: np.ndarray,
    template_to_route: np.ndarray,
) -> np.ndarray:
    """Invert the calibrated ROI -> template -> route mapping for the screen overlay."""
    roi_to_template_matrix = np.asarray(roi_to_template, dtype=np.float32)
    template_to_route_matrix = np.asarray(template_to_route, dtype=np.float32)
    if (
        roi_to_template_matrix.shape != (2, 3)
        or template_to_route_matrix.shape != (2, 3)
        or not np.isfinite(roi_to_template_matrix).all()
        or not np.isfinite(template_to_route_matrix).all()
    ):
        raise ValueError("invalid calibrated affine transform")
    roi_to_template_3x3 = np.vstack((roi_to_template_matrix, np.float32([0.0, 0.0, 1.0])))
    template_to_route_3x3 = np.vstack((template_to_route_matrix, np.float32([0.0, 0.0, 1.0])))
    roi_to_route = (template_to_route_3x3 @ roi_to_template_3x3)[:2]
    return cv2.invertAffineTransform(roi_to_route)


def template_to_screen_roi_hybrid_affine(
    *,
    best_position_full: tuple[float, float],
    observation_offset: tuple[float, float],
    roi_to_template_linear: np.ndarray,
) -> np.ndarray:
    """Invert a dynamic translation plus calibrated ROI-to-template linear map."""
    linear = np.asarray(roi_to_template_linear, dtype=np.float32)
    if linear.shape != (2, 2) or not np.isfinite(linear).all():
        raise ValueError("invalid ROI-to-template linear transform")
    if abs(float(np.linalg.det(linear))) < 1e-7:
        raise ValueError("singular ROI-to-template linear transform")
    origin = np.float32(best_position_full)
    observation = np.float32(observation_offset)
    translation = origin - linear @ observation
    roi_to_template = np.column_stack((linear, translation)).astype(np.float32)
    return cv2.invertAffineTransform(roi_to_template)


def semantic_cv_fixed_scale_for_map(
    app_config: dict,
    map_id: str,
    root_dir: Optional[Path] = None,
) -> Optional[float]:
    """Read the one current scale for this exact layout, without map groups."""
    normalized_map_id = str(map_id).strip().lower()
    configured_scales = app_config.get("semantic_cv_per_layout_scales")
    if isinstance(configured_scales, dict) and normalized_map_id in configured_scales:
        value = configured_scales.get(normalized_map_id)
        if isinstance(value, dict):
            value = value.get("scale")
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return None
        return scale if scale > 0 else None
    scale_path_value = str(app_config.get("semantic_cv_per_layout_scale_path", "")).strip()
    if scale_path_value:
        scale_path = Path(scale_path_value)
        if not scale_path.is_absolute():
            project_root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
            scale_path = project_root / scale_path
        try:
            data = json.loads(scale_path.read_text(encoding="utf-8-sig"))
            layouts = data.get("layouts", {}) if isinstance(data, dict) else {}
            value = layouts.get(normalized_map_id) if isinstance(layouts, dict) else None
            if isinstance(value, dict):
                value = value.get("scale")
            scale = float(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return scale if scale > 0 else None
    if isinstance(configured_scales, dict):
        return None
    if "semantic_cv_fixed_scale" not in app_config:
        return None
    try:
        scale = float(app_config.get("semantic_cv_fixed_scale"))
    except (TypeError, ValueError):
        return None
    return scale if scale > 0 else None


def semantic_cv_reference_scale_for_map(
    app_config: dict,
    map_id: str,
    root_dir: Path,
) -> Optional[float]:
    """Get a layout's reference-ROI scale from the durable or confirmed store."""
    scale = semantic_cv_fixed_scale_for_map(app_config, map_id, root_dir)
    if scale is not None:
        return scale
    reference_roi_size = semantic_cv_reference_map_roi_size(app_config)
    if reference_roi_size is None:
        return None
    return semantic_cv_confirmed_scale_for_map(
        semantic_cv_confirmed_scale_path(root_dir, app_config),
        map_id,
        reference_roi_size,
    )


def semantic_cv_resolution_scale_multiplier(app_config: dict, roi_size: tuple[int, int]) -> float:
    """Estimate the ROI pixel-density change relative to the checked-in reference ROI."""
    width, height = int(roi_size[0]), int(roi_size[1])
    if width <= 0 or height <= 0:
        return 1.0
    configured_size = app_config.get("semantic_cv_reference_map_roi_size")
    if isinstance(configured_size, (list, tuple)) and len(configured_size) >= 2:
        reference_width, reference_height = float(configured_size[0]), float(configured_size[1])
    else:
        reference_roi = app_config.get(str(app_config.get("alignment_map_roi_key", "alignment_map_roi")))
        if not isinstance(reference_roi, (list, tuple)) or len(reference_roi) < 4:
            return 1.0
        reference_width, reference_height = float(reference_roi[2]), float(reference_roi[3])
    if reference_width <= 0 or reference_height <= 0:
        return 1.0
    multiplier = float(np.sqrt((reference_width / width) * (reference_height / height)))
    minimum = max(0.1, float(app_config.get("semantic_cv_resolution_multiplier_min", 0.5)))
    maximum = max(minimum, float(app_config.get("semantic_cv_resolution_multiplier_max", 2.0)))
    return min(max(multiplier, minimum), maximum)


def semantic_cv_reference_map_roi_size(app_config: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    """Return the ROI size at which the checked-in layout scales were measured."""
    configured_size = app_config.get("semantic_cv_reference_map_roi_size")
    if isinstance(configured_size, (list, tuple)) and len(configured_size) >= 2:
        raw_size = configured_size
    else:
        reference_roi = app_config.get(str(app_config.get("alignment_map_roi_key", "alignment_map_roi")))
        raw_size = reference_roi[2:4] if isinstance(reference_roi, (list, tuple)) and len(reference_roi) >= 4 else None
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) < 2:
        return None
    try:
        width, height = int(raw_size[0]), int(raw_size[1])
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def semantic_cv_reference_roi_normalization(
    app_config: Mapping[str, Any],
    raw_roi_size: tuple[int, int],
    base_scale: Optional[float],
    *,
    force_scale_research: bool = False,
) -> Optional[dict[str, Any]]:
    """Describe raw-ROI to reference-ROI normalization for a known base scale."""
    if force_scale_research or not bool(app_config.get("semantic_cv_reference_roi_normalization_enabled", True)):
        return None
    try:
        raw_width, raw_height = int(raw_roi_size[0]), int(raw_roi_size[1])
        scale = float(base_scale) if base_scale is not None else 0.0
    except (TypeError, ValueError, IndexError):
        return None
    reference_size = semantic_cv_reference_map_roi_size(app_config)
    if raw_width <= 0 or raw_height <= 0 or reference_size is None or scale <= 0 or not np.isfinite(scale):
        return None
    reference_width, reference_height = reference_size
    if (raw_width, raw_height) == reference_size:
        return None
    return {
        "raw_roi_size": [raw_width, raw_height],
        "reference_roi_size": [reference_width, reference_height],
        "raw_to_reference_scale": [reference_width / raw_width, reference_height / raw_height],
    }


def reference_normalized_roi_to_template_affine(
    *,
    best_position_full: tuple[float, float],
    best_scale: float,
    raw_to_reference_scale: tuple[float, float],
    observation_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Map raw ROI pixels to template pixels after reference-ROI localization."""
    try:
        position = np.float32(best_position_full)
        reference_scale_x, reference_scale_y = (float(value) for value in raw_to_reference_scale)
        scale = float(best_scale)
        observation = np.float32(observation_offset)
    except (TypeError, ValueError):
        raise ValueError("invalid reference-normalized localization transform") from None
    if (
        position.shape != (2,)
        or observation.shape != (2,)
        or not np.isfinite(position).all()
        or not np.isfinite(observation).all()
        or not np.isfinite([reference_scale_x, reference_scale_y, scale]).all()
        or min(reference_scale_x, reference_scale_y, scale) <= 0
    ):
        raise ValueError("invalid reference-normalized localization transform")
    linear = np.float32([[scale * reference_scale_x, 0.0], [0.0, scale * reference_scale_y]])
    translation = position - scale * observation
    return np.column_stack((linear, translation)).astype(np.float32)


def semantic_cv_uses_reference_screen_base_scale(
    app_config: dict,
    screen_size: tuple[int, int],
    base_scale: Optional[float],
) -> bool:
    """Use a known base scale directly when capture runs at its reference resolution."""
    if base_scale is None or not np.isfinite(float(base_scale)) or float(base_scale) <= 0:
        return False
    configured_size = app_config.get("alignment_reference_capture_size", (2560, 1440))
    if not isinstance(configured_size, (list, tuple)) or len(configured_size) < 2:
        return False
    try:
        reference_size = (int(configured_size[0]), int(configured_size[1]))
        current_size = (int(screen_size[0]), int(screen_size[1]))
    except (TypeError, ValueError, IndexError):
        return False
    if min(*reference_size, *current_size) <= 0:
        return False
    canvas_scale, _offset_x, _offset_y = reference_canvas_transform(
        reference_size=reference_size,
        target_size=current_size,
    )
    return abs(canvas_scale - 1.0) <= 1e-6


def semantic_cv_adaptive_scale_window(
    base_scale: float,
    resolution_multiplier: float,
    relative_radius: float,
) -> tuple[float, float, float]:
    """Return predicted scale plus a symmetric coarse-search window."""
    predicted = float(base_scale) * float(resolution_multiplier)
    radius = min(max(float(relative_radius), 0.0), 0.8)
    return predicted, max(1e-4, predicted * (1.0 - radius)), predicted * (1.0 + radius)


def semantic_cv_unconfigured_scale_window(
    app_config: dict,
    roi_size: tuple[int, int],
    floor_suffix: Optional[str] = None,
) -> tuple[float, float, float]:
    """Scale the legacy-screen broad search range to the current ROI size."""
    floor_range = None
    configured_ranges = app_config.get("semantic_cv_unconfigured_scale_ranges")
    if isinstance(configured_ranges, dict):
        candidate = configured_ranges.get(str(floor_suffix or "").strip().lower())
        if isinstance(candidate, dict):
            floor_range = candidate
    try:
        base_minimum = float(
            floor_range.get("min")
            if floor_range is not None and "min" in floor_range
            else app_config.get("semantic_cv_unconfigured_base_scale_min", 0.50)
        )
        base_maximum = float(
            floor_range.get("max")
            if floor_range is not None and "max" in floor_range
            else app_config.get("semantic_cv_unconfigured_base_scale_max", 0.80)
        )
    except (TypeError, ValueError):
        base_minimum, base_maximum = 0.50, 0.80
    base_minimum = max(1e-4, base_minimum)
    base_maximum = max(base_minimum, base_maximum)
    resolution_multiplier = semantic_cv_resolution_scale_multiplier(app_config, roi_size)
    return (
        base_minimum * resolution_multiplier,
        base_maximum * resolution_multiplier,
        resolution_multiplier,
    )


def semantic_cv_unconfigured_scale_step(
    app_config: dict,
    floor_suffix: Optional[str] = None,
) -> float:
    """Return the coarse step for an unconfigured layout and floor."""
    value: object = app_config.get("semantic_cv_unconfigured_scale_step", 0.01)
    configured_ranges = app_config.get("semantic_cv_unconfigured_scale_ranges")
    if isinstance(configured_ranges, dict):
        candidate = configured_ranges.get(str(floor_suffix or "").strip().lower())
        if isinstance(candidate, dict) and "step" in candidate:
            value = candidate["step"]
    try:
        return max(1e-4, float(value))
    except (TypeError, ValueError):
        return 0.01


def semantic_cv_estimated_scale_window_for_map(
    root_dir: Path,
    app_config: dict,
    map_id: str,
    roi_size: tuple[int, int],
) -> Optional[tuple[float, float, float, float]]:
    """Load one geometry-estimated map range and scale it to the current ROI."""
    raw_path = str(app_config.get("semantic_cv_estimated_scale_ranges_path", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(root_dir) / path
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        layouts = data.get("layouts", {}) if isinstance(data, dict) else {}
        item = layouts.get(str(map_id).strip().lower()) if isinstance(layouts, dict) else None
        if not isinstance(item, dict):
            return None
        base_minimum = float(item["min"])
        base_maximum = float(item["max"])
        base_step = float(item.get("step", 0.02))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not (
        np.isfinite(base_minimum)
        and np.isfinite(base_maximum)
        and np.isfinite(base_step)
        and 0.0 < base_minimum <= base_maximum
        and base_step > 0.0
    ):
        return None
    resolution_multiplier = semantic_cv_resolution_scale_multiplier(app_config, roi_size)
    return (
        base_minimum * resolution_multiplier,
        base_maximum * resolution_multiplier,
        max(1e-4, base_step * resolution_multiplier),
        resolution_multiplier,
    )


def semantic_cv_scale_cache_key(map_id: str, roi_size: tuple[int, int]) -> tuple[str, int, int]:
    return str(map_id).strip().lower(), int(roi_size[0]), int(roi_size[1])


def semantic_cv_requires_initial_scale_confirmation(
    completed_floors: set[str],
    retry_floors: set[str],
    floor_suffix: str,
) -> bool:
    return floor_suffix not in completed_floors or floor_suffix in retry_floors


def semantic_cv_confirmed_scale_path(root_dir: Path, app_config: dict) -> Path:
    raw_path = str(
        app_config.get(
            "semantic_cv_confirmed_scale_path",
            "config/confirmed_semantic_cv_scales.json",
        )
    ).strip()
    path = Path(raw_path or "config/confirmed_semantic_cv_scales.json")
    return path if path.is_absolute() else Path(root_dir) / path


def load_semantic_cv_confirmed_scales(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    layouts = data.get("layouts")
    if not isinstance(layouts, dict):
        layouts = {}
    return {"version": 1, "layouts": layouts}


def semantic_cv_confirmed_scale_for_map(
    path: Path,
    map_id: str,
    roi_size: tuple[int, int],
) -> Optional[float]:
    data = load_semantic_cv_confirmed_scales(path)
    layout = data["layouts"].get(str(map_id).strip().lower())
    if not isinstance(layout, dict):
        return None
    roi_key = f"{int(roi_size[0])}x{int(roi_size[1])}"
    record = layout.get(roi_key)
    value = record.get("scale") if isinstance(record, dict) else record
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return None
    return scale if scale > 0 and np.isfinite(scale) else None


def save_semantic_cv_confirmed_scale(
    path: Path,
    map_id: str,
    roi_size: tuple[int, int],
    scale: float,
) -> None:
    scale_value = float(scale)
    if scale_value <= 0 or not np.isfinite(scale_value):
        raise ValueError("confirmed semantic CV scale must be positive and finite")
    path = Path(path)
    data = load_semantic_cv_confirmed_scales(path)
    layouts = data["layouts"]
    normalized_map_id = str(map_id).strip().lower()
    layout = layouts.setdefault(normalized_map_id, {})
    if not isinstance(layout, dict):
        layout = {}
        layouts[normalized_map_id] = layout
    width, height = int(roi_size[0]), int(roi_size[1])
    layout[f"{width}x{height}"] = {
        "scale": scale_value,
        "roi_size": [width, height],
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def semantic_cv_weight_overrides_for_map(app_config: dict, map_id: str) -> dict[str, float]:
    """Return calibrated matcher weights for a numbered map group."""
    match = re.fullmatch(r"map_(\d+)(?:_.+)?", str(map_id).strip().lower())
    if match is None:
        return {}
    map_number = int(match.group(1))
    groups = app_config.get("semantic_cv_weight_groups", [])
    if not isinstance(groups, list):
        return {}
    allowed = ("corridor_weight", "room_weight", "edge_weight", "stair_weight")
    for group in groups:
        if not isinstance(group, dict):
            continue
        try:
            minimum = int(group.get("min_map"))
            maximum = int(group.get("max_map"))
        except (TypeError, ValueError):
            continue
        if not minimum <= map_number <= maximum:
            continue
        weights: dict[str, float] = {}
        for name in allowed:
            try:
                value = float(group.get(name))
            except (TypeError, ValueError):
                continue
            if value >= 0:
                weights[name] = value
        return weights
    return {}


def semantic_cv_geometric_review_enabled_for_map(app_config: dict, map_id: str) -> bool:
    """Enable the experimental second-stage review only for validated groups."""
    match = re.fullmatch(r"map_(\d+)(?:_.+)?", str(map_id).strip().lower())
    if match is None:
        return False
    map_number = int(match.group(1))
    groups = app_config.get("semantic_cv_geometric_review_groups", [])
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            try:
                minimum = int(group.get("min_map"))
                maximum = int(group.get("max_map"))
            except (TypeError, ValueError):
                continue
            if minimum <= map_number <= maximum:
                return bool(group.get("enabled", False))
        return False
    return bool(app_config.get("semantic_cv_geometric_review_enabled", False))


def enforce_semantic_cv_geometric_review(cfg: dict) -> None:
    """Require every published localization to use the reviewed match."""
    cfg["geometric_review_enabled"] = True
    cfg["reuse_early_structure_match_for_fixed_scale"] = False
    cfg["semantic_cv_fast_fixed_scale_mode"] = False
    cfg["early_structure_overlay_enabled"] = False


class RouteMapOverlayWindow:
    """Borderless, click-through route overlay placed directly over the game map."""

    def __init__(self, master: tk.Tk, project_config: ProjectConfig) -> None:
        self.master = master
        self.project_config = project_config
        self.alpha = float(project_config.app.get("route_overlay_alpha", 0.62))
        self._transparent_color = "#ff00ff"
        self._native_layered_enabled = bool(
            sys.platform.startswith("win")
            and project_config.app.get("route_overlay_native_layered_enabled", True)
        )
        self._create_window()

    def _create_window(self) -> None:
        self.window = tk.Toplevel(self.master)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(bg=self._transparent_color)
        self.window.attributes("-topmost", True)
        if not self._native_layered_enabled:
            self.window.attributes("-alpha", self.alpha)
            try:
                self.window.attributes("-transparentcolor", self._transparent_color)
            except tk.TclError:
                pass
        self.label = tk.Label(self.window, bg=self._transparent_color, bd=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)
        self.photo: Optional[ImageTk.PhotoImage] = None
        self._photo_size: Optional[tuple[int, int]] = None
        self._magenta_buffer: Optional[np.ndarray] = None
        self._native_bgra_buffer: Optional[np.ndarray] = None
        self._native_hwnd: Optional[int] = None
        self._native_memory_dc: Optional[int] = getattr(self, "_native_memory_dc", None)
        self._native_bitmap: Optional[int] = getattr(self, "_native_bitmap", None)
        self._native_previous_bitmap: Optional[int] = getattr(self, "_native_previous_bitmap", None)
        self._native_bits: Optional[int] = getattr(self, "_native_bits", None)
        self._native_surface_size: Optional[tuple[int, int]] = getattr(self, "_native_surface_size", None)
        self._last_screen_roi: Optional[tuple[int, int, int, int]] = None
        self._last_show_image_ms: Optional[float] = None
        self._last_show_cached_ms: Optional[float] = None
        self._click_through_enabled = False
        self._overlay_initialized = False
        self._visible = False

    def set_alpha(self, value: float) -> None:
        self.alpha = max(0.15, min(1.0, float(value)))
        self.project_config.app["route_overlay_alpha"] = self.alpha
        native_bgra = getattr(self, "_native_bgra_buffer", None)
        if bool(getattr(self, "_native_layered_enabled", False)) and native_bgra is not None and self._last_screen_roi is not None:
            self._submit_native_bgra(native_bgra, self._last_screen_roi)
        else:
            self.window.attributes("-alpha", self.alpha)

    @staticmethod
    def _native_overlay_bgra(image: np.ndarray, *, image_is_rgb: bool, threshold: int) -> np.ndarray:
        conversion = cv2.COLOR_RGB2BGRA if image_is_rgb else cv2.COLOR_BGR2BGRA
        bgra = cv2.cvtColor(image, conversion)
        transparent = cv2.inRange(image, (0, 0, 0), (threshold, threshold, threshold))
        bgra[:, :, 3] = cv2.bitwise_not(transparent)
        return np.ascontiguousarray(bgra)

    def _native_window_handle(self) -> int:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.IsWindow.argtypes = [wintypes.HWND]
        existing = int(getattr(self, "_native_hwnd", 0) or 0)
        if existing and user32.IsWindow(wintypes.HWND(existing)):
            return existing
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            ctypes.c_void_p,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        ex_style = 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000
        hwnd = user32.CreateWindowExW(
            ex_style,
            "Static",
            "IdentityVRouteOverlay",
            0x80000000,
            0,
            0,
            1,
            1,
            wintypes.HWND(0),
            wintypes.HMENU(0),
            wintypes.HINSTANCE(0),
            None,
        )
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._native_hwnd = int(hwnd)
        return self._native_hwnd

    def _submit_native_bgra(
        self,
        bgra: np.ndarray,
        screen_roi: tuple[int, int, int, int],
        *,
        constant_alpha: Optional[float] = None,
    ) -> None:
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        class BLENDFUNCTION(ctypes.Structure):
            _fields_ = [
                ("BlendOp", wintypes.BYTE),
                ("BlendFlags", wintypes.BYTE),
                ("SourceConstantAlpha", wintypes.BYTE),
                ("AlphaFormat", wintypes.BYTE),
            ]

        x, y, width, height = screen_roi
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]
        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = ctypes.c_void_p
        user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
        gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
        gdi32.CreateDIBSection.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(BITMAPINFO),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        gdi32.CreateDIBSection.restype = ctypes.c_void_p
        gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi32.SelectObject.restype = ctypes.c_void_p
        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
        user32.UpdateLayeredWindow.argtypes = [
            wintypes.HWND,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.POINT),
            ctypes.POINTER(wintypes.SIZE),
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.POINT),
            wintypes.COLORREF,
            ctypes.POINTER(BLENDFUNCTION),
            wintypes.DWORD,
        ]
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
        hwnd = self._native_window_handle()
        diagnostic_event(
            "overlay_native_submit_begin",
            hwnd=hwnd,
            screen_roi=screen_roi,
            buffer_shape=list(bgra.shape),
            buffer_bytes=int(bgra.nbytes),
            surface_size=self._native_surface_size,
        )
        gwl_exstyle = -20
        ws_ex_layered = 0x00080000
        ws_ex_transparent = 0x00000020
        ws_ex_toolwindow = 0x00000080
        ws_ex_noactivate = 0x08000000
        style = int(user32.GetWindowLongW(hwnd, gwl_exstyle))
        user32.SetWindowLongW(
            hwnd,
            gwl_exstyle,
            style | ws_ex_layered | ws_ex_transparent | ws_ex_toolwindow | ws_ex_noactivate,
        )

        screen_dc = user32.GetDC(wintypes.HWND(0))
        try:
            if self._native_surface_size != (width, height):
                self._destroy_native_surface()
                memory_dc = gdi32.CreateCompatibleDC(screen_dc)
                if not memory_dc:
                    raise ctypes.WinError(ctypes.get_last_error())
                info = BITMAPINFO()
                info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                info.bmiHeader.biWidth = width
                info.bmiHeader.biHeight = -height
                info.bmiHeader.biPlanes = 1
                info.bmiHeader.biBitCount = 32
                bits = ctypes.c_void_p()
                bitmap = gdi32.CreateDIBSection(
                    memory_dc,
                    ctypes.byref(info),
                    0,
                    ctypes.byref(bits),
                    None,
                    0,
                )
                if not bitmap or not bits.value:
                    if bitmap:
                        gdi32.DeleteObject(bitmap)
                    gdi32.DeleteDC(memory_dc)
                    raise ctypes.WinError(ctypes.get_last_error())
                previous = gdi32.SelectObject(memory_dc, bitmap)
                self._native_memory_dc = int(memory_dc)
                self._native_bitmap = int(bitmap)
                self._native_previous_bitmap = int(previous) if previous else None
                self._native_bits = int(bits.value)
                self._native_surface_size = (width, height)
            memory_dc = int(self._native_memory_dc or 0)
            bits_address = int(self._native_bits or 0)
            if not memory_dc or not bits_address:
                raise RuntimeError("native overlay surface is unavailable")
            ctypes.memmove(bits_address, bgra.ctypes.data, bgra.nbytes)
            destination = wintypes.POINT(int(x), int(y))
            size = wintypes.SIZE(int(width), int(height))
            source = wintypes.POINT(0, 0)
            opacity = self.alpha if constant_alpha is None else constant_alpha
            blend = BLENDFUNCTION(0, 0, int(round(opacity * 255.0)), 1)
            user32.SetWindowPos(
                wintypes.HWND(hwnd),
                wintypes.HWND(-1),
                x,
                y,
                width,
                height,
                0x0010 | 0x0040,
            )
            user32.ShowWindow(wintypes.HWND(hwnd), 4)
            if not user32.UpdateLayeredWindow(
                hwnd,
                screen_dc,
                ctypes.byref(destination),
                ctypes.byref(size),
                memory_dc,
                ctypes.byref(source),
                0,
                ctypes.byref(blend),
                2,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            diagnostic_event(
                "overlay_native_submit_success",
                hwnd=hwnd,
                screen_roi=screen_roi,
                surface_size=self._native_surface_size,
            )
        finally:
            if screen_dc:
                user32.ReleaseDC(wintypes.HWND(0), screen_dc)

    def show_image(
        self,
        image: np.ndarray,
        screen_roi: tuple[int, int, int, int],
        *,
        image_is_rgb: bool = False,
    ) -> None:
        started = time.perf_counter()
        x, y, width, height = screen_roi
        if width <= 0 or height <= 0:
            self._last_show_image_ms = (time.perf_counter() - started) * 1000.0
            return
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        threshold = max(0, min(64, int(self.project_config.app.get("route_overlay_black_transparent_threshold", 16))))
        target_roi = (int(x), int(y), int(width), int(height))
        if bool(getattr(self, "_native_layered_enabled", False)):
            try:
                bgra = self._native_overlay_bgra(image, image_is_rgb=image_is_rgb, threshold=threshold)
                diagnostic_event(
                    "overlay_buffer_ready",
                    screen_roi=target_roi,
                    nontransparent_pixels=int(cv2.countNonZero(bgra[:, :, 3])),
                    total_pixels=int(width * height),
                    alpha=float(self.alpha),
                )
                self._submit_native_bgra(bgra, target_roi)
                self._native_bgra_buffer = bgra
                self._last_screen_roi = target_roi
                self._click_through_enabled = True
                self._overlay_initialized = True
                self._visible = True
                self._last_show_image_ms = (time.perf_counter() - started) * 1000.0
                return
            except Exception:
                diagnostic_event("overlay_native_submit_failed", screen_roi=target_roi)
                LOGGER.warning("native layered route overlay failed; falling back to Tk", exc_info=True)
                self._destroy_native_window()
                self._native_layered_enabled = False
                self.window.attributes("-alpha", self.alpha)
                try:
                    self.window.attributes("-transparentcolor", self._transparent_color)
                except tk.TclError:
                    pass
        image = image.copy()
        black_background = cv2.inRange(image, (0, 0, 0), (threshold, threshold, threshold))
        if self._magenta_buffer is None or self._magenta_buffer.shape != image.shape:
            self._magenta_buffer = np.empty_like(image)
            self._magenta_buffer[:] = (255, 0, 255)
        cv2.copyTo(self._magenta_buffer, black_background, image)
        pil = Image.fromarray(image if image_is_rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        was_visible = bool(getattr(self, "_visible", False))
        if was_visible and self.photo is not None and self._photo_size == (width, height):
            self.photo.paste(pil)
        else:
            # A hidden Tk window can briefly paint the previous pixels when a
            # reused PhotoImage is pasted and immediately deiconified. Bind a
            # fresh native image while hidden, then flush it before showing.
            self.photo = ImageTk.PhotoImage(pil)
            self._photo_size = (width, height)
            self.label.configure(image=self.photo)
        if target_roi != self._last_screen_roi:
            self.window.geometry(f"{width}x{height}+{x}+{y}")
        self._last_screen_roi = target_roi
        if not was_visible:
            self.window.update_idletasks()
            self.window.deiconify()
            self.window.lift()
            self.window.attributes("-topmost", True)
            self._visible = True
        # Window styles are process-local and do not change when the image is
        # replaced.  Applying them on every frame forces a costly Tk/Win32
        # round-trip and was visible as a late first aligned frame.
        if not self._overlay_initialized:
            self.window.update_idletasks()
            self._overlay_initialized = True
        if not self._click_through_enabled:
            self._enable_click_through()
            self._click_through_enabled = True
        self._last_show_image_ms = (time.perf_counter() - started) * 1000.0

    def show_cached(self, screen_roi: Optional[tuple[int, int, int, int]] = None) -> bool:
        """Restore the last rendered overlay without repeating image conversion."""
        started = time.perf_counter()
        if bool(getattr(self, "_native_layered_enabled", False)):
            native_bgra = getattr(self, "_native_bgra_buffer", None)
            if native_bgra is None or self._last_screen_roi is None:
                self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
                return False
            target = screen_roi or self._last_screen_roi
            if target[2:] != self._last_screen_roi[2:]:
                self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
                return False
            self._submit_native_bgra(native_bgra, target)
            self._last_screen_roi = tuple(int(value) for value in target)
            self._visible = True
            self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
            return True
        if self.photo is None or self._last_screen_roi is None:
            self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
            return False
        target = screen_roi or self._last_screen_roi
        _x, _y, width, height = target
        if width <= 0 or height <= 0:
            self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
            return False
        if (width, height) != self._last_screen_roi[2:]:
            self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
            return False
        was_visible = bool(getattr(self, "_visible", False))
        if not was_visible:
            self.label.configure(image=self.photo)
        target_roi = (int(target[0]), int(target[1]), int(width), int(height))
        if target_roi != self._last_screen_roi:
            self.window.geometry(f"{width}x{height}+{target_roi[0]}+{target_roi[1]}")
            self._last_screen_roi = target_roi
        if not was_visible:
            self.window.deiconify()
            self.window.lift()
            self.window.attributes("-topmost", True)
            self._visible = True
        self._last_show_cached_ms = (time.perf_counter() - started) * 1000.0
        return True

    def hide(self) -> None:
        diagnostic_event(
            "overlay_hide",
            visible=bool(getattr(self, "_visible", False)),
            screen_roi=self._last_screen_roi,
            hwnd=int(getattr(self, "_native_hwnd", 0) or 0),
        )
        if bool(getattr(self, "_native_layered_enabled", False)) and sys.platform.startswith("win"):
            try:
                ctypes.windll.user32.ShowWindow(self._native_window_handle(), 0)  # type: ignore[attr-defined]
            except Exception:
                self.window.withdraw()
        else:
            self.window.withdraw()
        self._visible = False
        self.window.update_idletasks()
        if sys.platform == "win32":
            try:
                ctypes.windll.dwmapi.DwmFlush()
            except (AttributeError, OSError):
                pass

    def reset(self) -> None:
        """Destroy the visible HWND while retaining the off-screen scratch surface."""
        diagnostic_event(
            "overlay_reset",
            visible=bool(getattr(self, "_visible", False)),
            screen_roi=self._last_screen_roi,
            hwnd=int(getattr(self, "_native_hwnd", 0) or 0),
            retained_surface_size=self._native_surface_size,
        )
        self._destroy_native_window(release_surface=False)
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        self._create_window()

    def warmup(self) -> None:
        """Create native window styles while the overlay remains hidden."""
        if bool(getattr(self, "_native_layered_enabled", False)):
            self._native_window_handle()
            self._overlay_initialized = True
            self._click_through_enabled = True
            return
        if not self._overlay_initialized:
            self.window.update_idletasks()
            self._overlay_initialized = True
        if not self._click_through_enabled:
            self._enable_click_through()
            self._click_through_enabled = True

    def destroy(self) -> None:
        self._destroy_native_window()
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def _destroy_native_window(self, *, release_surface: bool = True) -> None:
        if release_surface:
            self._destroy_native_surface()
        hwnd = int(getattr(self, "_native_hwnd", 0) or 0)
        self._native_hwnd = None
        if hwnd and sys.platform.startswith("win"):
            try:
                ctypes.windll.user32.DestroyWindow(wintypes.HWND(hwnd))  # type: ignore[attr-defined]
            except Exception:
                LOGGER.debug("failed to destroy native route overlay", exc_info=True)

    def _destroy_native_surface(self) -> None:
        memory_dc = int(getattr(self, "_native_memory_dc", 0) or 0)
        bitmap = int(getattr(self, "_native_bitmap", 0) or 0)
        previous = int(getattr(self, "_native_previous_bitmap", 0) or 0)
        self._native_memory_dc = None
        self._native_bitmap = None
        self._native_previous_bitmap = None
        self._native_bits = None
        self._native_surface_size = None
        if not sys.platform.startswith("win"):
            return
        try:
            gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]
            if memory_dc and previous:
                gdi32.SelectObject(ctypes.c_void_p(memory_dc), ctypes.c_void_p(previous))
            if bitmap:
                gdi32.DeleteObject(ctypes.c_void_p(bitmap))
            if memory_dc:
                gdi32.DeleteDC(ctypes.c_void_p(memory_dc))
        except Exception:
            LOGGER.debug("failed to destroy native route overlay surface", exc_info=True)

    def _enable_click_through(self) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            gwl_exstyle = -20
            ws_ex_transparent = 0x00000020
            ga_root = 2
            swp_framechanged = 0x0020
            swp_no_move = 0x0002
            swp_no_size = 0x0001
            swp_no_activate = 0x0010
            hwnd = int(self.window.winfo_id())
            root_hwnd = int(user32.GetAncestor(hwnd, ga_root) or hwnd)
            for target_hwnd in {hwnd, root_hwnd}:
                style = user32.GetWindowLongW(target_hwnd, gwl_exstyle)
                user32.SetWindowLongW(target_hwnd, gwl_exstyle, style | ws_ex_transparent)
                user32.SetWindowPos(
                    target_hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    swp_framechanged | swp_no_move | swp_no_size | swp_no_activate,
                )
        except Exception:
            LOGGER.debug("failed to enable route overlay click-through", exc_info=True)


class GameMapIdentifierGUI(tk.Tk):
    def __init__(
        self,
        project_config: ProjectConfig,
        preprocess_config: PreprocessConfig,
        map_db: MapDatabase,
        sample_db: ScreenshotSampleDatabase,
    ) -> None:
        super().__init__()
        self.project_config = project_config
        self.recognition_profile = get_recognition_profile(project_config.app.get("recognition_profile"))
        self.preprocess_config = preprocess_config
        self.map_db = map_db
        self.sample_db = sample_db
        variant_name = str(project_config.app.get("app_variant_name", "")).strip()
        if variant_name:
            self.title(f"游戏地图识别与对齐 - {variant_name}")
        else:
            self.title("游戏地图识别与对齐")
        self.geometry("1100x778")
        self.minsize(900, 636)
        self._main_background_source: Optional[Image.Image] = None
        self._main_background_photo: Optional[ImageTk.PhotoImage] = None
        self._main_background_item: Optional[int] = None
        self._main_background_resize_job: Optional[str] = None
        self._configure_main_window_styles()
        self._setup_main_window_background()

        self.last_capture: Optional[CaptureInfo] = None
        self.last_current_roi: Optional[np.ndarray] = None
        self._recognition_preview_roi: Optional[np.ndarray] = None
        self._recognition_preview_locked = False
        self.last_alignment_capture: Optional[CaptureInfo] = None
        self.last_alignment_roi: Optional[np.ndarray] = None
        self._alignment_preview_roi: Optional[np.ndarray] = None
        self._last_alignment_map_screen_roi: Optional[tuple[int, int, int, int]] = None
        self.last_summary: Optional[MatchSummary] = None
        self.cnn_load_error = ""
        self.cnn_classifier = self._load_cnn_classifier()
        self.last_cnn_scores: dict[str, float] = {}
        self._hotkeys = self._load_hotkeys()
        self._hotkey_configuring = False
        self._hotkey_capture_action: Optional[str] = None
        self._hotkey_settings_window: Optional[tk.Toplevel] = None
        self._game_map_toggle_settings_window: Optional[tk.Toplevel] = None
        self._copyright_statement_window: Optional[tk.Toplevel] = None
        self._scale_manager_window: Optional[tk.Toplevel] = None
        self._route_map_version_window: Optional[tk.Toplevel] = None
        self._settings_window: Optional[tk.Toplevel] = None
        self._game_map_binding_capturing = False
        self._hotkey_action_times: dict[str, float] = {}
        self._hotkey_down = False
        self._structure_confirm_key_down = False
        self._location_key_down = False
        self._second_floor_location_pending = False
        self._second_floor_transition_constraint_active = False
        self._floor_transition_constraint_source: Optional[str] = None
        self._first_floor_side_door_constraint_active = False
        self._last_second_floor_request_time = 0.0
        self._last_mouse_left_down = False
        self._route_overlay_dismiss_key_down = False
        self._route_overlay_escape_key_down = False
        self._route_overlay_dismiss_armed = False
        self._overlay_scale_research_context: Optional[dict[str, str]] = None
        self._manual_scale_research_return_content: Optional[tuple[str, tuple]] = None
        self._scale_research_return_context: Optional[dict[str, Any]] = None
        self._scale_search_active = False
        self._scale_search_cancel_requested = False
        self._scale_search_stop_key_down = False
        self._active_scale_search_return_content: Optional[tuple[str, tuple]] = None
        self._active_scale_search_map_id: Optional[str] = None
        self._active_scale_search_floor: Optional[str] = None
        self._program_paused = False
        self._route_overlay_request_id = 0
        self._active_route_overlay_request_id: Optional[int] = None
        self._cancelled_route_overlay_request_id = 0
        self._route_overlay_visible_request_id: Optional[int] = None
        self._route_overlay_close_detected_request_id: Optional[int] = None
        # The native close monitor must never call Tk. It only signals the
        # main loop, which owns hiding the overlay and all Tk state changes.
        self._route_overlay_close_requested = threading.Event()
        self._route_overlay_cancel_reason = "unspecified"
        self._route_overlay_close_monitor: Optional[threading.Thread] = None
        self._anchor_key_down = False
        self._awaiting_click_started_at = 0.0
        self._last_hotkey_time = 0.0
        self._recognition_running = False
        self._recognition_loop_active = False
        self._recognition_confirm_pending = False
        self._recognition_loop_from_hotkey = False
        self._recognition_frame_index = 0
        self._confirm_key_down = False
        self._side_door_structure_confirmed = False
        self._first_location_map_zoom_pending = False
        self._awaiting_unknown_anchor = False
        self._pending_alignment_map_id: Optional[str] = None
        self._manual_map_select_active = False
        self._confirmed_alignment_map_id: Optional[str] = None
        self._active_floor_suffix = self.recognition_profile.entry_floor
        self._last_unknown_anchor: Optional[tuple[float, float]] = None
        self.last_alignment_result = None
        self.last_structure_result: Optional[StructureRegistrationResult] = None
        self._semantic_cv_pf = None
        self._semantic_cv_pf_key: Optional[str] = None
        self._semantic_cv_pf_updated_at: Optional[float] = None
        self._semantic_cv_pf_last_motion_at: Optional[float] = None
        self._semantic_cv_pf_motion_hint: tuple[float, float] = (0.0, 0.0)
        self._semantic_cv_pf_motion_accum: tuple[float, float] = (0.0, 0.0)
        self._semantic_cv_pf_motion_radius_accum = 0.0
        self._semantic_cv_pf_motion_radius_max = 0.0
        self._semantic_cv_pf_motion_seconds = 0.0
        self._semantic_cv_pf_estimate_floor = self.recognition_profile.entry_floor
        self._semantic_cv_player_pfs: dict[str, Any] = {}
        self._semantic_cv_player_pf_key: Optional[str] = None
        self._semantic_cv_player_pf_updated_at: dict[str, float] = {}
        self._last_player_marker_candidates: list[tuple[float, float, float, tuple[int, int, int, int]]] = []
        self._last_player_marker_mask: Optional[np.ndarray] = None
        self._layout_switch_key_down = False
        self._variant_return_key_down = False
        self._side_door_layout_variants = self._load_side_door_layout_variants()
        self._active_route_layouts: tuple[str, ...] = ()
        self._active_route_layout_index = 0
        self._ambiguous_map_groups: dict[str, tuple[str, ...]] = {
            "map_01_18": ("map_01", "map_18"),
            "map_04_16_20_21": ("map_04", "map_16", "map_20", "map_21"),
        }
        self._ambiguous_map_ids = tuple(
            map_id for group in self._ambiguous_map_groups.values() for map_id in group
        )
        self._active_ambiguous_map_ids = self._ambiguous_map_groups["map_04_16_20_21"]
        self._ambiguous_choice_photos: dict[str, ImageTk.PhotoImage] = {}
        self._ambiguous_choice_buttons: dict[str, ttk.Button] = {}
        self._ambiguous_choice_visible = False
        self._ambiguous_cheatsheet_click_regions: list[tuple[str, tuple[int, int, int, int]]] = []
        self._manual_choice_window: Optional[tk.Toplevel] = None
        self._manual_choice_window_photos: list[ImageTk.PhotoImage] = []
        self._debug_cleanup_running = False
        self._yolo_server_prewarm_started = False
        self._semantic_cv_worker: Optional[subprocess.Popen] = None
        self._semantic_cv_worker_stderr_file = None
        self._semantic_cv_worker_warmup_pending = False
        self._semantic_cv_image_cache: dict[str, np.ndarray] = {}
        self._semantic_cv_route_crop_cache: dict[tuple[str, int], tuple[np.ndarray, tuple[int, int]]] = {}
        self._semantic_cv_affine_cache: dict[tuple[str, int, int, int, int], np.ndarray] = {}
        self._semantic_cv_piecewise_route_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self._semantic_cv_scale_cache: dict[tuple[str, int, int], float] = {}
        self._semantic_cv_device_multiplier_cache: dict[tuple[str, int, int], float] = {}
        self._semantic_cv_scale_confirmation_pending: Optional[dict[str, Any]] = None
        self._semantic_cv_scale_retry_floors: set[str] = set()
        self._semantic_cv_initial_location_floors: set[str] = set()
        self._semantic_cv_location_attempted_floors: set[str] = set()
        self._semantic_cv_postprocess_id = 0
        self._semantic_cv_last_overlay_map_id: Optional[str] = None
        self._semantic_cv_last_overlay_screen_roi: Optional[tuple[int, int, int, int]] = None

        self.result_var = tk.StringVar(value="尚未识别")
        cnn_text = "CNN启用" if self.cnn_classifier is not None else "CNN关闭"
        self.status_var = tk.StringVar(
            value=f"已加载 {len(map_db.templates)} 张地图模板 / {len(sample_db.samples)} 个截图样本 / "
            f"{sample_db.pending_count()} 个待标注样本 / {cnn_text}"
        )
        first_map = next(iter(project_config.maps.keys()), "")
        self.sample_map_var = tk.StringVar(value=first_map)
        self.cheatsheet_alpha_var = tk.DoubleVar(value=float(project_config.app.get("cheatsheet_alpha", 0.82)))
        self.route_overlay_alpha_var = tk.DoubleVar(value=float(project_config.app.get("route_overlay_alpha", 0.62)))
        self.project_config.app["route_overlay_content"] = "route"
        game_map_binding = self.project_config.app.get("game_map_toggle_button", "right")
        self.game_map_toggle_var = tk.StringVar(
            value=f"游戏地图键：{game_map_toggle_display_name(game_map_binding)}"
        )
        self.cheatsheet_base_width = 760
        self.cheatsheet_base_height = 480
        current_width = float(project_config.app.get("cheatsheet_width", self.cheatsheet_base_width))
        self.cheatsheet_size_var = tk.DoubleVar(value=max(0.5, min(1.6, current_width / self.cheatsheet_base_width)))
        self.cheatsheet = CheatSheetWindow(self, project_config)
        self.route_overlay = RouteMapOverlayWindow(self, project_config)

        self._build_widgets()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._shutdown_application)
        self.after(self._global_hotkey_poll_interval_ms(), self._poll_global_hotkey)
        self.after(1000, self._prewarm_yolo_player_server)
        self.after(1200, self._prewarm_semantic_cv_worker)

    def _load_cnn_classifier(self):
        return self._load_cnn_classifier_for_profile(self.recognition_profile)

    def _load_cnn_classifier_for_profile(self, profile: RecognitionProfile):
        if not bool(self.project_config.app.get("enable_cnn_match", True)):
            self.cnn_load_error = "配置 enable_cnn_match=false"
            return None
        try:
            service = CNNRecognitionService(self.project_config.root_dir, profile)
            return service.classifier
        except Exception as exc:
            self.cnn_load_error = str(exc)
            LOGGER.warning("CNN classifier for profile %s disabled: %s", profile.key, exc)
            return None

    def _reset_recognition_session_for_profile(self, profile: RecognitionProfile) -> None:
        self._recognition_running = False
        self._recognition_loop_active = False
        self._recognition_confirm_pending = False
        self._recognition_loop_from_hotkey = False
        self._recognition_frame_index = 0
        self._manual_map_select_active = False
        self._side_door_structure_confirmed = False
        self._first_location_map_zoom_pending = False
        self._pending_alignment_map_id = None
        self._confirmed_alignment_map_id = None
        self._active_floor_suffix = profile.entry_floor
        self._active_route_layouts = ()
        self._active_route_layout_index = 0
        self.last_capture = None
        self.last_current_roi = None
        self.last_summary = None
        self.last_cnn_scores = {}
        self.last_alignment_capture = None
        self.last_alignment_roi = None
        self.last_alignment_result = None
        self.last_structure_result = None
        self._recognition_preview_roi = None
        self._recognition_preview_locked = False
        self._alignment_preview_roi = None
        self._last_unknown_anchor = None
        self._semantic_cv_pf = None
        self._semantic_cv_pf_key = None
        self._semantic_cv_player_pfs.clear()
        self._semantic_cv_player_pf_key = None
        self._semantic_cv_player_pf_updated_at.clear()
        self._semantic_cv_initial_location_floors.clear()
        self._semantic_cv_location_attempted_floors.clear()
        self._semantic_cv_scale_retry_floors.clear()
        self._semantic_cv_scale_confirmation_pending = None
        self._semantic_cv_postprocess_id += 1
        self._route_overlay_cancel_reason = "recognition_profile_changed"
        self._cancel_route_overlay_generation()
        self._set_ambiguous_choice_visible(False)
        self.cheatsheet.clear_click_handler()
        self.cheatsheet.hide()

    def _switch_recognition_profile(self, profile_key: str) -> None:
        target = get_recognition_profile(profile_key)
        if target.key == self.recognition_profile.key:
            return
        if self._recognition_running:
            self.status_var.set("地图识别正在运行，请等待本次识别结束后再切换难度。")
            return

        previous_error = self.cnn_load_error
        self.cnn_load_error = ""
        classifier = self._load_cnn_classifier_for_profile(target)
        if classifier is None:
            error = self.cnn_load_error or f"无法加载{target.display_name}模型"
            self.cnn_load_error = previous_error
            self.status_var.set(error)
            messagebox.showerror("切换难度失败", error)
            return

        previous_profile = self.recognition_profile
        previous_classifier = self.cnn_classifier
        previous_config_value = self.project_config.app.get("recognition_profile")
        self.recognition_profile = target
        self.cnn_classifier = classifier
        self.project_config.app["recognition_profile"] = target.key
        try:
            save_app_config(self.project_config.app_config_path, self.project_config.app)
        except OSError as exc:
            self.recognition_profile = previous_profile
            self.cnn_classifier = previous_classifier
            if previous_config_value is None:
                self.project_config.app.pop("recognition_profile", None)
            else:
                self.project_config.app["recognition_profile"] = previous_config_value
            self.cnn_load_error = previous_error
            self.status_var.set(f"无法保存难度设置：{exc}")
            messagebox.showerror("切换难度失败", str(exc))
            return

        self.cnn_load_error = ""
        self._reset_recognition_session_for_profile(target)
        self.result_var.set(f"已切换至{target.display_name}")
        self.status_var.set(
            f"当前难度：{target.display_name}；识别入口："
            f"{'二楼侧门' if target.entry_floor == '2nd' else '一楼侧门'}。"
        )
        self._refresh_main_window_controls()

    def _configure_main_window_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        panel = "#1b1110"
        panel_hover = "#34201a"
        border = "#a97851"
        foreground = "#fbf1df"
        muted = "#d4bfa9"
        style.configure("Main.TFrame", background=panel)
        style.configure("Main.Panel.TFrame", background=panel)
        style.configure("Main.TLabel", background=panel, foreground=foreground)
        style.configure("Main.Panel.TLabel", background=panel, foreground=foreground)
        style.configure("Main.Preview.TLabel", background="#160e0d", foreground=muted, bordercolor=border)
        style.configure("Main.Result.TLabel", background=panel, foreground=foreground, padding=(12, 8))
        style.configure("Main.Status.TLabel", background="#160e0d", foreground=muted, padding=(10, 6))
        style.configure("Main.TButton", background="#382017", foreground=foreground, padding=(11, 6))
        style.map(
            "Main.TButton",
            background=[("active", "#5a3022"), ("pressed", "#24120f")],
            foreground=[("disabled", "#9f8b7b")],
        )
        style.configure("Main.TLabelframe", background=panel, bordercolor=border, relief="solid")
        style.configure("Main.TLabelframe.Label", background=panel, foreground=foreground)
        style.configure("Main.Horizontal.TScale", background=panel, troughcolor="#1c2b3f")

    def _setup_main_window_background(self) -> None:
        self._main_canvas = tk.Canvas(self, bg="#160d0c", bd=0, highlightthickness=0)
        self._main_canvas.pack(fill="both", expand=True)
        background_path = self.project_config.root_dir / "assets" / "ui" / "main_window_background.png"
        try:
            with Image.open(background_path) as image:
                self._main_background_source = image.convert("RGB").copy()
        except (OSError, ValueError):
            LOGGER.warning("main window background unavailable: %s", background_path)
        self.bind("<Configure>", self._schedule_main_background_resize, add="+")
        self.after_idle(self._render_main_window_background)

    def _schedule_main_background_resize(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if self._main_background_resize_job is not None:
            try:
                self.after_cancel(self._main_background_resize_job)
            except tk.TclError:
                pass
        self._main_background_resize_job = self.after(80, self._render_main_window_background)

    def _render_main_window_background(self) -> None:
        self._main_background_resize_job = None
        if self._main_background_source is not None:
            width = max(1, int(self._main_canvas.winfo_width()))
            height = max(1, int(self._main_canvas.winfo_height()))
            source_w, source_h = self._main_background_source.size
            scale = max(width / max(source_w, 1), height / max(source_h, 1))
            resized = self._main_background_source.resize(
                (max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))),
                Image.Resampling.LANCZOS,
            )
            left = max(0, (resized.width - width) // 2)
            top = max(0, (resized.height - height) // 2)
            cropped = resized.crop((left, top, left + width, top + height))
            muted = Image.blend(cropped, Image.new("RGB", cropped.size, (22, 10, 10)), 0.36)
            self._main_background_photo = ImageTk.PhotoImage(muted)
            if self._main_background_item is None:
                self._main_background_item = self._main_canvas.create_image(0, 0, anchor="nw", image=self._main_background_photo)
            else:
                self._main_canvas.itemconfigure(self._main_background_item, image=self._main_background_photo)
                self._main_canvas.coords(self._main_background_item, 0, 0)
            self._main_canvas.tag_lower(self._main_background_item)
        if getattr(self, "_main_controls_initialized", False):
            self._render_main_window_controls()

    def _build_widgets(self) -> None:
        self._main_controls_initialized = False
        self._main_button_rectangles: dict[str, int] = {}
        self._main_control_photos: list[ImageTk.PhotoImage] = []
        self._main_glass_photo_cache: dict[tuple[object, ...], ImageTk.PhotoImage] = {}
        self._main_controls_render_pending = False
        self._main_controls_render_after_id: Optional[str] = None
        self._main_result_text_item: Optional[int] = None
        self._main_result_text_width = 0
        self._main_status_text_item: Optional[int] = None
        self._main_status_text_width = 0
        self._main_button_icon_sources: dict[str, Image.Image] = {}
        button_icon_dir = self.project_config.root_dir / "assets" / "ui" / "button_icons"
        for icon_tag, icon_name in (
            ("main-settings", "scale.png"),
            ("main-copyright", "copyright.png"),
        ):
            icon_path = button_icon_dir / icon_name
            try:
                with Image.open(icon_path) as image:
                    self._main_button_icon_sources[icon_tag] = image.convert("RGBA").copy()
            except (OSError, ValueError):
                LOGGER.warning("main button icon unavailable: %s", icon_path)
        self._main_slider_knob_source: Optional[Image.Image] = None
        slider_knob_path = self.project_config.root_dir / "assets" / "ui" / "hydra_slider_knob.png"
        try:
            with Image.open(slider_knob_path) as image:
                self._main_slider_knob_source = image.convert("RGBA").copy()
        except (OSError, ValueError):
            LOGGER.warning("slider knob image unavailable: %s", slider_knob_path)
        self._main_slider_geometry: dict[int, tuple[int, int]] = {}
        self._main_slider_items: dict[int, tuple[int, int, int]] = {}
        self._main_slider_specs = (
            ("小抄透明度", 0.35, 1.0, self.cheatsheet_alpha_var, self._on_cheatsheet_alpha_change, "percent"),
            ("小抄大小", 0.5, 1.6, self.cheatsheet_size_var, self._on_cheatsheet_size_change, "scale"),
            ("路线图虚像透明度", 0.15, 1.0, self.route_overlay_alpha_var, self._on_route_overlay_alpha_change, "percent"),
        )

        # The raw capture is deliberately not shown. These two data sinks render only
        # after F8/G provides a meaningful ROI, so startup keeps the artwork clear.
        self.raw_panel = HiddenPreviewSink()
        self.roi_panel = HiddenPreviewSink()
        self._main_controls_initialized = True
        self.result_var.trace_add("write", self._refresh_main_result_text)
        self.status_var.trace_add("write", self._refresh_main_status_text)
        self.game_map_toggle_var.trace_add("write", self._refresh_main_window_controls)
        self._refresh_main_window_controls()

    def _set_main_recognition_preview(self, roi: np.ndarray) -> None:
        if self._recognition_preview_locked:
            return
        self._recognition_preview_roi = roi.copy()
        self._refresh_main_window_controls()

    def _lock_main_recognition_preview(self) -> None:
        if self._recognition_preview_roi is not None:
            self._recognition_preview_locked = True
            self._refresh_main_window_controls()

    def _start_main_recognition_preview_session(self) -> None:
        self._recognition_preview_locked = False
        self._recognition_preview_roi = None
        self._refresh_main_window_controls()

    def _set_main_alignment_preview(self, roi: np.ndarray) -> None:
        self._alignment_preview_roi = roi.copy()
        self._refresh_main_window_controls()

    def _draw_main_roi_preview(
        self,
        bounds: tuple[int, int, int, int],
        *,
        title: str,
        subtitle: str,
        image: np.ndarray,
        accent: str,
    ) -> None:
        x1, y1, x2, y2 = bounds
        self._draw_main_glass_surface(
            bounds,
            fill=(31, 12, 10, 52),
            outline=(236, 187, 122, 122),
            radius=7,
        )
        self._main_canvas.create_line(x1 + 16, y1 + 15, x1 + 16, y1 + 36, fill=accent, width=2, tags="main-control")
        self._main_canvas.create_text(
            x1 + 28, y1 + 20, anchor="w", text=title, fill="#fff0d8",
            font=("Microsoft YaHei UI", 10, "bold"), tags="main-control",
        )
        self._main_canvas.create_text(
            x1 + 28, y1 + 39, anchor="w", text=subtitle, fill="#ddc0a0",
            font=("Microsoft YaHei UI", 8), tags="main-control",
        )
        source_h, source_w = image.shape[:2]
        available_w = max(24, x2 - x1 - 24)
        available_h = max(24, y2 - y1 - 62)
        ratio = min(available_w / max(source_w, 1), available_h / max(source_h, 1))
        target_w = max(1, int(round(source_w * ratio)))
        target_h = max(1, int(round(source_h * ratio)))
        interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
        preview = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        if preview.ndim == 2:
            pil = Image.fromarray(preview).convert("RGB")
        else:
            pil = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(pil, master=self._main_canvas)
        self._main_control_photos.append(photo)
        px = x1 + (x2 - x1 - target_w) // 2
        py = y1 + 55 + (available_h - target_h) // 2
        self._main_canvas.create_image(px, py, anchor="nw", image=photo, tags="main-control")

    def _refresh_main_window_controls(self, *_args: object) -> None:
        if not getattr(self, "_main_controls_initialized", False):
            return
        if getattr(self, "_main_controls_render_pending", False):
            return
        self._main_controls_render_pending = True
        # The main panel is decorative. Batch rapid location/status updates so
        # they cannot compete with the aligned-overlay path or flood Tcl with
        # transient PhotoImages.
        self._main_controls_render_after_id = self.after(120, self._run_scheduled_main_controls_render)

    def _run_scheduled_main_controls_render(self) -> None:
        self._main_controls_render_pending = False
        self._main_controls_render_after_id = None
        self._render_main_window_controls()

    def _refresh_main_result_text(self, *_args: object) -> None:
        item = getattr(self, "_main_result_text_item", None)
        root = getattr(self, "_main_canvas", None)
        if item is None or root is None:
            self._refresh_main_window_controls()
            return
        try:
            root.itemconfigure(
                item,
                text=self._fit_main_canvas_text(
                    self.result_var.get(),
                    max_width=int(getattr(self, "_main_result_text_width", 0)),
                    font=("Microsoft YaHei UI", 12, "normal"),
                ),
            )
        except tk.TclError:
            self._refresh_main_window_controls()

    def _refresh_main_status_text(self, *_args: object) -> None:
        item = getattr(self, "_main_status_text_item", None)
        root = getattr(self, "_main_canvas", None)
        if item is None or root is None:
            self._refresh_main_window_controls()
            return
        try:
            root.itemconfigure(item, text=self.status_var.get())
        except tk.TclError:
            self._refresh_main_window_controls()

    def _make_main_glass_photo(
        self,
        width: int,
        height: int,
        *,
        fill: tuple[int, int, int, int] = (28, 12, 10, 94),
        outline: tuple[int, int, int, int] = (232, 191, 137, 84),
        radius: int = 6,
        master: Optional[tk.Misc] = None,
    ) -> ImageTk.PhotoImage:
        width = max(2, int(width))
        height = max(2, int(height))
        cache = getattr(self, "_main_glass_photo_cache", None)
        key = (width, height, fill, outline, int(radius))
        if isinstance(cache, dict):
            cached = cache.get(key)
            if cached is not None:
                return cached
        surface = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shadow = Image.new("RGBA", surface.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle(
            (3, 4, width - 3, height - 2),
            radius=radius,
            fill=(0, 0, 0, 76),
        )
        surface.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(3)))
        draw = ImageDraw.Draw(surface)
        draw.rounded_rectangle(
            (1, 1, width - 2, height - 2),
            radius=radius,
            fill=fill,
            outline=outline,
            width=1,
        )
        draw.line(
            (radius + 3, 2, width - radius - 4, 2),
            fill=(255, 227, 188, 50),
            width=1,
        )
        photo = ImageTk.PhotoImage(surface, master=master or self)
        if isinstance(cache, dict):
            # A resize can produce many one-off dimensions; retain the common
            # panel/button surfaces but bound cache growth.
            if len(cache) >= 48:
                cache.pop(next(iter(cache)))
            cache[key] = photo
        return photo

    def _draw_main_glass_surface(
        self,
        bounds: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int, int] = (28, 12, 10, 94),
        outline: tuple[int, int, int, int] = (232, 191, 137, 84),
        radius: int = 6,
    ) -> int:
        x1, y1, x2, y2 = bounds
        photo = self._make_main_glass_photo(x2 - x1, y2 - y1, fill=fill, outline=outline, radius=radius)
        self._main_control_photos.append(photo)
        return self._main_canvas.create_image(x1, y1, anchor="nw", image=photo, tags="main-control")

    def _draw_main_canvas_button(
        self,
        tag: str,
        bounds: tuple[int, int, int, int],
        text: str,
        command: Callable[[], None],
        *,
        align: str = "center",
        icon_tag: Optional[str] = None,
    ) -> None:
        x1, y1, x2, y2 = bounds
        root = self._main_canvas
        normal_photo = self._make_main_glass_photo(
            x2 - x1,
            y2 - y1,
            fill=(43, 20, 14, 76),
            outline=(236, 204, 161, 108),
            radius=5,
        )
        hover_photo = self._make_main_glass_photo(
            x2 - x1,
            y2 - y1,
            fill=(96, 39, 24, 132),
            outline=(255, 205, 136, 205),
            radius=5,
        )
        self._main_control_photos.extend((normal_photo, hover_photo))
        button_image = root.create_image(
            x1,
            y1,
            anchor="nw",
            image=normal_photo,
            tags=("main-control", tag),
        )
        icon_width = 0
        icon_source = self._main_button_icon_sources.get(icon_tag or "")
        if icon_source is not None:
            icon = icon_source.copy()
            icon.thumbnail((28, 28), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon, master=root)
            self._main_control_photos.append(icon_photo)
            icon_x = x1 + 8 + (28 - icon.width) // 2
            icon_y = y1 + ((y2 - y1) - icon.height) // 2
            root.create_image(icon_x, icon_y, anchor="nw", image=icon_photo, tags=("main-control", tag))
            icon_width = 38
        text_x = x1 + icon_width + (x2 - x1 - icon_width) // 2 if align == "center" else x1 + icon_width + 8
        root.create_text(
            text_x,
            (y1 + y2) // 2,
            anchor="center" if align == "center" else "w",
            text=text,
            fill="#edf3f7",
            font=("Microsoft YaHei UI", 10),
            tags=("main-control", tag),
        )
        self._main_button_rectangles[tag] = button_image

        def invoke(_event: tk.Event) -> None:
            self.after_idle(command)

        def set_hover(active: bool) -> None:
            if root.type(button_image):
                root.itemconfigure(button_image, image=hover_photo if active else normal_photo)
            root.configure(cursor="hand2" if active else "")

        root.tag_bind(tag, "<Button-1>", invoke)
        root.tag_bind(tag, "<Enter>", lambda _event: set_hover(True))
        root.tag_bind(tag, "<Leave>", lambda _event: set_hover(False))

    def _draw_main_profile_selector(self, bounds: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bounds
        root = self._main_canvas
        self._draw_main_glass_surface(
            bounds,
            fill=(25, 14, 13, 116),
            outline=(221, 177, 119, 92),
            radius=6,
        )
        label_width = 48
        root.create_text(
            x1 + 12,
            (y1 + y2) // 2,
            anchor="w",
            text="难度",
            fill="#d7c3ad",
            font=("Microsoft YaHei UI", 9),
            tags="main-control",
        )
        segment_left = x1 + label_width
        segment_width = max(62, (x2 - segment_left - 4) // 2)
        for index, (key, text_value) in enumerate((("difficulty", "困难"), ("nightmare", "噩梦"))):
            sx1 = segment_left + index * segment_width
            sx2 = x2 - 3 if index == 1 else sx1 + segment_width
            selected = self.recognition_profile.key == key
            segment = self._make_main_glass_photo(
                sx2 - sx1,
                y2 - y1 - 6,
                fill=(128, 57, 26, 180) if selected else (38, 23, 20, 82),
                outline=(255, 197, 112, 210) if selected else (168, 130, 93, 66),
                radius=5,
            )
            self._main_control_photos.append(segment)
            tag = f"main-profile-{key}"
            root.create_image(sx1, y1 + 3, anchor="nw", image=segment, tags=("main-control", tag))
            root.create_text(
                (sx1 + sx2) // 2,
                (y1 + y2) // 2,
                text=text_value,
                fill="#fff3df" if selected else "#cbb8a3",
                font=("Microsoft YaHei UI", 9, "bold" if selected else "normal"),
                tags=("main-control", tag),
            )
            root.tag_bind(tag, "<Button-1>", lambda _event, value=key: self.after_idle(lambda: self._switch_recognition_profile(value)))
            root.tag_bind(tag, "<Enter>", lambda _event: root.configure(cursor="hand2"))
            root.tag_bind(tag, "<Leave>", lambda _event: root.configure(cursor=""))

    def _format_main_slider_value(self, value: float, value_format: str) -> str:
        if value_format == "scale":
            return f"{value:.2f}x"
        return f"{value * 100:.0f}%"

    def _fit_main_canvas_text(
        self,
        value: object,
        *,
        max_width: int,
        font: tuple[str, int, str],
    ) -> str:
        """Keep status text inside its canvas card instead of relying on Canvas wrapping."""
        text = " ".join(str(value).split())
        if not text or max_width <= 0:
            return ""
        try:
            measure = tkfont.Font(
                master=self._main_canvas,
                family=font[0],
                size=font[1],
                weight=font[2],
            ).measure
        except tk.TclError:
            average_width = max(6, int(font[1] * 0.9))
            return text if len(text) * average_width <= max_width else text[: max(1, max_width // average_width - 1)] + "..."
        if measure(text) <= max_width:
            return text
        suffix = "..."
        shortened = text
        while shortened and measure(shortened + suffix) > max_width:
            shortened = shortened[:-1]
        return (shortened + suffix) if shortened else suffix

    def _make_main_slider_knob_photo(self) -> Optional[ImageTk.PhotoImage]:
        if self._main_slider_knob_source is None:
            return None
        icon = self._main_slider_knob_source.copy()
        icon.thumbnail((27, 27), Image.Resampling.LANCZOS)
        surface = Image.new("RGBA", (33, 33), (0, 0, 0, 0))
        x = (surface.width - icon.width) // 2
        y = (surface.height - icon.height) // 2
        glow_mask = Image.new("L", surface.size, 0)
        glow_mask.paste(icon.getchannel("A"), (x, y))
        glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(2))
        glow = Image.new("RGBA", surface.size, (82, 193, 236, 0))
        glow.putalpha(glow_mask.point(lambda value: int(value * 0.58)))
        surface.alpha_composite(glow)
        surface.alpha_composite(icon, (x, y))
        return ImageTk.PhotoImage(surface, master=self._main_canvas)

    def _set_main_slider_value(self, index: int, x: int) -> None:
        geometry = self._main_slider_geometry.get(index)
        items = self._main_slider_items.get(index)
        if geometry is None or items is None:
            return
        left, right = geometry
        label, minimum, maximum, variable, callback, value_format = self._main_slider_specs[index]
        del label
        ratio = max(0.0, min(1.0, (float(x) - left) / max(1, right - left)))
        value = minimum + ratio * (maximum - minimum)
        variable.set(value)
        callback(str(value))
        filled_line, knob, value_text = items
        knob_x = int(round(left + ratio * (right - left)))
        line_coords = self._main_canvas.coords(filled_line)
        knob_y = line_coords[1]
        self._main_canvas.coords(filled_line, left, knob_y, knob_x, knob_y)
        if self._main_canvas.type(knob) == "image":
            self._main_canvas.coords(knob, knob_x, knob_y)
        else:
            self._main_canvas.coords(knob, knob_x - 6, knob_y - 6, knob_x + 6, knob_y + 6)
        self._main_canvas.itemconfigure(value_text, text=self._format_main_slider_value(value, value_format))

    def _render_main_window_controls(self) -> None:
        if not getattr(self, "_main_controls_initialized", False):
            return
        root = self._main_canvas
        width = int(root.winfo_width())
        height = int(root.winfo_height())
        if width < 160 or height < 240:
            return

        root.delete("main-control")
        self._main_button_rectangles.clear()
        self._main_control_photos.clear()
        self._main_slider_geometry.clear()
        self._main_slider_items.clear()
        self._main_result_text_item = None
        self._main_status_text_item = None
        margin = max(24, int(width * 0.042))
        gap = max(10, int(width * 0.014))
        button_height = max(38, min(44, int(height * 0.052)))
        column_width = (width - 2 * margin - gap) // 2
        header_y = max(24, int(height * 0.034))
        root.create_line(
            margin,
            header_y - 8,
            margin,
            header_y + 13,
            fill="#f1be73",
            width=3,
            tags="main-control",
        )
        root.create_text(
            margin + 12,
            header_y + 2,
            anchor="w",
            text="游戏地图识别与对齐",
            fill="#fff2dd",
            font=("Microsoft YaHei UI", 14, "bold"),
            tags="main-control",
        )

        action_gap = max(8, int(width * 0.009))
        action_width = max(108, min(132, int(width * 0.118)))
        top = header_y - button_height // 2 + 2
        actions_bottom = top + button_height
        actions_left = width - margin - action_width * 3 - action_gap * 2
        selector_width = max(200, min(224, int(width * 0.205)))
        selector_right = actions_left - action_gap
        self._draw_main_profile_selector(
            (selector_right - selector_width, top, selector_right, actions_bottom)
        )
        self._draw_main_canvas_button(
            "main-pause",
            (actions_left, top, actions_left + action_width, actions_bottom),
            "▶ 已暂停" if self._program_paused else "Ⅱ 暂停",
            self._toggle_program_paused,
        )
        self._draw_main_canvas_button(
            "main-copyright",
            (
                actions_left + action_width + action_gap,
                top,
                actions_left + action_width * 2 + action_gap,
                actions_bottom,
            ),
            "版权声明",
            self.open_copyright_statement,
            icon_tag="main-copyright",
        )
        self._draw_main_canvas_button(
            "main-settings",
            (width - margin - action_width, top, width - margin, actions_bottom),
            "设置",
            self.open_settings_window,
            icon_tag="main-settings",
        )

        settings_top = header_y + max(42, int(height * 0.055))
        settings_height = max(154, min(166, int(height * 0.215)))
        settings_bottom = settings_top + settings_height
        self._draw_main_glass_surface(
            (margin, settings_top, width - margin, settings_bottom),
            fill=(35, 15, 11, 70),
            outline=(235, 190, 130, 88),
            radius=7,
        )
        root.create_text(
            margin + 18,
            settings_top + 20,
            anchor="w",
            text="显示设置",
            fill="#fff2dd",
            font=("Microsoft YaHei UI", 11, "bold"),
            tags="main-control",
        )
        root.create_line(
            margin + 18,
            settings_top + 39,
            width - margin - 18,
            settings_top + 39,
            fill="#8d684f",
            width=1,
            tags="main-control",
        )

        settings_inner_gap = max(14, int(width * 0.018))
        slider_group_width = (width - margin * 2 - settings_inner_gap * 2) // 3
        first_scale_y = settings_top + 70
        slider_knob_photo = self._make_main_slider_knob_photo()
        if slider_knob_photo is not None:
            self._main_control_photos.append(slider_knob_photo)
        for index, (label, minimum, maximum, variable, _callback, value_format) in enumerate(self._main_slider_specs):
            group_left = margin + index * (slider_group_width + settings_inner_gap)
            group_right = group_left + slider_group_width
            scale_left = group_left + 8
            scale_right = group_right - 38
            y = first_scale_y
            value = float(variable.get())
            ratio = max(0.0, min(1.0, (value - minimum) / max(0.0001, maximum - minimum)))
            knob_x = int(round(scale_left + ratio * (scale_right - scale_left)))
            root.create_text(
                group_left + 8,
                y - 22,
                anchor="w",
                text=label,
                fill="#f0ddc5",
                font=("Microsoft YaHei UI", 9),
                tags="main-control",
            )
            slider_tag = f"main-slider-{index}"
            root.create_line(
                scale_left,
                y,
                scale_right,
                y,
                fill="#4c3028",
                width=7,
                capstyle="round",
                tags=("main-control", slider_tag),
            )
            root.create_line(
                scale_left,
                y,
                scale_right,
                y,
                fill="#8f6b52",
                width=2,
                capstyle="round",
                tags=("main-control", slider_tag),
            )
            filled_line = root.create_line(
                scale_left,
                y,
                knob_x,
                y,
                fill="#f0b86c",
                width=3,
                capstyle="round",
                tags=("main-control", slider_tag),
            )
            if slider_knob_photo is not None:
                knob = root.create_image(
                    knob_x,
                    y,
                    image=slider_knob_photo,
                    tags=("main-control", slider_tag),
                )
            else:
                knob = root.create_oval(
                    knob_x - 6,
                    y - 6,
                    knob_x + 6,
                    y + 6,
                    fill="#fff0d5",
                    outline="#e9a95f",
                    width=2,
                    tags=("main-control", slider_tag),
                )
            value_text = root.create_text(
                group_right - 4,
                y,
                anchor="e",
                text=self._format_main_slider_value(value, value_format),
                fill="#f4cb91",
                font=("Microsoft YaHei UI", 9, "bold"),
                tags=("main-control", slider_tag),
            )
            self._main_slider_geometry[index] = (scale_left, scale_right)
            self._main_slider_items[index] = (filled_line, knob, value_text)
            root.tag_bind(slider_tag, "<Button-1>", lambda event, i=index: self._set_main_slider_value(i, event.x))
            root.tag_bind(slider_tag, "<B1-Motion>", lambda event, i=index: self._set_main_slider_value(i, event.x))
            root.tag_bind(slider_tag, "<Enter>", lambda _event: root.configure(cursor="hand2"))
            root.tag_bind(slider_tag, "<Leave>", lambda _event: root.configure(cursor=""))

        save_top = settings_top + 98
        self._draw_main_canvas_button(
            "main-save-display",
            (margin + 16, save_top, width - margin - 16, settings_bottom - 12),
            "保存显示设置",
            self.save_cheatsheet_settings,
        )

        preview_images = [
            (f"{configured_hotkey_display_name(self, 'recognize')} 地图识别 ROI", "已确认，保持本次识别画面" if self._recognition_preview_locked else "等待玩家确认侧门结构", self._recognition_preview_roi, "#71c7e8"),
            (f"{configured_hotkey_display_name(self, 'locate_current_floor')} 定位 ROI", "当前定位使用的地图裁剪画面", self._alignment_preview_roi, "#d8bc7a"),
        ]
        visible_previews = [item for item in preview_images if item[2] is not None]
        preview_bottom = settings_bottom
        if visible_previews:
            preview_top = settings_bottom + max(20, int(height * 0.025))
            preview_height = max(150, min(400, height - preview_top - 178))
            preview_bottom = preview_top + preview_height
            if len(visible_previews) == 1:
                title, subtitle, image, accent = visible_previews[0]
                self._draw_main_roi_preview(
                    (margin, preview_top, width - margin, preview_bottom),
                    title=title, subtitle=subtitle, image=image, accent=accent,
                )
            else:
                preview_gap = max(10, int(width * 0.018))
                preview_width = (width - margin * 2 - preview_gap) // 2
                for index, (title, subtitle, image, accent) in enumerate(visible_previews):
                    x1 = margin + index * (preview_width + preview_gap)
                    self._draw_main_roi_preview(
                        (x1, preview_top, x1 + preview_width, preview_bottom),
                        title=title, subtitle=subtitle, image=image, accent=accent,
                    )

        result_top = max(preview_bottom + 18, int(height * 0.69) if not visible_previews else preview_bottom + 18)
        result_height = max(74, min(88, int(height * 0.10)))
        result_bottom = min(height - 72, result_top + result_height)
        if result_bottom - result_top < 64:
            result_top = max(settings_bottom + 14, result_bottom - 64)
        self._draw_main_glass_surface(
            (margin, result_top, width - margin, result_bottom),
            fill=(16, 15, 17, 142),
            outline=(211, 158, 96, 118),
            radius=6,
        )
        root.create_line(
            margin + 17,
            result_top + 14,
            margin + 17,
            result_bottom - 14,
            fill="#d8a45f",
            width=2,
            tags="main-control",
        )
        root.create_text(
            margin + 31,
            result_top + 17,
            anchor="w",
            text="当前识别",
            fill="#cda26b",
            font=("Microsoft YaHei UI", 9, "bold"),
            tags="main-control",
        )
        result_font = ("Microsoft YaHei UI", 12, "normal")
        result_text = self._fit_main_canvas_text(
            self.result_var.get(),
            max_width=width - margin * 2 - 62,
            font=result_font,
        )
        self._main_result_text_width = width - margin * 2 - 62
        self._main_result_text_item = root.create_text(
            margin + 31,
            result_top + 47,
            anchor="w",
            text=result_text,
            fill="#ead6b8",
            font=result_font,
            tags="main-control",
        )

        status_top = height - 54
        self._draw_main_glass_surface(
            (margin, status_top, width - margin, height - 18),
            fill=(29, 11, 10, 78),
            outline=(225, 186, 139, 70),
            radius=5,
        )
        root.create_oval(
            margin + 14,
            status_top + 15,
            margin + 20,
            status_top + 21,
            fill="#f1bb70",
            outline="",
            tags="main-control",
        )
        self._main_status_text_width = width - margin * 2 - 43
        self._main_status_text_item = root.create_text(
            margin + 29,
            (status_top + height - 18) // 2,
            anchor="w",
            text=self.status_var.get(),
            width=width - margin * 2 - 43,
            fill="#f0ddc5",
            font=("Microsoft YaHei UI", 9),
            tags="main-control",
        )

    def _toggle_program_paused(self) -> None:
        next_state = not self._program_paused
        if next_state and self._scale_search_active:
            self._stop_semantic_cv_scale_search()
        self._program_paused = next_state
        if self._program_paused:
            self._route_overlay_cancel_reason = "program_paused"
            self._cancel_route_overlay_generation()
            self.status_var.set("程序已暂停锁定；识别与定位热键暂时停用。")
        else:
            self.status_var.set("程序已恢复启用；识别与定位热键可以正常使用。")
        self._render_main_window_controls()

    def _bind_keys(self) -> None:
        self.bind_all("<KeyPress>", self._handle_tk_hotkey)
        self.bind_all("<Return>", self._handle_return_key)
        # Debug/development hotkeys intentionally disabled:
        # F7 manual alignment, F9 save debug images, F10 save sample, F11 save pending sample.
        self.bind_all("<Escape>", lambda _event: self._cancel_route_overlay_generation())

    def _load_hotkeys(self) -> dict[str, str]:
        configured = self.project_config.app.get("hotkeys")
        values = dict(HOTKEY_DEFAULTS)
        used: set[tuple[str, object]] = set()
        reserved = {
            hotkey_binding_identity("F2"),
            hotkey_binding_identity(VARIANT_RETURN_HOTKEY),
            ("key", 0x1B),
        }
        game_map_identity = game_map_toggle_binding_identity(
            self.project_config.app.get("game_map_toggle_button", "right")
        )
        actions = tuple(HOTKEY_DEFAULTS)
        if isinstance(configured, dict):
            # Preserve every existing user binding before assigning a default
            # to a newly introduced action such as F4 confirmation.
            actions = tuple(sorted(actions, key=lambda action: action not in configured))
        for action in actions:
            default_key = HOTKEY_DEFAULTS[action]
            candidate = configured.get(action) if isinstance(configured, dict) else default_key
            binding = str(candidate).strip()
            identity = hotkey_binding_identity(binding)
            if identity is None or identity in reserved or identity == game_map_identity or identity in used:
                fallback_bindings = (default_key, *(f"F{index}" for index in range(1, 13)))
                binding = default_key
                identity = None
                for fallback in fallback_bindings:
                    fallback_identity = hotkey_binding_identity(fallback)
                    if (
                        fallback_identity is not None
                        and fallback_identity not in reserved
                        and fallback_identity != game_map_identity
                        and fallback_identity not in used
                    ):
                        binding = fallback
                        identity = fallback_identity
                        break
            values[action] = binding
            if identity is not None:
                used.add(identity)
        return values

    def _hotkey_text(self, action: str) -> str:
        """Return the current user-facing binding for a program action."""
        return configured_hotkey_display_name(self, action)

    def _refresh_hotkey_dependent_ui(self) -> None:
        """Refresh already-visible labels and route-sheet hints after a hotkey save."""
        render_main = getattr(self, "_render_main_window_controls", None)
        if callable(render_main):
            render_main()
        if not (
            getattr(self, "_side_door_structure_confirmed", False)
            and self._has_multiple_route_layouts()
        ):
            return
        content = getattr(self.cheatsheet, "_last_content", None)
        if not isinstance(content, tuple) or len(content) != 2:
            return
        kind, payload = content
        footer = self._route_layout_switch_hint()
        if kind == "image_with_message" and len(payload) == 4:
            image, title, _old_footer, compact_footer = payload
            self.cheatsheet.restore_content((kind, (image, title, footer, compact_footer)))
        elif kind == "image_with_bottom_action" and len(payload) == 6:
            image, title, _old_footer, compact_footer, button_text, on_click = payload
            self.cheatsheet.restore_content(
                (kind, (image, title, footer, compact_footer, button_text, on_click))
            )

    def _is_hotkey_action_down(self, action: str) -> bool:
        binding = self._hotkeys.get(action, HOTKEY_DEFAULTS[action])
        identity = hotkey_binding_identity(binding)
        if identity is None:
            return False
        kind, code = identity
        if kind == "mouse":
            virtual_key = GAME_MAP_MOUSE_VIRTUAL_KEYS.get(str(code))
            return virtual_key is not None and self._is_global_key_down(virtual_key)
        return self._is_global_key_down(int(code))

    def _action_for_hotkey_event(self, keysym: str, keycode: int) -> Optional[str]:
        normalized = str(keysym).upper()
        for action, binding in self._hotkeys.items():
            identity = hotkey_binding_identity(binding)
            if identity == ("key", int(keycode)):
                return action
            if str(binding).upper() == normalized:
                return action
        return None

    def _handle_tk_hotkey(self, event: tk.Event) -> Optional[str]:
        if self._hotkey_configuring:
            return None
        key = str(event.keysym).upper()
        if key == "F2" and self._scale_search_active:
            self._stop_semantic_cv_scale_search()
            return "break"
        if key == VARIANT_RETURN_HOTKEY and self._can_return_to_variant_layout_selection():
            self._return_to_variant_layout_selection()
            return "break"
        if key == VARIANT_RETURN_HOTKEY and self._can_return_to_scale_research_sheet():
            self._return_to_scale_research_sheet()
            return "break"
        try:
            keycode = int(event.keycode)
        except (AttributeError, TypeError, ValueError):
            identity = hotkey_binding_identity(key)
            keycode = int(identity[1]) if identity is not None and identity[0] == "key" else 0
        action = self._action_for_hotkey_event(key, keycode)
        if action is None:
            return None
        self._trigger_hotkey_action(action)
        return "break"

    def _handle_return_key(self, _event: tk.Event) -> str:
        enter_identity = ("key", 0x0D)
        structure_binding = self._hotkeys.get("confirm_structure", HOTKEY_DEFAULTS["confirm_structure"])
        if self._recognition_confirm_pending and hotkey_binding_identity(structure_binding) == enter_identity:
            self._trigger_hotkey_action("confirm_structure")
        return "break"

    def _trigger_hotkey_action(self, action: str) -> None:
        if self._hotkey_configuring:
            return
        now = time.monotonic()
        if now - self._hotkey_action_times.get(action, 0.0) < 0.15:
            return
        self._hotkey_action_times[action] = now
        if bool(getattr(self, "_program_paused", False)) and action in {"recognize", "locate_current_floor"}:
            self.status_var.set("程序当前处于暂停锁定状态；点击主窗口的 ▶ 已暂停 按钮恢复启用。")
            return
        if (
            self._semantic_cv_scale_confirmation_pending is not None
            and action == "locate_current_floor"
        ):
            self.status_var.set("请先在小抄界面选择虚像是否对齐。")
            return
        if action == "recognize":
            self.request_recognition_start(from_hotkey=True)
        elif action == "confirm_structure":
            if self._recognition_confirm_pending:
                self._accept_recognized_side_door_structure()
        elif action == "switch_layout":
            self.cycle_side_door_layout()
        elif action == "locate_current_floor":
            if not self._side_door_structure_confirmed or self._active_route_layout_id() is None:
                hotkey = hotkey_binding_display_name(
                    self._hotkeys.get("locate_current_floor", HOTKEY_DEFAULTS["locate_current_floor"])
                )
                self._show_layout_lock_required(hotkey)
                return
            self._request_map_location_after_toggle(None)

    def _build_ambiguous_choice_panel(self, parent: ttk.Frame) -> None:
        self.ambiguous_choice_frame = ttk.LabelFrame(parent, text="易混淆地图人工锁定")
        hint = ttk.Label(
            self.ambiguous_choice_frame,
            text="识别到 04 / 16 / 20 / 21 时显示。请在小抄界面点击入口结构图手动锁定。",
            wraplength=320,
        )
        hint.pack(fill="x", padx=6, pady=(6, 4))
        grid = ttk.Frame(self.ambiguous_choice_frame)
        grid.pack(fill="x", padx=6, pady=(4, 8))
        for idx, map_id in enumerate(self._ambiguous_map_ids):
            block = ttk.Frame(grid)
            block.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=4, pady=5)
            photo = self._make_ambiguous_entry_photo(map_id, max_width=150, max_height=70)
            image_label = ttk.Label(block, anchor="center", relief="solid")
            if photo is not None:
                self._ambiguous_choice_photos[map_id] = photo
                image_label.configure(image=photo)
            else:
                image_label.configure(text=f"{map_id}\n入口结构图未找到")
            image_label.pack(fill="x")
            button = ttk.Button(block, text=f"锁定 {map_id}", command=lambda value=map_id: self.confirm_manual_map_choice(value))
            button.pack(fill="x", pady=(3, 0))
            self._ambiguous_choice_buttons[map_id] = button
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        for row in range(2):
            grid.rowconfigure(row, weight=1)

    def _ambiguous_entry_dir(self) -> Path:
        raw = self.project_config.app.get(
            "ambiguous_map_entry_dir",
            r"C:\Users\LD\Desktop\摸金工具\地图合集\易混淆",
        )
        return resolve_config_path(self.project_config.root_dir, str(raw))

    def _ambiguous_entry_image_path(self, map_id: str) -> Optional[Path]:
        folder = self._ambiguous_entry_dir()
        if not folder.exists():
            return None
        target = int(map_id.split("_")[-1])
        suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in suffixes:
                continue
            prefix = ""
            for char in path.stem.strip():
                if char.isdigit():
                    prefix += char
                else:
                    break
            if prefix and int(prefix) == target:
                return path
        return None

    def _make_ambiguous_entry_photo(self, map_id: str, *, max_width: int, max_height: int) -> Optional[ImageTk.PhotoImage]:
        path = self._ambiguous_entry_image_path(map_id)
        if path is None:
            return None
        image = imread_unicode(path)
        if image is None:
            return None
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if image.ndim == 3 else Image.fromarray(image).convert("RGB")
        pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max_width, max_height), (32, 32, 32))
        x = (max_width - pil.width) // 2
        y = (max_height - pil.height) // 2
        canvas.paste(pil, (x, y))
        return ImageTk.PhotoImage(canvas)

    def _hide_manual_layout_chooser(self) -> None:
        if self._manual_choice_window is not None:
            try:
                self._manual_choice_window.destroy()
            except tk.TclError:
                pass
        self._manual_choice_window = None
        self._manual_choice_window_photos = []

    def _show_manual_layout_chooser(self, template_image: Optional[np.ndarray], template_title: str) -> None:
        self._hide_manual_layout_chooser()
        dialog = tk.Toplevel(self)
        dialog.title("选择正确布局")
        dialog.configure(bg="#111827")
        dialog.attributes("-topmost", True)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", dialog.withdraw)
        self._manual_choice_window = dialog

        body = tk.Frame(dialog, bg="#111827", padx=10, pady=10)
        body.pack(fill="both", expand=True)
        tk.Label(
            body,
            text=f"请选择正确布局：{' / '.join(self._active_ambiguous_map_ids)}",
            bg="#111827",
            fg="#f9fafb",
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        if template_image is not None:
            if template_image.ndim == 2:
                template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)
            template_pil = Image.fromarray(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
            template_pil.thumbnail((520, 190), Image.Resampling.LANCZOS)
            template_photo = ImageTk.PhotoImage(template_pil)
            self._manual_choice_window_photos.append(template_photo)
            tk.Label(
                body,
                image=template_photo,
                text=template_title,
                compound="top",
                bg="#0a0e14",
                fg="#f9fafb",
                font=("Microsoft YaHei UI", 9),
                padx=6,
                pady=6,
            ).pack(fill="x", pady=(0, 10))

        grid = tk.Frame(body, bg="#111827")
        grid.pack(fill="both", expand=True)
        is_two_choice_group = len(self._active_ambiguous_map_ids) == 2
        image_w = 300 if is_two_choice_group else 230
        image_h = 210 if is_two_choice_group else 145
        for idx, map_id in enumerate(self._active_ambiguous_map_ids):
            photo = self._make_ambiguous_entry_photo(map_id, max_width=image_w, max_height=image_h)
            if photo is not None:
                self._manual_choice_window_photos.append(photo)
            button = tk.Button(
                grid,
                text=map_id if photo is None else map_id,
                image=photo,
                compound="top",
                command=lambda value=map_id: self.confirm_manual_map_choice(value),
                bg="#1f2937",
                fg="#f9fafb",
                activebackground="#374151",
                activeforeground="#ffffff",
                relief="raised",
                bd=2,
                font=("Microsoft YaHei UI", 11, "bold"),
                padx=4,
                pady=4,
            )
            button.grid(row=idx // 2, column=idx % 2, padx=5, pady=5, sticky="nsew")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        for row in range((len(self._active_ambiguous_map_ids) + 1) // 2):
            grid.rowconfigure(row, weight=1)
        dialog.update_idletasks()
        dialog.lift()
        dialog.focus_force()

    def _set_ambiguous_choice_visible(self, visible: bool, summary: Optional[MatchSummary] = None) -> None:
        self._manual_map_select_active = visible
        self._ambiguous_choice_visible = False
        if not visible:
            self._hide_manual_layout_chooser()

    def _update_ambiguous_choice_button_scores(self, summary: Optional[MatchSummary]) -> None:
        scores = self.last_cnn_scores if self.last_cnn_scores else {}
        for map_id, button in self._ambiguous_choice_buttons.items():
            score = float(scores.get(map_id, 0.0))
            button.configure(text=f"锁定 {map_id}    CNN {score:.3f}")

    def _is_ambiguous_family_present(self, summary: Optional[MatchSummary]) -> bool:
        if summary is None or summary.best is None:
            return False
        return summary.best.map_id in self._ambiguous_map_ids

    def _ambiguous_map_ids_for_summary(self, summary: Optional[MatchSummary]) -> tuple[str, ...]:
        if summary is None or summary.best is None:
            return ()
        map_id = summary.best.map_id
        for group in self._ambiguous_map_groups.values():
            if map_id in group:
                return group
        return ()

    def _ambiguous_manual_prompt(self) -> str:
        map_names = " / ".join(self._active_ambiguous_map_ids)
        return f"当前结果属于易混淆组：{map_names}\n可直接点击下方入口结构图手动锁定。"

    @staticmethod
    def _draw_ambiguous_section_title(image: np.ndarray, title: str) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("msyhbd.ttc", 19)
        except OSError:
            font = ImageFont.load_default()
        draw.text((12, 6), title, fill=(249, 250, 251), font=font)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def _make_ambiguous_entry_contact_sheet(self) -> tuple[Optional[np.ndarray], list[tuple[str, tuple[int, int, int, int]]]]:
        tiles: list[np.ndarray] = []
        tile_regions: list[tuple[str, tuple[int, int, int, int]]] = []
        is_two_choice_group = len(self._active_ambiguous_map_ids) == 2
        tile_w = 360
        tile_h = 270 if is_two_choice_group else 210
        label_h = 30
        for map_id in self._active_ambiguous_map_ids:
            path = self._ambiguous_entry_image_path(map_id)
            image = imread_unicode(path) if path is not None else None
            tile = np.full((tile_h + label_h, tile_w, 3), (5, 7, 10), dtype=np.uint8)
            cv2.rectangle(tile, (0, 0), (tile_w, label_h), (10, 14, 20), -1)
            cv2.putText(tile, map_id, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (249, 250, 251), 1, cv2.LINE_AA)
            if image is not None:
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                h, w = image.shape[:2]
                scale = min(tile_w / max(w, 1), tile_h / max(h, 1))
                resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                x = (tile_w - resized.shape[1]) // 2
                y = label_h + (tile_h - resized.shape[0]) // 2
                tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
                tile_regions.append((map_id, (0, 0, tile_w, tile_h + label_h)))
            else:
                cv2.putText(tile, "entry image missing", (10, label_h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
                tile_regions.append((map_id, (0, label_h, tile_w, label_h + tile_h)))
            tiles.append(tile)
        if not tiles:
            return None, []
        rows = []
        regions: list[tuple[str, tuple[int, int, int, int]]] = []
        for idx in range(0, len(tiles), 2):
            row_tiles = tiles[idx : idx + 2]
            if len(row_tiles) == 1:
                row_tiles.append(np.full_like(row_tiles[0], (5, 7, 10)))
            row = idx // 2
            for col in range(min(2, len(tiles) - idx)):
                map_id, (x1, y1, x2, y2) = tile_regions[idx + col]
                regions.append((map_id, (x1 + col * tile_w, y1 + row * (tile_h + label_h), x2 + col * tile_w, y2 + row * (tile_h + label_h))))
            rows.append(np.hstack(row_tiles))
        return np.vstack(rows), regions

    def _make_ambiguous_confirmation_sheet(self, template_image: Optional[np.ndarray], template_title: str) -> tuple[Optional[np.ndarray], list[tuple[str, tuple[int, int, int, int]]]]:
        entry_sheet, entry_regions = self._make_ambiguous_entry_contact_sheet()
        if template_image is None:
            return entry_sheet, entry_regions
        if template_image.ndim == 2:
            template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)
        if entry_sheet is None:
            return template_image.copy(), []
        width = max(template_image.shape[1], entry_sheet.shape[1])
        title_h = 34

        def fit_section(image: np.ndarray, title: str, target_h: int) -> tuple[np.ndarray, tuple[int, int, float]]:
            section = np.full((target_h + title_h, width, 3), (5, 7, 10), dtype=np.uint8)
            cv2.rectangle(section, (0, 0), (width, title_h), (10, 14, 20), -1)
            section = self._draw_ambiguous_section_title(section, title)
            h, w = image.shape[:2]
            scale = min(width / max(w, 1), target_h / max(h, 1))
            resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            x = (width - resized.shape[1]) // 2
            y = title_h + (target_h - resized.shape[0]) // 2
            section[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            return section, (x, y, scale)

        is_two_choice_group = len(self._active_ambiguous_map_ids) == 2
        template_h = 300 if is_two_choice_group else 250
        entry_h = 330 if is_two_choice_group else 480
        template_section, _template_meta = fit_section(template_image, template_title, template_h)
        entry_title = "或者手动点击选择地图"
        entry_section, (entry_x, entry_y, entry_scale) = fit_section(entry_sheet, entry_title, entry_h)
        y_offset = template_section.shape[0]
        regions: list[tuple[str, tuple[int, int, int, int]]] = []
        for map_id, (x1, y1, x2, y2) in entry_regions:
            regions.append(
                (
                    map_id,
                    (
                        int(round(entry_x + x1 * entry_scale)),
                        int(round(y_offset + entry_y + y1 * entry_scale)),
                        int(round(entry_x + x2 * entry_scale)),
                        int(round(y_offset + entry_y + y2 * entry_scale)),
                    ),
                )
            )
        return np.vstack([template_section, entry_section]), regions

    def _register_ambiguous_cheatsheet_regions(self, regions: list[tuple[str, tuple[int, int, int, int]]]) -> None:
        self._ambiguous_cheatsheet_click_regions = regions

        def handle(point: tuple[float, float]) -> bool:
            x, y = point
            if not self._manual_map_select_active:
                return False
            for map_id, (x1, y1, x2, y2) in self._ambiguous_cheatsheet_click_regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.confirm_manual_map_choice(map_id)
                    return True
            return False

        self.cheatsheet.set_click_handler(handle)

    def _on_cheatsheet_alpha_change(self, _value: str) -> None:
        self.cheatsheet.set_alpha(self.cheatsheet_alpha_var.get())

    def _on_route_overlay_alpha_change(self, _value: str) -> None:
        self.route_overlay.set_alpha(self.route_overlay_alpha_var.get())

    def _current_cheatsheet_size(self) -> tuple[int, int]:
        scale = max(0.5, min(1.6, float(self.cheatsheet_size_var.get())))
        return int(round(self.cheatsheet_base_width * scale)), int(round(self.cheatsheet_base_height * scale))

    def _on_cheatsheet_size_change(self, _value: str) -> None:
        scale = max(0.5, min(1.6, float(self.cheatsheet_size_var.get())))
        adaptive_size = self.cheatsheet.adaptive_size_for_scale(scale)
        width, height = adaptive_size if adaptive_size is not None else self._current_cheatsheet_size()
        self.cheatsheet.set_size(width, height)

    def save_cheatsheet_settings(self) -> None:
        self._on_cheatsheet_size_change("")
        self.project_config.app["cheatsheet_alpha"] = float(self.cheatsheet_alpha_var.get())
        self.project_config.app["route_overlay_alpha"] = float(self.route_overlay_alpha_var.get())
        self.project_config.app["route_overlay_content"] = "route"
        self.project_config.app["cheatsheet_x"] = int(self.cheatsheet.x)
        self.project_config.app["cheatsheet_y"] = int(self.cheatsheet.y)
        path = self.project_config.app_config_path
        try:
            save_app_config(path, self.project_config.app)
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        width = int(self.project_config.app.get("cheatsheet_width", 0))
        height = int(self.project_config.app.get("cheatsheet_height", 0))
        self.status_var.set(
            f"显示设置已保存：小抄 {width} x {height}，位置 {self.cheatsheet.x},{self.cheatsheet.y}；"
            f"小抄透明度 {self.cheatsheet_alpha_var.get():.2f}；路线图虚像透明度 {self.route_overlay_alpha_var.get():.2f}"
        )

    def _game_map_keyboard_conflict(self, keysym: str, keycode: Optional[int] = None) -> Optional[str]:
        key = str(keysym).upper()
        identity = ("key", int(keycode)) if keycode is not None else hotkey_binding_identity(key)
        for action, assigned_key in self._hotkeys.items():
            if identity is not None and identity == hotkey_binding_identity(assigned_key):
                return HOTKEY_LABELS.get(action, action)
        fixed_keys = {
            "F2": "停止 SCALE 搜索",
            VARIANT_RETURN_HOTKEY: "返回上一步",
            "ESCAPE": "关闭或取消路线图虚像",
        }
        return fixed_keys.get(key)

    @staticmethod
    def _keyboard_key_display_label(keysym: str) -> str:
        key = str(keysym).strip()
        names = {
            "space": "Space",
            "tab": "Tab",
            "backspace": "Backspace",
            "delete": "Delete",
            "insert": "Insert",
            "home": "Home",
            "end": "End",
            "prior": "Page Up",
            "next": "Page Down",
            "left": "Left",
            "right": "Right",
            "up": "Up",
            "down": "Down",
            "shift_l": "Left Shift",
            "shift_r": "Right Shift",
            "control_l": "Left Ctrl",
            "control_r": "Right Ctrl",
            "alt_l": "Left Alt",
            "alt_r": "Right Alt",
            "caps_lock": "Caps Lock",
            "num_lock": "Num Lock",
            "scroll_lock": "Scroll Lock",
        }
        return names.get(key.lower(), key.upper() if len(key) == 1 else key)

    def _create_premium_settings_canvas(
        self,
        dialog: tk.Toplevel,
        *,
        width: int,
        height: int,
    ) -> tk.Canvas:
        dialog.configure(bg="#07111c")
        dialog.resizable(False, False)
        self.update_idletasks()
        x = max(0, self.winfo_rootx() + (self.winfo_width() - width) // 2)
        y = max(0, self.winfo_rooty() + (self.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        canvas = tk.Canvas(
            dialog,
            width=width,
            height=height,
            bg="#07111c",
            bd=0,
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)

        background_path = self.project_config.root_dir / "assets" / "ui" / "settings_window_background.png"
        try:
            with Image.open(background_path) as image:
                source = image.convert("RGB")
            source_w, source_h = source.size
            scale = max(width / max(1, source_w), height / max(1, source_h))
            resized = source.resize(
                (max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))),
                Image.Resampling.LANCZOS,
            )
            left = max(0, (resized.width - width) // 2)
            top = max(0, (resized.height - height) // 2)
            cropped = resized.crop((left, top, left + width, top + height)).convert("RGBA")
            veil = Image.new("RGBA", cropped.size, (0, 0, 0, 0))
            veil_draw = ImageDraw.Draw(veil)
            veil_draw.rectangle((0, 0, width, height), fill=(2, 8, 17, 82))
            for column in range(width):
                alpha = int(round(42 * (1.0 - column / max(1, width - 1))))
                veil_draw.line((column, 0, column, height), fill=(0, 6, 14, alpha))
            cropped.alpha_composite(veil)
            background_photo = ImageTk.PhotoImage(cropped, master=canvas)
            canvas.create_image(0, 0, anchor="nw", image=background_photo, tags="dialog-background")
            setattr(canvas, "_premium_background_photo", background_photo)
        except (OSError, ValueError):
            LOGGER.warning("settings window background unavailable: %s", background_path)
        setattr(canvas, "_premium_control_photos", [])
        return canvas

    def _draw_premium_dialog_surface(
        self,
        canvas: tk.Canvas,
        photos: list[ImageTk.PhotoImage],
        bounds: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int, int] = (4, 12, 22, 118),
        outline: tuple[int, int, int, int] = (174, 201, 220, 84),
        radius: int = 7,
    ) -> int:
        x1, y1, x2, y2 = bounds
        photo = self._make_main_glass_photo(
            x2 - x1,
            y2 - y1,
            fill=fill,
            outline=outline,
            radius=radius,
            master=canvas,
        )
        photos.append(photo)
        return canvas.create_image(x1, y1, anchor="nw", image=photo, tags="dialog-control")

    def _draw_premium_dialog_button(
        self,
        canvas: tk.Canvas,
        photos: list[ImageTk.PhotoImage],
        tag: str,
        bounds: tuple[int, int, int, int],
        text: str,
        command: Callable[[], None],
        *,
        selected: bool = False,
        primary: bool = False,
        font_size: int = 10,
        normal_fill: Optional[tuple[int, int, int, int]] = None,
        normal_outline: Optional[tuple[int, int, int, int]] = None,
        hover_fill: Optional[tuple[int, int, int, int]] = None,
        hover_outline: Optional[tuple[int, int, int, int]] = None,
        text_fill: str = "#f1f5f8",
    ) -> None:
        x1, y1, x2, y2 = bounds
        if normal_fill is None or normal_outline is None:
            if primary:
                normal_fill = (17, 74, 106, 152)
                normal_outline = (117, 208, 244, 190)
            elif selected:
                normal_fill = (18, 60, 86, 142)
                normal_outline = (111, 202, 241, 206)
            else:
                normal_fill = (5, 15, 26, 82)
                normal_outline = (181, 203, 219, 94)
        if hover_fill is None:
            hover_fill = (22, 70, 98, 162)
        if hover_outline is None:
            hover_outline = (126, 217, 250, 224)
        normal = self._make_main_glass_photo(
            x2 - x1,
            y2 - y1,
            fill=normal_fill,
            outline=normal_outline,
            radius=5,
            master=canvas,
        )
        hover = self._make_main_glass_photo(
            x2 - x1,
            y2 - y1,
            fill=hover_fill,
            outline=hover_outline,
            radius=5,
            master=canvas,
        )
        photos.extend((normal, hover))
        image_item = canvas.create_image(
            x1,
            y1,
            anchor="nw",
            image=normal,
            tags=("dialog-control", tag),
        )
        canvas.create_text(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            text=text,
            width=max(20, x2 - x1 - 16),
            justify="center",
            fill=text_fill,
            font=("Microsoft YaHei UI", font_size, "bold" if primary or selected else "normal"),
            tags=("dialog-control", tag),
        )

        def invoke(_event: tk.Event) -> None:
            self.after_idle(command)

        def hover_state(active: bool) -> None:
            if canvas.type(image_item):
                canvas.itemconfigure(image_item, image=hover if active else normal)
            canvas.configure(cursor="hand2" if active else "")

        canvas.tag_bind(tag, "<Button-1>", invoke)
        canvas.tag_bind(tag, "<Enter>", lambda _event: hover_state(True))
        canvas.tag_bind(tag, "<Leave>", lambda _event: hover_state(False))

    @staticmethod
    def _draw_premium_dialog_header(canvas: tk.Canvas, title: str, subtitle: str = "") -> None:
        canvas.create_line(40, 35, 40, 63, fill="#71c9ee", width=3, tags="dialog-control")
        canvas.create_text(
            54,
            42,
            anchor="w",
            text=title,
            fill="#f3f6f8",
            font=("Microsoft YaHei UI", 17, "bold"),
            tags="dialog-control",
        )
        if subtitle:
            canvas.create_text(
                54,
                66,
                anchor="w",
                text=subtitle,
                fill="#9fb2c0",
                font=("Microsoft YaHei UI", 9),
                tags="dialog-control",
            )

    def open_settings_window(self, initial_section: str = "hotkeys") -> None:
        """Open the unified settings workspace used by the main window."""
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_force()
            return

        dialog = tk.Toplevel(self)
        dialog.title("设置")
        dialog.transient(self)
        # Keep the unified settings workspace non-modal. The cheatsheet uses
        # sibling Toplevel windows for its controls, and a Tk grab here blocks
        # every one of those buttons while settings remains open.
        self._settings_window = dialog
        canvas = self._create_premium_settings_canvas(dialog, width=1180, height=760)
        photos: list[ImageTk.PhotoImage] = getattr(canvas, "_premium_control_photos")
        self._draw_premium_dialog_surface(canvas, photos, (24, 24, 250, 736), fill=(3, 10, 18, 24))
        self._draw_premium_dialog_surface(canvas, photos, (266, 24, 1156, 736), fill=(3, 11, 20, 16))
        canvas.create_text(50, 54, anchor="w", text="设置", fill="#f5eee3", font=("Microsoft YaHei UI", 21, "bold"))
        canvas.create_text(50, 85, anchor="w", text="摸金工具", fill="#aa987c", font=("Microsoft YaHei UI", 9))
        canvas.create_line(48, 106, 226, 106, fill="#665944", width=1)

        sidebar = tk.Canvas(canvas, bg="#07111b", bd=0, highlightthickness=0)
        content_viewport = tk.Canvas(
            canvas,
            bg="#07131f",
            bd=0,
            highlightthickness=0,
            yscrollincrement=24,
        )
        content = tk.Frame(content_viewport, bg="#07131f", bd=0, highlightthickness=0)
        content_window_id = content_viewport.create_window(
            0,
            0,
            anchor="nw",
            window=content,
            width=816,
        )
        content_scrollbar = tk.Canvas(
            canvas,
            width=14,
            bg="#081522",
            bd=0,
            highlightthickness=0,
        )
        canvas.create_window(38, 122, anchor="nw", window=sidebar, width=198, height=584)
        canvas.create_window(290, 48, anchor="nw", window=content_viewport, width=816, height=664)
        canvas.create_window(1112, 48, anchor="nw", window=content_scrollbar, width=14, height=664)
        scrollbar_track = content_scrollbar.create_rectangle(
            4, 2, 10, 662, fill="#132938", outline="", tags="settings-scrollbar"
        )
        scrollbar_thumb = content_scrollbar.create_rectangle(
            3, 2, 11, 80, fill="#5e879b", outline="#8eb5c4", width=1, tags="settings-scrollbar"
        )

        def sync_content_scrollbar(first: str, last: str) -> None:
            try:
                start, end = float(first), float(last)
            except (TypeError, ValueError):
                return
            height = max(1, int(content_scrollbar.winfo_height()))
            visible_fraction = max(0.0, min(1.0, end - start))
            thumb_height = max(42, int(round(height * visible_fraction)))
            thumb_height = min(height, thumb_height)
            y0 = int(round((height - thumb_height) * max(0.0, min(1.0, start))))
            content_scrollbar.coords(scrollbar_track, 4, 2, 10, max(2, height - 2))
            content_scrollbar.coords(scrollbar_thumb, 3, y0, 11, min(height - 2, y0 + thumb_height))

        content_viewport.configure(yscrollcommand=sync_content_scrollbar)

        def scrollbar_fraction_from_event(event: tk.Event) -> float:
            height = max(1, int(content_scrollbar.winfo_height()))
            thumb_box = content_scrollbar.coords(scrollbar_thumb)
            thumb_height = max(1.0, float(thumb_box[3] - thumb_box[1])) if len(thumb_box) >= 4 else 1.0
            return max(0.0, min(1.0, (float(event.y) - thumb_height / 2.0) / max(1.0, height - thumb_height)))

        scrollbar_drag_offset = {"value": 0.0}

        def start_scrollbar_drag(event: tk.Event) -> None:
            thumb_box = content_scrollbar.coords(scrollbar_thumb)
            if len(thumb_box) >= 4 and float(thumb_box[1]) <= float(event.y) <= float(thumb_box[3]):
                scrollbar_drag_offset["value"] = float(event.y) - float(thumb_box[1])
            else:
                scrollbar_drag_offset["value"] = 0.0
                content_viewport.yview_moveto(scrollbar_fraction_from_event(event))

        def drag_scrollbar(event: tk.Event) -> None:
            height = max(1, int(content_scrollbar.winfo_height()))
            thumb_box = content_scrollbar.coords(scrollbar_thumb)
            thumb_height = max(1.0, float(thumb_box[3] - thumb_box[1])) if len(thumb_box) >= 4 else 1.0
            fraction = (float(event.y) - scrollbar_drag_offset["value"]) / max(1.0, height - thumb_height)
            content_viewport.yview_moveto(max(0.0, min(1.0, fraction)))

        content_scrollbar.bind("<Button-1>", start_scrollbar_drag)
        content_scrollbar.bind("<B1-Motion>", drag_scrollbar)
        content_scrollbar.bind("<Enter>", lambda _event: content_scrollbar.itemconfigure(scrollbar_thumb, fill="#79aabd"))
        content_scrollbar.bind("<Leave>", lambda _event: content_scrollbar.itemconfigure(scrollbar_thumb, fill="#5e879b"))

        def update_content_scroll_region(_event: tk.Event | None = None) -> None:
            content_viewport.configure(scrollregion=content_viewport.bbox("all"))
            viewport_width = max(1, int(content_viewport.winfo_width()))
            content_viewport.itemconfigure(content_window_id, width=viewport_width)

        content.bind("<Configure>", update_content_scroll_region, add="+")
        content_viewport.bind("<Configure>", update_content_scroll_region, add="+")

        def scroll_settings_content(event: tk.Event) -> str:
            delta = getattr(event, "delta", 0)
            if delta:
                units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            elif getattr(event, "num", None) == 4:
                units = -1
            elif getattr(event, "num", None) == 5:
                units = 1
            else:
                units = 0
            if units:
                content_viewport.yview_scroll(units, "units")
            return "break"

        scroll_binding_ids = [
            dialog.bind("<MouseWheel>", scroll_settings_content, add="+"),
            dialog.bind("<Button-4>", scroll_settings_content, add="+"),
            dialog.bind("<Button-5>", scroll_settings_content, add="+"),
        ]

        settings_artwork: Optional[Image.Image] = None
        settings_artwork_photos: list[ImageTk.PhotoImage] = []
        try:
            with Image.open(self.project_config.root_dir / "assets" / "ui" / "settings_window_background.png") as image:
                source = image.convert("RGB")
            scale = max(1180 / max(1, source.width), 760 / max(1, source.height))
            rendered = source.resize(
                (max(1, int(round(source.width * scale))), max(1, int(round(source.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            left = max(0, (rendered.width - 1180) // 2)
            top = max(0, (rendered.height - 760) // 2)
            settings_artwork_rgba = rendered.crop((left, top, left + 1180, top + 760)).convert("RGBA")
            artwork_veil = Image.new("RGBA", settings_artwork_rgba.size, (0, 0, 0, 0))
            artwork_veil_draw = ImageDraw.Draw(artwork_veil)
            artwork_veil_draw.rectangle((0, 0, 1180, 760), fill=(2, 8, 17, 82))
            for column in range(1180):
                alpha = int(round(42 * (1.0 - column / max(1, 1179))))
                artwork_veil_draw.line((column, 0, column, 760), fill=(0, 6, 14, alpha))
            settings_artwork_rgba.alpha_composite(artwork_veil)
            settings_artwork = settings_artwork_rgba.convert("RGB")
        except (OSError, ValueError):
            LOGGER.warning("settings artwork unavailable")

        sections = (
            ("hotkeys", "自定义热键", "hotkeys.png"),
            ("map_key", "游戏地图键", "map_key.png"),
            ("scale", "Scale 管理", "scale.png"),
            ("cache", "清理识别缓存", "clear_cache.png"),
            ("route", "路线图版本选择", "route.png"),
        )
        section_keys = {item[0] for item in sections}
        active = {"key": initial_section if initial_section in section_keys else "hotkeys"}
        hotkey_working = dict(self._hotkeys)
        hotkey_capture: dict[str, Optional[str]] = {"action": None}
        map_working = {
            "binding": str(self.project_config.app.get("game_map_toggle_button", "right")).strip()
        }
        if map_working["binding"].lower() not in GAME_MAP_MOUSE_BINDINGS and re.fullmatch(
            r"key:\d{1,3}:.*", map_working["binding"]
        ) is None:
            map_working["binding"] = "right"
        map_capture = {"active": False}
        nav_buttons: dict[str, tuple[int, int, int, int]] = {}
        nav_photos: list[ImageTk.PhotoImage] = []

        icon_dir = self.project_config.root_dir / "assets" / "ui" / "button_icons"
        icons: dict[str, ImageTk.PhotoImage] = {}
        for key, _label, icon_name in sections:
            try:
                with Image.open(icon_dir / icon_name) as image:
                    icon = image.convert("RGBA")
                icon.thumbnail((25, 25), Image.Resampling.LANCZOS)
                icons[key] = ImageTk.PhotoImage(icon, master=dialog)
                nav_photos.append(icons[key])
            except (OSError, ValueError):
                LOGGER.debug("settings sidebar icon unavailable: %s", icon_name)
        setattr(dialog, "_settings_nav_photos", nav_photos)
        content_background_label: Optional[tk.Label] = None

        def clear_content() -> None:
            hotkey_capture["action"] = None
            map_capture["active"] = False
            self._hotkey_configuring = False
            self._hotkey_capture_action = None
            self._game_map_binding_capturing = False
            for child in content.winfo_children():
                if child is not content_background_label:
                    child.destroy()

        def artwork_photo(widget: tk.Misc, *, veil: float = 0.0) -> Optional[ImageTk.PhotoImage]:
            if settings_artwork is None:
                return None
            try:
                # The artwork is aligned to the client canvas, not the decorated
                # toplevel.  Using the toplevel origin shifts every crop by the
                # native Windows title-bar height.
                x = int(widget.winfo_rootx() - canvas.winfo_rootx())
                y = int(widget.winfo_rooty() - canvas.winfo_rooty())
                width = max(1, int(widget.winfo_width()))
                height = max(1, int(widget.winfo_height()))
                crop = settings_artwork.crop((max(0, x), max(0, y), min(1180, x + width), min(760, y + height)))
                if crop.size != (width, height):
                    crop = crop.resize((width, height), Image.Resampling.LANCZOS)
                if veil > 0:
                    crop = Image.blend(crop, Image.new("RGB", crop.size, (5, 17, 29)), veil)
                photo = ImageTk.PhotoImage(crop, master=dialog)
                settings_artwork_photos.append(photo)
                return photo
            except (tk.TclError, ValueError):
                return None

        content_background_pending = {"job": None}

        def paint_content_background() -> None:
            nonlocal content_background_label
            content_background_pending["job"] = None
            if not content.winfo_exists():
                return
            content.update_idletasks()
            photo = artwork_photo(content, veil=0.015)
            if photo is None:
                return
            if content_background_label is None or not content_background_label.winfo_exists():
                content_background_label = tk.Label(content, bd=0, highlightthickness=0, image=photo)
            else:
                content_background_label.configure(image=photo)
            content_background_label.place(x=0, y=0, relwidth=1.0, height=max(1, content.winfo_height()))
            content_background_label.lower()
            setattr(content_background_label, "_settings_background_photo", photo)

        def schedule_content_background(_event: Optional[tk.Event] = None) -> None:
            job = content_background_pending["job"]
            if job is not None:
                try:
                    content.after_cancel(job)
                except tk.TclError:
                    pass
            content_background_pending["job"] = content.after_idle(paint_content_background)

        content.bind("<Configure>", schedule_content_background, add="+")

        def decorate_canvas(surface: tk.Canvas, *, veil: float = 0.03) -> None:
            paint_pending = {"job": None}

            def paint() -> None:
                paint_pending["job"] = None
                photo = artwork_photo(surface, veil=veil)
                if photo is None or not surface.winfo_exists():
                    return
                item = getattr(surface, "_settings_background_item", None)
                if item is None:
                    item = surface.create_image(0, 0, anchor="nw", image=photo, tags="settings-surface-background")
                    setattr(surface, "_settings_background_item", item)
                else:
                    surface.itemconfigure(item, image=photo)
                surface.tag_lower(item)
                setattr(surface, "_settings_background_photo", photo)

            def schedule_paint(_event: Optional[tk.Event] = None) -> None:
                job = paint_pending["job"]
                if job is not None:
                    try:
                        surface.after_cancel(job)
                    except tk.TclError:
                        pass
                paint_pending["job"] = surface.after_idle(paint)

            surface.bind("<Configure>", schedule_paint, add="+")
            schedule_paint()

        class ArtworkLabel(tk.Canvas):
            """Canvas-backed text whose artwork does not affect requested size."""

            def __init__(self, parent: tk.Misc, **kwargs: Any) -> None:
                text = str(kwargs.pop("text", ""))
                self._textvariable = kwargs.pop("textvariable", None)
                self._text_color = kwargs.pop("fg", "#f2f5f6")
                self._text_font = kwargs.pop("font", ("Microsoft YaHei UI", 9))
                self._text_anchor = kwargs.pop("anchor", "center")
                self._wraplength = int(kwargs.pop("wraplength", 0) or 0)
                self._justify = kwargs.pop("justify", "center")
                kwargs.pop("bg", None)
                super().__init__(parent, width=1, height=20, bd=0, highlightthickness=0, bg="#07131f")
                if self._textvariable is not None:
                    text = str(self._textvariable.get())
                self._text_item = self.create_text(
                    0,
                    0,
                    text=text,
                    fill=self._text_color,
                    font=self._text_font,
                    anchor="w",
                    justify=self._justify,
                    width=self._wraplength if self._wraplength > 0 else 0,
                    tags="settings-label-text",
                )
                self._sync_requested_height()
                decorate_canvas(self, veil=0.0)
                self.bind("<Configure>", self._position_text, add="+")
                if self._textvariable is not None:
                    self._textvariable.trace_add("write", lambda *_args: self.configure(text=self._textvariable.get()))

            def _sync_requested_height(self) -> None:
                self.update_idletasks()
                bounds = self.bbox(self._text_item)
                height = max(18, (bounds[3] - bounds[1] + 4) if bounds else 20)
                width = max(1, (bounds[2] - bounds[0] + 4) if bounds else 1)
                tk.Canvas.configure(self, width=width, height=height)

            def _position_text(self, _event: Optional[tk.Event] = None) -> None:
                width = max(1, self.winfo_width())
                height = max(1, self.winfo_height())
                if self._text_anchor == "w":
                    x, anchor = 0, "w"
                elif self._text_anchor == "e":
                    x, anchor = width, "e"
                else:
                    x, anchor = width // 2, "center"
                self.coords(self._text_item, x, height // 2)
                self.itemconfigure(self._text_item, anchor=anchor)

            def configure(self, cnf: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
                options = dict(cnf or {})
                options.update(kwargs)
                redraw = False
                if "text" in options:
                    self.itemconfigure(self._text_item, text=str(options.pop("text")))
                    redraw = True
                if "fg" in options:
                    self._text_color = options.pop("fg")
                    self.itemconfigure(self._text_item, fill=self._text_color)
                options.pop("bg", None)
                result = tk.Canvas.configure(self, **options) if options else None
                if redraw:
                    self._sync_requested_height()
                    self._position_text()
                return result

            config = configure

        def transparent_label(parent: tk.Misc, **kwargs: Any) -> ArtworkLabel:
            return ArtworkLabel(parent, **kwargs)

        decorate_canvas(sidebar, veil=0.045)
        decorate_canvas(content_viewport, veil=0.015)

        def heading(title: str, subtitle: str) -> None:
            transparent_label(content, text=title, bg="#07131f", fg="#f2f5f6", anchor="w", font=("Microsoft YaHei UI", 19, "bold")).pack(fill="x")
            transparent_label(content, text=subtitle, bg="#07131f", fg="#94a9b6", anchor="w", font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(5, 21))

        def panel(parent: tk.Misc, *, pady: tuple[int, int] = (0, 14)) -> tk.Misc:
            surface = tk.Canvas(parent, bg="#07131f", highlightbackground="#496a7d", highlightthickness=1, bd=0)
            surface.pack(fill="x", pady=pady)
            setattr(surface, "_settings_panel", True)
            decorate_canvas(surface, veil=0.07)
            return surface

        def framed_panel(parent: tk.Misc, *, pady: tuple[int, int] = (0, 14)) -> tk.Misc:
            border = tk.Frame(parent, bg="#6f93a8", bd=0, highlightthickness=0)
            border.pack(fill="x", pady=pady)
            surface = tk.Canvas(border, bg="#07131f", highlightthickness=0, bd=0)
            surface.pack(fill="x", padx=1, pady=1)
            setattr(surface, "_settings_panel", True)
            decorate_canvas(surface, veil=0.07)
            return surface

        def action_button(parent: tk.Misc, text: str, command: Callable[[], None], *, primary: bool = False, danger: bool = False, width: int = 12) -> tk.Button:
            bg = "#155577" if primary else "#401f25" if danger else "#102534"
            active_bg = "#20749a" if primary else "#63313a" if danger else "#1a3a4d"
            return tk.Button(
                parent, text=text, command=command, width=width, relief="flat", bd=0,
                bg=bg, activebackground=active_bg, fg="#f2f5f6", activeforeground="#ffffff",
                cursor="hand2", font=("Microsoft YaHei UI", 9, "bold" if primary else "normal"),
                padx=12, pady=7,
            )

        def set_status(label: tk.Misc, message: str, *, warning: bool = False) -> None:
            label.configure(text=message, fg="#e1aa95" if warning else "#8fd0e8")

        def render_hotkeys() -> None:
            heading("自定义热键", "为识别、布局切换和楼层定位设置键盘或鼠标按键")
            rows = framed_panel(content, pady=(0, 14))
            status = transparent_label(content, text="点击右侧按键开始录入。", bg="#07131f", fg="#8fd0e8", anchor="w", font=("Microsoft YaHei UI", 9))
            value_buttons: dict[str, tk.Button] = {}

            def refresh_values() -> None:
                for action, button in value_buttons.items():
                    capturing = hotkey_capture["action"] == action
                    button.configure(
                        text="等待输入..." if capturing else hotkey_binding_display_name(hotkey_working[action]),
                        bg="#1b6689" if capturing else "#102b3c",
                    )

            def begin_capture(action: str) -> None:
                hotkey_capture["action"] = action
                self._hotkey_configuring = True
                self._hotkey_capture_action = action
                set_status(status, f"正在设置“{HOTKEY_LABELS[action]}”：请按键盘键，或点击下方鼠标按键。")
                refresh_values()
                dialog.focus_force()

            def conflict_for(action: str, binding: str) -> Optional[str]:
                identity = hotkey_binding_identity(binding)
                if identity is None:
                    return "无法识别该按键"
                fixed = {
                    hotkey_binding_identity("F2"): "停止 SCALE 搜索（F2）",
                    hotkey_binding_identity(VARIANT_RETURN_HOTKEY): "返回上一步（F3）",
                    ("key", 0x1B): "关闭或取消路线图虚像（Esc）",
                }
                if identity in fixed:
                    return fixed[identity]
                if identity == game_map_toggle_binding_identity(map_working["binding"]):
                    return f"游戏内打开/关闭地图（{game_map_toggle_display_name(map_working['binding'])}）"
                for owner, assigned in hotkey_working.items():
                    if owner != action and hotkey_binding_identity(assigned) == identity:
                        return HOTKEY_LABELS.get(owner, owner)
                return None

            def assign(binding: str) -> None:
                action = hotkey_capture["action"]
                if action is None:
                    set_status(status, "请先点击需要修改的功能按键。", warning=True)
                    return
                conflict = conflict_for(action, binding)
                if conflict is not None:
                    message = f"{hotkey_binding_display_name(binding)} 已用于“{conflict}”，请重新设置。"
                    set_status(status, message, warning=True)
                    messagebox.showwarning("热键冲突", message, parent=dialog)
                    return
                hotkey_working[action] = binding
                hotkey_capture["action"] = None
                self._hotkey_configuring = False
                self._hotkey_capture_action = None
                set_status(status, f"已设为 {hotkey_binding_display_name(binding)}，保存后立即生效。")
                refresh_values()

            setattr(dialog, "_settings_hotkey_assign", assign)
            for row, action in enumerate(HOTKEY_DEFAULTS):
                line = tk.Canvas(rows, bg="#0a1a27", height=52)
                line.pack(fill="x", padx=18, pady=(8 if row == 0 else 0, 0))
                line.pack_propagate(False)
                transparent_label(line, text=HOTKEY_LABELS[action], bg="#0a1a27", fg="#dce6eb", anchor="w", font=("Microsoft YaHei UI", 10)).pack(side="left")
                button = action_button(line, hotkey_binding_display_name(hotkey_working[action]), lambda item=action: begin_capture(item), width=20)
                button.pack(side="right", pady=7)
                value_buttons[action] = button
            bottom_guard = tk.Canvas(rows, bg="#0a1a27", height=12, bd=0, highlightthickness=0)
            bottom_guard.pack(fill="x")
            decorate_canvas(bottom_guard, veil=0.045)

            mouse_panel = panel(content)
            transparent_label(mouse_panel, text="鼠标按键", bg="#0a1a27", fg="#dce6eb", anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=18, pady=(13, 8))
            mouse_row = tk.Canvas(mouse_panel, bg="#0a1a27")
            mouse_row.pack(fill="x", padx=14, pady=(0, 14))
            for mouse_name in GAME_MAP_MOUSE_BINDINGS:
                button = action_button(mouse_row, GAME_MAP_MOUSE_BINDINGS[mouse_name][0], lambda value=mouse_name: assign(f"mouse:{value}"), width=12)
                button.pack(side="left", expand=True, fill="x", padx=4)
            status.pack(fill="x", pady=(0, 12))
            footer = tk.Canvas(content, bg="#07131f")
            footer.pack(fill="x", side="bottom")

            def reset() -> None:
                hotkey_working.clear()
                hotkey_working.update(HOTKEY_DEFAULTS)
                hotkey_capture["action"] = None
                refresh_values()
                set_status(status, "已恢复默认值，点击保存后生效。")

            def save() -> None:
                self._hotkeys = dict(hotkey_working)
                self.project_config.app["hotkeys"] = dict(hotkey_working)
                try:
                    save_app_config(self.project_config.app_config_path, self.project_config.app)
                except OSError as exc:
                    messagebox.showerror("保存失败", str(exc), parent=dialog)
                    return
                self._refresh_hotkey_dependent_ui()
                self.status_var.set("自定义热键已保存并立即生效")
                set_status(status, "已保存，新的热键现已生效。")

            action_button(footer, "恢复默认", reset).pack(side="left")
            action_button(footer, "保存更改", save, primary=True).pack(side="right")

        def render_map_key() -> None:
            heading(
                "游戏地图键",
                f"设置游戏内打开地图的按键；{configured_hotkey_display_name(self, 'locate_current_floor')} 定位时会自动调用",
            )
            status = transparent_label(content, text="选择鼠标按键，或录入一个键盘单键。", bg="#07131f", fg="#8fd0e8", anchor="w", font=("Microsoft YaHei UI", 9))
            selected = transparent_label(content, bg="#07131f", fg="#f0d4a4", anchor="w", font=("Microsoft YaHei UI", 11, "bold"))
            mouse_panel = panel(content)
            transparent_label(mouse_panel, text="鼠标按键", bg="#0a1a27", fg="#dce6eb", anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=18, pady=(16, 10))
            mouse_row = tk.Canvas(mouse_panel, bg="#0a1a27")
            mouse_row.pack(fill="x", padx=14, pady=(0, 16))
            buttons: dict[str, tk.Button] = {}

            def refresh() -> None:
                selected.configure(text=f"当前选择：{game_map_toggle_display_name(map_working['binding'])}")
                for binding, button in buttons.items():
                    button.configure(bg="#1b6689" if map_working["binding"].lower() == binding else "#102534")
                keyboard.configure(
                    text="等待键盘输入..." if map_capture["active"] else "录入键盘按键",
                    bg="#1b6689" if map_capture["active"] or map_working["binding"].startswith("key:") else "#102534",
                )

            def choose(binding: str) -> None:
                identity = game_map_toggle_binding_identity(binding)
                owner = next((action for action, assigned in hotkey_working.items() if identity is not None and hotkey_binding_identity(assigned) == identity), None)
                if owner is not None:
                    message = f"{game_map_toggle_display_name(binding)} 已用于“{HOTKEY_LABELS.get(owner, owner)}”，请重新设置。"
                    set_status(status, message, warning=True)
                    messagebox.showwarning("按键冲突", message, parent=dialog)
                    return
                map_working["binding"] = binding
                map_capture["active"] = False
                self._game_map_binding_capturing = False
                self._hotkey_configuring = False
                set_status(status, "选择完成，点击保存后立即生效。")
                refresh()

            setattr(dialog, "_settings_map_choose", choose)
            for binding in GAME_MAP_MOUSE_BINDINGS:
                button = action_button(mouse_row, GAME_MAP_MOUSE_BINDINGS[binding][0], lambda value=binding: choose(value), width=12)
                button.pack(side="left", expand=True, fill="x", padx=4)
                buttons[binding] = button
            keyboard_panel = panel(content)
            transparent_label(keyboard_panel, text="键盘按键", bg="#0a1a27", fg="#dce6eb", anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", padx=18, pady=(16, 10))

            def begin_keyboard() -> None:
                map_capture["active"] = True
                self._game_map_binding_capturing = True
                self._hotkey_configuring = True
                set_status(status, "请按下要用于游戏内打开地图的键盘按键...")
                refresh()
                dialog.focus_force()

            keyboard = action_button(keyboard_panel, "录入键盘按键", begin_keyboard, width=38)
            keyboard.pack(fill="x", padx=18, pady=(0, 18))
            selected.pack(fill="x", pady=(4, 8))
            status.pack(fill="x")

            def save() -> None:
                binding = map_working["binding"]
                self.project_config.app["game_map_toggle_button"] = binding
                try:
                    save_app_config(self.project_config.app_config_path, self.project_config.app)
                except OSError as exc:
                    messagebox.showerror("保存失败", str(exc), parent=dialog)
                    return
                display = game_map_toggle_display_name(binding)
                self.game_map_toggle_var.set(f"游戏地图键：{display}")
                self.status_var.set(f"游戏内打开地图按键已设置为：{display}")
                set_status(status, "已保存，新的游戏地图键现已生效。")

            action_button(content, "保存更改", save, primary=True).pack(anchor="e", side="bottom")
            refresh()

        def render_scale() -> None:
            heading("Scale 管理", "管理基准 Scale 与已确认分辨率的复用记录")
            profile_var = tk.StringVar(value=self.recognition_profile.key)
            profile_selector = tk.Canvas(content, bg="#07131f", height=44, highlightthickness=0, bd=0, cursor="hand2")
            profile_selector.pack(fill="x", pady=(0, 10))

            def profile_app_config() -> dict[str, Any]:
                return self._scale_storage_app_config(profile_var.get())

            def draw_profile_selector() -> None:
                profile_selector.delete("scale-profile")
                width = max(360, profile_selector.winfo_width())
                left = 4
                segment_width = max(120, (width - 8) // 2)
                for index, (key, label) in enumerate((("difficulty", "困难"), ("nightmare", "噩梦"))):
                    x1 = left + index * segment_width
                    x2 = width - 4 if index == 1 else x1 + segment_width
                    selected = profile_var.get() == key
                    profile_selector.create_rectangle(
                        x1,
                        4,
                        x2,
                        40,
                        fill="#80391a" if selected else "#261714",
                        outline="#ffc570" if selected else "#705746",
                        width=1,
                        tags="scale-profile",
                    )
                    profile_selector.create_text(
                        (x1 + x2) // 2,
                        22,
                        text=label,
                        fill="#fff3df" if selected else "#cbb8a3",
                        font=("Microsoft YaHei UI", 10, "bold" if selected else "normal"),
                        tags="scale-profile",
                    )

            def choose_profile(event: tk.Event) -> None:
                width = max(360, profile_selector.winfo_width())
                profile_var.set("difficulty" if event.x < width / 2 else "nightmare")
                selected_record[0] = None
                selected_index[0] = None
                map_var.set("")
                resolution_var.set("")
                scale_var.set("")
                set_identity_editable(True)
                status.configure(
                    text=(
                        "困难 Scale 表" if profile_var.get() == "difficulty" else "噩梦 Scale 表（地下室 / 一楼 / 二楼独立）"
                    )
                )
                draw_profile_selector()
                refresh()

            profile_selector.bind("<Button-1>", choose_profile)
            profile_selector.bind("<Configure>", lambda _event: draw_profile_selector())
            table_panel = panel(content, pady=(0, 12))
            table = tk.Canvas(table_panel, bg="#091522", height=286, highlightthickness=0, bd=0)
            table.pack(fill="x", padx=12, pady=12)
            decorate_canvas(table, veil=0.015)
            records: list[SavedScale] = []
            selected_record: list[Optional[SavedScale]] = [None]
            selected_index: list[Optional[int]] = [None]
            table_offset = [0]
            visible_rows = 10
            row_height = 25
            header_height = 31
            display_rows: list[tuple[str, Optional[str], Optional[int]]] = []
            scrollbar_drag: dict[str, Optional[float]] = {"start_y": None, "start_offset": None}
            columns = (
                ("类型", 104, "center"),
                ("布局", 112, "center"),
                ("ROI / 分辨率", 154, "center"),
                ("Scale", 116, "center"),
                ("来源 / 确认时间", 316, "w"),
            )
            edit = panel(content)
            fields = tk.Canvas(edit, bg="#07131f", height=78, highlightthickness=0, bd=0)
            fields.pack(fill="x", padx=14, pady=(13, 5))
            map_var, resolution_var, scale_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
            entries: list[tk.Entry] = []
            for column, (label, variable, width) in enumerate((("布局 ID", map_var, 22), ("ROI 尺寸", resolution_var, 18), ("Scale", scale_var, 18))):
                x = 0 if column == 0 else (218 if column == 1 else 398)
                fields.create_text(x, 4, anchor="nw", text=label, fill="#b8cbd4", font=("Microsoft YaHei UI", 8), tags="scale-field-label")
                entry = tk.Entry(
                    fields,
                    textvariable=variable,
                    width=width,
                    bg="#15344b",
                    fg="#edf6fa",
                    insertbackground="#edf6fa",
                    readonlybackground="#15344b",
                    relief="flat",
                    highlightthickness=1,
                    highlightbackground="#54788b",
                    highlightcolor="#73c8ed",
                    font=("Microsoft YaHei UI", 9),
                )
                fields.create_window(x, 24, anchor="nw", window=entry, width=(204 if column == 0 else 170), height=38)
                entries.append(entry)
            status = transparent_label(edit, text="选择记录可修改 Scale，或新增分辨率复用记录。", bg="#0a1a27", fg="#8fa6b4", anchor="w", font=("Microsoft YaHei UI", 8))
            status.pack(fill="x", padx=14, pady=(3, 9))
            button_row = tk.Canvas(edit, bg="#0a1a27")
            button_row.pack(fill="x", padx=14, pady=(0, 13))
            kind_labels = {"baseline": "基准 Scale", "confirmed": "分辨率复用", "legacy": "旧版对齐"}

            def cell_text(value: object, max_chars: int) -> str:
                text = str(value)
                return text if len(text) <= max_chars else text[: max(1, max_chars - 1)] + "…"

            def rebuild_display_rows() -> None:
                display_rows.clear()
                groups: dict[str, list[int]] = {}
                for index, record in enumerate(records):
                    groups.setdefault(scale_client_resolution(profile_app_config(), record), []).append(index)
                for client_resolution in sorted(groups, key=lambda value: (value == "unknown", value)):
                    display_rows.append(("group", client_resolution, None))
                    display_rows.extend(("record", None, index) for index in groups[client_resolution])

            def scrollbar_geometry() -> Optional[tuple[int, int, int, int]]:
                if len(display_rows) <= visible_rows:
                    return None
                width = max(760, table.winfo_width())
                track_top = header_height + 3
                track_bottom = header_height + visible_rows * row_height - 3
                track_height = max(1, track_bottom - track_top)
                thumb_height = max(26, int((visible_rows / len(display_rows)) * track_height))
                max_offset = max(1, len(display_rows) - visible_rows)
                thumb_top = track_top + int((table_offset[0] / max_offset) * (track_height - thumb_height))
                return width, track_top, track_bottom, thumb_top

            def set_table_offset(offset: int) -> None:
                table_offset[0] = max(0, min(max(0, len(display_rows) - visible_rows), int(offset)))
                draw_table()

            def draw_table() -> None:
                table.delete("scale-table")
                width = max(760, table.winfo_width())
                usable_width = max(100, width - 22)
                total_column_width = sum(column[1] for column in columns)
                scale = usable_width / max(1, total_column_width)
                x = 8
                table.create_line(8, header_height, width - 10, header_height, fill="#4a6a7d", width=1, tags="scale-table")
                for title, base_width, anchor in columns:
                    column_width = int(round(base_width * scale))
                    text_x = x + (column_width // 2 if anchor == "center" else 8)
                    table.create_text(
                        text_x,
                        16,
                        text=title,
                        anchor="center" if anchor == "center" else "w",
                        fill="#dfeaf0",
                        font=("Microsoft YaHei UI", 9, "bold"),
                        tags="scale-table",
                    )
                    x += column_width
                start = table_offset[0]
                visible = display_rows[start : start + visible_rows]
                for row, (row_kind, client_resolution, record_index) in enumerate(visible):
                    y1 = header_height + row * row_height
                    y2 = y1 + row_height
                    if row_kind == "group":
                        table.create_rectangle(8, y1 + 1, width - 18, y2 - 1, fill="#12364a", outline="", tags="scale-table")
                        table.create_text(18, y1 + row_height // 2, anchor="w", text=f"client {client_resolution}", fill="#a9d9eb", font=("Microsoft YaHei UI", 9, "bold"), tags="scale-table")
                        continue
                    if record_index is None:
                        continue
                    index = record_index
                    record = records[index]
                    if selected_index[0] == index:
                        table.create_rectangle(8, y1 + 1, width - 18, y2 - 1, fill="#184c65", outline="", tags="scale-table")
                    else:
                        table.create_line(8, y2, width - 18, y2, fill="#203948", width=1, tags="scale-table")
                    values = (
                        kind_labels.get(record.kind, record.kind),
                        record.map_id,
                        scale_resolution_display(profile_app_config(), record),
                        f"{record.scale:.12g}",
                        record.detail,
                    )
                    x = 8
                    for value, (_title, base_width, anchor) in zip(values, columns):
                        column_width = int(round(base_width * scale))
                        max_chars = max(8, column_width // 8)
                        text_x = x + (column_width // 2 if anchor == "center" else 8)
                        table.create_text(
                            text_x,
                            y1 + row_height // 2,
                            text=cell_text(value, max_chars),
                            anchor="center" if anchor == "center" else "w",
                            fill="#eef7fa",
                            font=("Microsoft YaHei UI", 9),
                            tags="scale-table",
                        )
                        x += column_width
                geometry = scrollbar_geometry()
                if geometry is not None:
                    _width, track_top, track_bottom, thumb_top = geometry
                    table.create_line(width - 8, track_top, width - 8, track_bottom, fill="#496172", width=2, tags="scale-table")
                    thumb_height = max(26, int((visible_rows / len(display_rows)) * (track_bottom - track_top)))
                    table.create_rectangle(width - 12, thumb_top, width - 4, thumb_top + thumb_height, fill="#7fb9d0", outline="", tags="scale-table")

            def select_current(index: int) -> None:
                if index < 0 or index >= len(records):
                    return
                record = records[index]
                selected_record[0] = record
                selected_index[0] = index
                map_var.set(record.map_id)
                resolution_var.set(record.resolution)
                scale_var.set(f"{record.scale:.12g}")
                set_identity_editable(False)
                status.configure(text=f"正在编辑：{kind_labels.get(record.kind, record.kind)}；仅修改 Scale 值")
                draw_table()

            def handle_table_click(event: tk.Event) -> None:
                geometry = scrollbar_geometry()
                if geometry is not None and event.x >= max(0, int(geometry[0]) - 24):
                    _width, track_top, track_bottom, thumb_top = geometry
                    thumb_height = max(26, int((visible_rows / len(display_rows)) * (track_bottom - track_top)))
                    if thumb_top <= event.y <= thumb_top + thumb_height:
                        scrollbar_drag["start_y"] = float(event.y)
                        scrollbar_drag["start_offset"] = float(table_offset[0])
                    elif track_top <= event.y <= track_bottom:
                        page = visible_rows - 1
                        set_table_offset(table_offset[0] + (page if event.y > thumb_top else -page))
                    return
                if event.y < header_height:
                    return
                row_index = table_offset[0] + (event.y - header_height) // row_height
                if 0 <= row_index < len(display_rows):
                    row_kind, _client_resolution, record_index = display_rows[row_index]
                    if row_kind == "record" and record_index is not None:
                        select_current(record_index)

            def handle_table_drag(event: tk.Event) -> None:
                if scrollbar_drag["start_y"] is None or scrollbar_drag["start_offset"] is None:
                    return
                geometry = scrollbar_geometry()
                if geometry is None:
                    return
                _width, track_top, track_bottom, thumb_top = geometry
                thumb_height = max(26, int((visible_rows / len(display_rows)) * (track_bottom - track_top)))
                travel = max(1, track_bottom - track_top - thumb_height)
                max_offset = max(1, len(display_rows) - visible_rows)
                delta = (float(event.y) - scrollbar_drag["start_y"]) / travel * max_offset
                set_table_offset(round(scrollbar_drag["start_offset"] + delta))

            def handle_table_release(_event: tk.Event) -> None:
                scrollbar_drag["start_y"] = None
                scrollbar_drag["start_offset"] = None

            def handle_table_scroll(event: tk.Event) -> str:
                if len(display_rows) <= visible_rows:
                    return "break"
                direction = -1 if event.delta > 0 else 1
                set_table_offset(table_offset[0] + direction)
                return "break"

            def set_identity_editable(enabled: bool) -> None:
                state = "normal" if enabled else "readonly"
                entries[0].configure(state=state)
                entries[1].configure(state=state)

            def refresh(select_record: Optional[SavedScale] = None) -> None:
                records.clear()
                records.extend(saved_scale_records(self.project_config.root_dir, profile_app_config()))
                rebuild_display_rows()
                selected_index[0] = None
                if table_offset[0] > max(0, len(display_rows) - visible_rows):
                    table_offset[0] = max(0, len(display_rows) - visible_rows)
                for index, record in enumerate(records):
                    if select_record is not None and record == select_record:
                        selected_index[0] = index
                        display_index = next(
                            (row_index for row_index, row in enumerate(display_rows) if row[0] == "record" and row[2] == index),
                            0,
                        )
                        if display_index < table_offset[0]:
                            table_offset[0] = display_index
                        elif display_index >= table_offset[0] + visible_rows:
                            table_offset[0] = max(0, display_index - visible_rows + 1)
                        break
                draw_table()

            def new_record() -> None:
                selected_record[0] = None
                selected_index[0] = None
                map_var.set("")
                resolution_var.set("")
                scale_var.set("")
                set_identity_editable(True)
                status.configure(text="新增：按精确 ROI 尺寸创建分辨率复用 Scale")
                entries[0].focus_set()
                draw_table()

            def save_current() -> None:
                try:
                    value = float(scale_var.get().strip())
                    normalized_map_id = map_var.get().strip().lower()
                    expected_prefix = "nightmare_" if profile_var.get() == "nightmare" else "map_"
                    if selected_record[0] is None and not normalized_map_id.startswith(expected_prefix):
                        raise ValueError(f"当前表格只允许 {expected_prefix} 开头的布局 ID")
                    if selected_record[0] is None:
                        saved = add_confirmed_scale(self.project_config.root_dir, profile_app_config(), map_var.get(), resolution_var.get(), value)
                    else:
                        old = selected_record[0]
                        update_saved_scale(self.project_config.root_dir, profile_app_config(), old, value)
                        saved = SavedScale(old.kind, old.map_id, old.resolution, value, old.detail)
                except (OSError, TypeError, ValueError) as exc:
                    messagebox.showerror("Scale 无法保存", str(exc), parent=dialog)
                    return
                self._semantic_cv_scale_cache.clear()
                self._semantic_cv_device_multiplier_cache.clear()
                selected_record[0] = saved
                self.status_var.set(f"已保存 {saved.map_id} / {saved.resolution} 的 Scale：{value:.12g}")
                refresh(saved)

            def delete_current() -> None:
                record = selected_record[0]
                if record is None:
                    return
                if not messagebox.askyesno("删除 Scale", f"删除 {record.map_id} / {record.resolution} 的记录？", parent=dialog):
                    return
                try:
                    delete_saved_scale(self.project_config.root_dir, profile_app_config(), record)
                except (OSError, ValueError) as exc:
                    messagebox.showerror("Scale 无法删除", str(exc), parent=dialog)
                    return
                self._semantic_cv_scale_cache.clear()
                self._semantic_cv_device_multiplier_cache.clear()
                new_record()
                refresh()

            table.bind("<Button-1>", handle_table_click)
            table.bind("<B1-Motion>", handle_table_drag)
            table.bind("<ButtonRelease-1>", handle_table_release)
            table.bind("<MouseWheel>", handle_table_scroll)
            table.bind("<Configure>", lambda _event: draw_table())
            action_button(button_row, "新增记录", new_record).pack(side="left")
            action_button(button_row, "删除选中", delete_current, danger=True).pack(side="right", padx=(8, 0))
            action_button(button_row, "保存", save_current, primary=True).pack(side="right")
            draw_profile_selector()
            refresh()

        def render_cache() -> None:
            heading("清理识别缓存", "删除识别和定位过程中生成的调试图片，不影响模板、Scale 或按键设置")
            cache_panel = panel(content)
            count_var = tk.StringVar()
            size_var = tk.StringVar()

            def update_stats() -> None:
                root = self.project_config.root_dir / "assets" / "debug"
                files = [path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"] if root.is_dir() else []
                total = sum(path.stat().st_size for path in files if path.exists())
                count_var.set(f"{len(files)} 个文件")
                size_var.set(f"{total / (1024 * 1024):.2f} MB")
                try:
                    stats_exists = stats.winfo_exists()
                except NameError:
                    stats_exists = False
                if stats_exists:
                    draw_cache_stats()

            transparent_label(cache_panel, text="当前缓存", bg="#0a1a27", fg="#dce6eb", anchor="w", font=("Microsoft YaHei UI", 11, "bold")).pack(fill="x", padx=20, pady=(18, 14))
            stats = tk.Canvas(cache_panel, bg="#0a1a27", height=86, highlightthickness=0, bd=0)
            stats.pack(fill="x", padx=20, pady=(0, 22))
            decorate_canvas(stats, veil=0.035)

            def draw_cache_stats(_event: tk.Event | None = None) -> None:
                stats.delete("cache-stat")
                width = max(stats.winfo_width(), 560)
                split_x = width // 2
                stats.create_line(split_x, 14, split_x, 72, fill="#5d7c8e", width=1, tags="cache-stat")
                stats.create_text(16, 24, text="文件数量", anchor="w", fill="#9fb4c1", font=("Microsoft YaHei UI", 9), tags="cache-stat")
                stats.create_text(16, 55, text=count_var.get(), anchor="w", fill="#f1d49c", font=("Microsoft YaHei UI", 16, "bold"), tags="cache-stat")
                stats.create_text(split_x + 24, 24, text="占用空间", anchor="w", fill="#9fb4c1", font=("Microsoft YaHei UI", 9), tags="cache-stat")
                stats.create_text(split_x + 24, 55, text=size_var.get(), anchor="w", fill="#f1d49c", font=("Microsoft YaHei UI", 16, "bold"), tags="cache-stat")

            stats.bind("<Configure>", draw_cache_stats, add="+")
            transparent_label(content, text="清理后无法恢复，但不会删除干净模板、路线图、识别样本和配置文件。", bg="#07131f", fg="#a9bac4", anchor="w", font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(2, 18))

            def clean() -> None:
                if not messagebox.askyesno("清理识别缓存", "确定删除全部识别与定位调试缓存？", parent=dialog):
                    return
                self.clear_recognition_cache()
                update_stats()

            action_button(content, "清理识别缓存", clean, danger=True, width=18).pack(anchor="w")
            update_stats()

        def render_route() -> None:
            heading("路线图版本选择", "困难与噩梦模式独立配置，小抄会按当前难度自动切换")
            current = str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower()
            choice = tk.StringVar(value="zhanshi" if current == "zhanshi" else "lianghapi")
            configured_mode = str(self.project_config.app.get("lianghapi_cheatsheet_mode", "single")).strip().lower()
            display_mode = tk.StringVar(value="dual" if configured_mode == "dual" else "single")
            configured_nightmare_mode = str(self.project_config.app.get("nightmare_cheatsheet_mode", "single")).strip().lower()
            nightmare_mode = tk.StringVar(value="triple" if configured_nightmare_mode == "triple" else "single")
            nightmare_lianghapi_mode = tk.StringVar(
                value=nightmare_lianghapi_cheatsheet_mode(self.project_config.app)
            )
            nightmare_version = tk.StringVar(value=nightmare_route_map_version(self.project_config.app))
            nightmare_zhanshi_variant = tk.StringVar(value=nightmare_zhanshi_route_variant(self.project_config.app))
            difficulty_zhanshi_variant = tk.StringVar(value=difficulty_zhanshi_route_variant(self.project_config.app))
            status = transparent_label(
                content,
                text="",
                bg="#07131f",
                fg="#8fd0e8",
                anchor="w",
                font=("Microsoft YaHei UI", 9),
            )

            def status_text(saved: bool = False) -> str:
                difficult_mode = "单图" if display_mode.get() == "single" else "双图"
                nightmare_text = (
                    f"展十老师版本 / {'全棺版' if nightmare_zhanshi_variant.get() == 'full_coffin' else '速刷版'}"
                    if nightmare_version.get() == "zhanshi"
                    else f"凉哈皮老师版本 / {'单图' if nightmare_lianghapi_mode.get() == 'single' else '三图'}"
                    if nightmare_version.get() == "lianghapi"
                    else f"干净模板 / {'单图' if nightmare_mode.get() == 'single' else '三图'}"
                )
                suffix = "（已保存）" if saved else ""
                return (
                    f"困难：{self._route_map_version_name()} / {'速刷版' if difficulty_zhanshi_variant.get() == 'fast' else '全棺版'} / {difficult_mode}    "
                    f"噩梦：{nightmare_text}{suffix}"
                )

            set_status(status, status_text())

            def save_choice() -> None:
                value = choice.get()
                previous = str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower()
                previous_mode = self.project_config.app.get("lianghapi_cheatsheet_mode", "single")
                previous_nightmare_mode = self.project_config.app.get("nightmare_cheatsheet_mode", "single")
                previous_nightmare_lianghapi_mode = self.project_config.app.get("nightmare_lianghapi_cheatsheet_mode", "triple")
                previous_nightmare_version = nightmare_route_map_version(self.project_config.app)
                previous_nightmare_zhanshi_variant = nightmare_zhanshi_route_variant(self.project_config.app)
                previous_difficulty_zhanshi_variant = difficulty_zhanshi_route_variant(self.project_config.app)
                self.project_config.app["route_map_version"] = value
                self.project_config.app["lianghapi_cheatsheet_mode"] = display_mode.get()
                self.project_config.app["nightmare_cheatsheet_mode"] = nightmare_mode.get()
                self.project_config.app["nightmare_lianghapi_cheatsheet_mode"] = nightmare_lianghapi_mode.get()
                self.project_config.app["nightmare_route_map_version"] = nightmare_version.get()
                self.project_config.app["nightmare_zhanshi_route_variant"] = nightmare_zhanshi_variant.get()
                self.project_config.app["difficulty_zhanshi_route_variant"] = difficulty_zhanshi_variant.get()
                try:
                    save_app_config(self.project_config.app_config_path, self.project_config.app)
                except OSError as exc:
                    self.project_config.app["route_map_version"] = previous
                    self.project_config.app["lianghapi_cheatsheet_mode"] = previous_mode
                    self.project_config.app["nightmare_cheatsheet_mode"] = previous_nightmare_mode
                    self.project_config.app["nightmare_lianghapi_cheatsheet_mode"] = previous_nightmare_lianghapi_mode
                    self.project_config.app["nightmare_route_map_version"] = previous_nightmare_version
                    self.project_config.app["nightmare_zhanshi_route_variant"] = previous_nightmare_zhanshi_variant
                    self.project_config.app["difficulty_zhanshi_route_variant"] = previous_difficulty_zhanshi_variant
                    choice.set(previous)
                    messagebox.showerror("保存失败", str(exc), parent=dialog)
                    return
                self._semantic_cv_image_cache.clear()
                self._semantic_cv_route_crop_cache.clear()
                self._semantic_cv_affine_cache.clear()
                self._semantic_cv_piecewise_route_cache.clear()
                self._reset_route_overlay_for_route_version_change()
                self._render_main_window_controls()
                if self._side_door_structure_confirmed and self._active_route_layout_id() is not None:
                    self._show_active_route_layout()
                set_status(status, status_text(saved=True))
                self.status_var.set("困难/噩梦小抄显示设置已保存。")

            difficult_panel = panel(content)
            difficult_panel.configure(height=270, cursor="hand2")
            difficult_panel.pack_propagate(False)
            difficult_panel.pack(fill="x", pady=(0, 12))
            nightmare_panel = panel(content)
            nightmare_panel.configure(height=342, cursor="hand2")
            nightmare_panel.pack_propagate(False)
            nightmare_panel.pack(fill="x", pady=(0, 8))

            def draw_option(canvas: tk.Canvas, value: str, selected: bool, label: str, y: int, *, indent: int = 18) -> None:
                canvas.create_oval(
                    indent, y - 7, indent + 14, y + 7,
                    outline="#9ed7ee" if selected else "#668594",
                    width=1,
                    tags="route-option",
                )
                if selected:
                    canvas.create_oval(indent + 4, y - 3, indent + 10, y + 3, fill="#8fd5f2", outline="", tags="route-option")
                canvas.create_text(
                    indent + 26, y, anchor="w", text=label,
                    fill="#f0f6f8" if selected else "#aabdc7",
                    font=("Microsoft YaHei UI", 10 if indent == 18 else 9),
                    tags="route-option",
                )

            def draw_route_options() -> None:
                difficult_panel.delete("route-option")
                nightmare_panel.delete("route-option")
                difficult_panel.create_text(18, 24, anchor="w", text="困难模式", fill="#f1d49c", font=("Microsoft YaHei UI", 11, "bold"), tags="route-option")
                difficult_panel.create_text(104, 24, anchor="w", text="路线图版本与画面组合", fill="#7896a6", font=("Microsoft YaHei UI", 9), tags="route-option")
                legacy_selected = choice.get() == "lianghapi"
                draw_option(difficult_panel, "lianghapi", legacy_selected, "凉哈皮老师版本（旧版路线图）", 62)
                difficult_panel.create_text(
                    44, 92, anchor="w", text="凉哈皮版本的小抄显示模式",
                    fill="#dcebf0" if legacy_selected else "#668594",
                    font=("Microsoft YaHei UI", 9), tags="route-option",
                )
                for value, label, y in (("single", "单图：仅显示当前所在楼层", 120), ("dual", "双图：同时显示一楼、二楼", 150)):
                    selected = legacy_selected and display_mode.get() == value
                    draw_option(difficult_panel, value, selected, label, y, indent=48)
                draw_option(
                    difficult_panel,
                    "zhanshi",
                    choice.get() == "zhanshi",
                    "展十老师版本（一二楼合并路线图）",
                    190,
                )
                zhanshi_selected = choice.get() == "zhanshi"
                draw_option(difficult_panel, "full_coffin", zhanshi_selected and difficulty_zhanshi_variant.get() == "full_coffin", "全棺版", 220, indent=48)
                draw_option(difficult_panel, "fast", zhanshi_selected and difficulty_zhanshi_variant.get() == "fast", "速刷版", 250, indent=48)

                nightmare_panel.create_text(18, 24, anchor="w", text="噩梦模式", fill="#f1d49c", font=("Microsoft YaHei UI", 11, "bold"), tags="route-option")
                nightmare_panel.create_text(104, 24, anchor="w", text="小抄与虚像共用同一版本", fill="#7896a6", font=("Microsoft YaHei UI", 9), tags="route-option")
                clean_selected = nightmare_version.get() == "clean_template"
                draw_option(nightmare_panel, "clean_template", clean_selected, "干净模板底图", 62)
                draw_option(nightmare_panel, "single", clean_selected and nightmare_mode.get() == "single", "单图：仅显示当前所在楼层", 94, indent=48)
                draw_option(nightmare_panel, "triple", clean_selected and nightmare_mode.get() == "triple", "三图：竖向同时显示地下室、一楼、二楼", 126, indent=48)
                zhanshi_selected = nightmare_version.get() == "zhanshi"
                draw_option(nightmare_panel, "zhanshi", zhanshi_selected, "展十老师版本（三层合并路线图）", 160)
                draw_option(nightmare_panel, "fast", zhanshi_selected and nightmare_zhanshi_variant.get() == "fast", "速刷版", 192, indent=48)
                draw_option(nightmare_panel, "full_coffin", zhanshi_selected and nightmare_zhanshi_variant.get() == "full_coffin", "全棺版", 224, indent=48)
                lianghapi_selected = nightmare_version.get() == "lianghapi"
                draw_option(nightmare_panel, "lianghapi", lianghapi_selected, "凉哈皮老师版本（三层合并路线图）", 254)
                draw_option(nightmare_panel, "single", lianghapi_selected and nightmare_lianghapi_mode.get() == "single", "单图：仅显示当前所在楼层", 286, indent=48)
                draw_option(nightmare_panel, "triple", lianghapi_selected and nightmare_lianghapi_mode.get() == "triple", "三图：显示现有三层合并路线图", 316, indent=48)

            def choose_difficulty(event: tk.Event) -> None:
                if event.y < 80:
                    choice.set("lianghapi")
                elif choice.get() == "lianghapi" and event.y < 136:
                    display_mode.set("single")
                elif choice.get() == "lianghapi" and event.y < 168:
                    display_mode.set("dual")
                elif event.y < 210:
                    choice.set("zhanshi")
                elif event.y < 236:
                    choice.set("zhanshi")
                    difficulty_zhanshi_variant.set("full_coffin")
                elif event.y < 266:
                    choice.set("zhanshi")
                    difficulty_zhanshi_variant.set("fast")
                else:
                    return
                draw_route_options()
                save_choice()

            def choose_nightmare(event: tk.Event) -> None:
                if event.y < 78:
                    nightmare_version.set("clean_template")
                elif event.y < 110 and nightmare_version.get() == "clean_template":
                    nightmare_mode.set("single")
                elif event.y < 144 and nightmare_version.get() == "clean_template":
                    nightmare_mode.set("triple")
                elif event.y < 178:
                    nightmare_version.set("zhanshi")
                elif event.y < 208:
                    nightmare_version.set("zhanshi")
                    nightmare_zhanshi_variant.set("fast")
                elif event.y < 240:
                    nightmare_version.set("zhanshi")
                    nightmare_zhanshi_variant.set("full_coffin")
                elif event.y < 274:
                    nightmare_version.set("lianghapi")
                elif event.y < 302:
                    nightmare_version.set("lianghapi")
                    nightmare_lianghapi_mode.set("single")
                else:
                    nightmare_version.set("lianghapi")
                    nightmare_lianghapi_mode.set("triple")
                draw_route_options()
                save_choice()

            difficult_panel.bind("<Button-1>", choose_difficulty)
            nightmare_panel.bind("<Button-1>", choose_nightmare)
            draw_route_options()

        renderers = {
            "hotkeys": render_hotkeys,
            "map_key": render_map_key,
            "scale": render_scale,
            "cache": render_cache,
            "route": render_route,
        }

        def redraw_sidebar_navigation() -> None:
            sidebar.delete("settings-nav")
            nav_buttons.clear()
            for index, (key, label, _icon_name) in enumerate(sections):
                y1 = 70 + index * 54
                y2 = y1 + 48
                selected = active["key"] == key
                if selected:
                    photo = self._make_main_glass_photo(
                        188,
                        48,
                        fill=(18, 72, 99, 118),
                        outline=(96, 184, 222, 98),
                        radius=3,
                        master=sidebar,
                    )
                    nav_photos.append(photo)
                    sidebar.create_image(5, y1, anchor="nw", image=photo, tags=("settings-nav", f"settings-nav-{key}"))
                else:
                    photo = self._make_main_glass_photo(
                        188,
                        48,
                        fill=(4, 13, 23, 26),
                        outline=(0, 0, 0, 0),
                        radius=3,
                        master=sidebar,
                    )
                    nav_photos.append(photo)
                    sidebar.create_image(5, y1, anchor="nw", image=photo, tags=("settings-nav", f"settings-nav-{key}"))
                icon = icons.get(key)
                if icon is not None:
                    sidebar.create_image(25, y1 + 24, anchor="center", image=icon, tags=("settings-nav", f"settings-nav-{key}"))
                sidebar.create_text(
                    50,
                    y1 + 24,
                    anchor="w",
                    text=label,
                    fill="#ffffff" if selected else "#c7d3d9",
                    font=("Microsoft YaHei UI", 9, "bold" if selected else "normal"),
                    tags=("settings-nav", f"settings-nav-{key}"),
                )
                nav_buttons[key] = (5, y1, 193, y2)
                sidebar.tag_bind(f"settings-nav-{key}", "<Button-1>", lambda _event, value=key: select_section(value))
                sidebar.tag_bind(f"settings-nav-{key}", "<Enter>", lambda _event: sidebar.configure(cursor="hand2"))
                sidebar.tag_bind(f"settings-nav-{key}", "<Leave>", lambda _event: sidebar.configure(cursor=""))

        def decorate_descendant_canvases(widget: tk.Misc) -> None:
            for child in widget.winfo_children():
                if isinstance(child, tk.Canvas) and child not in {sidebar, content}:
                    if not getattr(child, "_settings_no_artwork", False) and not getattr(child, "_settings_panel", False):
                        child.configure(highlightthickness=0, bd=0)
                    if not getattr(child, "_settings_no_artwork", False):
                        if not hasattr(child, "_settings_background_photo"):
                            decorate_canvas(child, veil=0.045)
                decorate_descendant_canvases(child)

        def select_section(key: str) -> None:
            active["key"] = key
            redraw_sidebar_navigation()
            clear_content()
            content_viewport.yview_moveto(0.0)
            renderers[key]()
            dialog.after_idle(
                lambda: (
                    update_content_scroll_region(),
                    paint_content_background(),
                    decorate_descendant_canvases(content),
                )
            )

        def handle_key(event: tk.Event) -> str:
            if active["key"] == "hotkeys" and hotkey_capture["action"] is not None:
                try:
                    virtual_key = int(event.keycode)
                except (TypeError, ValueError):
                    virtual_key = 0
                if 1 <= virtual_key <= 255:
                    label = self._keyboard_key_display_label(str(event.keysym)).replace(":", " ")
                    assign = getattr(dialog, "_settings_hotkey_assign", None)
                    if callable(assign):
                        assign(f"key:{virtual_key}:{label}")
                return "break"
            if active["key"] == "map_key" and map_capture["active"]:
                try:
                    virtual_key = int(event.keycode)
                except (TypeError, ValueError):
                    virtual_key = 0
                if 1 <= virtual_key <= 255:
                    conflict = self._game_map_keyboard_conflict(str(event.keysym), virtual_key)
                    if conflict is not None:
                        messagebox.showwarning("按键冲突", f"该键已用于“{conflict}”，请重新设置。", parent=dialog)
                    else:
                        label = self._keyboard_key_display_label(str(event.keysym)).replace(":", " ")
                        choose = getattr(dialog, "_settings_map_choose", None)
                        if callable(choose):
                            choose(f"key:{virtual_key}:{label}")
                return "break"
            return ""

        def close() -> None:
            clear_content()
            for sequence, binding_id in zip(
                ("<MouseWheel>", "<Button-4>", "<Button-5>"),
                scroll_binding_ids,
            ):
                try:
                    dialog.unbind(sequence, binding_id)
                except tk.TclError:
                    pass
            self._settings_window = None
            dialog.destroy()

        dialog.bind("<KeyPress>", handle_key)
        dialog.bind("<Escape>", lambda _event: close())
        dialog.protocol("WM_DELETE_WINDOW", close)
        select_section(active["key"])
        dialog.lift()
        dialog.focus_force()

    def open_scale_manager(self) -> None:
        """Open the development-only editor for every persisted scale source."""
        if self._scale_manager_window is not None and self._scale_manager_window.winfo_exists():
            self._scale_manager_window.lift()
            self._scale_manager_window.focus_force()
            return

        dialog = tk.Toplevel(self)
        dialog.title("Scale 管理")
        dialog.transient(self)
        dialog.grab_set()
        self._scale_manager_window = dialog
        canvas = self._create_premium_settings_canvas(dialog, width=1040, height=720)
        photos = getattr(canvas, "_premium_control_photos", [])
        self._draw_premium_dialog_header(
            canvas,
            "Scale 管理",
            "基准 scale 与已确认分辨率复用记录",
        )
        self._draw_premium_dialog_surface(canvas, photos, (34, 92, 1006, 428), fill=(3, 12, 23, 155))
        self._draw_premium_dialog_surface(canvas, photos, (34, 448, 1006, 652), fill=(3, 12, 23, 155))
        canvas.create_text(
            54, 112, anchor="w", text="已保存记录", fill="#d9e7ef", font=("Microsoft YaHei UI", 10, "bold")
        )
        record_count_item = canvas.create_text(
            986,
            112,
            anchor="e",
            text="加载中",
            fill="#9fb2c0",
            font=("Microsoft YaHei UI", 9),
        )
        canvas.create_text(
            54,
            467,
            anchor="w",
            text="编辑选中记录，或新增一个按 ROI 尺寸保存的复用 scale",
            fill="#d9e7ef",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        canvas.create_text(
            54,
            675,
            anchor="w",
            text="按客户区分辨率分类：基准 scale 适用于参考分辨率，复用 scale 适用于已保存的 ROI 尺寸。",
            fill="#9fb2c0",
            font=("Microsoft YaHei UI", 9),
        )

        style = ttk.Style(dialog)
        style.configure(
            "ScaleManager.Treeview",
            background="#091522",
            fieldbackground="#091522",
            foreground="#e6f0f4",
            rowheight=27,
            borderwidth=0,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "ScaleManager.Treeview.Heading",
            background="#14283a",
            foreground="#d9e7ef",
            relief="flat",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map("ScaleManager.Treeview", background=[("selected", "#1e5874")])
        table_frame = tk.Frame(canvas, bg="#091522", highlightbackground="#40637a", highlightthickness=1)
        tree = ttk.Treeview(
            table_frame,
            columns=("kind", "layout", "resolution", "scale", "detail"),
            show="tree headings",
            style="ScaleManager.Treeview",
            selectmode="browse",
        )
        tree.heading("#0", text="客户区分辨率")
        tree.column("#0", width=155, anchor="w", stretch=False)
        columns = {
            "kind": ("类型", 125),
            "layout": ("布局", 150),
            "resolution": ("ROI / 分辨率", 150),
            "scale": ("Scale", 130),
            "detail": ("来源 / 确认时间", 375),
        }
        for key, (label, width) in columns.items():
            tree.heading(key, text=label)
            tree.column(key, width=width, anchor="center" if key != "detail" else "w", stretch=key == "detail")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.create_window(54, 130, anchor="nw", window=table_frame, width=932, height=276)

        kind_labels = {"baseline": "基准 scale", "confirmed": "分辨率复用", "legacy": "旧版对齐"}
        records_by_item: dict[str, SavedScale] = {}
        selected_record: list[Optional[SavedScale]] = [None]
        map_var = tk.StringVar()
        resolution_var = tk.StringVar()
        scale_var = tk.StringVar()
        mode_item = canvas.create_text(
            54,
            586,
            anchor="w",
            text="选择一条记录后可修改其 Scale",
            fill="#9fb2c0",
            font=("Microsoft YaHei UI", 9),
        )

        def field(label: str, variable: tk.StringVar, x: int, width: int) -> tk.Entry:
            canvas.create_text(x, 511, anchor="w", text=label, fill="#afc4d1", font=("Microsoft YaHei UI", 9))
            entry = tk.Entry(
                dialog,
                textvariable=variable,
                bg="#0b1b29",
                readonlybackground="#0b1b29",
                fg="#edf6fa",
                insertbackground="#edf6fa",
                relief="flat",
                highlightthickness=1,
                highlightbackground="#4f7184",
                highlightcolor="#73c8ed",
                font=("Microsoft YaHei UI", 10),
            )
            canvas.create_window(x, 529, anchor="nw", window=entry, width=width, height=31)
            return entry

        map_entry = field("布局 ID", map_var, 54, 230)
        resolution_entry = field("ROI 尺寸", resolution_var, 306, 190)
        scale_entry = field("Scale", scale_var, 518, 180)
        def set_mode(message: str) -> None:
            canvas.itemconfigure(mode_item, text=message)

        def set_identity_editable(enabled: bool) -> None:
            state = "normal" if enabled else "readonly"
            map_entry.configure(state=state)
            resolution_entry.configure(state=state)

        def refresh(select_record: Optional[SavedScale] = None) -> None:
            records_by_item.clear()
            for item_id in tree.get_children():
                tree.delete(item_id)
            records = saved_scale_records(self.project_config.root_dir, self.project_config.app)
            canvas.itemconfigure(record_count_item, text=f"{len(records)} 条记录")
            groups: dict[str, list[SavedScale]] = {}
            for record in records:
                groups.setdefault(scale_client_resolution(self.project_config.app, record), []).append(record)
            row_index = 0
            for client_resolution, group_records in groups.items():
                group_id = f"resolution-{client_resolution}"
                tree.insert("", "end", iid=group_id, text=f"client {client_resolution}", open=True)
                for record in group_records:
                    item_id = f"scale-{row_index}"
                    row_index += 1
                    records_by_item[item_id] = record
                    tree.insert(
                        group_id,
                        "end",
                        iid=item_id,
                        values=(
                            kind_labels.get(record.kind, record.kind),
                            record.map_id,
                            scale_resolution_display(self.project_config.app, record),
                            f"{record.scale:.12g}",
                            record.detail,
                        ),
                    )
                    if select_record is not None and record == select_record:
                        tree.selection_set(item_id)
                        tree.focus(item_id)

        def new_resolution_record() -> None:
            selected_record[0] = None
            map_var.set("")
            resolution_var.set("")
            scale_var.set("")
            set_identity_editable(True)
            set_mode("新增：将创建按精确 ROI 尺寸保存的分辨率复用 scale")
            map_entry.focus_set()

        def select_current(_event: Optional[tk.Event] = None) -> None:
            selection = tree.selection()
            if not selection:
                return
            record = records_by_item.get(selection[0])
            if record is None:
                return
            selected_record[0] = record
            map_var.set(record.map_id)
            resolution_var.set(record.resolution)
            scale_var.set(f"{record.scale:.12g}")
            set_identity_editable(False)
            set_mode(f"正在编辑：{kind_labels.get(record.kind, record.kind)}；仅修改 Scale 值")
            scale_entry.focus_set()
            scale_entry.selection_range(0, "end")

        def save_current() -> None:
            try:
                scale_value = float(scale_var.get().strip())
                if selected_record[0] is None:
                    saved = add_confirmed_scale(
                        self.project_config.root_dir,
                        self.project_config.app,
                        map_var.get(),
                        resolution_var.get(),
                        scale_value,
                    )
                else:
                    saved = selected_record[0]
                    update_saved_scale(self.project_config.root_dir, self.project_config.app, saved, scale_value)
                    saved = SavedScale(
                        saved.kind,
                        saved.map_id,
                        saved.resolution,
                        scale_value,
                        saved.detail,
                    )
            except (TypeError, ValueError) as exc:
                messagebox.showerror("Scale 无法保存", str(exc), parent=dialog)
                return
            self._semantic_cv_scale_cache.clear()
            self._semantic_cv_device_multiplier_cache.clear()
            self.status_var.set(f"已保存 {saved.map_id} / {saved.resolution} 的 Scale：{scale_value:.12g}")
            refresh(saved)
            selected_record[0] = saved
            select_current()

        def delete_current() -> None:
            record = selected_record[0]
            if record is None:
                return
            if not messagebox.askyesno(
                "删除 Scale", f"删除 {record.map_id} / {record.resolution} 的 {kind_labels.get(record.kind, record.kind)}？", parent=dialog
            ):
                return
            try:
                delete_saved_scale(self.project_config.root_dir, self.project_config.app, record)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Scale 无法删除", str(exc), parent=dialog)
                return
            self._semantic_cv_scale_cache.clear()
            self._semantic_cv_device_multiplier_cache.clear()
            self.status_var.set(f"已删除 {record.map_id} / {record.resolution} 的 Scale")
            new_resolution_record()
            refresh()

        tree.bind("<<TreeviewSelect>>", select_current)
        self._draw_premium_dialog_button(canvas, photos, "scale-new", (716, 526, 842, 560), "新增分辨率", new_resolution_record)
        self._draw_premium_dialog_button(canvas, photos, "scale-save", (856, 526, 986, 560), "保存", save_current, primary=True)
        self._draw_premium_dialog_button(
            canvas,
            photos,
            "scale-delete",
            (716, 574, 842, 608),
            "删除选中",
            delete_current,
            normal_fill=(71, 24, 29, 138),
            normal_outline=(211, 111, 116, 178),
        )

        def close() -> None:
            if dialog.winfo_exists():
                dialog.grab_release()
                dialog.destroy()
            self._scale_manager_window = None

        self._draw_premium_dialog_button(canvas, photos, "scale-close", (856, 574, 986, 608), "关闭", close)
        dialog.protocol("WM_DELETE_WINDOW", close)
        refresh()

    def open_copyright_statement(self) -> None:
        if (
            self._copyright_statement_window is not None
            and self._copyright_statement_window.winfo_exists()
        ):
            self._copyright_statement_window.lift()
            self._copyright_statement_window.focus_force()
            return

        dialog = tk.Toplevel(self)
        dialog.title("版权与开放声明")
        dialog.transient(self)
        dialog.grab_set()
        self._copyright_statement_window = dialog
        width, height = 980, 680
        dialog.configure(bg="#fdfcf8")
        dialog.resizable(False, False)
        self.update_idletasks()
        x = max(0, self.winfo_rootx() + (self.winfo_width() - width) // 2)
        y = max(0, self.winfo_rooty() + (self.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        canvas = tk.Canvas(
            dialog,
            width=width,
            height=height,
            bg="#fdfcf8",
            bd=0,
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        photos: list[ImageTk.PhotoImage] = []

        background_path = self.project_config.root_dir / "assets" / "ui" / "copyright_statement_background.png"
        try:
            with Image.open(background_path) as image:
                source = image.convert("RGB")
            image_area = (300, 0, width, height)
            area_width = image_area[2] - image_area[0]
            area_height = image_area[3] - image_area[1]
            scale = min(area_width / max(1, source.width), area_height / max(1, source.height))
            rendered = source.resize(
                (
                    max(1, int(round(source.width * scale))),
                    max(1, int(round(source.height * scale))),
                ),
                Image.Resampling.LANCZOS,
            )
            composition = Image.new("RGB", (width, height), (253, 252, 248))
            image_x = image_area[0] + (area_width - rendered.width) // 2
            image_y = image_area[1] + (area_height - rendered.height) // 2
            composition.paste(rendered, (image_x, image_y))

            # Fade only the source's empty left margin into the statement panel.
            transition = Image.new("RGBA", composition.size, (0, 0, 0, 0))
            transition_draw = ImageDraw.Draw(transition)
            for column in range(260, 440):
                alpha = int(round(220 * (1.0 - (column - 260) / 180)))
                transition_draw.line((column, 0, column, height), fill=(253, 252, 248, alpha))
            composition = Image.alpha_composite(composition.convert("RGBA"), transition)

            shadow = Image.new("RGBA", composition.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rounded_rectangle((38, 32, 438, 655), radius=18, fill=(28, 38, 42, 42))
            composition = Image.alpha_composite(
                composition,
                shadow.filter(ImageFilter.GaussianBlur(12)),
            )
            panel = Image.new("RGBA", composition.size, (0, 0, 0, 0))
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rounded_rectangle(
                (28, 22, 428, 645),
                radius=16,
                fill=(255, 255, 253, 242),
                outline=(184, 148, 73, 105),
                width=1,
            )
            composition = Image.alpha_composite(composition, panel)
            background_photo = ImageTk.PhotoImage(composition, master=canvas)
            canvas.create_image(0, 0, anchor="nw", image=background_photo)
            setattr(canvas, "_copyright_background_photo", background_photo)
        except (OSError, ValueError):
            LOGGER.warning("copyright statement background unavailable: %s", background_path)

        def close() -> None:
            self._copyright_statement_window = None
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        canvas.create_text(
            56,
            52,
            anchor="w",
            text="OPEN SOURCE  /  ORIGINAL WORK",
            fill="#a57b31",
            font=("Segoe UI", 8, "bold"),
        )
        canvas.create_text(
            56,
            88,
            anchor="w",
            text="版权与开放声明",
            fill="#27363a",
            font=("Microsoft YaHei UI", 21, "bold"),
        )
        canvas.create_text(
            56,
            119,
            anchor="w",
            text="创作者  B站_飞侠樂迪",
            fill="#6f7b79",
            font=("Microsoft YaHei UI", 9),
        )
        canvas.create_line(56, 145, 394, 145, fill="#d8c69e", width=1)
        canvas.create_text(
            56,
            174,
            anchor="w",
            text="永久免费开放  ·  禁止二次贩卖",
            fill="#2e6b70",
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        canvas.create_text(
            56,
            202,
            anchor="nw",
            text="本软件免费提供并开放源代码。严禁任何个人或组织以软件本体、激活资格、安装服务等名义进行二次贩卖。",
            width=338,
            justify="left",
            fill="#4f5c5d",
            font=("Microsoft YaHei UI", 9),
        )
        canvas.create_line(56, 267, 394, 267, fill="#e3ddd0", width=1)
        canvas.create_text(
            56,
            292,
            anchor="w",
            text="原创与权利说明",
            fill="#27363a",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        rights_text = (
            "本软件的程序代码、功能逻辑与交互流程由 B站_飞侠樂迪 独立开发。未经书面许可，不得冒充官方、删除或篡改署名，亦不得用于违法侵权活动。\n"
            "本软件为非官方辅助工具，与游戏权利人无隶属、合作或授权关系；第三方名称、地图、标识及素材的合法权利归其权利人所有。"
        )
        canvas.create_text(
            56,
            316,
            anchor="nw",
            text=rights_text,
            width=338,
            justify="left",
            fill="#586264",
            font=("Microsoft YaHei UI", 9),
        )
        canvas.create_line(56, 459, 394, 459, fill="#e3ddd0", width=1)
        canvas.create_text(
            56,
            484,
            anchor="w",
            text="付费风险提示",
            fill="#a0473e",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        canvas.create_text(
            56,
            508,
            anchor="nw",
            text=(
                "若您付费获得本软件，请停止付款并保留交易凭证，联系作者 B站_飞侠樂迪 核实。"
                "作者未授权任何第三方销售本软件。"
            ),
            width=338,
            justify="left",
            fill="#6c5551",
            font=("Microsoft YaHei UI", 9),
        )
        canvas.create_text(
            56,
            579,
            anchor="w",
            text="© 2026 B站_飞侠樂迪  ·  免费开放  ·  禁止转售",
            fill="#8a8172",
            font=("Microsoft YaHei UI", 8),
        )
        self._draw_premium_dialog_button(
            canvas,
            photos,
            "copyright-close",
            (56, 598, 394, 632),
            "我已知悉",
            close,
            primary=True,
            font_size=9,
            normal_fill=(43, 102, 103, 236),
            normal_outline=(184, 148, 73, 210),
            hover_fill=(54, 119, 117, 246),
            hover_outline=(203, 165, 83, 238),
            text_fill="#fffdf7",
        )

        dialog.bind("<Escape>", lambda _event: close())
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.lift()
        dialog.focus_force()

    def _route_map_version_name(self) -> str:
        version = str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower()
        return "展十老师版本" if version == "zhanshi" else "凉哈皮老师版本"

    def open_route_map_version_settings(self) -> None:
        if self._route_map_version_window is not None and self._route_map_version_window.winfo_exists():
            self._route_map_version_window.lift()
            self._route_map_version_window.focus_force()
            return
        dialog = tk.Toplevel(self)
        dialog.title("路线图版本选择")
        dialog.transient(self)
        dialog.grab_set()
        self._route_map_version_window = dialog
        canvas = self._create_premium_settings_canvas(dialog, width=760, height=820)
        photos = getattr(canvas, "_premium_control_photos", [])
        self._draw_premium_dialog_header(canvas, "路线图版本选择", "困难与噩梦模式使用独立的小抄显示设置")
        self._draw_premium_dialog_surface(canvas, photos, (34, 104, 726, 720), fill=(3, 12, 23, 155))
        current = str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower()
        choice = tk.StringVar(value="zhanshi" if current == "zhanshi" else "lianghapi")
        display_mode = tk.StringVar(
            value="dual"
            if str(self.project_config.app.get("lianghapi_cheatsheet_mode", "single")).strip().lower() == "dual"
            else "single"
        )
        nightmare_mode = tk.StringVar(
            value="triple"
            if str(self.project_config.app.get("nightmare_cheatsheet_mode", "single")).strip().lower() == "triple"
            else "single"
        )
        nightmare_lianghapi_mode = tk.StringVar(value=nightmare_lianghapi_cheatsheet_mode(self.project_config.app))
        nightmare_version = tk.StringVar(value=nightmare_route_map_version(self.project_config.app))
        nightmare_zhanshi_variant = tk.StringVar(value=nightmare_zhanshi_route_variant(self.project_config.app))
        difficulty_zhanshi_variant = tk.StringVar(value=difficulty_zhanshi_route_variant(self.project_config.app))
        ttk.Label(dialog, text="困难模式", style="Main.Panel.TLabel").place(x=58, y=135)
        ttk.Label(
            dialog,
            text="噩梦可选择干净模板、展十两版或凉哈皮老师路线图；凉哈皮可单图或三图显示。",
            style="Main.Panel.TLabel",
        ).place(x=72, y=682)
        status = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=status, style="Main.Status.TLabel").place(x=58, y=712, width=640)

        def update_status(saved: bool = False) -> None:
            status.set(
                f"困难：{self._route_map_version_name()} / {'速刷版' if difficulty_zhanshi_variant.get() == 'fast' else '全棺版'} / {'单图' if display_mode.get() == 'single' else '双图'}    "
                f"噩梦：{('展十老师版本 / ' + ('全棺版' if nightmare_zhanshi_variant.get() == 'full_coffin' else '速刷版')) if nightmare_version.get() == 'zhanshi' else '凉哈皮老师版本 / ' + ('单图' if nightmare_lianghapi_mode.get() == 'single' else '三图') if nightmare_version.get() == 'lianghapi' else '干净模板 / ' + ('单图' if nightmare_mode.get() == 'single' else '三图')}"
                f"{'（已保存）' if saved else ''}"
            )

        def apply_choice(*, close_after: bool) -> bool:
            value = choice.get()
            previous_value = str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower()
            previous_mode = self.project_config.app.get("lianghapi_cheatsheet_mode", "single")
            previous_nightmare_mode = self.project_config.app.get("nightmare_cheatsheet_mode", "single")
            previous_nightmare_lianghapi_mode = self.project_config.app.get("nightmare_lianghapi_cheatsheet_mode", "triple")
            previous_nightmare_version = nightmare_route_map_version(self.project_config.app)
            previous_nightmare_zhanshi_variant = nightmare_zhanshi_route_variant(self.project_config.app)
            previous_difficulty_zhanshi_variant = difficulty_zhanshi_route_variant(self.project_config.app)
            if value not in {"lianghapi", "zhanshi"}:
                value = "lianghapi"
                choice.set(value)
            self.project_config.app["route_map_version"] = value
            self.project_config.app["lianghapi_cheatsheet_mode"] = display_mode.get()
            self.project_config.app["nightmare_cheatsheet_mode"] = nightmare_mode.get()
            self.project_config.app["nightmare_lianghapi_cheatsheet_mode"] = nightmare_lianghapi_mode.get()
            self.project_config.app["nightmare_route_map_version"] = nightmare_version.get()
            self.project_config.app["nightmare_zhanshi_route_variant"] = nightmare_zhanshi_variant.get()
            self.project_config.app["difficulty_zhanshi_route_variant"] = difficulty_zhanshi_variant.get()
            try:
                save_app_config(self.project_config.app_config_path, self.project_config.app)
            except OSError as exc:
                self.project_config.app["route_map_version"] = previous_value
                self.project_config.app["lianghapi_cheatsheet_mode"] = previous_mode
                self.project_config.app["nightmare_cheatsheet_mode"] = previous_nightmare_mode
                self.project_config.app["nightmare_lianghapi_cheatsheet_mode"] = previous_nightmare_lianghapi_mode
                self.project_config.app["nightmare_route_map_version"] = previous_nightmare_version
                self.project_config.app["nightmare_zhanshi_route_variant"] = previous_nightmare_zhanshi_variant
                self.project_config.app["difficulty_zhanshi_route_variant"] = previous_difficulty_zhanshi_variant
                choice.set(previous_value)
                messagebox.showerror("保存失败", str(exc), parent=dialog)
                return False
            self._semantic_cv_image_cache.clear()
            self._semantic_cv_route_crop_cache.clear()
            self._semantic_cv_affine_cache.clear()
            self._semantic_cv_piecewise_route_cache.clear()
            self._reset_route_overlay_for_route_version_change()
            self._render_main_window_controls()
            if self._side_door_structure_confirmed and self._active_route_layout_id() is not None:
                self._show_active_route_layout()
            update_status(saved=True)
            self.status_var.set("困难/噩梦小抄显示设置已保存。")
            if close_after:
                close()
            return True

        def save() -> None:
            apply_choice(close_after=True)

        def save_on_selection() -> None:
            apply_choice(close_after=False)

        def select_nightmare_zhanshi_variant() -> None:
            nightmare_version.set("zhanshi")
            save_on_selection()

        def select_nightmare_lianghapi_mode() -> None:
            nightmare_version.set("lianghapi")
            save_on_selection()

        def select_difficulty_zhanshi_variant() -> None:
            choice.set("zhanshi")
            save_on_selection()

        def close() -> None:
            self._route_map_version_window = None
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        ttk.Radiobutton(dialog, text="凉哈皮老师版本（旧版路线图）", variable=choice, value="lianghapi", command=save_on_selection).place(x=72, y=175)
        ttk.Label(dialog, text="凉哈皮版本的小抄显示模式", style="Main.Panel.TLabel").place(x=98, y=207)
        ttk.Radiobutton(dialog, text="单图：仅显示当前楼层", variable=display_mode, value="single", command=save_on_selection).place(x=112, y=233)
        ttk.Radiobutton(dialog, text="双图：同时显示一楼、二楼", variable=display_mode, value="dual", command=save_on_selection).place(x=112, y=264)
        ttk.Radiobutton(dialog, text="展十老师版本（一二楼合并路线图）", variable=choice, value="zhanshi", command=save_on_selection).place(x=72, y=296)
        ttk.Radiobutton(dialog, text="全棺版", variable=difficulty_zhanshi_variant, value="full_coffin", command=select_difficulty_zhanshi_variant).place(x=98, y=326)
        ttk.Radiobutton(dialog, text="速刷版", variable=difficulty_zhanshi_variant, value="fast", command=select_difficulty_zhanshi_variant).place(x=98, y=356)
        ttk.Label(dialog, text="噩梦模式", style="Main.Panel.TLabel").place(x=58, y=365)
        ttk.Radiobutton(dialog, text="干净模板底图", variable=nightmare_version, value="clean_template", command=save_on_selection).place(x=72, y=400)
        ttk.Radiobutton(dialog, text="单图：仅显示当前楼层", variable=nightmare_mode, value="single", command=save_on_selection).place(x=98, y=428)
        ttk.Radiobutton(dialog, text="三图：竖向显示地下室、一楼、二楼", variable=nightmare_mode, value="triple", command=save_on_selection).place(x=98, y=458)
        ttk.Radiobutton(dialog, text="展十老师版本（三层合并路线图）", variable=nightmare_version, value="zhanshi", command=save_on_selection).place(x=72, y=490)
        ttk.Radiobutton(dialog, text="速刷版", variable=nightmare_zhanshi_variant, value="fast", command=select_nightmare_zhanshi_variant).place(x=98, y=520)
        ttk.Radiobutton(dialog, text="全棺版", variable=nightmare_zhanshi_variant, value="full_coffin", command=select_nightmare_zhanshi_variant).place(x=98, y=550)
        ttk.Radiobutton(dialog, text="凉哈皮老师版本（三层合并路线图）", variable=nightmare_version, value="lianghapi", command=save_on_selection).place(x=72, y=580)
        ttk.Radiobutton(dialog, text="单图：仅显示当前所在楼层", variable=nightmare_lianghapi_mode, value="single", command=select_nightmare_lianghapi_mode).place(x=98, y=610)
        ttk.Radiobutton(dialog, text="三图：显示现有三层合并路线图", variable=nightmare_lianghapi_mode, value="triple", command=select_nightmare_lianghapi_mode).place(x=98, y=640)
        ttk.Button(dialog, text="保存并关闭", command=save, style="Main.TButton").place(x=440, y=752, width=130, height=34)
        ttk.Button(dialog, text="关闭", command=close, style="Main.TButton").place(x=585, y=752, width=80, height=34)
        update_status()
        dialog.protocol("WM_DELETE_WINDOW", close)

    def open_game_map_toggle_settings(self) -> None:
        if (
            self._game_map_toggle_settings_window is not None
            and self._game_map_toggle_settings_window.winfo_exists()
        ):
            self._game_map_toggle_settings_window.focus_force()
            return

        dialog = tk.Toplevel(self)
        dialog.title("游戏地图唤出按键")
        dialog.transient(self)
        dialog.grab_set()
        self._game_map_toggle_settings_window = dialog
        canvas = self._create_premium_settings_canvas(dialog, width=780, height=540)
        photos: list[ImageTk.PhotoImage] = getattr(canvas, "_premium_control_photos")

        current = str(self.project_config.app.get("game_map_toggle_button", "right")).strip()
        if current.lower() not in GAME_MAP_MOUSE_BINDINGS and re.fullmatch(r"key:\d{1,3}:.*", current) is None:
            current = "right"
        working = {"binding": current}
        status_var = tk.StringVar(value="请选择鼠标按键，或点击录入键盘按键。")

        def stop_keyboard_capture() -> None:
            self._game_map_binding_capturing = False
            self._hotkey_configuring = False

        def choose(binding: str) -> None:
            identity = game_map_toggle_binding_identity(binding)
            owner = next(
                (
                    action
                    for action, assigned in self._hotkeys.items()
                    if identity is not None and hotkey_binding_identity(assigned) == identity
                ),
                None,
            )
            if owner is not None:
                display = game_map_toggle_display_name(binding)
                conflict = HOTKEY_LABELS.get(owner, owner)
                message = f"{display} 已用于“{conflict}”，请重新设置。"
                status_var.set(message)
                render()
                messagebox.showwarning("按键冲突", message, parent=dialog)
                dialog.focus_force()
                return
            stop_keyboard_capture()
            working["binding"] = binding
            status_var.set("选择完成，点击保存后立即生效。")
            render()

        def begin_keyboard_capture() -> None:
            self._game_map_binding_capturing = True
            self._hotkey_configuring = True
            status_var.set("请按下要用于游戏内打开地图的任意单个键盘按键……")
            render()
            dialog.focus_force()

        def handle_key(event: tk.Event) -> str:
            if not self._game_map_binding_capturing:
                return "break"
            keysym = str(event.keysym)
            try:
                virtual_key = int(event.keycode)
            except (TypeError, ValueError):
                virtual_key = 0
            if not 1 <= virtual_key <= 255:
                status_var.set("无法识别该键的 Windows 键码，请重新按键。")
                render()
                return "break"
            conflict = self._game_map_keyboard_conflict(keysym, virtual_key)
            if conflict is not None:
                message = f"{self._keyboard_key_display_label(keysym)} 已用于“{conflict}”，请重新设置。"
                status_var.set(message)
                render()
                messagebox.showwarning("按键冲突", message, parent=dialog)
                dialog.focus_force()
                return "break"
            label = self._keyboard_key_display_label(keysym).replace(":", " ")
            choose(f"key:{virtual_key}:{label}")
            return "break"

        def close() -> None:
            stop_keyboard_capture()
            self._game_map_toggle_settings_window = None
            dialog.grab_release()
            dialog.destroy()

        def save() -> None:
            binding = working["binding"]
            self.project_config.app["game_map_toggle_button"] = binding
            try:
                save_app_config(self.project_config.app_config_path, self.project_config.app)
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=dialog)
                return
            display = game_map_toggle_display_name(binding)
            self.game_map_toggle_var.set(f"游戏地图键：{display}")
            self.status_var.set(f"游戏内打开地图按键已设置为：{display}；G 将自动调用该按键。")
            close()

        def render() -> None:
            canvas.delete("dialog-control")
            photos.clear()
            selected_name = game_map_toggle_display_name(working["binding"])
            self._draw_premium_dialog_header(canvas, "游戏地图唤出按键", f"当前选择  ·  {selected_name}")

            self._draw_premium_dialog_surface(canvas, photos, (40, 100, 740, 232))
            canvas.create_text(
                58,
                124,
                anchor="w",
                text="鼠标按键",
                fill="#e9f0f4",
                font=("Microsoft YaHei UI", 11, "bold"),
                tags="dialog-control",
            )
            mouse_bindings = tuple(GAME_MAP_MOUSE_BINDINGS)
            mouse_gap = 8
            mouse_left = 58
            mouse_right = 722
            mouse_width = (mouse_right - mouse_left - mouse_gap * (len(mouse_bindings) - 1)) // len(mouse_bindings)
            for column, binding in enumerate(mouse_bindings):
                x1 = mouse_left + column * (mouse_width + mouse_gap)
                x2 = mouse_right if column == len(mouse_bindings) - 1 else x1 + mouse_width
                self._draw_premium_dialog_button(
                    canvas,
                    photos,
                    f"map-mouse-{binding}",
                    (x1, 154, x2, 204),
                    GAME_MAP_MOUSE_BINDINGS[binding][0],
                    lambda value=binding: choose(value),
                    selected=working["binding"].lower() == binding,
                    font_size=9,
                )

            self._draw_premium_dialog_surface(canvas, photos, (40, 248, 740, 360))
            canvas.create_text(
                58,
                272,
                anchor="w",
                text="键盘按键",
                fill="#e9f0f4",
                font=("Microsoft YaHei UI", 11, "bold"),
                tags="dialog-control",
            )
            keyboard_text = "等待键盘输入…" if self._game_map_binding_capturing else "录入任意键盘单键"
            if working["binding"].startswith("key:") and not self._game_map_binding_capturing:
                keyboard_text = f"当前键盘按键  ·  {selected_name}"
            self._draw_premium_dialog_button(
                canvas,
                photos,
                "map-keyboard-capture",
                (58, 294, 722, 340),
                keyboard_text,
                begin_keyboard_capture,
                selected=self._game_map_binding_capturing or working["binding"].startswith("key:"),
            )

            self._draw_premium_dialog_surface(
                canvas,
                photos,
                (40, 378, 740, 444),
                fill=(4, 12, 22, 132),
            )
            canvas.create_oval(58, 407, 64, 413, fill="#72c8eb", outline="", tags="dialog-control")
            canvas.create_text(
                76,
                410,
                anchor="w",
                text=status_var.get(),
                width=640,
                fill="#d4e0e7",
                font=("Microsoft YaHei UI", 9),
                tags="dialog-control",
            )
            self._draw_premium_dialog_button(
                canvas,
                photos,
                "map-cancel",
                (526, 470, 628, 516),
                "取消",
                close,
            )
            self._draw_premium_dialog_button(
                canvas,
                photos,
                "map-save",
                (640, 470, 740, 516),
                "保存",
                save,
                primary=True,
            )

        dialog.bind("<KeyPress>", handle_key)
        dialog.protocol("WM_DELETE_WINDOW", close)
        render()
        dialog.lift()
        dialog.focus_force()

    def open_hotkey_settings(self) -> None:
        if self._hotkey_settings_window is not None and self._hotkey_settings_window.winfo_exists():
            self._hotkey_settings_window.focus_force()
            return
        dialog = tk.Toplevel(self)
        dialog.title("自定义程序热键")
        dialog.transient(self)
        dialog.grab_set()
        self._hotkey_settings_window = dialog
        canvas = self._create_premium_settings_canvas(dialog, width=820, height=650)
        photos: list[ImageTk.PhotoImage] = getattr(canvas, "_premium_control_photos")
        working = dict(self._hotkeys)
        capture_status = tk.StringVar(value="点击一个功能的当前按键，然后按键盘键或选择鼠标按键。")

        def refresh() -> None:
            render()

        def begin_capture(action: str) -> None:
            self._hotkey_configuring = True
            self._hotkey_capture_action = action
            capture_status.set(f"正在设置“{HOTKEY_LABELS[action]}”：请按任意键盘单键，或点击鼠标按键。")
            refresh()
            dialog.focus_force()

        def cancel_capture() -> None:
            self._hotkey_configuring = False
            self._hotkey_capture_action = None
            refresh()

        def conflict_for(action: str, binding: str) -> Optional[str]:
            identity = hotkey_binding_identity(binding)
            if identity is None:
                return "无法识别该按键"
            fixed = {
                hotkey_binding_identity("F2"): "停止 SCALE 搜索（F2）",
                hotkey_binding_identity(VARIANT_RETURN_HOTKEY): "返回上一步（F3）",
                ("key", 0x1B): "关闭或取消路线图虚像（Esc）",
            }
            if identity in fixed:
                return fixed[identity]
            game_binding = self.project_config.app.get("game_map_toggle_button", "right")
            if identity == game_map_toggle_binding_identity(game_binding):
                return f"游戏内打开/关闭地图（{game_map_toggle_display_name(game_binding)}）"
            for owner, assigned in working.items():
                if owner != action and hotkey_binding_identity(assigned) == identity:
                    return HOTKEY_LABELS.get(owner, owner)
            return None

        def assign_binding(binding: str) -> None:
            action = self._hotkey_capture_action
            if action is None:
                capture_status.set("请先点击需要修改的功能按键。")
                return
            conflict = conflict_for(action, binding)
            if conflict is not None:
                display = hotkey_binding_display_name(binding)
                message = f"{display} 已用于“{conflict}”，请重新设置。"
                capture_status.set(message)
                messagebox.showwarning("热键冲突", message, parent=dialog)
                dialog.focus_force()
                return
            working[action] = binding
            selected = HOTKEY_LABELS[action]
            cancel_capture()
            capture_status.set(f"“{selected}”已设为 {hotkey_binding_display_name(binding)}，保存后立即生效。")

        def handle_key(event: tk.Event) -> str:
            if self._hotkey_capture_action is None:
                return "break"
            try:
                virtual_key = int(event.keycode)
            except (TypeError, ValueError):
                virtual_key = 0
            if not 1 <= virtual_key <= 255:
                capture_status.set("无法识别该键的 Windows 键码，请重新按键。")
                return "break"
            label = self._keyboard_key_display_label(str(event.keysym)).replace(":", " ")
            assign_binding(f"key:{virtual_key}:{label}")
            return "break"

        def save() -> None:
            self._hotkeys = dict(working)
            self.project_config.app["hotkeys"] = dict(working)
            try:
                save_app_config(self.project_config.app_config_path, self.project_config.app)
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=dialog)
                return
            self._refresh_hotkey_dependent_ui()
            summary = "；".join(
                f"{HOTKEY_LABELS[action]}={hotkey_binding_display_name(binding)}"
                for action, binding in working.items()
            )
            self.status_var.set(f"自定义热键已保存并立即生效：{summary}")
            close()

        def reset() -> None:
            working.clear()
            working.update(HOTKEY_DEFAULTS)
            refresh()
            cancel_capture()

        def close() -> None:
            cancel_capture()
            self._hotkey_settings_window = None
            dialog.grab_release()
            dialog.destroy()

        def render() -> None:
            canvas.delete("dialog-control")
            photos.clear()
            self._draw_premium_dialog_header(canvas, "自定义程序热键", "键盘与鼠标按键均可录入")

            self._draw_premium_dialog_surface(canvas, photos, (40, 96, 780, 356))
            canvas.create_text(
                60,
                120,
                anchor="w",
                text="功能按键",
                fill="#eaf1f5",
                font=("Microsoft YaHei UI", 11, "bold"),
                tags="dialog-control",
            )
            row_top = 146
            row_height = 42
            for row, action in enumerate(HOTKEY_DEFAULTS):
                y1 = row_top + row * row_height
                y2 = y1 + 35
                if row:
                    canvas.create_line(
                        60,
                        y1 - 5,
                        760,
                        y1 - 5,
                        fill="#3d5363",
                        width=1,
                        tags="dialog-control",
                    )
                canvas.create_text(
                    60,
                    (y1 + y2) // 2,
                    anchor="w",
                    text=HOTKEY_LABELS[action],
                    fill="#dbe5eb",
                    font=("Microsoft YaHei UI", 10),
                    tags="dialog-control",
                )
                capturing = self._hotkey_capture_action == action
                binding_text = "等待输入…" if capturing else hotkey_binding_display_name(working[action])
                self._draw_premium_dialog_button(
                    canvas,
                    photos,
                    f"hotkey-{action}",
                    (526, y1, 760, y2),
                    binding_text,
                    lambda value=action: begin_capture(value),
                    selected=capturing,
                )

            self._draw_premium_dialog_surface(canvas, photos, (40, 374, 780, 484))
            canvas.create_text(
                60,
                398,
                anchor="w",
                text="鼠标按键",
                fill="#eaf1f5",
                font=("Microsoft YaHei UI", 11, "bold"),
                tags="dialog-control",
            )
            mouse_bindings = tuple(GAME_MAP_MOUSE_BINDINGS)
            mouse_gap = 8
            mouse_left = 60
            mouse_right = 760
            mouse_width = (mouse_right - mouse_left - mouse_gap * (len(mouse_bindings) - 1)) // len(mouse_bindings)
            for column, mouse_name in enumerate(mouse_bindings):
                x1 = mouse_left + column * (mouse_width + mouse_gap)
                x2 = mouse_right if column == len(mouse_bindings) - 1 else x1 + mouse_width
                self._draw_premium_dialog_button(
                    canvas,
                    photos,
                    f"hotkey-mouse-{mouse_name}",
                    (x1, 420, x2, 464),
                    GAME_MAP_MOUSE_BINDINGS[mouse_name][0],
                    lambda value=mouse_name: assign_binding(f"mouse:{value}"),
                    font_size=9,
                )

            self._draw_premium_dialog_surface(
                canvas,
                photos,
                (40, 500, 780, 558),
                fill=(4, 12, 22, 136),
            )
            canvas.create_oval(58, 526, 64, 532, fill="#72c8eb", outline="", tags="dialog-control")
            canvas.create_text(
                76,
                529,
                anchor="w",
                text=capture_status.get(),
                width=680,
                fill="#d4e0e7",
                font=("Microsoft YaHei UI", 9),
                tags="dialog-control",
            )

            self._draw_premium_dialog_button(
                canvas,
                photos,
                "hotkey-stop",
                (40, 582, 168, 628),
                "停止录入",
                cancel_capture,
                selected=self._hotkey_capture_action is not None,
            )
            self._draw_premium_dialog_button(
                canvas,
                photos,
                "hotkey-reset",
                (180, 582, 308, 628),
                "恢复默认",
                reset,
            )
            self._draw_premium_dialog_button(
                canvas,
                photos,
                "hotkey-cancel",
                (568, 582, 668, 628),
                "取消",
                close,
            )
            self._draw_premium_dialog_button(
                canvas,
                photos,
                "hotkey-save",
                (680, 582, 780, 628),
                "保存",
                save,
                primary=True,
            )

        dialog.bind("<KeyPress>", handle_key)
        dialog.protocol("WM_DELETE_WINDOW", close)
        refresh()
        dialog.lift()
        dialog.focus_force()

    def _poll_global_hotkey(self) -> None:
        try:
            pressed = self._is_hotkey_action_down("recognize")
            now = time.monotonic()
            if pressed and not self._hotkey_down and now - self._last_hotkey_time > 0.45:
                self._last_hotkey_time = now
                self._trigger_hotkey_action("recognize")
            self._hotkey_down = pressed
            self._poll_layout_switch_key()
            self._poll_variant_return_key()
            self._poll_location_key()
            self._poll_route_overlay_dismiss_key()
            self._poll_structure_confirmation_key()
            self._poll_pf_motion_keys()
        finally:
            self.after(self._global_hotkey_poll_interval_ms(), self._poll_global_hotkey)

    def _global_hotkey_poll_interval_ms(self) -> int:
        return max(10, int(self.project_config.app.get("global_hotkey_poll_interval_ms", 10)))

    def _is_global_key_down(self, virtual_key: int) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000)  # type: ignore[attr-defined]
        except Exception:
            return False

    def _poll_location_key(self) -> None:
        pressed = self._is_hotkey_action_down("locate_current_floor")
        if pressed and not self._location_key_down:
            self._trigger_hotkey_action("locate_current_floor")
        self._location_key_down = pressed

    def _poll_layout_switch_key(self) -> None:
        pressed = self._is_hotkey_action_down("switch_layout")
        if pressed and not self._layout_switch_key_down:
            self._trigger_hotkey_action("switch_layout")
        self._layout_switch_key_down = pressed

    def _poll_variant_return_key(self) -> None:
        pressed = self._is_global_key_down(0x72)  # VK_F3
        if pressed and not self._variant_return_key_down:
            if self._can_return_to_variant_layout_selection():
                self._return_to_variant_layout_selection()
            elif self._can_return_to_scale_research_sheet():
                self._return_to_scale_research_sheet()
        self._variant_return_key_down = pressed

    def _poll_structure_confirmation_key(self) -> None:
        pressed = self._is_hotkey_action_down("confirm_structure")
        if pressed and not self._structure_confirm_key_down:
            self._trigger_hotkey_action("confirm_structure")
        self._structure_confirm_key_down = pressed

    def _poll_route_overlay_dismiss_key(self) -> None:
        """Cancel pending output or hide the overlay when the game map closes."""
        toggle_pressed = self._is_game_map_toggle_down()
        escape_pressed = self._is_global_key_down(0x1B)  # VK_ESCAPE
        toggle_edge = toggle_pressed and not self._route_overlay_dismiss_key_down
        escape_edge = escape_pressed and not self._route_overlay_escape_key_down
        close_detected_request_id = getattr(
            self,
            "_route_overlay_close_detected_request_id",
            None,
        )
        close_event = getattr(self, "_route_overlay_close_requested", None)
        # Keep the prior request-id latch behavior for old callers/tests while
        # production uses the thread-safe event set by the close monitor.
        close_requested = bool(close_event.is_set()) if close_event is not None else close_detected_request_id is not None
        latched_close = close_requested and close_detected_request_id is not None and close_detected_request_id in {
            self._active_route_overlay_request_id,
            getattr(self, "_route_overlay_visible_request_id", None),
        }
        if latched_close or ((toggle_edge or escape_edge) and (
            self._route_overlay_dismiss_armed or self._active_route_overlay_request_id is not None
        )):
            self._route_overlay_cancel_reason = (
                "close_monitor" if latched_close else ("toggle_key" if toggle_edge else "escape_key")
            )
            self._cancel_route_overlay_generation()
            if latched_close:
                close_event = getattr(self, "_route_overlay_close_requested", None)
                if close_event is not None:
                    close_event.clear()
        self._route_overlay_dismiss_key_down = toggle_pressed
        self._route_overlay_escape_key_down = escape_pressed

    def _is_game_map_toggle_down(self) -> bool:
        binding = str(self.project_config.app.get("game_map_toggle_button", "right")).strip()
        mouse_vk = GAME_MAP_MOUSE_VIRTUAL_KEYS.get(binding.lower())
        if mouse_vk is not None:
            return self._is_global_key_down(mouse_vk)
        match = re.fullmatch(r"key:(\d{1,3}):.*", binding)
        if match is None:
            return False
        virtual_key = int(match.group(1))
        return 1 <= virtual_key <= 255 and self._is_global_key_down(virtual_key)

    def _send_game_map_toggle(self) -> None:
        """Send the configured game-map mouse or keyboard input."""
        if not sys.platform.startswith("win"):
            return
        binding = str(self.project_config.app.get("game_map_toggle_button", "right")).strip()
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        mouse = GAME_MAP_MOUSE_BINDINGS.get(binding.lower())
        if mouse is not None:
            _label, down_flag, up_flag, mouse_data = mouse
            user32.mouse_event(down_flag, 0, 0, mouse_data, 0)
            user32.mouse_event(up_flag, 0, 0, mouse_data, 0)
            return

        match = re.fullmatch(r"key:(\d{1,3}):.*", binding)
        if match is None:
            raise ValueError(f"unsupported game map toggle binding: {binding}")
        virtual_key = int(match.group(1))
        if not 1 <= virtual_key <= 255:
            raise ValueError(f"invalid game map virtual key: {virtual_key}")
        user32.keybd_event(virtual_key, 0, 0, 0)
        user32.keybd_event(virtual_key, 0, 0x0002, 0)

    def _send_game_map_zoom_out_step(self) -> None:
        """Send one mouse-wheel-down step to the open game map."""
        if not sys.platform.startswith("win"):
            return
        wheel_delta = ctypes.c_uint32(-120).value
        ctypes.windll.user32.mouse_event(0x0800, 0, 0, wheel_delta, 0)  # type: ignore[attr-defined]

    def _preload_route_overlay_assets(self, floor_suffix: str) -> None:
        """Decode the first-frame overlay assets before the game map opens."""
        try:
            map_id = self._semantic_cv_map_id(floor_suffix)
            localizer_root = self._semantic_cv_project_root()
            processed_dir = localizer_root / str(
                self.project_config.app.get("semantic_cv_processed_template_dir", "templates_processed")
            )
            paths: list[Optional[Path]] = [
                self._resolve_route_template_path(map_id),
                processed_dir / map_id / "image.png",
                resolve_floor_template_path(self.project_config.app, map_id, floor_suffix),
            ]
            if self._should_show_lianghapi_dual_cheatsheet():
                other_floor = "2nd" if floor_suffix == "1st" else "1st"
                paths.append(self._resolve_route_template_path(self._floor_map_id(map_id, other_floor)))
            if self._is_nightmare_mode() and nightmare_three_cheatsheet_enabled(self.project_config.app):
                for other_floor in ("b1", "1st", "2nd"):
                    if other_floor != floor_suffix:
                        paths.append(
                            self._resolve_nightmare_clean_template_path(map_id, other_floor)
                        )
            for path in paths:
                if path is not None:
                    self._semantic_cv_cached_image(path)
            route_path = self._resolve_route_template_path(map_id)
            locate_path = processed_dir / map_id / "image.png"
            route_image = self._semantic_cv_cached_image(route_path) if route_path is not None else None
            locate_image = self._semantic_cv_cached_image(locate_path)
            if route_image is not None and locate_image is not None:
                self._template_to_route_affine(
                    map_id=map_id,
                    locate_size=(locate_image.shape[1], locate_image.shape[0]),
                    route_size=(route_image.shape[1], route_image.shape[0]),
                )
        except Exception:
            LOGGER.debug("failed to preload route overlay assets", exc_info=True)

    def _schedule_route_overlay_warmup(self) -> None:
        """Move first-use Tk and route asset work to idle time after F8."""
        def warm() -> None:
            try:
                self.route_overlay.warmup()
                self._preload_route_overlay_assets("1st")
                self._preload_route_overlay_assets("2nd")
            except Exception:
                LOGGER.debug("failed to warm route overlay after recognition", exc_info=True)

        self.after_idle(warm)

    def _request_map_location_after_toggle(
        self,
        floor_suffix: Optional[str] = None,
    ) -> None:
        if bool(getattr(self, "_program_paused", False)):
            self.status_var.set("程序当前处于暂停锁定状态，定位请求已忽略。")
            return
        if not self._side_door_structure_confirmed or self._active_route_layout_id() is None:
            hotkey = hotkey_binding_display_name(
                getattr(self, "_hotkeys", {}).get(
                    "locate_current_floor",
                    HOTKEY_DEFAULTS["locate_current_floor"],
                )
            )
            self._show_layout_lock_required(hotkey)
            return
        # A manually cancelled scale search leaves this latch set so its worker
        # can exit promptly.  A later G request for an already-confirmed floor
        # has no scale-search progress stage to clear it, so reset only stale
        # cancellation state before opening the next game map.
        if not bool(getattr(self, "_scale_search_active", False)):
            self._scale_search_cancel_requested = False
            self._scale_search_stop_key_down = False
        if self._active_route_overlay_request_id is not None:
            self._route_overlay_cancel_reason = "superseded_by_new_request"
            self._cancel_route_overlay_generation()
        self._route_overlay_request_id += 1
        request_id = self._route_overlay_request_id
        self._active_route_overlay_request_id = request_id
        diagnostic_event(
            "location_request_started",
            request_id=request_id,
            requested_floor=floor_suffix,
            active_layout=self._active_route_layout_id(),
            previous_floor=getattr(self, "_active_floor_suffix", None),
        )
        self._route_overlay_visible_request_id = None
        self._route_overlay_close_detected_request_id = None
        close_event = getattr(self, "_route_overlay_close_requested", None)
        if close_event is not None:
            close_event.clear()
        route_overlay = getattr(self, "route_overlay", None)
        hide_overlay = getattr(route_overlay, "hide", None)
        if callable(hide_overlay):
            hide_overlay()
        self._route_overlay_dismiss_armed = False
        self._overlay_scale_research_context = None
        cheatsheet = getattr(self, "cheatsheet", None)
        clear_bottom_action = getattr(cheatsheet, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()
        try:
            preload_overlay = getattr(self, "_preload_route_overlay_assets", None)
            if callable(preload_overlay) and floor_suffix in {"1st", "2nd"}:
                preload_overlay(floor_suffix)
            self._send_game_map_toggle()
            diagnostic_event("game_map_toggle_sent", request_id=request_id, requested_floor=floor_suffix)
        except Exception as exc:
            diagnostic_event("game_map_toggle_failed", request_id=request_id, exception=repr(exc))
            self._active_route_overlay_request_id = None
            self.status_var.set(f"failed to open the game map: {exc}")
            return
        delay_ms = max(0, int(self.project_config.app.get("game_map_open_stable_delay_ms", 25)))
        if bool(getattr(self, "_first_location_map_zoom_pending", False)):
            zoom_steps = max(
                1,
                int(self.project_config.app.get("game_map_first_location_zoom_out_steps", 6)),
            )
            self.status_var.set(
                f"game map opened; waiting {delay_ms} ms, then zooming to minimum before CV location"
            )
            self.after(
                delay_ms,
                lambda: self._run_first_location_map_zoom_out(floor_suffix, request_id, zoom_steps),
            )
            return
        self.status_var.set(f"game map opened; waiting {delay_ms} ms before CV location")
        self.after(delay_ms, lambda: self._run_delayed_map_location(floor_suffix, request_id))

    def _run_first_location_map_zoom_out(
        self,
        floor_suffix: Optional[str],
        request_id: int,
        steps_remaining: int,
    ) -> None:
        """Zoom the first post-F8 map view to minimum before capturing its ROI."""
        if request_id <= self._cancelled_route_overlay_request_id:
            if self._active_route_overlay_request_id == request_id:
                self._active_route_overlay_request_id = None
            return
        try:
            self._send_game_map_zoom_out_step()
        except Exception as exc:
            if self._active_route_overlay_request_id == request_id:
                self._active_route_overlay_request_id = None
            self.status_var.set(f"failed to zoom the game map to minimum: {exc}")
            return

        if steps_remaining > 1:
            interval_ms = max(
                0,
                int(self.project_config.app.get("game_map_first_location_zoom_interval_ms", 5)),
            )
            self.after(
                interval_ms,
                lambda: self._run_first_location_map_zoom_out(
                    floor_suffix,
                    request_id,
                    steps_remaining - 1,
                ),
            )
            return

        self._first_location_map_zoom_pending = False
        settle_ms = max(
            0,
            int(self.project_config.app.get("game_map_first_location_zoom_settle_delay_ms", 5)),
        )
        self.status_var.set(
            f"game map zoomed to minimum; waiting {settle_ms} ms before CV location"
        )
        self.after(settle_ms, lambda: self._run_delayed_map_location(floor_suffix, request_id))

    def _run_delayed_map_location(self, floor_suffix: Optional[str], request_id: int) -> None:
        if request_id <= self._cancelled_route_overlay_request_id:
            if self._active_route_overlay_request_id == request_id:
                self._active_route_overlay_request_id = None
            return
        self._active_route_overlay_request_id = request_id
        diagnostic_event("location_processing_begin", request_id=request_id, requested_floor=floor_suffix)
        self._start_route_overlay_close_monitor(request_id)
        try:
            if floor_suffix is None:
                self.relocate_player_marker_auto_floor()
            elif floor_suffix == "2nd":
                self._request_second_floor_location()
            else:
                self.relocate_player_marker()
            diagnostic_event(
                "location_processing_returned",
                request_id=request_id,
                active_floor=self._active_floor_suffix,
                visible_request_id=self._route_overlay_visible_request_id,
            )
        except BaseException as exc:
            diagnostic_event(
                "location_processing_exception",
                request_id=request_id,
                exception_type=type(exc).__name__,
                exception=repr(exc),
            )
            raise
        finally:
            if self._active_route_overlay_request_id == request_id:
                self._active_route_overlay_request_id = None

    def _start_route_overlay_close_monitor(self, request_id: int) -> None:
        def monitor() -> None:
            # The synthetic opening input has already completed before this starts.
            toggle_was_down = self._is_game_map_toggle_down()
            escape_was_down = self._is_global_key_down(0x1B)
            while (
                self._active_route_overlay_request_id == request_id
                or getattr(self, "_route_overlay_visible_request_id", None) == request_id
            ):
                toggle_down = self._is_game_map_toggle_down()
                escape_down = self._is_global_key_down(0x1B)
                if (toggle_down and not toggle_was_down) or (escape_down and not escape_was_down):
                    self._cancelled_route_overlay_request_id = max(
                        self._cancelled_route_overlay_request_id,
                        request_id,
                    )
                    self._route_overlay_close_detected_request_id = request_id
                    self._route_overlay_close_requested.set()
                    diagnostic_event(
                        "overlay_close_input_detected",
                        request_id=request_id,
                        toggle_edge=bool(toggle_down and not toggle_was_down),
                        escape_edge=bool(escape_down and not escape_was_down),
                    )
                    return
                toggle_was_down = toggle_down
                escape_was_down = escape_down
                time.sleep(0.01)

        self._route_overlay_close_monitor = threading.Thread(
            target=monitor,
            name="route-overlay-close-monitor",
            daemon=True,
        )
        self._route_overlay_close_monitor.start()

    def _cancel_route_overlay_generation(self) -> None:
        request_id = self._active_route_overlay_request_id
        reason = str(getattr(self, "_route_overlay_cancel_reason", "unspecified"))
        self._route_overlay_cancel_reason = "unspecified"
        diagnostic_event(
            "overlay_generation_cancelled",
            reason=reason,
            active_request_id=request_id,
            visible_request_id=getattr(self, "_route_overlay_visible_request_id", None),
            close_detected_request_id=getattr(self, "_route_overlay_close_detected_request_id", None),
        )
        if request_id is not None:
            self._cancelled_route_overlay_request_id = max(self._cancelled_route_overlay_request_id, request_id)
        self._active_route_overlay_request_id = None
        route_overlay = getattr(self, "route_overlay", None)
        hide_overlay = getattr(route_overlay, "hide", None)
        if callable(hide_overlay):
            hide_overlay()
        self._route_overlay_dismiss_armed = False
        self._route_overlay_visible_request_id = None
        self._route_overlay_close_detected_request_id = None
        self._overlay_scale_research_context = None
        cheatsheet = getattr(self, "cheatsheet", None)
        clear_bottom_action = getattr(cheatsheet, "clear_bottom_action", None)
        if callable(clear_bottom_action):
            clear_bottom_action()

    def _reset_route_overlay_for_route_version_change(self) -> None:
        """End the old overlay session before changing the route-map coordinate space."""
        self._route_overlay_cancel_reason = "route_version_changed"
        self._cancel_route_overlay_generation()
        route_overlay = getattr(self, "route_overlay", None)
        reset_overlay = getattr(route_overlay, "reset", None)
        if callable(reset_overlay):
            reset_overlay()
        close_event = getattr(self, "_route_overlay_close_requested", None)
        if close_event is not None:
            close_event.clear()
        self._route_overlay_visible_request_id = None
        self._route_overlay_close_detected_request_id = None
        self._route_overlay_dismiss_armed = False
        self._overlay_scale_research_context = None
        # A right-click that is already held while the settings window closes
        # belongs to the old map view, not to the next G request.
        self._route_overlay_dismiss_key_down = self._is_game_map_toggle_down()
        self._route_overlay_escape_key_down = self._is_global_key_down(0x1B)

    def _poll_pf_motion_keys(self) -> None:
        if not bool(self.project_config.app.get("semantic_cv_enable_particle_filter", True)):
            self._semantic_cv_player_pfs.clear()
            self._semantic_cv_player_pf_updated_at.clear()
            self._semantic_cv_player_pf_key = None
            self._semantic_cv_pf_motion_hint = (0.0, 0.0)
            self._semantic_cv_pf_last_motion_at = None
            return
        if not bool(self.project_config.app.get("semantic_cv_pf_keyboard_motion_enabled", True)):
            self._semantic_cv_pf_motion_hint = (0.0, 0.0)
            self._semantic_cv_pf_last_motion_at = None
            return
        dx = 0.0
        dy = 0.0
        if self._is_global_key_down(0x41):  # A
            dx -= 1.0
        if self._is_global_key_down(0x44):  # D
            dx += 1.0
        if self._is_global_key_down(0x57):  # W
            dy -= 1.0
        if self._is_global_key_down(0x53):  # S
            dy += 1.0
        if dx != 0.0 and dy != 0.0:
            inv = 2 ** -0.5
            dx *= inv
            dy *= inv
        now = time.monotonic()
        self._semantic_cv_pf_motion_hint = (dx, dy)
        if dx == 0.0 and dy == 0.0:
            self._semantic_cv_pf_last_motion_at = None
            return
        active_filters = {
            player_id: particle_filter
            for player_id, particle_filter in self._semantic_cv_player_pfs.items()
            if getattr(particle_filter, "initialized", False)
        }
        if not active_filters:
            self._semantic_cv_pf_last_motion_at = now
            return
        last = self._semantic_cv_pf_last_motion_at or now
        dt = max(0.0, min(0.25, now - last))
        self._semantic_cv_pf_last_motion_at = now
        if dt <= 0:
            return
        speed_modes = self.project_config.app.get("semantic_cv_pf_keyboard_speed_modes")
        pixels_per_second = float(self.project_config.app.get("semantic_cv_pf_keyboard_speed", 90.0))
        noise = float(self.project_config.app.get("semantic_cv_pf_keyboard_noise", 2.0))
        try:
            if isinstance(speed_modes, list) and speed_modes:
                radii: list[tuple[float, float]] = []
                for player_id, particle_filter in active_filters.items():
                    radii.append(
                        particle_filter.apply_motion_diffusion(
                            speed_modes,
                            dt,
                            direction_bias=(dx, dy) if player_id == "1" else None,
                            direction_bias_strength=(
                                float(self.project_config.app.get("semantic_cv_pf_keyboard_direction_bias_strength", 0.0))
                                if player_id == "1"
                                else 0.0
                            ),
                            noise=noise,
                        )
                    )
                    self._diffuse_player_filter_transitions(particle_filter, self._semantic_cv_player_pf_key)
                mean_radius = float(np.mean([item[0] for item in radii])) if radii else 0.0
                max_radius = float(max((item[1] for item in radii), default=0.0))
                move_dx = 0.0
                move_dy = 0.0
            else:
                move_dx = dx * pixels_per_second * dt
                move_dy = dy * pixels_per_second * dt
                for player_id, particle_filter in active_filters.items():
                    particle_filter.apply_motion_hint(
                        move_dx if player_id == "1" else 0.0,
                        move_dy if player_id == "1" else 0.0,
                        noise=noise,
                    )
                    self._diffuse_player_filter_transitions(particle_filter, self._semantic_cv_player_pf_key)
                mean_radius = float((move_dx * move_dx + move_dy * move_dy) ** 0.5)
                max_radius = mean_radius
        except AttributeError:
            return
        accum_x, accum_y = self._semantic_cv_pf_motion_accum
        self._semantic_cv_pf_motion_accum = (accum_x + move_dx, accum_y + move_dy)
        self._semantic_cv_pf_motion_radius_accum += float(mean_radius)
        self._semantic_cv_pf_motion_radius_max = max(self._semantic_cv_pf_motion_radius_max, float(max_radius))
        self._semantic_cv_pf_motion_seconds += dt

    def _global_cursor_position(self) -> Optional[tuple[int, int]]:
        if not sys.platform.startswith("win"):
            return None
        try:
            point = wintypes.POINT()
            if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):  # type: ignore[attr-defined]
                return None
            return int(point.x), int(point.y)
        except Exception:
            return None

    def _poll_side_door_anchor_key(self) -> None:
        if not self._awaiting_unknown_anchor:
            self._anchor_key_down = self._is_global_key_down(0x70)
            return
        key_down = self._is_global_key_down(0x70)
        if not key_down or self._anchor_key_down:
            self._anchor_key_down = key_down
            return
        self._anchor_key_down = key_down
        self.confirm_side_door_cursor()

    def _use_structure_registration_location(self) -> bool:
        return str(self.project_config.app.get("player_location_method", "")).strip().lower() == "structure_registration"

    def _use_roi_affine_location(self) -> bool:
        return str(self.project_config.app.get("player_location_method", "")).strip().lower() == "roi_affine"

    def _use_semantic_cv_location(self) -> bool:
        return str(self.project_config.app.get("player_location_method", "")).strip().lower() in {
            "semantic_cv",
            "masked_template",
            "cv_localizer",
        }

    def confirm_side_door_cursor(self) -> None:
        if not self._awaiting_unknown_anchor:
            return
        if self._use_semantic_cv_location():
            try:
                self._capture_and_run_semantic_cv_location("1st")
            except Exception as exc:
                self.status_var.set(f"CV定位失败：{exc}")
                self.result_var.set("CV定位失败")
                messagebox.showerror("CV定位失败", str(exc))
            finally:
                self._awaiting_unknown_anchor = False
            return
        if self._use_roi_affine_location():
            try:
                self._capture_and_run_roi_affine_location()
            except Exception as exc:
                self.status_var.set(f"全局仿射定位失败：{exc}")
                self.result_var.set("全局仿射定位失败")
                messagebox.showerror("全局仿射定位失败", str(exc))
            finally:
                self._awaiting_unknown_anchor = False
            return
        if self._use_structure_registration_location():
            try:
                self._capture_and_run_structure_registration()
            except Exception as exc:
                self.status_var.set(f"结构配准定位失败：{exc}")
                self.result_var.set("结构配准定位失败")
                messagebox.showerror("结构配准定位失败", str(exc))
            finally:
                self._awaiting_unknown_anchor = False
            return
        point = self._global_cursor_position()
        if point is None:
            locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
            self.status_var.set(f"无法读取鼠标位置，请重新移动光标并按 {locate_hotkey}。")
            self.cheatsheet.show_message(
                f"无法读取鼠标位置\n请重新移动光标到侧门锚点并按 {locate_hotkey}",
                title="侧门锚点失败",
            )
            return
        self.status_var.set(f"收到侧门锚点：屏幕坐标 {point}")
        unknown_anchor = self._capture_alignment_roi_and_point(point)
        if unknown_anchor is None:
            locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
            self.status_var.set(f"光标不在本次地图截图区域内，请把光标移到侧门锚点上方并按 {locate_hotkey}。")
            app = self.project_config.app
            capture_key = str(app.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))
            self.cheatsheet.show_message(
                f"光标未落入对齐截图区域\n光标坐标：{point}\n{capture_key}={app.get(capture_key)}\n请确认游戏地图在该截图区域内",
                title="侧门锚点未命中",
            )
            return
        try:
            self._last_unknown_anchor = unknown_anchor
            self._run_background_alignment(unknown_anchor)
        except Exception as exc:
            self.status_var.set(f"后台对齐失败：{exc}")
            self.result_var.set("后台对齐失败")
            messagebox.showerror("后台对齐失败", str(exc))
        finally:
            self._awaiting_unknown_anchor = False

    def _capture_alignment_roi_and_point(self, screen_point: tuple[int, int]) -> Optional[tuple[float, float]]:
        app = self.project_config.app
        capture_key = str(app.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))
        map_roi_key = str(app.get("alignment_map_roi_key", "alignment_map_roi"))
        try:
            capture, alignment_roi, scaled_map_roi = capture_location_roi(
                app,
                capture_key=capture_key,
                map_roi_key=map_roi_key,
                message="对齐比例尺截图成功",
            )
        except (CaptureError, ValueError, RuntimeError) as exc:
            self.status_var.set(f"对齐截图失败：{exc}")
            self.cheatsheet.show_message(
                f"对齐截图失败\n{capture_key}={app.get(capture_key)}\n{map_roi_key}={app.get(map_roi_key)}\n{exc}",
                title="对齐截图失败",
            )
            return None
        capture_x, capture_y, capture_w, capture_h = capture.roi
        screen_x, screen_y = screen_point
        local_x = screen_x - capture_x
        local_y = screen_y - capture_y
        if local_x < 0 or local_y < 0 or local_x >= capture_w or local_y >= capture_h:
            return None
        map_x, map_y, map_w, map_h = scaled_map_roi
        if local_x < map_x or local_y < map_y or local_x >= map_x + map_w or local_y >= map_y + map_h:
            return None
        roi_x = local_x - map_x
        roi_y = local_y - map_y
        image_h, image_w = alignment_roi.shape[:2]
        if image_w <= 0 or image_h <= 0:
            return None
        self.last_alignment_capture = capture
        self.last_alignment_roi = alignment_roi
        self.raw_panel.set_image(capture.image)
        self.roi_panel.set_image(alignment_roi)
        self._set_main_alignment_preview(alignment_roi)
        return (
            max(0.0, min(float(image_w - 1), float(roi_x))),
            max(0.0, min(float(image_h - 1), float(roi_y))),
        )

    def _selected_sample_map_id(self) -> str:
        raw = self.sample_map_var.get().strip()
        return raw.split()[0] if raw else ""

    def _load_side_door_layout_variants(self) -> dict[str, tuple[str, ...]]:
        configured = self.project_config.app.get(
            "side_door_layout_variants_path",
            "config/side_door_layout_variants.json",
        )
        path = resolve_config_path(self.project_config.root_dir, str(configured))
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            LOGGER.debug("failed to load side-door layout variants: %s", path, exc_info=True)
            return {}
        groups = data.get("groups") if isinstance(data, dict) else None
        if not isinstance(groups, list):
            return {}
        variants: dict[str, tuple[str, ...]] = {}
        for item in groups:
            if not isinstance(item, dict):
                continue
            layouts = item.get("layouts")
            if not isinstance(layouts, list):
                continue
            normalized = tuple(dict.fromkeys(str(layout).strip() for layout in layouts if str(layout).strip()))
            if len(normalized) < 2:
                continue
            for layout in normalized:
                variants[layout] = normalized
        return variants

    def _set_active_route_layouts(self, map_id: str) -> None:
        base = base_map_id_from_alignment(map_id)
        layouts = self._side_door_layout_variants.get(base, (base,))
        self._active_route_layouts = layouts
        self._active_route_layout_index = layouts.index(base) if base in layouts else 0

    def _active_route_layout_id(self) -> Optional[str]:
        if self._active_route_layouts:
            return self._active_route_layouts[self._active_route_layout_index]
        map_id = self._confirmed_alignment_map_id or self._pending_alignment_map_id
        return base_map_id_from_alignment(map_id) if map_id else None

    def _location_layout_map_id(self) -> Optional[str]:
        return self._active_route_layout_id()

    def _has_multiple_route_layouts(self) -> bool:
        return len(self._active_route_layouts) > 1

    def _route_layout_switch_hint(self) -> str:
        if self._side_door_structure_confirmed and self._has_multiple_route_layouts():
            switch_hotkey = configured_hotkey_display_name(self, "switch_layout")
            locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
            return f"当前侧门结构对应多种布局：按{switch_hotkey}切换；按{locate_hotkey}定位当前显示布局"
        return ""

    def _show_layout_lock_required(self, hotkey: str) -> None:
        if self._recognition_confirm_pending:
            message = "请先在小抄界面确认当前侧门结构是否正确。"
            title = "等待确认侧门结构"
        else:
            message = f"请先按{configured_hotkey_display_name(self, 'recognize')}识别侧门结构，并在小抄界面确认。"
            title = "尚未确认侧门结构"
        self.status_var.set(message.replace("\n", " "))
        self.cheatsheet.show_message(message, title=title)

    def _return_to_variant_layout_selection(self) -> None:
        return

    def _can_return_to_variant_layout_selection(self) -> bool:
        return False

    def _should_show_lianghapi_dual_cheatsheet(self) -> bool:
        profile = getattr(getattr(self, "recognition_profile", None), "key", None)
        return lianghapi_dual_cheatsheet_enabled(self.project_config.app, profile_key=profile)

    def _is_nightmare_mode(self) -> bool:
        return getattr(self, "recognition_profile", None) is not None and self.recognition_profile.key == "nightmare"

    def _uses_nightmare_zhanshi_routes(self) -> bool:
        return self._is_nightmare_mode() and nightmare_zhanshi_route_enabled(self.project_config.app)

    def _uses_nightmare_lianghapi_routes(self) -> bool:
        return self._is_nightmare_mode() and nightmare_lianghapi_route_enabled(self.project_config.app)

    def _nightmare_lianghapi_single_floor_image(
        self,
        route_image: np.ndarray,
        floor_map_id: str,
    ) -> Optional[np.ndarray]:
        """Extract the current floor from Lianghapi's reviewed three-floor canvas."""
        template_path = resolve_floor_template_path(
            self.project_config.app,
            floor_map_id,
            "b1" if floor_map_id.endswith("_b1") else "2nd" if floor_map_id.endswith("_2nd") else "1st",
        )
        template_image = self._semantic_cv_cached_image(template_path) if template_path is not None else None
        matrix = self._new_layout_route_affine_matrix(
            floor_map_id,
            (route_image.shape[1], route_image.shape[0]),
        )
        if template_image is None or matrix is None:
            return None
        try:
            return crop_route_image_to_template_bounds(
                route_image,
                (template_image.shape[1], template_image.shape[0]),
                matrix,
                padding=int(self.project_config.app.get("nightmare_lianghapi_single_floor_crop_padding", 16)),
            )
        except ValueError:
            LOGGER.warning("failed to crop Nightmare Lianghapi route for %s", floor_map_id, exc_info=True)
            return None

    def _uses_nightmare_route_maps(self) -> bool:
        return self._is_nightmare_mode() and nightmare_route_map_enabled(self.project_config.app)

    def _uses_versioned_route_for_map(self, map_id: str) -> bool:
        base = base_map_id_from_alignment(map_id)
        if base.startswith("nightmare_"):
            return nightmare_route_map_enabled(self.project_config.app)
        return str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower() == "zhanshi"

    def _show_active_route_layout(self) -> bool:
        layout_id = self._active_route_layout_id()
        if layout_id is None:
            return False
        floor_map_id = self._floor_map_id(layout_id, self._active_floor_suffix)
        if self._is_nightmare_mode() and not self._uses_nightmare_route_maps():
            image = self._nightmare_cheatsheet_image(layout_id, self._active_floor_suffix)
            if image is None:
                self.cheatsheet.show_message(
                    f"缺少噩梦干净模板：{floor_map_id}",
                    title="噩梦小抄",
                )
                return False
            floor_map_id = (
                f"{base_map_id_from_alignment(layout_id)}_三层"
                if nightmare_three_cheatsheet_enabled(self.project_config.app)
                else floor_map_id
            )
        elif self._uses_nightmare_lianghapi_routes() and nightmare_lianghapi_single_cheatsheet_enabled(self.project_config.app):
            route_path = self._resolve_route_template_path(floor_map_id)
            route_image = imread_unicode(route_path) if route_path is not None else None
            image = (
                self._nightmare_lianghapi_single_floor_image(route_image, floor_map_id)
                if route_image is not None
                else None
            )
            if image is None:
                self.cheatsheet.show_message(
                    f"缺少凉哈皮单图裁切标定：{floor_map_id}",
                    title="路线图",
                )
                return False
        elif self._should_show_lianghapi_dual_cheatsheet():
            route_images: list[np.ndarray] = []
            for suffix in ("1st", "2nd"):
                route_path = self._resolve_route_template_path(self._floor_map_id(layout_id, suffix))
                image_for_floor = imread_unicode(route_path) if route_path is not None else None
                if image_for_floor is None:
                    self.cheatsheet.show_message(
                        f"缺少路线图：{self._floor_map_id(layout_id, suffix)}",
                        title="路线图",
                    )
                    return False
                route_images.append(image_for_floor)
            if len(route_images) != 2:
                self.cheatsheet.show_message(
                    f"缺少路线图：{base_map_id_from_alignment(layout_id)}",
                    title="路线图",
                )
                return False
            image = make_route_floor_contact_sheet(layout_id, route_images[0], route_images[1])
            floor_map_id = base_map_id_from_alignment(layout_id)
        else:
            route_path = self._resolve_route_template_path(floor_map_id)
            if route_path is None:
                self.cheatsheet.show_message(
                    f"缺少路线图：{floor_map_id}",
                    title="路线图",
                )
                return False
            image = imread_unicode(route_path)
        if self._uses_versioned_route_for_map(floor_map_id):
            floor_map_id = base_map_id_from_alignment(layout_id)
        footer = self._route_layout_switch_hint()
        title_prefix = "干净模板" if self._is_nightmare_mode() and not self._uses_nightmare_route_maps() else "路线图"
        self.cheatsheet.show_image(image, title=f"{title_prefix}：{floor_map_id}", footer=footer, compact_footer=True)
        return True

    def cycle_side_door_layout(self) -> None:
        if not self._side_door_structure_confirmed or not self._has_multiple_route_layouts():
            return
        self._active_route_layout_index = (self._active_route_layout_index + 1) % len(self._active_route_layouts)
        layout_id = self._active_route_layout_id()
        if layout_id is not None:
            self._pending_alignment_map_id = self._floor_map_id(layout_id, "1st")
            self._confirmed_alignment_map_id = self._pending_alignment_map_id
        if self._show_active_route_layout() and layout_id is not None:
            locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
            self.status_var.set(
                f"已切换路线布局：{layout_id}_{self._active_floor_suffix}；"
                f"按{locate_hotkey}将定位当前显示布局"
            )

    def _summary_from_cnn(self, current_roi: np.ndarray, cnn_scores: dict[str, float]) -> MatchSummary:
        current_pre = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        ranked = sorted(cnn_scores.items(), key=lambda item: item[1], reverse=True)
        results: list[MatchResult] = []
        for rank, (map_id, score) in enumerate(ranked, 1):
            cfg = self.project_config.maps.get(map_id)
            if cfg is None and map_id not in self.recognition_profile.labels:
                continue
            map_name = cfg.name if cfg is not None else self.recognition_profile.label_name(map_id)
            results.append(
                MatchResult(
                    map_id=map_id,
                    map_name=map_name,
                    color_score=0.0,
                    ssim_score=0.0,
                    deep_score=0.0,
                    candidate_score=score,
                    template_score=0.0,
                    feature_score=0.0,
                    structure_score=0.0,
                    partial_structure_score=0.0,
                    entrance_patch_score=0.0,
                    entrance_shape_score=0.0,
                    entrance_direction_score=0.0,
                    entrance_corridor_pair_score=0.0,
                    anchored_subset_score=0.0,
                    corridor_component_score=0.0,
                    entry_corridor_score=0.0,
                    door_anchor_score=0.0,
                    final_score=score,
                    cnn_score=score,
                    cnn_rank=rank,
                    source="cnn",
                    sample_count=len([sample for sample in self.sample_db.samples if sample.map_id == map_id]),
                )
            )
        display_n = int(self.project_config.app.get("display_top_n", 5))
        top = results[:display_n]
        best = top[0] if top else None
        reliable = False
        status = "cnn no result"
        if best is not None:
            second = top[1].cnn_score if len(top) > 1 else 0.0
            second_map_id = top[1].map_id if len(top) > 1 else ""
            margin = best.cnn_score - second
            threshold = float(self.project_config.app.get("cnn_only_threshold", 0.50))
            required_margin = float(self.project_config.app.get("cnn_only_margin", 0.15))
            force_confirm_pair = {best.map_id, second_map_id} == {"map_01", "map_18"}
            if force_confirm_pair:
                status = (
                    f"cnn-only ambiguous: {best.map_id} vs {second_map_id}, "
                    f"cnn={best.cnn_score:.3f}, second={second:.3f}, margin={margin:.3f}"
                )
            elif best.cnn_score >= threshold and margin >= required_margin:
                reliable = True
                status = f"cnn-only matched: {best.map_id} / {best.map_name}, cnn={best.cnn_score:.3f}, margin={margin:.3f}"
            else:
                status = f"cnn-only uncertain: cnn={best.cnn_score:.3f}, margin={margin:.3f}"
            best.is_reliable = reliable
            best.reason = status
        return MatchSummary(
            results=results,
            top3=top,
            top_candidates=[],
            best=best,
            is_reliable=reliable,
            status=status,
            current_preprocessed=current_pre,
            best_template_preprocessed=None,
        )

    def save_current_roi_as_sample(self) -> None:
        if self.last_current_roi is None:
            messagebox.showinfo("没有截图", f"请先按 {configured_hotkey_display_name(self, 'recognize')} 截取当前游戏地图 ROI。")
            return
        map_id = self._selected_sample_map_id()
        if map_id not in self.project_config.maps:
            messagebox.showerror("地图无效", f"请选择有效地图：{map_id}")
            return
        try:
            sample = self.sample_db.save_sample(map_id, self.last_current_roi, app_config=self.project_config.app)
        except Exception as exc:
            messagebox.showerror("保存样本失败", str(exc))
            return
        self.status_var.set(f"已保存截图样本：{sample.sample_id} -> {sample.map_id}，版本 {sample.roi_version}，样本总数 {len(self.sample_db.samples)}")
        messagebox.showinfo("保存成功", f"已把当前 ROI 保存为样本：\n{sample.sample_id}\n绑定地图：{sample.map_id} / {sample.map_name}\nROI版本：{sample.roi_version}")

    def save_current_roi_as_pending(self) -> None:
        if self.last_current_roi is None:
            messagebox.showinfo("没有截图", f"请先按 {configured_hotkey_display_name(self, 'recognize')} 截取当前游戏地图 ROI。")
            return
        try:
            item = self.sample_db.save_pending_sample(self.last_current_roi, app_config=self.project_config.app)
        except Exception as exc:
            messagebox.showerror("保存待标注失败", str(exc))
            return
        self.status_var.set(f"已保存待标注样本：{item['sample_id']}，版本 {item.get('roi_version', '')}，待标注总数 {self.sample_db.pending_count()}")
        messagebox.showinfo("保存成功", f"已保存为待标注样本：\n{item['sample_id']}\nROI版本：{item.get('roi_version', '')}\n之后可以再绑定到新地图。")

    def _update_cheatsheet(self, summary: MatchSummary) -> None:
        if summary.best is None:
            return
        min_score = float(self.project_config.app.get("cheatsheet_min_confidence", 0.35))
        score = summary.best.cnn_score if summary.best.cnn_score > 0 else summary.best.final_score
        if score >= min_score:
            alignment_map_id = self._alignment_map_id(summary.best.map_id)
            image = load_entry_template(self.project_config.app, alignment_map_id)
            if image is None:
                image = load_floor_template(self.project_config.app, alignment_map_id, self.recognition_profile.entry_floor)
            if image is not None:
                self.cheatsheet.show_image(image, title=f"{alignment_map_id} / 入口结构    CNN {score:.3f}")
            else:
                self.cheatsheet.show_map(summary.best.map_id, score)
        else:
            self.cheatsheet.show_message(
                f"识别置信度偏低：{summary.best.map_id} / {summary.best.map_name}  {score:.3f}",
                title="暂不显示完整地图",
            )

    def _best_confidence(self, summary: MatchSummary) -> float:
        if summary.best is None:
            return 0.0
        return float(summary.best.cnn_score if summary.best.cnn_score > 0 else summary.best.final_score)

    def _recognition_confirm_threshold(self, summary: Optional[MatchSummary] = None) -> float:
        if self._is_ambiguous_family_present(summary):
            return float(self.project_config.app.get("ambiguous_map_confirm_threshold", self.project_config.app.get("map04_map16_confirm_threshold", 0.60)))
        return float(self.project_config.app.get("recognition_confirm_threshold", 0.70))

    def _recognition_loop_interval_ms(self) -> int:
        return max(80, int(self.project_config.app.get("recognition_loop_interval_ms", 180)))

    def _show_map_confirmation(self, summary: MatchSummary) -> None:
        if summary.best is None:
            return
        self._active_ambiguous_map_ids = self._ambiguous_map_ids_for_summary(summary)
        self._manual_map_select_active = bool(self._active_ambiguous_map_ids)
        score = self._best_confidence(summary)
        confirm_map_id = self._alignment_map_id(summary.best.map_id)
        confirm_hotkey = hotkey_binding_display_name(
            self._hotkeys.get("confirm_structure", HOTKEY_DEFAULTS["confirm_structure"])
        )
        confirm_yes_text = f"是，结构正确（{confirm_hotkey}）"
        message = f"识别置信度 {score:.3f}\n当前侧门结构是否正确？"
        template_image = load_entry_template(self.project_config.app, confirm_map_id)
        if template_image is None:
            template_image = load_floor_template(
                self.project_config.app,
                confirm_map_id,
                self.recognition_profile.entry_floor,
            )
        try:
            if template_image is None:
                known_anchors_path = resolve_config_path(
                    self.project_config.root_dir,
                    str(self.project_config.app.get("alignment_known_anchors_path", "config/side_door_anchors_1st.json")),
                )
                _known_point, full_image_path = load_known_anchor(self.project_config.root_dir, known_anchors_path, confirm_map_id)
                template_image = imread_unicode(full_image_path)
            if template_image is None:
                raise RuntimeError(f"无法读取入口结构例图：{confirm_map_id}")
        except Exception as exc:
            LOGGER.debug("failed to load entry structure image for confirmation", exc_info=True)
            template_image = None
            message = f"{summary.best.map_id} / {summary.best.map_name}\n识别置信度 {score:.3f}\n当前侧门结构是否正确？\n{exc}"
        choices_image = None
        choice_regions: list[tuple[str, tuple[int, int, int, int]]] = []
        if template_image is not None and self._active_ambiguous_map_ids:
            choices_image, choice_regions = self._make_ambiguous_entry_contact_sheet()
        if template_image is not None and choices_image is not None and choice_regions:
            self.cheatsheet.show_image_confirmation_with_choices(
                template_image,
                message,
                choices_image,
                choice_regions,
                title=f"确认当前侧门结构：{confirm_map_id}",
                yes_text=confirm_yes_text,
                no_text="否，重新识别",
                on_yes=self._accept_recognized_side_door_structure,
                on_no=self._reject_recognized_side_door_structure,
                on_choice=self.confirm_manual_map_choice,
            )
        elif template_image is not None:
            self.cheatsheet.show_image_confirmation(
                template_image,
                message,
                title=f"确认当前侧门结构：{confirm_map_id}",
                yes_text=confirm_yes_text,
                no_text="否，重新识别",
                on_yes=self._accept_recognized_side_door_structure,
                on_no=self._reject_recognized_side_door_structure,
            )
        else:
            self.cheatsheet.show_confirmation(
                message,
                title="确认当前侧门结构",
                yes_text=confirm_yes_text,
                no_text="否，重新识别",
                on_yes=self._accept_recognized_side_door_structure,
                on_no=self._reject_recognized_side_door_structure,
            )
        self.status_var.set(f"等待玩家确认侧门结构：{summary.best.map_id}，置信度 {score:.3f}")
        self.result_var.set(f"确认侧门结构：{summary.best.map_id} / {summary.best.map_name}  {score:.3f}")

    def start_recognition_loop(self, from_hotkey: bool = False) -> None:
        self._awaiting_unknown_anchor = False
        self._pending_alignment_map_id = None
        self._set_ambiguous_choice_visible(False)
        self._recognition_confirm_pending = False
        self._recognition_loop_active = False
        self._recognition_loop_from_hotkey = from_hotkey
        self._recognition_frame_index = 0
        clear_temporary_size = getattr(self.cheatsheet, "clear_temporary_size", None)
        if callable(clear_temporary_size):
            clear_temporary_size()
        self.cheatsheet.clear_click_handler()
        self.status_var.set("开始单帧识别，识别后请在小抄界面确认侧门结构。")
        self.run_recognition(from_hotkey=from_hotkey, loop_mode=False)

    def request_recognition_start(self, from_hotkey: bool = False) -> None:
        if bool(getattr(self, "_program_paused", False)):
            self.status_var.set("程序当前处于暂停锁定状态，识别请求已忽略。")
            return
        recognize_hotkey = configured_hotkey_display_name(self, "recognize")
        now = time.monotonic()
        last_start = float(getattr(self, "_last_recognition_start_time", 0.0))
        if now - last_start < 0.65:
            self.status_var.set(f"{recognize_hotkey} 识别正在启动，请稍候。")
            return
        if bool(getattr(self, "_recognition_running", False)):
            self.status_var.set(f"{recognize_hotkey} 识别仍在运行，请稍候。")
            return
        self._last_recognition_start_time = now
        start_preview_session = getattr(self, "_start_main_recognition_preview_session", None)
        if callable(start_preview_session):
            start_preview_session()
        self._confirmed_alignment_map_id = None
        self._pending_alignment_map_id = None
        self._side_door_structure_confirmed = False
        self._first_location_map_zoom_pending = False
        self._active_route_layouts = ()
        self._active_route_layout_index = 0
        initial_location_floors = getattr(self, "_semantic_cv_initial_location_floors", None)
        if isinstance(initial_location_floors, set):
            initial_location_floors.clear()
        else:
            self._semantic_cv_initial_location_floors = set()
        attempted_floors = getattr(self, "_semantic_cv_location_attempted_floors", None)
        if isinstance(attempted_floors, set):
            attempted_floors.clear()
        else:
            self._semantic_cv_location_attempted_floors = set()
        self._last_unknown_anchor = None
        self._first_floor_side_door_constraint_active = False
        # F8 starts a new layout session. The first G initializes the detected
        # own floor-specific scales after the route layout is locked.
        self._semantic_cv_scale_cache.clear()
        self._semantic_cv_device_multiplier_cache.clear()
        self._semantic_cv_scale_confirmation_pending = None
        self._manual_scale_research_return_content = None
        self._scale_research_return_context = None
        retry_floors = getattr(self, "_semantic_cv_scale_retry_floors", None)
        if isinstance(retry_floors, set):
            retry_floors.clear()
        else:
            self._semantic_cv_scale_retry_floors = set()
        self._semantic_cv_postprocess_id = int(getattr(self, "_semantic_cv_postprocess_id", 0)) + 1
        cheatsheet = getattr(self, "cheatsheet", None)
        if cheatsheet is not None:
            expand_on_next_show = getattr(cheatsheet, "expand_on_next_show", None)
            if callable(expand_on_next_show):
                expand_on_next_show()
            clear_handler = getattr(cheatsheet, "clear_click_handler", None)
            if callable(clear_handler):
                clear_handler()
            hide_sheet = getattr(cheatsheet, "hide", None)
            if callable(hide_sheet):
                hide_sheet()
        route_overlay = getattr(self, "route_overlay", None)
        hide_overlay = getattr(route_overlay, "hide", None)
        if callable(hide_overlay):
            hide_overlay()
        self.start_recognition_loop(from_hotkey=from_hotkey)

    def confirm_recognized_map(self) -> None:
        """Compatibility entry point for accepting the on-sheet structure choice."""
        self._accept_recognized_side_door_structure()

    def _accept_recognized_side_door_structure(self) -> None:
        if not self._recognition_confirm_pending:
            return
        if self.last_summary is None or self.last_summary.best is None:
            return
        self._recognition_confirm_pending = False
        self._recognition_loop_active = False
        lock_preview = getattr(self, "_lock_main_recognition_preview", None)
        if callable(lock_preview):
            lock_preview()
        record_capture_profile = getattr(self, "_record_confirmed_window_capture_profile", None)
        if callable(record_capture_profile):
            record_capture_profile()
        self._manual_map_select_active = False
        self._set_ambiguous_choice_visible(False)
        clear_temporary_size = getattr(self.cheatsheet, "clear_temporary_size", None)
        if callable(clear_temporary_size):
            clear_temporary_size()
        self.cheatsheet.clear_click_handler()
        self._confirm_key_down = self._is_global_key_down(0x0D)
        entry_floor = getattr(getattr(self, "recognition_profile", None), "entry_floor", "1st")
        self._active_floor_suffix = entry_floor
        self._side_door_structure_confirmed = True
        self._set_active_route_layouts(self.last_summary.best.map_id)
        layout_id = self._active_route_layout_id()
        if layout_id is None:
            return
        self._first_location_map_zoom_pending = True
        self._pending_alignment_map_id = self._floor_map_id(layout_id, entry_floor)
        self._confirmed_alignment_map_id = self._pending_alignment_map_id
        self._show_active_route_layout()
        schedule_warmup = getattr(self, "_schedule_route_overlay_warmup", None)
        if callable(schedule_warmup):
            schedule_warmup()
        self.result_var.set(f"侧门结构正确，当前布局：{layout_id}")
        locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
        if self._has_multiple_route_layouts():
            switch_hotkey = configured_hotkey_display_name(self, "switch_layout")
            self.status_var.set(
                f"侧门结构已确认；按{switch_hotkey}切换变体布局，"
                f"按{locate_hotkey}定位当前显示布局。"
            )
        else:
            self.status_var.set(f"已确认唯一布局：{layout_id}；现在可以按 {locate_hotkey} 自动识别楼层并定位。")

    def _record_confirmed_window_capture_profile(self) -> None:
        """Persist the client size validated by the player's F8 confirmation."""
        capture = self.last_capture
        if capture is None or capture.mode != "scaled_cnn_game_client" or not capture.target_size:
            return
        relative_path = str(self.project_config.app.get("game_window_profile_path", "config/window_capture_profiles.json"))
        profile_path = self.project_config.root_dir / relative_path
        try:
            payload: dict[str, Any] = {"profiles": {}}
            if profile_path.exists():
                loaded = json.loads(profile_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload.update(loaded)
            profiles = payload.setdefault("profiles", {})
            if not isinstance(profiles, dict):
                profiles = payload["profiles"] = {}
            width, height = capture.target_size
            title = capture.window_title or "game_client"
            profiles[f"{title}|{width}x{height}"] = {
                "window_title": title,
                "client_size": [int(width), int(height)],
                "cnn_reference_size": list(capture.reference_size or (2560, 1351)),
                "alignment_reference_size": list(self.project_config.app.get("alignment_reference_capture_size", (2560, 1440))),
                "confirmed_at": timestamp(),
            }
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.warning("Unable to save window capture profile: %s", exc)

    def _reject_recognized_side_door_structure(self) -> None:
        if not self._recognition_confirm_pending:
            return
        self._recognition_confirm_pending = False
        self._recognition_loop_active = False
        self._manual_map_select_active = False
        self._side_door_structure_confirmed = False
        self._first_location_map_zoom_pending = False
        self._pending_alignment_map_id = None
        self._confirmed_alignment_map_id = None
        self._active_route_layouts = ()
        clear_temporary_size = getattr(self.cheatsheet, "clear_temporary_size", None)
        if callable(clear_temporary_size):
            clear_temporary_size()
        self.cheatsheet.clear_click_handler()
        recognize_hotkey = configured_hotkey_display_name(self, "recognize")
        self.cheatsheet.show_message(
            f"当前侧门结构识别不正确。\n请按{recognize_hotkey}重新识别。",
            title="请重新识别",
        )
        self.result_var.set("侧门结构未确认")
        self.status_var.set(f"玩家否认当前侧门结构；等待按{recognize_hotkey}重新识别。")

    def confirm_manual_map_choice(self, map_id: str) -> None:
        if not self._manual_map_select_active or self.last_summary is None:
            return
        selected = next((result for result in self.last_summary.results if result.map_id == map_id), None)
        if selected is None:
            cfg = self.project_config.maps.get(map_id)
            if cfg is None:
                return
            selected = MatchResult(
                map_id=map_id,
                map_name=cfg.name,
                color_score=0.0,
                ssim_score=0.0,
                deep_score=0.0,
                candidate_score=0.0,
                template_score=0.0,
                feature_score=0.0,
                structure_score=0.0,
                partial_structure_score=0.0,
                entrance_patch_score=0.0,
                entrance_shape_score=0.0,
                entrance_direction_score=0.0,
                entrance_corridor_pair_score=0.0,
                anchored_subset_score=0.0,
                corridor_component_score=0.0,
                entry_corridor_score=0.0,
                door_anchor_score=0.0,
                final_score=1.0,
                cnn_score=float(self.last_cnn_scores.get(map_id, 0.0)),
                source="manual",
            )
        self.last_summary.best = selected
        self.last_summary.is_reliable = True
        self.last_summary.status = f"manual selected: {map_id}"
        selected.is_reliable = True
        selected.reason = self.last_summary.status
        self._set_ambiguous_choice_visible(False)
        self._recognition_confirm_pending = True
        self._accept_recognized_side_door_structure()

    def reject_recognized_map(self) -> None:
        confirm_hotkey = configured_hotkey_display_name(self, "confirm_structure")
        recognize_hotkey = configured_hotkey_display_name(self, "recognize")
        self.status_var.set(f"{confirm_hotkey} 已停用；如需重新识别请按 {recognize_hotkey}。")

    def run_recognition(self, from_hotkey: bool = False, loop_mode: bool = False) -> None:
        if loop_mode and (not self._recognition_loop_active or self._recognition_confirm_pending):
            return
        if self._recognition_running:
            return
        self._recognition_running = True
        try:
            capture, current_roi = capture_scaled_cnn_recognition_roi(self.project_config.app)
            cnn_scores: dict[str, float] = {}
            if self.cnn_classifier is None:
                detail = self.cnn_load_error or "未知原因"
                raise RuntimeError(
                    "CNN 未启用，已取消识别。当前版本只允许使用 CNN，不再回退到 map_template。"
                    f"\n原因：{detail}"
                    "\n请确认 models/map_cnn.onnx 和 models/map_cnn_info.json 与 exe 在同一个解压目录内。"
                )
            cnn_scores = dict(self.cnn_classifier.predict_proba(current_roi))
            if not cnn_scores:
                raise RuntimeError("CNN 没有返回任何分数，已取消识别。")
            self.last_cnn_scores = cnn_scores
            summary = self._summary_from_cnn(current_roi, cnn_scores)
        except (CaptureError, ValueError, RuntimeError) as exc:
            self.status_var.set(f"识别失败：{exc}")
            self.result_var.set("识别失败")
            if from_hotkey:
                self.cheatsheet.show_message(f"识别失败：{exc}", title="识别失败")
            else:
                messagebox.showerror("识别失败", str(exc))
            self._recognition_running = False
            return

        try:
            self.last_capture = capture
            self.last_current_roi = current_roi
            self.last_summary = summary
            self.raw_panel.set_image(capture.image)
            self.roi_panel.set_image(current_roi)
            self._set_main_recognition_preview(current_roi)

            if summary.best is None:
                self.result_var.set("无法识别")
            else:
                prefix = "[OK]" if summary.is_reliable else "[?]"
                self.result_var.set(
                    f"{prefix} {summary.best.map_id} / {summary.best.map_name}  分数 {summary.best.final_score:.3f}"
                )

            self.status_var.set(f"{capture.message}；{summary.status}")
            LOGGER.info("\n%s", self._format_log(capture, summary))

            if bool(self.project_config.app.get("debug_save", True)):
                self.save_debug_images(silent=True)
            if summary.best is not None:
                self._recognition_loop_active = False
                self._recognition_confirm_pending = True
                self._manual_map_select_active = False
                self._confirm_key_down = False
                self._show_map_confirmation(summary)
            else:
                self._update_cheatsheet(summary)
        finally:
            self._recognition_running = False

    def _alignment_map_id(self, base_map_id: str) -> str:
        entry_floor = getattr(getattr(self, "recognition_profile", None), "entry_floor", "1st")
        floor_suffix = str(
            self.project_config.app.get("alignment_known_map_suffix", entry_floor)
        ).strip()
        if floor_suffix and not base_map_id.endswith(f"_{floor_suffix}"):
            return f"{base_map_id}_{floor_suffix}"
        return base_map_id

    def _floor_map_id(self, alignment_map_id: str, floor_suffix: str) -> str:
        return f"{base_map_id_from_alignment(alignment_map_id)}_{floor_suffix}"

    def _request_second_floor_location(self) -> None:
        now = time.monotonic()
        if self._second_floor_location_pending or now - self._last_second_floor_request_time < 0.35:
            return
        self._last_second_floor_request_time = now
        self.switch_to_second_floor()

    def switch_to_second_floor(self) -> None:
        if self._active_floor_suffix == "1st":
            self._second_floor_transition_constraint_active = True
        self._active_floor_suffix = "2nd"
        if self._confirmed_alignment_map_id is None:
            self._show_layout_lock_required("G")
            return
        self._show_active_route_layout()
        if self._second_floor_location_pending:
            return
        self._second_floor_location_pending = True
        delay_ms = max(0, int(self.project_config.app.get("second_floor_location_delay_ms", 250)))
        self.status_var.set("已切换到二楼，正在准备 CV 定位玩家。")
        if delay_ms == 0:
            self._run_second_floor_location()
        else:
            self.after(delay_ms, self._run_second_floor_location)

    def _run_second_floor_location(self) -> None:
        self._second_floor_location_pending = False
        if self._confirmed_alignment_map_id is None:
            return
        if self._use_semantic_cv_location():
            self.status_var.set("已切换到二楼定位，正在使用 CV 语义匹配定位玩家。")
            try:
                require_confirmation = semantic_cv_requires_initial_scale_confirmation(
                    self._semantic_cv_initial_location_floors,
                    self._semantic_cv_scale_retry_floors,
                    "2nd",
                )
                self._capture_and_run_semantic_cv_location(
                    "2nd",
                    require_scale_confirmation=require_confirmation,
                )
                self._semantic_cv_initial_location_floors.add("2nd")
                self._second_floor_transition_constraint_active = False
            except Exception as exc:
                self.status_var.set(f"二楼CV定位失败：{exc}")
                self.cheatsheet.show_message(f"二楼CV定位失败\n{exc}", title="二楼定位失败")
            return
        try:
            self.status_var.set("已切换到二楼定位，正在重新定位玩家。")
            self._capture_and_run_second_floor_roi_affine_location()
        except Exception as exc:
            self.status_var.set(f"二楼定位失败：{exc}")
            self.cheatsheet.show_message(f"二楼定位失败\n{exc}", title="二楼定位失败")

    def _active_template_image(self, map_id: str, fallback: np.ndarray) -> tuple[str, np.ndarray]:
        floor_map_id = self._floor_map_id(map_id, self._active_floor_suffix)
        image = load_floor_template(self.project_config.app, map_id, self._active_floor_suffix)
        if image is None:
            return map_id, fallback
        return floor_map_id, image

    def request_side_door_alignment(self, silent: bool = False) -> None:
        recognize_hotkey = configured_hotkey_display_name(self, "recognize")
        locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
        if self.last_current_roi is None or self.last_summary is None or self.last_summary.best is None:
            if not silent:
                messagebox.showinfo("无法对齐", f"请先按 {recognize_hotkey} 识别当前地图。")
            return
        self._recognition_loop_active = False
        self._recognition_confirm_pending = False
        self._manual_map_select_active = False
        self._set_active_route_layouts(self.last_summary.best.map_id)
        self._pending_alignment_map_id = self._alignment_map_id(self.last_summary.best.map_id)
        self._active_floor_suffix = "1st"
        if self._use_semantic_cv_location():
            self._confirmed_alignment_map_id = self._pending_alignment_map_id
            self._awaiting_unknown_anchor = False
            prompt = f"已确认侧门结构，请切换到全览后按 {locate_hotkey} 进行首次自动楼层 CV 定位"
            self.status_var.set(f"{prompt}：{self._pending_alignment_map_id}")
            self.result_var.set(f"等待 {locate_hotkey} CV定位")
            self._show_active_route_layout()
            if not silent:
                messagebox.showinfo("确认定位", prompt)
            self._run_initial_side_door_cv_location()
            return
        self._awaiting_unknown_anchor = True
        self._awaiting_click_started_at = time.monotonic()
        self._anchor_key_down = self._is_global_key_down(0x71)
        if self._use_semantic_cv_location():
            prompt = f"已确认，请将游戏内地图切换到全览（固定比例尺），然后按 {locate_hotkey} 自动识别楼层并使用 CV 语义匹配定位"
        elif self._use_roi_affine_location():
            prompt = f"已确认，请将游戏内地图切换到全览（固定比例尺），然后按 {locate_hotkey} 使用全局仿射矩阵定位"
        elif self._use_structure_registration_location():
            prompt = f"已确认，请将游戏内地图切换到全览（固定比例尺），然后按 {locate_hotkey} 进行结构配准定位"
        else:
            prompt = f"已确认，请将游戏内地图切换到全览（刚刚所固定的截图规范比例尺），再移动光标至侧门锚点上方，并按 {locate_hotkey}"
        self.status_var.set(f"{prompt}：{self._pending_alignment_map_id}")
        if self._use_semantic_cv_location():
            self.result_var.set(f"等待 {locate_hotkey} CV定位")
            title = "等待CV定位"
        elif self._use_roi_affine_location():
            self.result_var.set(f"等待 {locate_hotkey} 全局仿射定位")
            title = "等待全局仿射定位"
        elif self._use_structure_registration_location():
            self.result_var.set(f"等待 {locate_hotkey} 结构配准定位")
            title = "等待结构配准"
        else:
            self.result_var.set(f"等待 {locate_hotkey} 确认侧门锚点")
            title = "等待侧门锚点"
        self.result_var.set(f"等待 {locate_hotkey} 定位")
        title = "等待定位"
        self.cheatsheet.show_message(prompt, title=title)
        if not silent:
            messagebox.showinfo("确认定位", prompt)

    def _run_initial_side_door_cv_location(self) -> None:
        self._first_floor_side_door_constraint_active = True
        try:
            self._capture_and_run_semantic_cv_location("1st", require_scale_confirmation=True)
        except SemanticCVSearchCancelled:
            return
        except Exception as exc:
            self.status_var.set(f"initial side-door CV location failed: {exc}")
            self.cheatsheet.show_message(f"initial CV location failed\n{exc}", title="Initial location failed")
        finally:
            # Later G requests intentionally search the full first-floor template.
            self._first_floor_side_door_constraint_active = False

    def _run_background_alignment(self, unknown_anchor: tuple[float, float]) -> None:
        if self.last_alignment_roi is None or not self._pending_alignment_map_id:
            raise RuntimeError("缺少对齐比例尺截图或地图编号")
        app = self.project_config.app
        known_anchors_path = resolve_config_path(
            self.project_config.root_dir,
            str(app.get("alignment_known_anchors_path", "config/side_door_anchors_1st.json")),
        )
        scales_path = resolve_config_path(
            self.project_config.root_dir,
            str(app.get("alignment_scale_config_path", "config/screenshot_scale_calibration.json")),
        )
        result = align_unknown_to_known(
            root_dir=self.project_config.root_dir,
            map_id=self._pending_alignment_map_id,
            unknown_image=self.last_alignment_roi,
            unknown_anchor=unknown_anchor,
            known_anchors_path=known_anchors_path,
            scales_path=scales_path,
            preprocess_config=self.preprocess_config,
            default_scale=float(app.get("alignment_default_scale", 1.0)),
            overlay_alpha=float(app.get("alignment_overlay_alpha", 0.45)),
        )
        self.last_alignment_result = result
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug" / "alignment_preview")
        ts = timestamp()
        overlay_path = debug_dir / f"{ts}_{self._pending_alignment_map_id}_overlay.png"
        diff_path = debug_dir / f"{ts}_{self._pending_alignment_map_id}_diff.png"
        heatmap_path = debug_dir / f"{ts}_{self._pending_alignment_map_id}_heatmap.png"
        imwrite_unicode(overlay_path, result.overlay)
        imwrite_unicode(diff_path, result.diff)
        imwrite_unicode(heatmap_path, result.heatmap)
        self._confirmed_alignment_map_id = self._pending_alignment_map_id
        self._active_floor_suffix = "1st"
        transform = result.transform
        self.result_var.set(
            f"已后台对齐：{self._pending_alignment_map_id} "
            f"scale={transform.scale:.4f} unknown=({unknown_anchor[0]:.1f},{unknown_anchor[1]:.1f})"
        )
        LOGGER.info(
            "alignment completed map=%s scale=%.4f known_anchor=(%.1f, %.1f) "
            "unknown_anchor=(%.1f, %.1f) extra_offset=(%.1f, %.1f) top_left=(%.1f, %.1f)",
            transform.map_id,
            transform.scale,
            transform.known_anchor[0],
            transform.known_anchor[1],
            transform.unknown_anchor[0],
            transform.unknown_anchor[1],
            transform.extra_offset[0],
            transform.extra_offset[1],
            transform.top_left[0],
            transform.top_left[1],
        )
        self._render_player_location(result, self.last_alignment_roi, debug_dir, timestamp_value=ts, fallback_image=result.diff)

    def relocate_player_marker(self) -> None:
        locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
        recognize_hotkey = configured_hotkey_display_name(self, "recognize")
        self.status_var.set(f"收到 {locate_hotkey}，正在重新定位玩家。")
        self._active_floor_suffix = "1st"
        if not self._side_door_structure_confirmed or self._active_route_layout_id() is None:
            self._show_layout_lock_required(locate_hotkey)
            return
        if self._awaiting_unknown_anchor:
            self.confirm_side_door_cursor()
            return
        if self._use_semantic_cv_location():
            if self._confirmed_alignment_map_id is None:
                self._show_layout_lock_required(locate_hotkey)
                return
            first_floor_side_door_constraint = self._first_floor_side_door_constraint_active
            try:
                require_confirmation = semantic_cv_requires_initial_scale_confirmation(
                    self._semantic_cv_initial_location_floors,
                    self._semantic_cv_scale_retry_floors,
                    "1st",
                )
                self._capture_and_run_semantic_cv_location(
                    "1st",
                    require_scale_confirmation=require_confirmation,
                )
                self._semantic_cv_initial_location_floors.add("1st")
            except SemanticCVSearchCancelled:
                return
            except Exception as exc:
                self.status_var.set(f"CV重新定位失败：{exc}")
                self.cheatsheet.show_message(f"CV重新定位失败\n{exc}", title="重新定位失败")
            finally:
                if first_floor_side_door_constraint:
                    self._first_floor_side_door_constraint_active = False
            return
        if self._use_roi_affine_location():
            if self._confirmed_alignment_map_id is None:
                self.cheatsheet.show_message(f"请先按{recognize_hotkey}识别并确认侧门结构。", title="无法重新定位")
                self.status_var.set("无法重新定位玩家：缺少已确认地图。")
                return
            try:
                self._capture_and_run_roi_affine_location()
            except Exception as exc:
                self.status_var.set(f"全局仿射重新定位失败：{exc}")
                self.cheatsheet.show_message(f"全局仿射重新定位失败\n{exc}", title="重新定位失败")
            return
        if self._use_structure_registration_location():
            if self._confirmed_alignment_map_id is None:
                self.cheatsheet.show_message(f"请先按{recognize_hotkey}识别并确认侧门结构。", title="无法重新定位")
                self.status_var.set("无法重新定位玩家：缺少已确认地图。")
                return
            try:
                self._capture_and_run_structure_registration()
            except Exception as exc:
                self.status_var.set(f"结构配准重新定位失败：{exc}")
                self.cheatsheet.show_message(f"结构配准重新定位失败\n{exc}", title="重新定位失败")
            return
        if self.last_alignment_result is None or self._confirmed_alignment_map_id is None or self._last_unknown_anchor is None:
            self.cheatsheet.show_message(f"请先按{recognize_hotkey}识别并确认侧门结构。", title="无法重新定位")
            self.status_var.set("无法重新定位玩家：缺少已确认地图或侧门锚点。")
            return
        app = self.project_config.app
        capture_key = str(app.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))
        map_roi_key = str(app.get("alignment_map_roi_key", "alignment_map_roi"))
        try:
            capture, alignment_roi, _scaled_map_roi = capture_location_roi(
                app,
                capture_key=capture_key,
                map_roi_key=map_roi_key,
                message="玩家重新定位截图成功",
            )
        except (CaptureError, ValueError, RuntimeError) as exc:
            self.status_var.set(f"玩家重新定位截图失败：{exc}")
            self.cheatsheet.show_message(f"玩家重新定位截图失败\n{capture_key}={app.get(capture_key)}\n{exc}", title="重新定位失败")
            return
        self.last_alignment_capture = capture
        self.last_alignment_roi = alignment_roi
        self.raw_panel.set_image(capture.image)
        self.roi_panel.set_image(alignment_roi)
        self._set_main_alignment_preview(alignment_roi)
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug" / "alignment_preview")
        self._render_player_location(self.last_alignment_result, alignment_roi, debug_dir, fallback_image=self.last_alignment_result.diff)

    def relocate_player_marker_auto_floor(self) -> None:
        """Detect the open map floor and run the existing floor-specific CV path."""
        hotkey = hotkey_binding_display_name(
            self._hotkeys.get("locate_current_floor", HOTKEY_DEFAULTS["locate_current_floor"])
        )
        self.status_var.set(f"收到 {hotkey}，正在识别当前地图楼层。")
        if not self._side_door_structure_confirmed or self._active_route_layout_id() is None:
            self._show_layout_lock_required(hotkey)
            return
        if not self._use_semantic_cv_location():
            raise RuntimeError("自动楼层定位当前仅支持 CV 结构匹配模式")

        first_floor_constraint_was_active = self._first_floor_side_door_constraint_active
        try:
            self._capture_and_run_semantic_cv_location(None)
            detected_floor = self._active_floor_suffix
            self._semantic_cv_initial_location_floors.add(detected_floor)
            if detected_floor == "2nd":
                self._second_floor_transition_constraint_active = False
        except SemanticCVSearchCancelled:
            diagnostic_event(
                "auto_floor_location_cancelled",
                active_request_id=self._active_route_overlay_request_id,
                active_floor=getattr(self, "_active_floor_suffix", None),
                active_layout=self._active_route_layout_id(),
            )
            return
        except Exception as exc:
            diagnostic_event(
                "auto_floor_location_failed",
                exception_type=type(exc).__name__,
                exception=repr(exc),
                active_request_id=self._active_route_overlay_request_id,
                active_floor=getattr(self, "_active_floor_suffix", None),
                active_layout=self._active_route_layout_id(),
            )
            LOGGER.exception("automatic-floor CV location failed")
            self.status_var.set(f"自动楼层 CV 定位失败：{exc}")
            self.cheatsheet.show_message(f"自动楼层 CV 定位失败\n{exc}", title="定位失败")
        finally:
            if first_floor_constraint_was_active and self._active_floor_suffix == "1st":
                self._first_floor_side_door_constraint_active = False

    def _second_floor_map_id(self) -> str:
        map_id = self._location_layout_map_id()
        if map_id is None:
            raise RuntimeError("缺少 CNN 已确认地图编号")
        return self._floor_map_id(map_id, "2nd")

    def _capture_alignment_roi(self, message: str) -> tuple[CaptureInfo, np.ndarray]:
        app = self.project_config.app
        capture_key = str(app.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))
        map_roi_key = str(app.get("alignment_map_roi_key", "alignment_map_roi"))
        capture, alignment_roi, scaled_map_roi = capture_location_roi(
            app,
            capture_key=capture_key,
            map_roi_key=map_roi_key,
            message=message,
        )
        self.last_alignment_capture = capture
        self.last_alignment_roi = alignment_roi
        capture_x, capture_y, _capture_w, _capture_h = capture.roi
        map_x, map_y, map_w, map_h = scaled_map_roi
        self._last_alignment_map_screen_roi = (capture_x + map_x, capture_y + map_y, map_w, map_h)
        self.raw_panel.set_image(capture.image)
        self.roi_panel.set_image(alignment_roi)
        self._set_main_alignment_preview(alignment_roi)
        return capture, alignment_roi

    def _semantic_cv_project_root(self) -> Path:
        app = self.project_config.app
        configured = app.get("semantic_cv_project_dir")
        if configured:
            configured_root = resolve_config_path(self.project_config.root_dir, str(configured))
            if (
                (configured_root / "GameMapLocalizer.exe").exists()
                or (configured_root / "localize_frame.py").exists()
            ):
                return configured_root
        package_root = self.project_config.root_dir
        candidates = (
            package_root / "game_map_localizer",
            package_root.parent / "game_map_localizer",
        )
        for candidate in candidates:
            if (
                (candidate / "GameMapLocalizer.exe").exists()
                or (candidate / "localize_frame.py").exists()
            ):
                return candidate
        return candidates[0]

    def _semantic_cv_python(self, localizer_root: Path) -> Path:
        configured = self.project_config.app.get("semantic_cv_python")
        if configured:
            return resolve_config_path(self.project_config.root_dir, str(configured))
        candidates = [
            self.project_config.root_dir.parent.parent / ".venv" / "Scripts" / "python.exe",
            self.project_config.root_dir / ".venv-cnn" / "Scripts" / "python.exe",
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def _semantic_cv_command(self, localizer_root: Path, config_path: Path, image_path: Path, map_id: str) -> list[str]:
        configured_exe = self.project_config.app.get("semantic_cv_executable")
        localizer_exe = (
            resolve_config_path(self.project_config.root_dir, str(configured_exe))
            if configured_exe
            else localizer_root / "GameMapLocalizer.exe"
        )
        args = ["--config", str(config_path), "--image", str(image_path), "--map-id", map_id]
        if localizer_exe.exists():
            return [str(localizer_exe), *args]

        python_exe = self._semantic_cv_python(localizer_root)
        if not python_exe.exists():
            raise RuntimeError(f"找不到 CV 定位程序：{localizer_exe}；也找不到 Python 解释器：{python_exe}")
        return [str(python_exe), str(localizer_root / "localize_frame.py"), *args]

    def _semantic_cv_worker_command(self, localizer_root: Path) -> list[str]:
        configured_exe = self.project_config.app.get("semantic_cv_executable")
        localizer_exe = (
            resolve_config_path(self.project_config.root_dir, str(configured_exe))
            if configured_exe
            else localizer_root / "GameMapLocalizer.exe"
        )
        if localizer_exe.exists():
            return [str(localizer_exe), "--worker"]
        python_exe = self._semantic_cv_python(localizer_root)
        if not python_exe.exists():
            raise RuntimeError(f"找不到 CV 定位工作进程：{localizer_exe}；也找不到 Python 解释器：{python_exe}")
        return [str(python_exe), str(localizer_root / "localize_frame.py"), "--worker"]

    def _stop_semantic_cv_worker(self) -> None:
        worker = self._semantic_cv_worker
        self._semantic_cv_worker = None
        self._semantic_cv_worker_warmup_pending = False
        if worker is None:
            stderr_file = getattr(self, "_semantic_cv_worker_stderr_file", None)
            self._semantic_cv_worker_stderr_file = None
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except OSError:
                    pass
            return
        try:
            if worker.poll() is None and worker.stdin is not None:
                worker.stdin.write(json.dumps({"operation": "shutdown"}, ensure_ascii=False) + "\n")
                worker.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            worker.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=0.8)
        finally:
            for stream in (worker.stdin, worker.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            stderr_file = getattr(self, "_semantic_cv_worker_stderr_file", None)
            self._semantic_cv_worker_stderr_file = None
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except OSError:
                    pass

    def _start_semantic_cv_worker(self, localizer_root: Path) -> Optional[subprocess.Popen]:
        if not bool(self.project_config.app.get("semantic_cv_persistent_worker_enabled", True)):
            return None
        worker = self._semantic_cv_worker
        if worker is not None and worker.poll() is None:
            return worker
        if worker is not None:
            self._stop_semantic_cv_worker()
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        if sys.platform == "win32" and bool(
            self.project_config.app.get("semantic_cv_worker_above_normal_priority", True)
        ):
            creationflags |= getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0x00008000)
        worker_env = os.environ.copy()
        worker_env["PYTHONUTF8"] = "1"
        worker_env["PYTHONIOENCODING"] = "utf-8"
        try:
            worker_stderr = diagnostic_path("cv_worker_stderr.log").open("a", encoding="utf-8", buffering=1)
            self._semantic_cv_worker_stderr_file = worker_stderr
            worker = subprocess.Popen(
                self._semantic_cv_worker_command(localizer_root),
                cwd=str(localizer_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=worker_stderr,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=worker_env,
            )
            self._semantic_cv_worker = worker
            diagnostic_event(
                "cv_worker_started",
                worker_pid=worker.pid,
                command=self._semantic_cv_worker_command(localizer_root),
                above_normal_priority=bool(
                    sys.platform == "win32"
                    and self.project_config.app.get("semantic_cv_worker_above_normal_priority", True)
                ),
            )
            config_name = str(self.project_config.app.get("semantic_cv_config", "config.yaml"))
            config_path = resolve_config_path(localizer_root, config_name)
            if worker.stdin is None:
                raise RuntimeError("CV定位工作进程没有标准输入通道")
            # Keep the pipe payload ASCII-only. Frozen Python workers on
            # Windows may otherwise decode Chinese package paths using the
            # active legacy code page and exit during warmup.
            worker.stdin.write(json.dumps({"operation": "warmup", "config": str(config_path)}, ensure_ascii=True) + "\n")
            worker.stdin.flush()
            self._semantic_cv_worker_warmup_pending = True
            return worker
        except Exception:
            diagnostic_event("cv_worker_start_failed")
            self._stop_semantic_cv_worker()
            LOGGER.debug("failed to start persistent OpenCV DNN localizer", exc_info=True)
            return None

    def _semantic_cv_worker_response(self, worker: subprocess.Popen) -> dict[str, Any]:
        if worker.stdout is None:
            raise RuntimeError("CV定位工作进程没有标准输出通道")
        responses: queue.Queue[tuple[Optional[str], Optional[BaseException]]] = queue.Queue(maxsize=1)

        def read_response() -> None:
            try:
                responses.put((worker.stdout.readline(), None))
            except BaseException as exc:
                responses.put((None, exc))

        threading.Thread(target=read_response, daemon=True).start()
        timeout_seconds = float(self.project_config.app.get("semantic_cv_timeout_seconds", 30))
        deadline = time.monotonic() + timeout_seconds
        while True:
            if bool(getattr(self, "_scale_search_cancel_requested", False)):
                raise SemanticCVSearchCancelled("玩家已停止 SCALE 搜索")
            scale_search_active = bool(getattr(self, "_scale_search_active", False))
            f2_down = self._is_global_key_down(0x71) if scale_search_active else False
            if f2_down and not bool(getattr(self, "_scale_search_stop_key_down", False)):
                self._scale_search_stop_key_down = True
                self._stop_semantic_cv_scale_search()
                raise SemanticCVSearchCancelled("玩家已按 F2 停止 SCALE 搜索")
            self._scale_search_stop_key_down = f2_down
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("CV定位工作进程响应超时")
            try:
                raw, read_error = responses.get(timeout=min(0.05, remaining))
                break
            except queue.Empty:
                # The worker wait runs on Tk's thread. Pump UI events so the
                # cheatsheet stop button remains clickable during CV search.
                if scale_search_active:
                    try:
                        self.update()
                    except tk.TclError:
                        pass
        if read_error is not None:
            raise RuntimeError("读取 CV定位工作进程响应失败") from read_error
        if not raw:
            raise RuntimeError("CV定位工作进程已退出且未返回结果")
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise RuntimeError("CV定位工作进程返回格式无效")
        if not bool(response.get("ok", False)):
            raise RuntimeError(str(response.get("error", "CV定位工作进程失败")))
        return response

    def _run_semantic_cv_worker(
        self,
        localizer_root: Path,
        config_data: dict[str, Any],
        image_data: np.ndarray,
        map_id: str,
        *,
        structure_callback=None,
    ) -> Optional[dict[str, Any]]:
        worker = self._start_semantic_cv_worker(localizer_root)
        if worker is None:
            return None
        image = np.ascontiguousarray(image_data)
        image_memory = shared_memory.SharedMemory(create=True, size=image.nbytes)
        np.ndarray(image.shape, dtype=image.dtype, buffer=image_memory.buf)[:] = image
        try:
            if self._semantic_cv_worker_warmup_pending:
                self._semantic_cv_worker_response(worker)
                self._semantic_cv_worker_warmup_pending = False
            if worker.stdin is None:
                raise RuntimeError("CV定位工作进程没有标准输入通道")
            worker.stdin.write(
                json.dumps(
                    {
                        "operation": "localize",
                        "config_data": config_data,
                        "image_shared_memory": {
                            "name": image_memory.name,
                            "shape": list(image.shape),
                            "dtype": image.dtype.str,
                            "nbytes": image.nbytes,
                        },
                        "map_id": map_id,
                        "write_result": bool(
                            config_data.get("save_debug_images", False)
                            or config_data.get("save_full_debug_images", False)
                        ),
                        "wait_for_alignment_visible_ack": True,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            worker.stdin.flush()
            response = self._semantic_cv_worker_response(worker)
            while str(response.get("phase", "")) != "complete":
                diagnostic_event(
                    "cv_worker_phase_received",
                    map_id=map_id,
                    phase=response.get("phase"),
                    worker_pid=getattr(worker, "pid", None),
                )
                if structure_callback is not None:
                    structure_callback(response)
                if str(response.get("phase", "")) == "aligned":
                    if worker.stdin is None:
                        raise RuntimeError("CV localizer worker stdin closed before alignment acknowledgement")
                    worker.stdin.write(
                        json.dumps({"operation": "alignment_visible"}, ensure_ascii=True) + "\n"
                    )
                    worker.stdin.flush()
                response = self._semantic_cv_worker_response(worker)
            diagnostic_event("cv_worker_complete", map_id=map_id, worker_pid=getattr(worker, "pid", None))
            return response
        except Exception as exc:
            diagnostic_event(
                "cv_worker_failed",
                map_id=map_id,
                worker_pid=getattr(worker, "pid", None),
                exception_type=type(exc).__name__,
                exception=repr(exc),
            )
            LOGGER.warning("persistent CV localizer failed", exc_info=True)
            self._stop_semantic_cv_worker()
            if isinstance(exc, SemanticCVSearchCancelled) or bool(
                getattr(self, "_scale_search_cancel_requested", False)
            ):
                raise SemanticCVSearchCancelled(str(exc)) from exc
            self._scale_search_active = False
            self._active_scale_search_return_content = None
            self._active_scale_search_map_id = None
            self._active_scale_search_floor = None
            if bool(
                self.project_config.app.get(
                    "semantic_cv_require_alignment_visible_before_player_detection",
                    True,
                )
            ):
                raise RuntimeError(
                    "CV localizer worker failed before the aligned-overlay/player-detection handshake completed"
                )
            return None
        finally:
            image_memory.close()
            try:
                image_memory.unlink()
            except FileNotFoundError:
                pass

    def _prewarm_semantic_cv_worker(self) -> None:
        try:
            localizer_root = self._semantic_cv_project_root()
            if localizer_root.exists():
                self._start_semantic_cv_worker(localizer_root)
        except Exception:
            LOGGER.debug("failed to prewarm persistent CV localizer", exc_info=True)

    def _shutdown_application(self) -> None:
        diagnostic_event("application_shutdown_requested")
        self._stop_semantic_cv_worker()
        self.route_overlay.destroy()
        self.destroy()

    def _semantic_cv_map_id(self, floor_suffix: str) -> str:
        map_id = self._location_layout_map_id()
        if map_id is None:
            raise RuntimeError("缺少 CNN 已确认地图编号")
        return self._floor_map_id(map_id, floor_suffix)

    def _semantic_cv_base_map_id(self, map_id: str) -> str:
        return base_map_id_from_alignment(map_id)

    def _is_new_layout_map(self, map_id: str) -> bool:
        base = self._semantic_cv_base_map_id(map_id)
        configured = self.project_config.app.get("new_layout_route_only_maps")
        if isinstance(configured, list) and configured:
            return base in {str(item) for item in configured}
        return base in {"map_18", "map_19", "map_20", "map_21", "map_22"}

    def _is_semantic_cv_route_only_map(self, map_id: str) -> bool:
        if bool(self.project_config.app.get("semantic_cv_route_only_all_maps", False)):
            return True
        return self._is_new_layout_map(map_id)

    def _shared_new_layout_route_affine_matrix(
        self,
        map_id: str,
        route_size: Optional[tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        configured = self.project_config.app.get("new_layout_route_map_affine_transform")
        if not isinstance(configured, dict):
            return None
        map_ids = configured.get("map_ids")
        base_map_id = base_map_id_from_alignment(map_id)
        if not isinstance(map_ids, list) or base_map_id not in {str(item) for item in map_ids}:
            return None
        expected_route_size = configured.get("route_size")
        if (
            not isinstance(expected_route_size, list)
            or len(expected_route_size) != 2
            or not all(isinstance(value, (int, float)) for value in expected_route_size)
        ):
            return None
        if route_size is not None and route_size != (int(expected_route_size[0]), int(expected_route_size[1])):
            return None
        try:
            matrix = np.asarray(configured.get("matrix"), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            return None
        return matrix

    def _new_layout_route_affine_matrix(
        self,
        map_id: str,
        route_size: Optional[tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        base = base_map_id_from_alignment(map_id)
        if self._uses_versioned_route_for_map(map_id):
            nightmare_version = nightmare_route_map_version(self.project_config.app)
            if base.startswith("nightmare_") and nightmare_version == "lianghapi":
                configured_key, default_path = (
                    "nightmare_lianghapi_route_affine_path",
                    "config/nightmare_lianghapi_route_map_affines.json",
                )
            elif base.startswith("nightmare_") and nightmare_zhanshi_route_variant(self.project_config.app) == "full_coffin":
                configured_key, default_path = (
                    "nightmare_zhanshi_full_coffin_route_affine_path",
                    "config/nightmare_zhanshi_full_coffin_route_map_affines.json",
                )
            elif base.startswith("nightmare_"):
                configured_key, default_path = (
                    "nightmare_zhanshi_fast_route_affine_path",
                    str(self.project_config.app.get("nightmare_zhanshi_route_affine_path", "config/nightmare_zhanshi_route_map_affines.json")),
                )
            else:
                if difficulty_zhanshi_route_variant(self.project_config.app) == "fast":
                    configured_key, default_path = "difficulty_zhanshi_fast_route_affine_path", "config/zhanshi_fast_route_map_affines.json"
                else:
                    configured_key = "difficulty_zhanshi_full_coffin_route_affine_path"
                    default_path = str(self.project_config.app.get("zhanshi_route_affine_path", "config/zhanshi_route_map_affines.json"))
            configured = self.project_config.app.get(configured_key, default_path)
            path = resolve_config_path(self.project_config.root_dir, str(configured))
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                item = data.get("maps", {}).get(map_id)
                matrix = np.asarray(item.get("matrix"), dtype=np.float32) if isinstance(item, dict) else None
                if matrix is not None and matrix.shape == (2, 3) and np.isfinite(matrix).all():
                    return matrix
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                LOGGER.debug("failed to load versioned route affine for %s", map_id, exc_info=True)
            return None
        configured = self.project_config.app.get("new_layout_route_map_affine_overrides")
        item = configured.get(map_id) if isinstance(configured, dict) else None
        if isinstance(item, dict):
            expected_route_size = item.get("route_size")
            if (
                isinstance(expected_route_size, list)
                and len(expected_route_size) == 2
                and all(isinstance(value, (int, float)) for value in expected_route_size)
                and (route_size is None or route_size == (int(expected_route_size[0]), int(expected_route_size[1])))
            ):
                try:
                    matrix = np.asarray(item.get("matrix"), dtype=np.float64)
                except (TypeError, ValueError):
                    matrix = None
                if matrix is not None and matrix.shape == (2, 3) and np.isfinite(matrix).all():
                    return matrix
        return self._shared_new_layout_route_affine_matrix(map_id, route_size)

    def _has_route_map_coordinate_calibration(self, map_id: str) -> bool:
        if self._route_map_coordinate_override(map_id) is not None or self._new_layout_route_affine_matrix(map_id) is not None:
            return True
        return str(self.project_config.app.get("route_map_coordinate_mode", "scale_to_route")).strip().lower() == "direct"

    def _load_semantic_cv_config(self, localizer_root: Path) -> dict:
        config_name = str(self.project_config.app.get("semantic_cv_config", "config.yaml"))
        config_path = resolve_config_path(localizer_root, config_name)
        try:
            import yaml  # type: ignore

            with config_path.open("r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            # The localizer accepts JSON as YAML, but its checked-in config is YAML.
            # Keep this fallback explicit so a missing PyYAML error is easy to understand.
            raise RuntimeError("缺少 PyYAML，无法读取 game_map_localizer/config.yaml")
        if not isinstance(data, dict):
            raise RuntimeError(f"CV定位配置格式无效：{config_path}")
        return data

    def _prewarm_yolo_player_server(self) -> None:
        if self._yolo_server_prewarm_started:
            return
        self._yolo_server_prewarm_started = True
        try:
            localizer_root = self._semantic_cv_project_root()
            cfg = self._load_semantic_cv_config(localizer_root)
            detector = cfg.get("player_detector", {})
            if not isinstance(detector, dict) or str(detector.get("backend", "")).lower() != "yolo":
                return
            if not bool(detector.get("yolo_server_enabled", False)):
                return
            host = str(detector.get("yolo_server_host", "127.0.0.1"))
            port = int(detector.get("yolo_server_port", 49285))
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    return
            except OSError:
                pass
            model_path = resolve_config_path(localizer_root, str(detector["yolo_model_path"]))
            python_exe = self._semantic_cv_python(localizer_root)
            server_script = localizer_root / "yolo_player_server.py"
            subprocess.Popen(
                [
                    str(python_exe), str(server_script), "--model", str(model_path), "--port", str(port),
                    "--imgsz", str(detector.get("yolo_imgsz", 1280)),
                    "--confidence", str(detector.get("yolo_confidence", 0.25)),
                    "--iou", str(detector.get("yolo_iou", 0.5)),
                ],
                cwd=str(localizer_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception:
            LOGGER.debug("failed to prewarm YOLO player server", exc_info=True)

    def _route_template_dir(self) -> Path:
        app = self.project_config.app
        if str(app.get("route_map_version", "lianghapi")).strip().lower() == "zhanshi":
            if difficulty_zhanshi_route_variant(app) == "fast":
                configured = app.get("difficulty_zhanshi_fast_route_maps_dir", "assets/route_maps_zhanshi_fast")
            else:
                configured = app.get(
                    "difficulty_zhanshi_full_coffin_route_maps_dir",
                    app.get("zhanshi_route_maps_dir", "assets/route_maps_zhanshi"),
                )
            return resolve_config_path(self.project_config.root_dir, str(configured))
        configured = app.get("route_template_maps_dir")
        if configured:
            return resolve_config_path(self.project_config.root_dir, str(configured))
        first_floor_dir = floor_template_dir(app, "1st")
        return first_floor_dir.parent.parent / "maps"

    def _nightmare_clean_template_root(self) -> Path:
        configured = self.project_config.app.get(
            "nightmare_clean_template_dir",
            r"C:\Users\LD\Desktop\摸金工具\nightmare_maps\clean_template",
        )
        return resolve_config_path(self.project_config.root_dir, str(configured))

    def _nightmare_zhanshi_route_root(self) -> Path:
        app = self.project_config.app
        if nightmare_zhanshi_route_variant(app) == "full_coffin":
            configured = app.get(
                "nightmare_zhanshi_full_coffin_route_maps_dir",
                "assets/nightmare_route_maps_zhanshi_full_coffin_rebuilt",
            )
        else:
            configured = app.get(
                "nightmare_zhanshi_fast_route_maps_dir",
                app.get("nightmare_zhanshi_route_maps_dir", r"C:\Users\LD\Desktop\摸金工具\nightmare_maps\route\zhanshi"),
            )
        return resolve_config_path(self.project_config.root_dir, str(configured))

    def _nightmare_lianghapi_route_root(self) -> Path:
        configured = self.project_config.app.get(
            "nightmare_lianghapi_route_maps_dir",
            r"C:\Users\LD\Desktop\摸金工具\nightmare_maps\route\lianghapi",
        )
        return resolve_config_path(self.project_config.root_dir, str(configured))

    def _nightmare_route_affine_record(self, map_id: str) -> Optional[dict[str, Any]]:
        base = base_map_id_from_alignment(map_id)
        if not base.startswith("nightmare_") or not nightmare_route_map_enabled(self.project_config.app):
            return None
        version = nightmare_route_map_version(self.project_config.app)
        if version == "lianghapi":
            key, default = "nightmare_lianghapi_route_affine_path", "config/nightmare_lianghapi_route_map_affines.json"
        elif nightmare_zhanshi_route_variant(self.project_config.app) == "full_coffin":
            key, default = "nightmare_zhanshi_full_coffin_route_affine_path", "config/nightmare_zhanshi_full_coffin_route_map_affines.json"
        else:
            key = "nightmare_zhanshi_fast_route_affine_path"
            default = str(self.project_config.app.get("nightmare_zhanshi_route_affine_path", "config/nightmare_zhanshi_route_map_affines.json"))
        path = resolve_config_path(self.project_config.root_dir, str(self.project_config.app.get(key, default)))
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            item = data.get("maps", {}).get(map_id) if isinstance(data, dict) else None
            return item if isinstance(item, dict) else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _route_affine_record(self, map_id: str) -> Optional[dict[str, Any]]:
        """Load the active versioned route-transform record for either difficulty profile."""
        base = base_map_id_from_alignment(map_id)
        if base.startswith("nightmare_"):
            return self._nightmare_route_affine_record(map_id)
        if str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower() != "zhanshi":
            return None
        app = self.project_config.app
        if difficulty_zhanshi_route_variant(app) == "fast":
            configured = app.get("difficulty_zhanshi_fast_route_affine_path", "config/zhanshi_fast_route_map_affines.json")
        else:
            configured = app.get(
                "difficulty_zhanshi_full_coffin_route_affine_path",
                app.get("zhanshi_route_affine_path", "config/zhanshi_route_map_affines.json"),
            )
        path = resolve_config_path(
            self.project_config.root_dir,
            str(configured),
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            item = data.get("maps", {}).get(map_id) if isinstance(data, dict) else None
            return item if isinstance(item, dict) else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _resolve_nightmare_clean_template_path(
        self,
        map_id: str,
        floor_suffix: str,
    ) -> Optional[Path]:
        base = base_map_id_from_alignment(map_id)
        match = re.fullmatch(r"nightmare_(\d+)", base)
        if match is None or floor_suffix not in {"b1", "1st", "2nd"}:
            return None
        number = int(match.group(1))
        source_number = 14 if number == 13 else number
        root = self._nightmare_clean_template_root()
        directories = [root / f"{source_number}_TBD", root / str(source_number)]
        patterns = {
            "b1": (r"^负一楼$", r"^f-?1$", r"^b1$"),
            "1st": (r"^1楼$", r"^f1$", r".*1楼$"),
            "2nd": (r"^2楼$", r"^f2$", r".*2楼$"),
        }
        for directory in directories:
            if not directory.exists():
                continue
            candidates: list[tuple[int, str, Path]] = []
            for path in directory.iterdir():
                if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                stem = path.stem.strip().lower().replace(" ", "")
                for rank, pattern in enumerate(patterns[floor_suffix]):
                    if re.fullmatch(pattern, stem):
                        candidates.append((rank, path.name.lower(), path))
                        break
            if candidates:
                return min(candidates)[2]
        return None

    def _nightmare_cheatsheet_image(
        self,
        map_id: str,
        floor_suffix: str,
        *,
        current_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if nightmare_three_cheatsheet_enabled(self.project_config.app):
            images: list[tuple[str, np.ndarray]] = []
            for suffix, label in (("b1", "B1"), ("1st", "1F"), ("2nd", "2F")):
                image = current_image if suffix == floor_suffix and current_image is not None else None
                if image is None:
                    path = self._resolve_nightmare_clean_template_path(map_id, suffix)
                    image = imread_unicode(path) if path is not None else None
                if image is None:
                    return None
                images.append((label, image))
            return make_nightmare_three_floor_sheet(map_id, images)
        image = current_image
        if image is None:
            path = self._resolve_nightmare_clean_template_path(map_id, floor_suffix)
            image = imread_unicode(path) if path is not None else None
        return trim_dark_image_border(image) if image is not None else None

    def _resolve_route_template_path(self, map_id: str) -> Optional[Path]:
        base = base_map_id_from_alignment(map_id)
        if base.startswith("nightmare_"):
            version = nightmare_route_map_version(self.project_config.app)
            if version in {"zhanshi", "lianghapi"}:
                match = re.fullmatch(r"nightmare_(\d+)", base)
                if match is None:
                    return None
                root = self._nightmare_lianghapi_route_root() if version == "lianghapi" else self._nightmare_zhanshi_route_root()
                for extension in (".jpg", ".png", ".jpeg"):
                    path = root / f"{int(match.group(1))}{extension}"
                    if path.exists():
                        return path
                return None
            floor_suffix = "b1" if map_id.endswith("_b1") else "2nd" if map_id.endswith("_2nd") else "1st"
            return self._resolve_nightmare_clean_template_path(map_id, floor_suffix)
        if str(self.project_config.app.get("route_map_version", "lianghapi")).strip().lower() == "zhanshi":
            for extension in (".jpg", ".png", ".jpeg"):
                path = self._route_template_dir() / f"{base}{extension}"
                if path.exists():
                    return path
            return None
        directory = self._route_template_dir()
        for ext in (".jpg", ".png", ".jpeg"):
            path = directory / f"{map_id}{ext}"
            if path.exists():
                return path
        floor_suffix = "b1" if map_id.endswith("_b1") else "2nd" if map_id.endswith("_2nd") else "1st"
        route_path = resolve_floor_template_path(self.project_config.app, map_id, floor_suffix)
        if route_path is not None:
            return route_path
        return None

    def _semantic_cv_crop_offset(self, processed_dir: Path, map_id: str) -> tuple[float, float]:
        meta_path = processed_dir / map_id / "meta.json"
        try:
            with meta_path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            offset = data.get("crop_offset", [0, 0]) if isinstance(data, dict) else [0, 0]
            return float(offset[0]), float(offset[1])
        except Exception:
            return 0.0, 0.0

    def _route_map_coordinate_override(self, map_id: str) -> Optional[dict]:
        configured = self.project_config.app.get("route_map_coordinate_overrides")
        if not configured:
            return None
        path = resolve_config_path(self.project_config.root_dir, str(configured))
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            return None
        maps = data.get("maps") if isinstance(data, dict) else None
        item = maps.get(map_id) if isinstance(maps, dict) else None
        return item if isinstance(item, dict) else None

    def _map_clean_template_point_to_route(
        self,
        point: tuple[float, float],
        *,
        map_id: str,
        locate_size: tuple[int, int],
        route_size: tuple[int, int],
    ) -> tuple[float, float]:
        locate_w, locate_h = locate_size
        route_w, route_h = route_size
        override = self._route_map_coordinate_override(map_id)
        route_record = self._route_affine_record(map_id)
        piecewise = route_record.get("piecewise") if isinstance(route_record, dict) else None
        if isinstance(piecewise, Mapping):
            try:
                affine_matrix = piecewise_diagonal_matrix_at_y(piecewise, float(point[1]))
            except (TypeError, ValueError):
                affine_matrix = None
        else:
            try:
                global_matrix = np.asarray(route_record.get("matrix"), dtype=np.float32) if isinstance(route_record, Mapping) else None
                regions = route_record.get("regions") if isinstance(route_record, Mapping) else None
                affine_matrix = route_region_matrix_at_point(global_matrix, regions, point) if global_matrix is not None else None
            except (TypeError, ValueError):
                affine_matrix = None
        if affine_matrix is not None:
            rx = float(affine_matrix[0, 0] * point[0] + affine_matrix[0, 1] * point[1] + affine_matrix[0, 2])
            ry = float(affine_matrix[1, 0] * point[0] + affine_matrix[1, 1] * point[1] + affine_matrix[1, 2])
            return min(max(rx, 0.0), float(route_w - 1)), min(max(ry, 0.0), float(route_h - 1))
        affine_matrix = self._new_layout_route_affine_matrix(map_id, route_size)
        if affine_matrix is not None:
            rx = float(affine_matrix[0, 0] * point[0] + affine_matrix[0, 1] * point[1] + affine_matrix[0, 2])
            ry = float(affine_matrix[1, 0] * point[0] + affine_matrix[1, 1] * point[1] + affine_matrix[1, 2])
            return min(max(rx, 0.0), float(route_w - 1)), min(max(ry, 0.0), float(route_h - 1))
        if override is not None:
            mode = str(override.get("mode", "")).strip().lower()
            crop_box = override.get("crop_box")
            if mode == "crop_box" and isinstance(crop_box, list) and len(crop_box) >= 4:
                x0, y0, crop_w, crop_h = [float(v) for v in crop_box[:4]]
                if crop_w > 0 and crop_h > 0:
                    rx = (point[0] - x0) * route_w / crop_w
                    ry = (point[1] - y0) * route_h / crop_h
                    return min(max(rx, 0.0), float(route_w - 1)), min(max(ry, 0.0), float(route_h - 1))

        mode = str(self.project_config.app.get("route_map_coordinate_mode", "scale_to_route")).strip().lower()
        if mode == "direct":
            return point
        return point[0] * route_w / locate_w, point[1] * route_h / locate_h

    def _template_to_route_affine(
        self,
        *,
        map_id: str,
        locate_size: tuple[int, int],
        route_size: tuple[int, int],
    ) -> np.ndarray:
        """Return the unbounded template-to-route transform used by screen overlays."""
        locate_w, locate_h = locate_size
        route_w, route_h = route_size
        if min(locate_w, locate_h, route_w, route_h) <= 0:
            raise ValueError("invalid template or route size")
        cache_key = (str(map_id), int(locate_w), int(locate_h), int(route_w), int(route_h))
        affine_cache = getattr(self, "_semantic_cv_affine_cache", None)
        if affine_cache is None:
            affine_cache = {}
            self._semantic_cv_affine_cache = affine_cache
        cached = affine_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        affine_matrix = self._new_layout_route_affine_matrix(map_id, route_size)
        if affine_matrix is not None:
            result = np.asarray(affine_matrix, dtype=np.float32)
            affine_cache[cache_key] = result
            return result.copy()
        override = self._route_map_coordinate_override(map_id)
        if override is not None:
            mode = str(override.get("mode", "")).strip().lower()
            crop_box = override.get("crop_box")
            if mode == "crop_box" and isinstance(crop_box, list) and len(crop_box) >= 4:
                x0, y0, crop_w, crop_h = [float(value) for value in crop_box[:4]]
                if crop_w > 0 and crop_h > 0:
                    result = np.float32(
                        [
                            [route_w / crop_w, 0.0, -x0 * route_w / crop_w],
                            [0.0, route_h / crop_h, -y0 * route_h / crop_h],
                        ]
                    )
                    affine_cache[cache_key] = result
                    return result.copy()
        mode = str(self.project_config.app.get("route_map_coordinate_mode", "scale_to_route")).strip().lower()
        if mode == "direct":
            result = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            result = np.float32([[route_w / locate_w, 0.0, 0.0], [0.0, route_h / locate_h, 0.0]])
        affine_cache[cache_key] = result
        return result.copy()

    @staticmethod
    def _semantic_cv_players(result_data: dict) -> list[tuple[str, tuple[float, float], tuple[int, int, int]]]:
        players: list[tuple[str, tuple[float, float], tuple[int, int, int]]] = []
        raw_players = result_data.get("players") if isinstance(result_data, dict) else None
        if isinstance(raw_players, list):
            for item in raw_players:
                if not isinstance(item, dict):
                    continue
                point = item.get("template")
                if not isinstance(point, list) or len(point) < 2:
                    continue
                try:
                    player_id = str(item.get("player_id", "?"))
                    color_raw = item.get("color_bgr", [255, 255, 255])
                    color = tuple(int(max(0, min(255, value))) for value in color_raw[:3])
                    players.append((player_id, (float(point[0]), float(point[1])), color))
                except (TypeError, ValueError):
                    continue
        if players:
            return players
        player_full = result_data.get("player_position_in_template") if isinstance(result_data, dict) else None
        if isinstance(player_full, list) and len(player_full) >= 2:
            return [("1", (float(player_full[0]), float(player_full[1])), (0, 255, 255))]
        return []

    @staticmethod
    def _draw_semantic_cv_player(
        image: np.ndarray,
        point: tuple[float, float],
        player_id: str,
        color: tuple[int, int, int],
        marker_scale: float = 1.0,
    ) -> None:
        x, y = int(round(point[0])), int(round(point[1]))
        scale = max(0.5, float(marker_scale))
        fill_radius = max(5, int(round(10 * scale)))
        outline_radius = max(fill_radius + 2, int(round(15 * scale)))
        text_offset_x = max(8, int(round(14 * scale)))
        text_offset_y = max(5, int(round(8 * scale)))
        font_scale = max(0.35, 0.62 * scale)
        thickness = max(1, int(round(2 * scale)))
        cv2.circle(image, (x, y), fill_radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), outline_radius, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(
            image,
            f"P{player_id}",
            (x + text_offset_x, y - text_offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _render_semantic_cv_route_map(
        self,
        *,
        map_id: str,
        floor_suffix: str,
        result_data: dict,
        processed_dir: Path,
        filtered_players: dict[str, tuple[tuple[float, float], str, tuple[int, int, int]]],
        output_path: Optional[Path],
    ) -> Optional[tuple[np.ndarray, str]]:
        route_path = self._resolve_route_template_path(map_id)
        processed_template_path = processed_dir / map_id / "image.png"
        if route_path is None or not processed_template_path.exists():
            return None
        route_image = self._semantic_cv_cached_image(route_path)
        locate_image = self._semantic_cv_cached_image(processed_template_path)
        if route_image is None or locate_image is None:
            return None
        locate_h, locate_w = locate_image.shape[:2]
        route_h, route_w = route_image.shape[:2]
        if locate_w <= 0 or locate_h <= 0 or route_w <= 0 or route_h <= 0:
            return None

        def map_point(point: tuple[float, float]) -> tuple[float, float]:
            return self._map_clean_template_point_to_route(
                point,
                map_id=map_id,
                locate_size=(locate_w, locate_h),
                route_size=(route_w, route_h),
            )

        out = route_image.copy()
        label_parts = [map_id]
        players_to_render = [
            (player_id, point, color)
            for player_id, (point, estimate_floor, color) in filtered_players.items()
            if estimate_floor == floor_suffix
        ]
        if not filtered_players:
            players_to_render = self._semantic_cv_players(result_data)
        marker_scale = semantic_cv_player_marker_scale(
            self.project_config.app,
            (route_w, route_h),
        )
        for player_id, template_point, color in players_to_render:
            rx, ry = map_point(template_point)
            self._draw_semantic_cv_player(out, (rx, ry), player_id, color, marker_scale=marker_scale)
            label_parts.append(f"P{player_id}=({rx:.1f},{ry:.1f})")
        if self._is_nightmare_mode() and nightmare_lianghapi_single_cheatsheet_enabled(self.project_config.app):
            cropped = self._nightmare_lianghapi_single_floor_image(out, map_id)
            if cropped is not None:
                return cropped, f"{map_id}  单图"
        if output_path is not None:
            imwrite_unicode(output_path, out)
        return out, "  ".join(label_parts)

    def _make_lianghapi_dual_route_sheet(
        self,
        *,
        map_id: str,
        floor_suffix: str,
        current_route_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Keep both legacy route floors visible while marking only the active floor."""
        if not self._should_show_lianghapi_dual_cheatsheet():
            return None
        if floor_suffix not in {"1st", "2nd"}:
            return None
        other_floor = "2nd" if floor_suffix == "1st" else "1st"
        other_map_id = self._floor_map_id(map_id, other_floor)
        other_path = self._resolve_route_template_path(other_map_id)
        other_image = imread_unicode(other_path) if other_path is not None else None
        if other_image is None:
            LOGGER.warning("dual legacy route sheet is missing %s", other_map_id)
            return None
        base_map_id = base_map_id_from_alignment(map_id)
        if floor_suffix == "1st":
            return make_route_floor_contact_sheet(base_map_id, current_route_image, other_image)
        return make_route_floor_contact_sheet(base_map_id, other_image, current_route_image)

    def _show_semantic_cv_route_overlay(
        self,
        *,
        map_id: str,
        result_data: dict,
        processed_dir: Path,
        output_path: Path,
    ) -> bool:
        """Warp the route map into the captured game-map ROI."""
        request_id = self._active_route_overlay_request_id
        if request_id is not None and request_id <= self._cancelled_route_overlay_request_id:
            self.route_overlay.hide()
            return False
        if not bool(self.project_config.app.get("route_overlay_enabled", True)):
            self.route_overlay.hide()
            return False
        screen_roi = self._last_alignment_map_screen_roi
        best_position = result_data.get("best_position") if isinstance(result_data, dict) else None
        best_scale = result_data.get("best_scale") if isinstance(result_data, dict) else None
        observation_bbox = result_data.get("observation_bbox") if isinstance(result_data, dict) else None
        if (
            screen_roi is None
            or not isinstance(best_position, list)
            or len(best_position) < 2
            or not isinstance(best_scale, (int, float))
        ):
            return False
        observation_offset = (0.0, 0.0)
        if isinstance(observation_bbox, list) and len(observation_bbox) >= 2:
            try:
                observation_offset = (float(observation_bbox[0]), float(observation_bbox[1]))
            except (TypeError, ValueError):
                pass
        floor_suffix = "b1" if map_id.endswith("_b1") else "2nd" if map_id.endswith("_2nd") else "1st"
        full_template_path = resolve_floor_template_path(self.project_config.app, map_id, floor_suffix)
        route_path = self._resolve_route_template_path(map_id)
        if full_template_path is None or route_path is None:
            return False
        full_template_image = self._semantic_cv_cached_image(full_template_path)
        route_image = self._semantic_cv_cached_image(route_path)
        if full_template_image is None or route_image is None:
            return False
        template_h, template_w = full_template_image.shape[:2]
        route_h, route_w = route_image.shape[:2]
        screen_x, screen_y, screen_w, screen_h = screen_roi
        if min(template_w, template_h, route_w, route_h, screen_w, screen_h) <= 0:
            return False

        route_record = self._route_affine_record(map_id)
        piecewise = route_record.get("piecewise") if isinstance(route_record, dict) else None
        regions = route_record.get("regions") if isinstance(route_record, dict) else None
        if isinstance(piecewise, Mapping):
            piecewise_key = (str(route_path.resolve()), template_w, template_h)
            adjusted_route = self._semantic_cv_piecewise_route_cache.get(piecewise_key)
            if adjusted_route is None:
                adjusted_route = remap_piecewise_route_to_template(
                    route_image,
                    (template_w, template_h),
                    piecewise,
                )
                self._semantic_cv_piecewise_route_cache[piecewise_key] = adjusted_route
            route_image = adjusted_route
            route_h, route_w = route_image.shape[:2]
            template_to_route = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elif isinstance(regions, list) and regions:
            try:
                global_matrix = np.asarray(route_record.get("matrix"), dtype=np.float32)
                regions_key = json.dumps(regions, ensure_ascii=False, sort_keys=True)
                region_key = (f"regions:{route_path.resolve()}:{regions_key}", template_w, template_h)
                adjusted_route = self._semantic_cv_piecewise_route_cache.get(region_key)
                if adjusted_route is None:
                    adjusted_route = remap_region_route_to_template(
                        route_image,
                        (template_w, template_h),
                        global_matrix,
                        regions,
                    )
                    self._semantic_cv_piecewise_route_cache[region_key] = adjusted_route
                route_image = adjusted_route
                route_h, route_w = route_image.shape[:2]
                template_to_route = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            except (TypeError, ValueError, json.JSONDecodeError):
                template_to_route = self._template_to_route_affine(
                    map_id=map_id,
                    locate_size=(template_w, template_h),
                    route_size=(route_w, route_h),
                )
        else:
            template_to_route = self._template_to_route_affine(
                map_id=map_id,
                locate_size=(template_w, template_h),
                route_size=(route_w, route_h),
            )
        crop_x, crop_y = self._semantic_cv_crop_offset(processed_dir, map_id)
        normalization = result_data.get("reference_roi_normalization") if isinstance(result_data, dict) else None
        raw_to_reference_scale = (
            normalization.get("raw_to_reference_scale") if isinstance(normalization, dict) else None
        )
        if isinstance(raw_to_reference_scale, (list, tuple)) and len(raw_to_reference_scale) >= 2:
            dynamic_roi_to_template = reference_normalized_roi_to_template_affine(
                best_position_full=(float(best_position[0]) + crop_x, float(best_position[1]) + crop_y),
                best_scale=float(best_scale),
                raw_to_reference_scale=(float(raw_to_reference_scale[0]), float(raw_to_reference_scale[1])),
                observation_offset=observation_offset,
            )
        else:
            template_to_roi = route_to_screen_roi_affine(
                best_position=(float(best_position[0]) + crop_x, float(best_position[1]) + crop_y),
                best_scale=float(best_scale),
                roi_size=(screen_w, screen_h),
                template_to_route=np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                observation_offset=observation_offset,
            )
            dynamic_roi_to_template = cv2.invertAffineTransform(template_to_roi)
        source_to_roi = route_to_screen_roi_from_calibrated_affines(
            roi_to_template=dynamic_roi_to_template,
            template_to_route=template_to_route,
        )
        threshold = max(
            0,
            min(64, int(self.project_config.app.get("route_overlay_black_transparent_threshold", 16))),
        )
        crop_key = (
            f"piecewise:{route_path.resolve()}"
            if isinstance(piecewise, Mapping)
            else f"regions:{route_path.resolve()}" if isinstance(regions, list) and regions else str(route_path.resolve()),
            threshold,
        )
        cropped_route = self._semantic_cv_route_crop_cache.get(crop_key)
        if cropped_route is None:
            cropped_image, cropped_offset = crop_route_overlay_content(
                route_image,
                black_threshold=threshold,
            )
            cropped_route = (cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), cropped_offset)
            self._semantic_cv_route_crop_cache[crop_key] = cropped_route
        route_content, (route_offset_x, route_offset_y) = cropped_route
        cropped_source_to_roi = np.float32(source_to_roi).copy()
        cropped_source_to_roi[:, 2] += cropped_source_to_roi[:, :2] @ np.float32(
            [route_offset_x, route_offset_y]
        )
        overlay_image, visible_roi = warp_route_overlay_visible_region(
            route_content,
            cropped_source_to_roi,
            (screen_w, screen_h),
        )
        visible_x, visible_y, visible_w, visible_h = visible_roi
        if visible_w <= 0 or visible_h <= 0:
            self.route_overlay.hide()
            return False
        if bool(self.project_config.app.get("semantic_cv_save_overlay_debug_image", False)):
            debug_image = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            debug_image[visible_y : visible_y + visible_h, visible_x : visible_x + visible_w] = cv2.cvtColor(
                overlay_image,
                cv2.COLOR_RGB2BGR,
            )
            imwrite_unicode(output_path, debug_image)
        if request_id is not None and request_id <= self._cancelled_route_overlay_request_id:
            self.route_overlay.hide()
            return False
        self.route_overlay.show_image(
            overlay_image,
            (screen_x + visible_x, screen_y + visible_y, visible_w, visible_h),
            image_is_rgb=True,
        )
        self._semantic_cv_last_overlay_map_id = map_id
        self._semantic_cv_last_overlay_screen_roi = screen_roi
        self._route_overlay_visible_request_id = request_id
        self._route_overlay_dismiss_armed = True
        self._overlay_scale_research_context = {
            "map_id": map_id,
            "floor_suffix": floor_suffix,
        }
        return True

    def _render_semantic_cv_clean_template_map(
        self,
        *,
        map_id: str,
        floor_suffix: str,
        result_data: dict,
        processed_dir: Path,
        filtered_players: dict[str, tuple[tuple[float, float], str, tuple[int, int, int]]],
        output_path: Optional[Path],
    ) -> Optional[np.ndarray]:
        locate_path = resolve_floor_template_path(self.project_config.app, map_id, floor_suffix)
        if locate_path is None:
            return None
        template_image = self._semantic_cv_cached_image(locate_path)
        if template_image is None:
            return None
        out = template_image.copy()
        players_to_render = [
            (player_id, point, color)
            for player_id, (point, estimate_floor, color) in filtered_players.items()
            if estimate_floor == floor_suffix
        ]
        if not filtered_players:
            players_to_render = self._semantic_cv_players(result_data)
        marker_scale = semantic_cv_player_marker_scale(
            self.project_config.app,
            (template_image.shape[1], template_image.shape[0]),
        )
        for player_id, template_point, color in players_to_render:
            self._draw_semantic_cv_player(
                out,
                template_point,
                player_id,
                color,
                marker_scale=marker_scale,
            )
        if output_path is not None:
            imwrite_unicode(output_path, out)
        return out

    def _semantic_cv_cached_image(self, path: Path) -> Optional[np.ndarray]:
        key = str(path.resolve())
        image = self._semantic_cv_image_cache.get(key)
        if image is None:
            image = imread_unicode(path)
            if image is not None:
                self._semantic_cv_image_cache[key] = image
        return image

    def _make_semantic_cv_display_sheet(
        self,
        route_image: np.ndarray,
        template_image: np.ndarray,
        *,
        route_title: str,
        template_title: str,
    ) -> np.ndarray:
        def fit_panel(image: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
            target_w, target_h = size
            title_h = 30
            panel = np.full((target_h, target_w, 3), (5, 7, 10), dtype=np.uint8)
            cv2.rectangle(panel, (0, 0), (target_w, title_h), (10, 14, 20), -1)
            cv2.putText(panel, title[:52], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            h, w = image.shape[:2]
            scale = min(target_w / max(1, w), (target_h - title_h) / max(1, h))
            resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            y = title_h + (target_h - title_h - resized.shape[0]) // 2
            x = (target_w - resized.shape[1]) // 2
            panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            return panel

        route_h, route_w = route_image.shape[:2]
        template_h, template_w = template_image.shape[:2]
        panel_w = max(route_w, template_w)
        route_panel_h = max(1, int(route_h * panel_w / max(1, route_w)))
        template_panel_h = max(1, int(template_h * panel_w / max(1, template_w)))
        max_total_h = 1800
        total_h = route_panel_h + template_panel_h
        if total_h > max_total_h:
            shrink = max_total_h / total_h
            panel_w = max(320, int(panel_w * shrink))
            route_panel_h = max(1, int(route_panel_h * shrink))
            template_panel_h = max(1, int(template_panel_h * shrink))
        return np.vstack(
            [
                fit_panel(route_image, route_title, (panel_w, route_panel_h)),
                fit_panel(template_image, template_title, (panel_w, template_panel_h)),
            ]
        )

    def _semantic_cv_particle_filter_class(self, localizer_root: Path):
        module_path = localizer_root / "src" / "particle_filter.py"
        if not module_path.exists():
            raise RuntimeError(f"找不到粒子滤波模块：{module_path}")
        module_name = "_game_map_localizer_particle_filter"
        if module_name in sys.modules:
            return getattr(sys.modules[module_name], "ParticleFilterLocalizer")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载粒子滤波模块：{module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, "ParticleFilterLocalizer")

    def _semantic_cv_walkable_mask(self, processed_dir: Path, floor_map_id: str) -> Optional[np.ndarray]:
        mode = str(self.project_config.app.get("semantic_cv_pf_walkable_mask", "map_mask")).strip().lower()
        floor_dir = processed_dir / floor_map_id
        if mode in {"", "none", "off", "disabled"}:
            return None
        if mode in {"corridor_room", "corridor+room", "corridor_and_room"}:
            masks: list[np.ndarray] = []
            for name in ("corridor_mask.png", "room_mask.png"):
                image = imread_unicode(floor_dir / name)
                if image is None:
                    continue
                if image.ndim == 3:
                    image = image.max(axis=2)
                masks.append(image > 0)
            if not masks:
                return None
            combined = np.logical_or.reduce(masks)
            return combined.astype(np.uint8) * 255
        filename = mode if mode.endswith((".png", ".jpg", ".jpeg")) else f"{mode}.png"
        image = imread_unicode(floor_dir / filename)
        if image is not None:
            return image
        return imread_unicode(floor_dir / "map_mask.png")

    def _configure_semantic_cv_pf_walkable_masks(self, processed_dir: Path, map_id: str) -> None:
        if self._semantic_cv_pf is None:
            return
        enabled = bool(self.project_config.app.get("semantic_cv_pf_walkable_constraint_enabled", True))
        base = base_map_id_from_alignment(map_id)
        try:
            self._semantic_cv_pf.set_walkable_constraint_options(
                local_radius=float(self.project_config.app.get("semantic_cv_pf_walkable_local_radius", 70.0)),
                attempts=int(self.project_config.app.get("semantic_cv_pf_walkable_attempts", 12)),
                step_size=float(self.project_config.app.get("semantic_cv_pf_walkable_step_size", 3.0)),
                turn_attempts=int(self.project_config.app.get("semantic_cv_pf_walkable_turn_attempts", 18)),
                liveliness=float(self.project_config.app.get("semantic_cv_pf_walkable_liveliness", 2.4)),
                lookahead_steps=int(self.project_config.app.get("semantic_cv_pf_walkable_lookahead_steps", 6)),
                direction_jitter_degrees=float(self.project_config.app.get("semantic_cv_pf_walkable_direction_jitter_degrees", 8.0)),
                radius_tolerance=float(self.project_config.app.get("semantic_cv_pf_walkable_radius_tolerance", 12.0)),
                radius_weight=float(self.project_config.app.get("semantic_cv_pf_walkable_radius_weight", 2.5)),
            )
        except AttributeError:
            return
        for floor_suffix in self.recognition_profile.floors:
            floor_map_id = self._floor_map_id(base, floor_suffix)
            mask = self._semantic_cv_walkable_mask(processed_dir, floor_map_id) if enabled else None
            try:
                self._semantic_cv_pf.set_walkable_mask(mask, floor=floor_suffix)
            except AttributeError:
                return

    def _semantic_cv_candidate_observations(
        self,
        result_data: dict,
        *,
        top_n: Optional[int] = None,
        score_power: Optional[float] = None,
    ) -> list[tuple[float, float, float, float]]:
        player_roi = result_data.get("player_roi")
        if not isinstance(player_roi, list) or len(player_roi) < 2:
            return []
        if top_n is None:
            top_n = int(self.project_config.app.get("semantic_cv_pf_top_n_observations", 1))
        if score_power is None:
            score_power = float(self.project_config.app.get("semantic_cv_pf_score_power", 10.0))
        candidates = result_data.get("top_k_candidates")
        if not isinstance(candidates, list):
            return []
        sorted_candidates = sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        observations: list[tuple[float, float, float, float]] = []
        for item in sorted_candidates[: max(1, top_n)]:
            top_left = item.get("top_left")
            scale = item.get("scale")
            score = item.get("score")
            if not isinstance(top_left, list) or len(top_left) < 2 or scale is None or score is None:
                continue
            x = float(top_left[0]) + float(player_roi[0]) * float(scale)
            y = float(top_left[1]) + float(player_roi[1]) * float(scale)
            observations.append((x, y, float(scale), float(score) ** score_power))
        return observations

    def _semantic_cv_first_second_floor_transition_support(self, map_id: str) -> list[dict]:
        if self._semantic_cv_pf is None or not getattr(self._semantic_cv_pf, "initialized", False):
            return []
        particles = getattr(self._semantic_cv_pf, "particles", None)
        if particles is None:
            return []
        try:
            particles_array = np.asarray(particles, dtype=np.float64)
        except Exception:
            return []
        if particles_array.ndim != 2 or particles_array.shape[1] < 5 or particles_array.shape[0] == 0:
            return []
        transitions = self._semantic_cv_floor_transitions(map_id)
        if not transitions:
            return []
        app = self.project_config.app
        radius = max(1.0, float(app.get("semantic_cv_pf_first_second_floor_particle_radius", 90.0)))
        radius2 = radius * radius
        min_weight = max(0.0, float(app.get("semantic_cv_pf_first_second_floor_min_particle_weight", 0.002)))
        supports: list[dict] = []
        second_mask = particles_array[:, 4] == 1.0
        weights = particles_array[:, 3]
        for item in transitions:
            second = item.get("2nd") or item.get("second") or item.get("to")
            if not isinstance(second, list) or len(second) < 2:
                continue
            if not isinstance(second[0], (int, float)) or not isinstance(second[1], (int, float)):
                continue
            dx = particles_array[:, 0] - float(second[0])
            dy = particles_array[:, 1] - float(second[1])
            near = second_mask & ((dx * dx + dy * dy) <= radius2)
            support_weight = float(weights[near].sum()) if bool(near.any()) else 0.0
            if support_weight < min_weight:
                continue
            supports.append(
                {
                    "point": (float(second[0]), float(second[1])),
                    "weight": support_weight,
                    "count": int(near.sum()),
                }
            )
        return supports

    def _semantic_cv_apply_first_second_floor_prior(
        self,
        observations: list[tuple[float, float, float, float]],
        *,
        map_id: str,
        result_data: dict,
    ) -> list[tuple[float, float, float, float]]:
        if not observations or not bool(self.project_config.app.get("semantic_cv_pf_first_second_floor_prior_enabled", True)):
            return observations
        supports = self._semantic_cv_first_second_floor_transition_support(map_id)
        if not supports:
            result_data["first_second_floor_transition_prior"] = {
                "enabled": True,
                "applied": False,
                "reason": "no_second_floor_transition_particle_support",
            }
            return observations
        app = self.project_config.app
        radius = max(1.0, float(app.get("semantic_cv_pf_first_second_floor_prior_radius", 180.0)))
        strength = max(0.0, float(app.get("semantic_cv_pf_first_second_floor_prior_strength", 8.0)))
        support_ref = max(1e-6, float(app.get("semantic_cv_pf_first_second_floor_particle_support_ref", 0.03)))
        adjusted: list[tuple[float, float, float, float]] = []
        debug_items: list[dict] = []
        for x, y, scale, weight in observations:
            best_prior = 0.0
            best_support: Optional[dict] = None
            for support in supports:
                sx, sy = support["point"]
                dist = float(np.hypot(float(x) - sx, float(y) - sy))
                proximity = float(np.exp(-(dist * dist) / (2.0 * radius * radius)))
                support_factor = min(1.0, float(support["weight"]) / support_ref)
                prior = proximity * support_factor
                if prior > best_prior:
                    best_prior = prior
                    best_support = support
            multiplier = 1.0 + strength * best_prior
            adjusted_weight = float(weight) * multiplier
            adjusted.append((float(x), float(y), float(scale), adjusted_weight))
            debug_items.append(
                {
                    "point": [float(x), float(y)],
                    "scale": float(scale),
                    "base_weight": float(weight),
                    "adjusted_weight": adjusted_weight,
                    "multiplier": multiplier,
                    "prior": best_prior,
                    "support": {
                        "point": list(best_support["point"]),
                        "weight": float(best_support["weight"]),
                        "count": int(best_support["count"]),
                    }
                    if best_support is not None
                    else None,
                }
            )
        result_data["first_second_floor_transition_prior"] = {
            "enabled": True,
            "applied": True,
            "radius": radius,
            "strength": strength,
            "support_ref": support_ref,
            "supports": [
                {"point": list(item["point"]), "weight": float(item["weight"]), "count": int(item["count"])}
                for item in supports
            ],
            "observations": debug_items,
        }
        return adjusted

    def _semantic_cv_side_door_entry_points(self, map_id: str) -> list[list[float]]:
        base = base_map_id_from_alignment(map_id)
        config_key = (
            "nightmare_second_floor_side_door_entry_points_path"
            if base.startswith("nightmare_")
            else "side_door_entry_points_path"
        )
        default_path = (
            "config/nightmare_second_floor_side_door_entry_points.json"
            if base.startswith("nightmare_")
            else "config/side_door_entry_points.json"
        )
        configured = self.project_config.app.get(config_key, default_path)
        path = resolve_config_path(self.project_config.root_dir, str(configured))
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            LOGGER.debug("failed to load side-door entry points: %s", path, exc_info=True)
            return []
        if not isinstance(data, dict):
            return []
        maps = data.get("maps")
        value = maps.get(base) if isinstance(maps, dict) else data.get(base)
        if not isinstance(value, list):
            return []
        points: list[list[float]] = []
        for item in value:
            if (
                isinstance(item, list)
                and len(item) >= 2
                and isinstance(item[0], (int, float))
                and isinstance(item[1], (int, float))
            ):
                points.append([float(item[0]), float(item[1])])
        return points

    def _semantic_cv_basement_tp_points(self, map_id: str) -> list[list[float]]:
        """Return the configured first-entry teleport point for a Nightmare B1 map."""
        base = base_map_id_from_alignment(map_id)
        if not base.startswith("nightmare_"):
            return []
        configured = self.project_config.app.get(
            "nightmare_basement_tp_points_path",
            "config/nightmare_basement_tp_points.json",
        )
        path = resolve_config_path(self.project_config.root_dir, str(configured))
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            LOGGER.debug("failed to load Nightmare basement TP points: %s", path, exc_info=True)
            return []
        if not isinstance(data, dict):
            return []
        maps = data.get("maps")
        value = maps.get(base) if isinstance(maps, dict) else data.get(base)
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            return [[float(value[0]), float(value[1])]]
        return []

    def _semantic_cv_floor_transitions(self, map_id: str) -> list[dict]:
        base = base_map_id_from_alignment(map_id)
        config_key = "nightmare_floor_transition_links_path" if base.startswith("nightmare_") else "floor_transition_links_path"
        default_path = "config/nightmare_floor_transition_links.json" if base.startswith("nightmare_") else "config/floor_transition_links.json"
        configured = self.project_config.app.get(config_key, default_path)
        path = resolve_config_path(self.project_config.root_dir, str(configured))
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            LOGGER.debug("failed to load floor transition links: %s", path, exc_info=True)
            return []
        if not isinstance(data, dict):
            return []
        maps = data.get("maps")
        if isinstance(maps, dict):
            value = maps.get(base) or maps.get(map_id)
        else:
            value = data.get(base) or data.get(map_id)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            combined: list[dict] = []
            for key in ("b1_1st", "1st_2nd"):
                links = value.get(key)
                if isinstance(links, list):
                    combined.extend(item for item in links if isinstance(item, dict))
            return combined
        return []

    def _semantic_cv_apply_floor_transitions(self, map_id: str, *, probability: Optional[float] = None) -> None:
        if self._semantic_cv_pf is None or not bool(self.project_config.app.get("semantic_cv_pf_floor_transitions_enabled", True)):
            return
        transitions = self._semantic_cv_floor_transitions(map_id)
        if not transitions:
            return
        try:
            self._semantic_cv_pf.diffuse_via_transitions(
                transitions,
                probability=float(self.project_config.app.get("semantic_cv_pf_floor_transition_probability", 0.12) if probability is None else probability),
                radius=float(self.project_config.app.get("semantic_cv_pf_floor_transition_radius", 55.0)),
                noise=float(self.project_config.app.get("semantic_cv_pf_floor_transition_noise", 18.0)),
            )
        except AttributeError:
            return

    def _save_semantic_cv_particle_debug(
        self,
        *,
        call_dir: Path,
        map_id: str,
        floor_suffix: str,
        processed_dir: Path,
        result_data: dict,
        estimate_point: Optional[tuple[float, float]],
        phase: str = "after_observation",
    ) -> None:
        try:
            self._save_semantic_cv_particle_debug_impl(
                call_dir=call_dir,
                map_id=map_id,
                floor_suffix=floor_suffix,
                processed_dir=processed_dir,
                result_data=result_data,
                estimate_point=estimate_point,
                phase=phase,
            )
        except (OSError, cv2.error) as exc:
            LOGGER.warning("Particle-filter debug output skipped: %s", exc)

    def _save_semantic_cv_particle_debug_impl(
        self,
        *,
        call_dir: Path,
        map_id: str,
        floor_suffix: str,
        processed_dir: Path,
        result_data: dict,
        estimate_point: Optional[tuple[float, float]],
        phase: str = "after_observation",
    ) -> None:
        if self._semantic_cv_pf is None or not getattr(self._semantic_cv_pf, "initialized", False):
            return
        particles = getattr(self._semantic_cv_pf, "particles", None)
        if particles is None:
            return
        try:
            particles_array = np.asarray(particles, dtype=np.float64)
        except Exception:
            return
        if particles_array.ndim != 2 or particles_array.shape[1] < 4 or particles_array.shape[0] == 0:
            return
        debug_dir = ensure_dir(call_dir / "particle_filter_debug")
        safe_phase = re.sub(r"[^a-zA-Z0-9_]+", "_", str(phase).strip() or "after_observation")
        np.save(debug_dir / f"particles_{safe_phase}.npy", particles_array)
        if safe_phase == "after_observation":
            np.save(debug_dir / "particles.npy", particles_array)

        floor_counts: dict[str, int] = {}
        floor_weight_sums: dict[str, float] = {}
        for code, name in ((-1.0, "b1"), (0.0, "1st"), (1.0, "2nd")):
            mask = particles_array[:, 4] == code if particles_array.shape[1] >= 5 else np.ones(len(particles_array), dtype=bool)
            floor_counts[name] = int(mask.sum())
            floor_weight_sums[name] = float(particles_array[mask, 3].sum()) if mask.any() else 0.0

        motion_accum = self._semantic_cv_pf_motion_accum
        summary = {
            "phase": safe_phase,
            "map_id": map_id,
            "active_floor": floor_suffix,
            "estimate_floor": self._semantic_cv_pf_estimate_floor,
            "estimate_point": list(estimate_point) if estimate_point is not None else None,
            "wasd_motion_accum": [float(motion_accum[0]), float(motion_accum[1])],
            "wasd_motion_radius_accum": float(self._semantic_cv_pf_motion_radius_accum),
            "wasd_motion_radius_max": float(self._semantic_cv_pf_motion_radius_max),
            "wasd_motion_seconds": float(self._semantic_cv_pf_motion_seconds),
            "wasd_last_direction": [float(self._semantic_cv_pf_motion_hint[0]), float(self._semantic_cv_pf_motion_hint[1])],
            "floor_counts": floor_counts,
            "floor_weight_sums": floor_weight_sums,
            "particle_count": int(particles_array.shape[0]),
        }
        (debug_dir / f"summary_{safe_phase}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if safe_phase == "after_observation":
            (debug_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        cloud = self._render_semantic_cv_particle_cloud(map_id, processed_dir, particles_array, result_data, estimate_point)
        if cloud is not None:
            imwrite_unicode(debug_dir / f"particle_cloud_{safe_phase}.png", cloud)
            if safe_phase == "after_observation":
                imwrite_unicode(debug_dir / "particle_cloud.png", cloud)

    def _render_semantic_cv_particle_cloud(
        self,
        map_id: str,
        processed_dir: Path,
        particles: np.ndarray,
        result_data: dict,
        estimate_point: Optional[tuple[float, float]],
    ) -> Optional[np.ndarray]:
        base = base_map_id_from_alignment(map_id)
        panels: list[np.ndarray] = []
        max_particles = int(self.project_config.app.get("semantic_cv_pf_debug_max_particles", 500))
        for floor_suffix in self.recognition_profile.floors:
            floor_map_id = self._floor_map_id(base, floor_suffix)
            image = load_floor_template(self.project_config.app, base, floor_suffix)
            if image is None:
                processed_path = processed_dir / floor_map_id / "image.png"
                image = imread_unicode(processed_path)
            if image is None:
                continue
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            out = image.copy()
            floor_code = {"b1": -1.0, "1st": 0.0, "2nd": 1.0}.get(floor_suffix, 0.0)
            floor_particles = particles[particles[:, 4] == floor_code] if particles.shape[1] >= 5 else particles
            if len(floor_particles):
                weights = floor_particles[:, 3]
                keep_count = min(max_particles, len(floor_particles))
                keep = np.argsort(weights)[-keep_count:]
                max_weight = max(float(weights[keep].max()), 1e-12)
                for idx in keep:
                    x = int(round(float(floor_particles[idx, 0])))
                    y = int(round(float(floor_particles[idx, 1])))
                    if x < 0 or y < 0 or x >= out.shape[1] or y >= out.shape[0]:
                        continue
                    intensity = int(80 + 175 * min(1.0, float(floor_particles[idx, 3]) / max_weight))
                    color = (255, intensity, 40) if floor_suffix == "1st" else (40, intensity, 255)
                    cv2.circle(out, (x, y), 2, color, -1, cv2.LINE_AA)
            active_map_floor = "b1" if map_id.endswith("_b1") else "2nd" if map_id.endswith("_2nd") else "1st"
            player_full = result_data.get("player_position_in_template") if floor_suffix == active_map_floor else None
            if isinstance(player_full, list) and len(player_full) >= 2:
                cv2.circle(out, (int(round(float(player_full[0]))), int(round(float(player_full[1])))), 10, (0, 255, 255), 2, cv2.LINE_AA)
            if estimate_point is not None and floor_suffix == self._semantic_cv_pf_estimate_floor:
                cv2.circle(out, (int(round(estimate_point[0])), int(round(estimate_point[1]))), 12, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(out, (int(round(estimate_point[0])), int(round(estimate_point[1]))), 3, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.rectangle(out, (0, 0), (min(out.shape[1] - 1, 460), 32), (0, 0, 0), -1)
            cv2.putText(
                out,
                f"{floor_map_id} particles={len(floor_particles)}",
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(out)
        if not panels:
            return None
        target_h = max(panel.shape[0] for panel in panels)
        normalized = []
        for panel in panels:
            if panel.shape[0] == target_h:
                normalized.append(panel)
                continue
            w = int(round(panel.shape[1] * target_h / max(1, panel.shape[0])))
            normalized.append(cv2.resize(panel, (w, target_h), interpolation=cv2.INTER_AREA))
        return np.hstack(normalized) if len(normalized) > 1 else normalized[0]

    def _diffuse_player_filter_transitions(self, particle_filter: Any, map_id: Optional[str]) -> None:
        if not map_id or not bool(self.project_config.app.get("semantic_cv_pf_floor_transitions_enabled", True)):
            return
        transitions = self._semantic_cv_floor_transitions(map_id)
        if not transitions:
            return
        particle_filter.diffuse_via_transitions(
            transitions,
            probability=float(self.project_config.app.get("semantic_cv_pf_floor_transition_probability", 0.5)),
            radius=float(self.project_config.app.get("semantic_cv_pf_floor_transition_radius", 45.0)),
            noise=float(self.project_config.app.get("semantic_cv_pf_floor_transition_noise", 16.0)),
        )

    def _update_semantic_cv_player_filters(
        self,
        *,
        localizer_root: Path,
        map_id: str,
        floor_suffix: str,
        result_data: dict,
    ) -> dict[str, tuple[tuple[float, float], str, tuple[int, int, int]]]:
        if not bool(self.project_config.app.get("semantic_cv_enable_particle_filter", False)):
            self._semantic_cv_player_pfs.clear()
            self._semantic_cv_player_pf_updated_at.clear()
            self._semantic_cv_player_pf_key = None
            return {}
        players = self._semantic_cv_players(result_data)
        if not players:
            return {}
        app = self.project_config.app
        filter_key = base_map_id_from_alignment(map_id)
        if self._semantic_cv_player_pf_key != filter_key:
            self._semantic_cv_player_pfs.clear()
            self._semantic_cv_player_pf_updated_at.clear()
            self._semantic_cv_player_pf_key = filter_key
        particle_filter_class = self._semantic_cv_particle_filter_class(localizer_root)
        now = time.monotonic()
        estimates: dict[str, tuple[tuple[float, float], str, tuple[int, int, int]]] = {}
        debug_players: dict[str, dict[str, object]] = {}
        for player_id, point, color in players:
            particle_filter = self._semantic_cv_player_pfs.get(player_id)
            if particle_filter is None:
                particle_filter = particle_filter_class(
                    count=int(app.get("semantic_cv_pf_particle_count", 1600)),
                    motion_noise=float(app.get("semantic_cv_pf_motion_noise", 45.0)),
                    scale_noise=float(app.get("semantic_cv_pf_scale_noise", 0.0005)),
                    observation_noise=float(app.get("semantic_cv_pf_observation_noise", 10.0)),
                )
                self._semantic_cv_player_pfs[player_id] = particle_filter
            if getattr(particle_filter, "initialized", False):
                previous = self._semantic_cv_player_pf_updated_at.get(player_id, now)
                dt = max(0.1, min(float(app.get("semantic_cv_pf_max_predict_dt", 2.0)), now - previous))
                # No walkable-mask or channel-candidate restriction: this is plain circular diffusion.
                particle_filter.predict_without_observation(dt)
                self._diffuse_player_filter_transitions(particle_filter, filter_key)
                particle_filter.update_by_observation(
                    point[0], point[1], confidence=max(0.15, float(app.get("semantic_cv_pf_confidence_bias", 0.5))), floor=floor_suffix
                )
            else:
                particle_filter.reset(
                    point[0],
                    point[1],
                    spread=float(app.get("semantic_cv_pf_init_spread", 14.0)),
                    floor=floor_suffix,
                )
            self._semantic_cv_player_pf_updated_at[player_id] = now
            estimate = particle_filter.estimate()
            if estimate is None:
                continue
            estimates[player_id] = ((float(estimate.x), float(estimate.y)), str(estimate.floor), color)
            debug_players[player_id] = {
                "point": [float(estimate.x), float(estimate.y)],
                "floor": str(estimate.floor),
                "particle_count": int(getattr(particle_filter, "count", 0)),
            }
        result_data["player_particle_filters"] = {
            "enabled": True,
            "mode": "independent_circular_velocity_diffusion",
            "walkable_constraint": False,
            "transition_probability": float(app.get("semantic_cv_pf_floor_transition_probability", 0.5)),
            "players": debug_players,
        }
        return estimates

    def _update_semantic_cv_particle_filter(
        self,
        *,
        localizer_root: Path,
        map_id: str,
        floor_suffix: str,
        call_dir: Path,
        processed_dir: Path,
        result_data: dict,
    ) -> Optional[tuple[float, float]]:
        if not bool(self.project_config.app.get("semantic_cv_enable_particle_filter", True)):
            self._semantic_cv_pf = None
            self._semantic_cv_pf_key = None
            self._semantic_cv_pf_updated_at = None
            return None
        app = self.project_config.app
        first_second_floor_location = (
            floor_suffix == "2nd"
            and self._semantic_cv_pf is not None
            and getattr(self._semantic_cv_pf, "initialized", False)
            and self._semantic_cv_pf_estimate_floor != "2nd"
        )
        observation_top_n = None
        if first_second_floor_location:
            observation_top_n = int(app.get("semantic_cv_pf_first_second_floor_top_n_observations", 6))
        observations = self._semantic_cv_candidate_observations(result_data, top_n=observation_top_n)
        if not observations:
            return None
        pf_key = base_map_id_from_alignment(map_id)
        pf_class = self._semantic_cv_particle_filter_class(localizer_root)
        if self._semantic_cv_pf is None or self._semantic_cv_pf_key != pf_key:
            self._semantic_cv_pf = pf_class(
                count=int(app.get("semantic_cv_pf_particle_count", 1600)),
                motion_noise=float(app.get("semantic_cv_pf_motion_noise", 45.0)),
                scale_noise=float(app.get("semantic_cv_pf_scale_noise", 0.0005)),
                observation_noise=float(app.get("semantic_cv_pf_observation_noise", 10.0)),
            )
            self._semantic_cv_pf_key = pf_key
            self._semantic_cv_pf_updated_at = None
            self._semantic_cv_pf_last_motion_at = None
        self._configure_semantic_cv_pf_walkable_masks(processed_dir, map_id)

        now = time.monotonic()
        if getattr(self._semantic_cv_pf, "initialized", False):
            dt = 1.0 if self._semantic_cv_pf_updated_at is None else max(0.1, now - self._semantic_cv_pf_updated_at)
            relocalize_dt = float(app.get("semantic_cv_pf_relocalize_dt", 3.0))
            max_predict_dt = float(app.get("semantic_cv_pf_max_predict_dt", 2.0))
            predict_dt = dt if max_predict_dt <= 0 else min(dt, max_predict_dt)
            if relocalize_dt > 0 and dt >= relocalize_dt:
                self._semantic_cv_pf.relocalize(observations, floor=floor_suffix)
            else:
                self._semantic_cv_pf.predict_without_observation(predict_dt, floor=floor_suffix)
            transition_probability = None
            if first_second_floor_location:
                transition_probability = float(app.get("semantic_cv_pf_first_second_floor_transition_probability", 0.5))
                result_data["first_second_floor_pre_observation_transition_probability"] = transition_probability
            self._semantic_cv_apply_floor_transitions(map_id, probability=transition_probability)
        if not getattr(self._semantic_cv_pf, "initialized", False):
            x, y, scale, _score = observations[0]
            self._semantic_cv_pf.reset(x, y, scale, spread=float(app.get("semantic_cv_pf_init_spread", 14.0)), floor=floor_suffix)

        before_estimate = self._semantic_cv_pf.estimate()
        before_point: Optional[tuple[float, float]] = None
        if before_estimate is not None:
            self._semantic_cv_pf_estimate_floor = str(getattr(before_estimate, "floor", floor_suffix))
            if self._semantic_cv_pf_estimate_floor == floor_suffix:
                before_point = (float(before_estimate.x), float(before_estimate.y))
        self._save_semantic_cv_particle_debug(
            call_dir=call_dir,
            map_id=map_id,
            floor_suffix=floor_suffix,
            processed_dir=processed_dir,
            result_data=result_data,
            estimate_point=before_point,
            phase="before_observation",
        )

        if first_second_floor_location:
            observations = self._semantic_cv_apply_first_second_floor_prior(
                observations,
                map_id=map_id,
                result_data=result_data,
            )

        confidence = max(
            0.15,
            float(result_data.get("confidence") or 0.0) + float(app.get("semantic_cv_pf_confidence_bias", 2.0)),
        )
        try:
            self._semantic_cv_pf.inject_observations(
                observations,
                floor=floor_suffix,
                fraction=float(app.get("semantic_cv_pf_observation_injection_fraction", 0.04)),
                spread=float(app.get("semantic_cv_pf_observation_injection_spread", app.get("semantic_cv_pf_init_spread", 14.0))),
            )
        except AttributeError:
            pass
        self._semantic_cv_pf.update_by_observations(observations, confidence=confidence, floor=floor_suffix)
        self._semantic_cv_apply_floor_transitions(map_id)
        self._semantic_cv_pf_updated_at = now
        estimate = self._semantic_cv_pf.estimate()
        if estimate is None:
            return None
        self._semantic_cv_pf_estimate_floor = str(getattr(estimate, "floor", floor_suffix))
        if self._semantic_cv_pf_estimate_floor != floor_suffix:
            return None
        return float(estimate.x), float(estimate.y)

    def _confirmed_semantic_cv_scale_path(self) -> Path:
        return semantic_cv_confirmed_scale_path(
            self.project_config.root_dir,
            self._scale_storage_app_config(),
        )

    def _scale_storage_app_config(self, profile_key: Optional[str] = None) -> dict[str, Any]:
        key = profile_key or self.recognition_profile.key
        return scale_profile_app_config(self.project_config.app, key)

    def _show_semantic_cv_scale_search_progress(self, map_id: str, floor_suffix: str, mode: str) -> None:
        """Keep the cheatsheet on a dedicated waiting page while SCALE is being searched."""
        floor_label = {"b1": "地下室", "1st": "一楼", "2nd": "二楼"}.get(floor_suffix, floor_suffix)
        mode_label = "重新搜索" if mode == "manual_overlay_research" else "首次搜索"
        message = (
            f"正在{mode_label} {map_id} {floor_label}的 SCALE。\n\n"
            "请保持游戏地图开启，不要移动地图或玩家。\n"
            "搜索完成后将显示虚像，并询问是否确认保存 SCALE。"
        )
        return_content = (
            getattr(self, "_manual_scale_research_return_content", None)
            if mode == "manual_overlay_research"
            and isinstance(getattr(self, "_manual_scale_research_return_content", None), tuple)
            else getattr(self.cheatsheet, "_last_content", None)
        )
        self._active_scale_search_return_content = (
            return_content if isinstance(return_content, tuple) else None
        )
        self._scale_search_active = True
        self._scale_search_cancel_requested = False
        self._scale_search_stop_key_down = False
        self._active_scale_search_map_id = map_id
        self._active_scale_search_floor = floor_suffix
        self.status_var.set(f"正在{mode_label} {map_id} {floor_label} SCALE，请保持地图开启，不要移动")
        self.cheatsheet.show_action_button(
            message,
            title=f"{mode_label} SCALE：{map_id}",
            button_text="停止搜索（F2）",
            on_click=self._stop_semantic_cv_scale_search,
        )

    def _stop_semantic_cv_scale_search(self) -> None:
        if not self._scale_search_active:
            return
        map_id = str(self._active_scale_search_map_id or "").strip()
        floor_suffix = str(self._active_scale_search_floor or "").strip()
        return_content = self._active_scale_search_return_content
        self._scale_search_cancel_requested = True
        self._scale_search_active = False
        request_id = self._active_route_overlay_request_id
        if request_id is not None:
            self._cancelled_route_overlay_request_id = max(
                self._cancelled_route_overlay_request_id,
                request_id,
            )
        self._semantic_cv_scale_confirmation_pending = None
        self._stop_semantic_cv_worker()
        self.route_overlay.hide()
        self._route_overlay_dismiss_armed = False
        self._overlay_scale_research_context = None
        diagnostic_event(
            "semantic_cv_scale_search_stopped",
            map_id=map_id,
            floor=floor_suffix,
            request_id=request_id,
            trigger="button_or_f2",
        )
        if isinstance(return_content, tuple) and map_id and floor_suffix in {"b1", "1st", "2nd"}:
            self._scale_research_return_context = {
                "map_id": map_id,
                "floor_suffix": floor_suffix,
                "content": return_content,
            }
            self._return_to_scale_research_sheet()
            self.status_var.set(f"已停止 {map_id} 的 SCALE 搜索并返回路线图。")
        else:
            self.cheatsheet.show_message("SCALE 搜索已停止。", title="已停止搜索")
            self.status_var.set("SCALE 搜索已停止。")

    def _show_pending_semantic_cv_scale_confirmation(self, map_id: str, floor_suffix: str) -> None:
        pending = self._semantic_cv_scale_confirmation_pending
        if (
            not isinstance(pending, dict)
            or pending.get("map_id") != map_id
            or pending.get("floor") != floor_suffix
        ):
            return
        pending["return_content"] = self.cheatsheet._last_content
        scale = float(pending["scale"])
        hotkey = hotkey_binding_display_name(
            self._hotkeys.get("locate_current_floor", HOTKEY_DEFAULTS["locate_current_floor"])
        )
        self.status_var.set(f"等待确认 {map_id} 的路线图虚像是否对齐（scale={scale:.4f}）。")
        self.cheatsheet.show_confirmation(
            "请观察游戏内地图上方的路线图虚像。\n当前虚像是否准确对齐？\n\n选择“是”后，该布局和当前分辨率以后将直接复用此 scale。",
            title=f"确认虚像对齐 · {map_id} · scale={scale:.4f}",
            yes_text="是，固定此 SCALE",
            no_text=f"否，探索后再按 {hotkey}",
            on_yes=lambda: self._accept_semantic_cv_scale_confirmation(pending),
            on_no=lambda: self._reject_semantic_cv_scale_confirmation(pending),
        )

    def _accept_semantic_cv_scale_confirmation(self, pending: dict[str, Any]) -> None:
        if pending is not self._semantic_cv_scale_confirmation_pending:
            return
        map_id = str(pending["map_id"])
        floor_suffix = str(pending["floor"])
        roi_size = tuple(int(value) for value in pending["roi_size"][:2])
        scale = float(pending["scale"])
        save_scope = str(pending.get("save_scope", "confirmed"))
        try:
            if save_scope == "baseline":
                reference_size = self.project_config.app.get("alignment_reference_capture_size", (2560, 1440))
                reference_resolution = (
                    f"{int(reference_size[0])}x{int(reference_size[1])}"
                    if isinstance(reference_size, (list, tuple)) and len(reference_size) >= 2
                    else "2560x1440"
                )
                update_saved_scale(
                    self.project_config.root_dir,
                    self._scale_storage_app_config(),
                    SavedScale("baseline", map_id, reference_resolution, scale),
                    scale,
                )
            else:
                save_semantic_cv_confirmed_scale(self._confirmed_semantic_cv_scale_path(), map_id, roi_size, scale)
        except Exception as exc:
            LOGGER.exception("failed to persist semantic CV scale")
            self.status_var.set(f"保存确认 scale 失败：{exc}")
            self.cheatsheet.show_message(f"无法保存确认结果\n{exc}", title="保存 SCALE 失败")
            return
        self._semantic_cv_scale_cache[semantic_cv_scale_cache_key(map_id, roi_size)] = scale
        base_scale = pending.get("base_scale")
        if floor_suffix == "1st" and isinstance(base_scale, (int, float)) and float(base_scale) > 0:
            multiplier_key = (base_map_id_from_alignment(map_id), roi_size[0], roi_size[1])
            self._semantic_cv_device_multiplier_cache[multiplier_key] = scale / float(base_scale)
        self._semantic_cv_scale_retry_floors.discard(floor_suffix)
        self._semantic_cv_scale_confirmation_pending = None
        self._manual_scale_research_return_content = None
        self._scale_research_return_context = None
        self.cheatsheet.restore_content(pending.get("return_content"))
        self.status_var.set(
            f"已保存 {map_id} 在 {roi_size[0]}×{roi_size[1]} ROI 下的 scale={scale:.4f}；以后识别到该布局将直接复用。"
        )

    def _reject_semantic_cv_scale_confirmation(self, pending: dict[str, Any]) -> None:
        if pending is not self._semantic_cv_scale_confirmation_pending:
            return
        map_id = str(pending["map_id"])
        floor_suffix = str(pending["floor"])
        roi_size = tuple(int(value) for value in pending["roi_size"][:2])
        self._semantic_cv_scale_cache.pop(semantic_cv_scale_cache_key(map_id, roi_size), None)
        # Unconfirmed runs are never inserted into the multiplier cache. Keep
        # an already confirmed 1F multiplier as the prediction centre for a
        # subsequent 2F scale retry.
        self._semantic_cv_scale_retry_floors.add(floor_suffix)
        self._semantic_cv_scale_confirmation_pending = None
        self.route_overlay.hide()
        hotkey = hotkey_binding_display_name(
            self._hotkeys.get("locate_current_floor", HOTKEY_DEFAULTS["locate_current_floor"])
        )
        manual_return_content = pending.get("manual_research_return_content")
        if bool(pending.get("manual_overlay_research")) and isinstance(manual_return_content, tuple):
            self._scale_research_return_context = {
                "map_id": map_id,
                "floor_suffix": floor_suffix,
                "content": manual_return_content,
            }
            self.status_var.set(f"{map_id} 的手动 SCALE 重搜未对齐；可按 F3 返回路线图后再次搜索。")
            self.cheatsheet.show_action_button(
                "本次手动重新搜索的 SCALE 未对齐，错误虚像已隐藏。\n可继续探索、去除更多迷雾后，再返回路线图重新搜索。",
                title="虚像未对齐",
                button_text="返回路线图（F3）",
                on_click=self._return_to_scale_research_sheet,
            )
            return
        # Initial scale searches also start from a route-map cheatsheet. Keep
        # that page recoverable when the player rejects the first alignment.
        return_content = pending.get("return_content")
        if isinstance(return_content, tuple):
            self._scale_research_return_context = {
                "map_id": map_id,
                "floor_suffix": floor_suffix,
                "content": return_content,
            }
            self.status_var.set(f"{map_id} 的 SCALE 未固定；可返回路线图（F3）继续探索，之后按 {hotkey} 重新搜索。")
            self.cheatsheet.show_action_button(
                f"本次 SCALE 未固定，错误虚像已隐藏。\n请返回路线图继续探索、去除更多迷雾，之后再按 {hotkey} 重新搜索 scale 并定位。",
                title="虚像未对齐",
                button_text="返回路线图（F3）",
                on_click=self._return_to_scale_research_sheet,
            )
            return
        self.status_var.set(f"{map_id} 的 scale 未固定；探索更多地图后按 {hotkey} 重新搜索并定位。")
        self.cheatsheet.show_message(
            f"本次 SCALE 未固定，错误虚像已隐藏。\n请继续探索、去除更多迷雾后再按 {hotkey}。\n下一次会重新搜索 scale 并定位。",
            title="虚像未对齐",
        )

    def _can_return_to_scale_research_sheet(self) -> bool:
        context = self._scale_research_return_context
        return (
            isinstance(context, dict)
            and isinstance(context.get("content"), tuple)
            and bool(str(context.get("map_id", "")).strip())
            and str(context.get("floor_suffix", "")) in {"b1", "1st", "2nd"}
        )

    def _return_to_scale_research_sheet(self) -> None:
        if not self._can_return_to_scale_research_sheet():
            return
        context = self._scale_research_return_context
        assert isinstance(context, dict)
        map_id = str(context["map_id"])
        floor_suffix = str(context["floor_suffix"])
        content = context["content"]
        self._scale_research_return_context = None
        self._route_overlay_dismiss_armed = False
        self._overlay_scale_research_context = None
        if content[0] == "image_with_bottom_action":
            image, title, footer, compact_footer, _button_text, _on_click = content[1]
            content = (
                ("image_with_message", (image, title, footer, compact_footer))
                if footer
                else ("image", (image, title))
            )
        self.cheatsheet.restore_content(content)
        hotkey = hotkey_binding_display_name(
            self._hotkeys.get("locate_current_floor", HOTKEY_DEFAULTS["locate_current_floor"])
        )
        self.status_var.set(f"已返回 {map_id} 路线图；当前虚像已关闭，按 {hotkey} 重新定位后可再次搜索。")

    def _show_semantic_cv_location_sheet(
        self,
        image: np.ndarray,
        *,
        map_id: str,
        title: str,
        footer: str,
    ) -> None:
        """Append the scale-search row only while this layout's overlay is visible."""
        context = self._overlay_scale_research_context
        if (
            self._route_overlay_dismiss_armed
            and isinstance(context, dict)
            and context.get("map_id") == map_id
        ):
            self.cheatsheet.show_image_with_bottom_action(
                image,
                title=title,
                footer=footer,
                compact_footer=True,
                button_text="重新搜索 SCALE",
                on_click=self._research_active_overlay_scale,
            )
            return
        self.cheatsheet.show_image(image, title=title, footer=footer, compact_footer=True)

    def _research_active_overlay_scale(self) -> None:
        """Re-run the first-use scale search while the already-open game map remains visible."""
        context = self._overlay_scale_research_context
        if not self._route_overlay_dismiss_armed or not isinstance(context, dict):
            self.cheatsheet.show_message("当前没有可重新搜索的路线图虚像。", title="SCALE 搜索不可用")
            return
        map_id = str(context.get("map_id", "")).strip()
        floor_suffix = str(context.get("floor_suffix", "")).strip()
        if not map_id or floor_suffix not in set(self.recognition_profile.floors):
            locate_hotkey = configured_hotkey_display_name(self, "locate_current_floor")
            self.cheatsheet.show_message(
                f"缺少当前虚像的布局信息，请重新按 {locate_hotkey} 定位。",
                title="SCALE 搜索不可用",
            )
            return
        self._manual_scale_research_return_content = self.cheatsheet._last_content
        self._scale_research_return_context = None
        self.route_overlay.hide()
        self._route_overlay_dismiss_armed = False
        self._overlay_scale_research_context = None
        self.cheatsheet.show_message(
            "正在按当前客户区的首次缺失 SCALE 范围重新搜索。\n搜索完成后请确认新的虚像是否对齐。",
            title=f"重新搜索 SCALE：{map_id}",
        )
        self.status_var.set(f"正在重新搜索 {map_id} 的 SCALE（{floor_suffix}）。")

        def run() -> None:
            try:
                self._capture_and_run_semantic_cv_location(
                    floor_suffix,
                    require_scale_confirmation=True,
                    force_scale_research=True,
                )
            except SemanticCVSearchCancelled:
                return
            except Exception as exc:
                LOGGER.exception("manual overlay scale research failed")
                self.status_var.set(f"SCALE 重新搜索失败：{exc}")
                self.cheatsheet.show_message(f"SCALE 重新搜索失败\n{exc}", title="SCALE 重新搜索失败")

        self.after(10, run)

    def _capture_and_run_semantic_cv_location(
        self,
        floor_suffix: Optional[str],
        *,
        require_scale_confirmation: bool = False,
        force_scale_research: bool = False,
    ) -> None:
        diagnostic_event(
            "semantic_cv_location_begin",
            requested_floor=floor_suffix,
            active_request_id=self._active_route_overlay_request_id,
            active_layout=self._active_route_layout_id(),
        )
        route_overlay = getattr(self, "route_overlay", None)
        prior_overlay_hidden = bool(getattr(route_overlay, "_visible", False))
        reset_overlay = getattr(route_overlay, "reset", None)
        if callable(reset_overlay):
            reset_overlay()
            # Destroy the prior HWND before capture. No old native surface or
            # texture can participate in this request's first visible frame.
            self.update_idletasks()
        auto_detect_floor = floor_suffix is None
        capture_label = "自动楼层" if auto_detect_floor else floor_suffix
        _capture, alignment_roi = self._capture_alignment_roi(f"{capture_label} CV定位截图成功")
        capture = _capture
        floor_detection_debug: Optional[dict[str, Any]] = None
        floor_detection_strip: Optional[np.ndarray] = None
        deferred_diagnostic_writers: list[Callable[[], None]] = []
        if auto_detect_floor:
            previous_floor = self._active_floor_suffix
            floor_detector_details: dict[str, Any] = {}
            floor_search_roi = floor_detection_search_roi_for_profile(
                self.project_config.app,
                self.recognition_profile.key,
            )
            floor_suffix, floor_confidence, first_score, second_score = detect_open_map_floor(
                alignment_roi,
                minimum_margin=float(
                    self.project_config.app.get("map_floor_detection_minimum_margin", 12.0)
                ),
                search_roi_normalized=floor_search_roi,
                available_floors=tuple(self.recognition_profile.floors),
                debug_details=floor_detector_details,
            )
            diagnostic_event(
                "floor_detected",
                detected_floor=floor_suffix,
                previous_floor=previous_floor,
                confidence=float(floor_confidence),
                first_score=float(first_score),
                second_score=float(second_score),
                capture_roi=capture.roi,
                target_size=capture.target_size,
            )
            if floor_suffix != previous_floor and {floor_suffix, previous_floor} in (
                {"b1", "1st"},
                {"1st", "2nd"},
            ):
                self._floor_transition_constraint_source = previous_floor
            else:
                self._floor_transition_constraint_source = None
            if floor_suffix == "2nd" and previous_floor == "1st":
                self._second_floor_transition_constraint_active = True
            self._active_floor_suffix = floor_suffix
            require_scale_confirmation = semantic_cv_requires_initial_scale_confirmation(
                self._semantic_cv_initial_location_floors,
                self._semantic_cv_scale_retry_floors,
                floor_suffix,
            )
            LOGGER.info(
                "map floor detected floor=%s confidence=%.2f first_score=%.2f second_score=%.2f",
                floor_suffix,
                floor_confidence,
                first_score,
                second_score,
            )
            roi_height, roi_width = alignment_roi.shape[:2]
            strip_x0, strip_y0, strip_x1, strip_y1 = floor_detection_search_roi(
                (roi_width, roi_height),
                floor_search_roi,
            )
            floor_detection_strip = alignment_roi[strip_y0:strip_y1, strip_x0:strip_x1].copy()
            floor_detection_debug = {
                "detected_floor": floor_suffix,
                "previous_floor": previous_floor,
                "confidence": floor_confidence,
                "first_score": first_score,
                "second_score": second_score,
                **floor_detector_details,
                "alignment_roi_size": [int(roi_width), int(roi_height)],
                "saved_strip_roi": [
                    int(strip_x0),
                    int(strip_y0),
                    int(strip_x1 - strip_x0),
                    int(strip_y1 - strip_y0),
                ],
                "capture_mode": capture.mode,
                "capture_roi": [int(value) for value in capture.roi],
                "window_title": capture.window_title,
                "target_size": list(capture.target_size) if capture.target_size else None,
                "reference_size": list(capture.reference_size) if capture.reference_size else None,
                "recognition_profile": self.recognition_profile.key,
            }
            floor_label = {"b1": "地下室", "1st": "一楼", "2nd": "二楼"}.get(floor_suffix, floor_suffix)
            self.status_var.set(f"已自动识别当前地图为{floor_label}，正在执行 CV 定位。")
        assert floor_suffix in set(self.recognition_profile.floors)
        map_id = self._semantic_cv_map_id(floor_suffix)
        debug_root = self.project_config.root_dir / "assets" / "debug" / "semantic_cv_location"
        ts = timestamp()
        call_dir = debug_root / f"{ts}_{map_id}"
        image_path = call_dir / "input_roi.png"
        config_path = call_dir / "config_runtime.json"
        development_capture_enabled = bool(
            self.project_config.app.get("semantic_cv_development_roi_capture_enabled", False)
        )
        if development_capture_enabled:
            save_started = time.perf_counter()
            try:
                save_semantic_cv_development_capture(
                    call_dir,
                    alignment_roi,
                    floor_detection_strip=floor_detection_strip,
                    floor_detection_debug=floor_detection_debug,
                )
                diagnostic_event(
                    "development_roi_saved",
                    map_id=map_id,
                    path=str(image_path),
                    save_ms=round((time.perf_counter() - save_started) * 1000.0, 3),
                    before_cv_worker=True,
                )
            except Exception as exc:
                diagnostic_event(
                    "development_roi_save_failed",
                    map_id=map_id,
                    path=str(image_path),
                    exception_type=type(exc).__name__,
                    exception=repr(exc),
                )
                LOGGER.warning("failed to save development localization ROI before CV", exc_info=True)

        localizer_root = self._semantic_cv_project_root()
        if not localizer_root.exists():
            raise RuntimeError(f"找不到 CV 定位工程：{localizer_root}")
        processed_dir = localizer_root / str(self.project_config.app.get("semantic_cv_processed_template_dir", "templates_processed"))
        if not (processed_dir / map_id).exists():
            raise RuntimeError(f"缺少 CV 预处理模板：{processed_dir / map_id}，请先预处理该楼层模板")

        if floor_detection_debug is not None and floor_detection_strip is not None:
            debug_payload = dict(floor_detection_debug)
            debug_strip = floor_detection_strip

            def save_floor_detection_debug() -> None:
                try:
                    ensure_dir(call_dir)
                    imwrite_unicode(call_dir / "floor_tabs.png", debug_strip)
                    (call_dir / "floor_detection.json").write_text(
                        json.dumps(debug_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    LOGGER.debug("failed to save floor detection diagnostics", exc_info=True)

            if not development_capture_enabled:
                deferred_diagnostic_writers.append(save_floor_detection_debug)

        cfg = self._load_semantic_cv_config(localizer_root)
        h, w = alignment_roi.shape[:2]
        raw_roi_size = (int(w), int(h))
        cfg["map_roi"] = [0, 0, int(w), int(h)]
        cfg["processed_template_dir"] = str(processed_dir)
        cfg["output_dir"] = str(call_dir)
        cfg["output_dir_direct"] = True
        particle_filter_enabled = bool(self.project_config.app.get("semantic_cv_enable_particle_filter", True))
        initial_location_floors = getattr(self, "_semantic_cv_initial_location_floors", set())
        attempted_floors = getattr(self, "_semantic_cv_location_attempted_floors", None)
        if not isinstance(attempted_floors, set):
            attempted_floors = set()
            self._semantic_cv_location_attempted_floors = attempted_floors
        floor_already_attempted = floor_suffix in attempted_floors
        first_first_floor_location = consume_initial_first_floor_location_attempt(
            attempted_floors,
            floor_suffix,
            consume=not force_scale_research,
        )
        first_entry_floor_location = (
            not force_scale_research
            and not floor_already_attempted
            and floor_suffix == self.recognition_profile.entry_floor
        )
        transition_source_floor = getattr(self, "_floor_transition_constraint_source", None)
        floor_transition_location = (
            transition_source_floor in {"b1", "1st", "2nd"}
            and transition_source_floor != floor_suffix
            and floor_suffix not in initial_location_floors
        )
        first_second_floor_location = (
            floor_suffix == "2nd"
            and transition_source_floor == "1st"
            and "2nd" not in initial_location_floors
        )
        first_basement_location = (
            not force_scale_research
            and not floor_already_attempted
            and floor_suffix == "b1"
        )
        b1_to_first_floor_location = (
            floor_transition_location
            and transition_source_floor == "b1"
            and floor_suffix == "1st"
        )
        bound_transition_match_location = (
            first_second_floor_location
            or b1_to_first_floor_location
            or (
                floor_transition_location
                and {transition_source_floor, floor_suffix} == {"1st", "2nd"}
            )
        )
        first_floor_side_door_allowed_points: list[list[float]] = []
        side_door_constraint_requested = (
            first_entry_floor_location
            if self.recognition_profile.key == "nightmare"
            else first_first_floor_location
        )
        if side_door_constraint_requested and bool(self.project_config.app.get("semantic_cv_first_f1_match_only_side_doors", True)):
            first_floor_side_door_allowed_points = self._semantic_cv_side_door_entry_points(map_id)
            if first_floor_side_door_allowed_points:
                cfg["allowed_player_template_points"] = first_floor_side_door_allowed_points
                cfg["allowed_player_template_radius"] = float(
                    self.project_config.app.get("semantic_cv_first_f1_side_door_match_radius", 110.0)
                )
                cfg["allowed_player_constraint_semantic_component_max"] = max(
                    0,
                    int(
                        self.project_config.app.get(
                            "semantic_cv_first_f1_constraint_semantic_component_max",
                            2,
                        )
                    ),
                )
        first_basement_allowed_points: list[list[float]] = []
        if first_basement_location and bool(
            self.project_config.app.get("semantic_cv_first_b1_match_only_bound_transitions", True)
        ):
            # A first B1 view can originate at either a bound inter-floor
            # passage or its layout-specific teleport point. Later B1
            # relocalizations deliberately remain unconstrained.
            for item in self._semantic_cv_floor_transitions(map_id):
                point = item.get("b1")
                if (
                    isinstance(point, list)
                    and len(point) >= 2
                    and isinstance(point[0], (int, float))
                    and isinstance(point[1], (int, float))
                ):
                    first_basement_allowed_points.append([float(point[0]), float(point[1])])
            first_basement_allowed_points.extend(self._semantic_cv_basement_tp_points(map_id))
            unique_points: list[list[float]] = []
            for point in first_basement_allowed_points:
                if point not in unique_points:
                    unique_points.append(point)
            first_basement_allowed_points = unique_points
            if first_basement_allowed_points:
                cfg["allowed_player_template_points"] = first_basement_allowed_points
                cfg["allowed_player_template_radius"] = float(
                    self.project_config.app.get("semantic_cv_first_b1_match_radius", 110.0)
                )
                cfg["allowed_player_constraint_semantic_component_max"] = max(
                    0,
                    int(
                        self.project_config.app.get(
                            "semantic_cv_first_b1_constraint_semantic_component_max",
                            2,
                        )
                    ),
                )
        first_second_floor_allowed_points: list[list[float]] = []
        if bound_transition_match_location and bool(self.project_config.app.get("semantic_cv_first_second_floor_match_only_bound_transitions", True)):
            for item in self._semantic_cv_floor_transitions(map_id):
                second = item.get(floor_suffix)
                if (
                    isinstance(second, list)
                    and len(second) >= 2
                    and isinstance(second[0], (int, float))
                    and isinstance(second[1], (int, float))
                ):
                    first_second_floor_allowed_points.append([float(second[0]), float(second[1])])
            if first_second_floor_allowed_points:
                cfg["allowed_player_template_points"] = first_second_floor_allowed_points
                cfg["allowed_player_template_radius"] = float(
                    self.project_config.app.get("semantic_cv_first_second_floor_match_radius", 220.0)
                )
                cfg["allowed_player_constraint_semantic_component_max"] = max(
                    0,
                    int(
                        self.project_config.app.get(
                            "semantic_cv_first_second_floor_constraint_semantic_component_max",
                            2,
                        )
                    ),
                )
        if floor_transition_location:
            self._floor_transition_constraint_source = None
        if first_second_floor_location:
            first_second_floor_top_n = int(self.project_config.app.get("semantic_cv_pf_first_second_floor_top_n_observations", 6))
            cfg["allowed_player_constraint_top_k"] = first_second_floor_top_n
        cfg["candidate_pool_size"] = max(
            int(cfg.get("top_k", 8)),
            int(self.project_config.app.get("semantic_cv_candidate_pool_size", 48)),
        )
        cfg["candidate_border_padding"] = max(
            0,
            int(self.project_config.app.get("semantic_cv_candidate_border_padding", 8)),
        )
        cfg["geometric_review_enabled"] = semantic_cv_geometric_review_enabled_for_map(
            self.project_config.app,
            map_id,
        )
        for config_key, default in (
            ("geometric_review_candidate_count", 32),
            ("geometric_review_sparse_semantic_only_enabled", True),
            ("geometric_review_sparse_semantic_component_max", 2),
            ("geometric_review_semantic_component_min_area", 60),
            ("geometric_review_sparse_semantic_min_stage1_margin", 0.01),
            ("geometric_review_strength", 0.50),
            ("geometric_review_semantic_weight", 0.55),
            ("geometric_review_directional_weight", 0.45),
            ("geometric_review_forward_weight", 0.60),
            ("geometric_review_trim_ratio", 0.15),
            ("geometric_review_hit_distance", 3.0),
            ("geometric_review_max_distance", 20.0),
            ("geometric_review_valid_erosion", 3),
            ("geometric_review_edge_support_radius", 4),
        ):
            cfg[config_key] = self.project_config.app.get(f"semantic_cv_{config_key}", default)
        scale_app_config = self._scale_storage_app_config()
        base_scale = semantic_cv_reference_scale_for_map(
            scale_app_config,
            map_id,
            self.project_config.root_dir,
        )
        roi_size = raw_roi_size
        scale_cache_key = semantic_cv_scale_cache_key(map_id, roi_size)
        persistent_confirmed_scale = None
        if not force_scale_research:
            persistent_confirmed_scale = semantic_cv_confirmed_scale_for_map(
                self._confirmed_semantic_cv_scale_path(),
                map_id,
                roi_size,
            )
            if persistent_confirmed_scale is not None:
                self._semantic_cv_scale_cache[scale_cache_key] = persistent_confirmed_scale
        saved_current_roi_scale = persistent_confirmed_scale
        if saved_current_roi_scale is None and not force_scale_research:
            saved_current_roi_scale = self._semantic_cv_scale_cache.get(scale_cache_key)
        reference_roi_normalization = semantic_cv_reference_roi_normalization(
            scale_app_config,
            raw_roi_size,
            base_scale,
            force_scale_research=force_scale_research or saved_current_roi_scale is not None,
        )
        localization_roi = alignment_roi
        if reference_roi_normalization is not None:
            reference_width, reference_height = reference_roi_normalization["reference_roi_size"]
            localization_roi = cv2.resize(
                alignment_roi,
                (int(reference_width), int(reference_height)),
                interpolation=cv2.INTER_LINEAR,
            )
            cfg["map_roi"] = [0, 0, int(reference_width), int(reference_height)]
        capture_target_size = capture.target_size
        if capture_target_size is None:
            _screen_left, _screen_top, screen_width, screen_height = primary_screen_bounds()
            capture_target_size = (screen_width, screen_height)
        screen_width, screen_height = capture_target_size
        reference_base_scale_active = semantic_cv_uses_reference_screen_base_scale(
            scale_app_config,
            capture_target_size,
            base_scale,
        )
        device_multiplier_key = (
            base_map_id_from_alignment(map_id),
            roi_size[0],
            roi_size[1],
        )
        if force_scale_research:
            # Deliberately bypass every stored value. This is the same broad
            # first-use range used when a scale is absent for this client ROI.
            cached_scale = None
        elif saved_current_roi_scale is not None:
            cached_scale = float(saved_current_roi_scale)
        elif reference_roi_normalization is not None:
            cached_scale = float(base_scale)
        elif reference_base_scale_active:
            cached_scale = float(base_scale)
            self._semantic_cv_scale_cache[scale_cache_key] = cached_scale
        else:
            cached_scale = None
        confirmation_required_for_run = bool(
            force_scale_research
            or (
                require_scale_confirmation
                and not reference_base_scale_active
                and reference_roi_normalization is None
                and persistent_confirmed_scale is None
            )
        )
        adaptive_scale_enabled = bool(
            self.project_config.app.get("semantic_cv_adaptive_scale_enabled", True)
        )
        scale_runtime: dict[str, Any] = {
            "map_id": map_id,
            "floor": floor_suffix,
            "roi_size": [roi_size[0], roi_size[1]],
            "cache_key": [scale_cache_key[0], scale_cache_key[1], scale_cache_key[2]],
            "base_scale": base_scale,
            "screen_size": [screen_width, screen_height],
            "reference_base_scale_active": reference_base_scale_active,
            "reference_roi_normalization": reference_roi_normalization,
        }
        estimated_layout_scale_window = semantic_cv_estimated_scale_window_for_map(
            self.project_config.root_dir,
            scale_app_config,
            map_id,
            roi_size,
        )
        # The GUI owns the session-specific scale. Prevent localize_frame.py
        # from replacing it with the old per-layout value a second time.
        cfg["per_layout_scale_path"] = ""
        cfg["early_structure_overlay_enabled"] = bool(
            self.project_config.app.get("semantic_cv_early_structure_overlay_enabled", False)
        )
        # A known-scale G run can publish a structure-only route image
        # before the slower player-marker detector finishes. Player/floor
        # constraints remain active for the final result below.
        cfg["early_structure_overlay_ignore_player_priors"] = True
        cfg["early_structure_preview_scale"] = max(
            0.25,
            min(1.0, float(self.project_config.app.get("semantic_cv_early_structure_preview_scale", 1.0))),
        )
        cfg["early_matched_overlay_enabled"] = bool(
            self.project_config.app.get("semantic_cv_early_matched_overlay_enabled", True)
        )
        cfg["early_matched_overlay_min_score"] = float(
            self.project_config.app.get("semantic_cv_early_matched_overlay_min_score", 0.30)
        )
        cfg["early_shortlist_overlay_enabled"] = bool(
            self.project_config.app.get("semantic_cv_early_shortlist_overlay_enabled", True)
        )
        cfg["early_shortlist_overlay_min_score"] = float(
            self.project_config.app.get("semantic_cv_early_shortlist_overlay_min_score", 0.30)
        )
        cfg["early_shortlist_overlay_consensus_count"] = int(
            self.project_config.app.get("semantic_cv_early_shortlist_overlay_consensus_count", 4)
        )
        cfg["early_shortlist_overlay_max_position_span_px"] = float(
            self.project_config.app.get("semantic_cv_early_shortlist_overlay_max_position_span_px", 6.0)
        )
        cfg["early_shortlist_overlay_max_relative_scale_span"] = float(
            self.project_config.app.get("semantic_cv_early_shortlist_overlay_max_relative_scale_span", 0.04)
        )
        cfg["adaptive_scale_parallel_structure_coarse_enabled"] = bool(
            self.project_config.app.get("semantic_cv_adaptive_scale_parallel_structure_coarse_enabled", True)
        )
        cfg["adaptive_scale_staged_fine_enabled"] = bool(
            self.project_config.app.get("semantic_cv_adaptive_scale_staged_fine_enabled", True)
        )
        cfg["adaptive_scale_fine_fast_top_k"] = int(
            self.project_config.app.get("semantic_cv_adaptive_scale_fine_fast_top_k", 8)
        )
        cfg["adaptive_scale_fine_fast_candidate_pool_size"] = int(
            self.project_config.app.get("semantic_cv_adaptive_scale_fine_fast_candidate_pool_size", 16)
        )
        cfg["adaptive_scale_fine_review_max_scales"] = int(
            self.project_config.app.get("semantic_cv_adaptive_scale_fine_review_max_scales", 4)
        )
        cfg["adaptive_scale_fine_review_total_candidates"] = int(
            self.project_config.app.get("semantic_cv_adaptive_scale_fine_review_total_candidates", 48)
        )
        cfg["adaptive_scale_staged_min_score"] = float(
            self.project_config.app.get("semantic_cv_adaptive_scale_staged_min_score", 0.30)
        )
        cfg["adaptive_scale_staged_full_fallback_enabled"] = bool(
            self.project_config.app.get("semantic_cv_adaptive_scale_staged_full_fallback_enabled", True)
        )
        if force_scale_research:
            cfg.pop("fixed_scale", None)
            if estimated_layout_scale_window is not None:
                scale_min, scale_max, scale_step, resolution_multiplier = estimated_layout_scale_window
                search_range_source = "layout_geometry_estimate"
            else:
                if bool(scale_app_config.get("semantic_cv_estimated_scale_ranges_required", False)):
                    raise RuntimeError(f"缺少 {map_id} 的专属 SCALE 搜索范围配置")
                scale_min, scale_max, resolution_multiplier = semantic_cv_unconfigured_scale_window(
                    scale_app_config,
                    roi_size,
                    floor_suffix,
                )
                scale_step = semantic_cv_unconfigured_scale_step(scale_app_config, floor_suffix)
                search_range_source = "floor_fallback"
            cfg["scale_min"] = scale_min
            cfg["scale_max"] = scale_max
            cfg["scale_step"] = max(1e-4, scale_step)
            cfg["adaptive_scale_wide_fallback_enabled"] = False
            cfg["adaptive_scale_refine_enabled"] = adaptive_scale_enabled
            cfg["adaptive_scale_refine_relative_radius"] = float(
                self.project_config.app.get("semantic_cv_initial_scale_fine_relative_radius", 0.025)
            )
            cfg["adaptive_scale_refine_step"] = max(
                1e-4,
                float(self.project_config.app.get("semantic_cv_initial_scale_fine_step", 0.004)),
            )
            cfg["adaptive_scale_coarse_fast_mode_enabled"] = False
            scale_runtime.update(
                {
                    "mode": "manual_overlay_research",
                    "search_range_source": search_range_source,
                    "base_screen_scale_min": float(
                        scale_min / resolution_multiplier
                    ),
                    "base_screen_scale_max": float(
                        scale_max / resolution_multiplier
                    ),
                    "resolution_multiplier": resolution_multiplier,
                    "coarse_scale_min": scale_min,
                    "coarse_scale_max": scale_max,
                    "coarse_scale_step": cfg["scale_step"],
                    "fine_relative_radius": cfg["adaptive_scale_refine_relative_radius"],
                    "fine_scale_step": cfg["adaptive_scale_refine_step"],
                }
            )
        elif cached_scale is not None:
            cfg["fixed_scale"] = float(cached_scale)
            cfg["adaptive_scale_refine_enabled"] = False
            if bool(self.project_config.app.get("semantic_cv_fast_fixed_scale_mode", True)):
                cfg["candidate_pool_size"] = min(int(cfg.get("candidate_pool_size", 48)), 16)
                cfg["top_k"] = min(int(cfg.get("top_k", 8)), 4)
            cfg["early_structure_overlay_enabled"] = bool(
                self.project_config.app.get("semantic_cv_early_structure_overlay_enabled", True)
            )
            scale_runtime.update(
                {
                    "mode": (
                        "reference_roi_normalized_fixed"
                        if reference_roi_normalization is not None
                        else
                        "confirmed_persistent"
                        if persistent_confirmed_scale is not None
                        else
                        "reference_base_fixed"
                        if reference_base_scale_active
                        else "cached"
                    ),
                    "cached_scale": float(cached_scale),
                    "persistently_confirmed": persistent_confirmed_scale is not None,
                }
            )
        elif adaptive_scale_enabled and base_scale is not None:
            learned_multiplier = self._semantic_cv_device_multiplier_cache.get(device_multiplier_key)
            if floor_suffix == "2nd" and learned_multiplier is not None:
                resolution_multiplier = float(learned_multiplier)
                fallback_relative_radius = float(
                    self.project_config.app.get("semantic_cv_second_floor_scale_search_relative_radius", 0.08)
                )
                fast_relative_radius = float(
                    self.project_config.app.get("semantic_cv_second_floor_scale_fast_relative_radius", 0.05)
                )
                multiplier_source = "first_floor_refined"
            else:
                resolution_multiplier = semantic_cv_resolution_scale_multiplier(
                    self.project_config.app,
                    roi_size,
                )
                fallback_relative_radius = float(
                    self.project_config.app.get("semantic_cv_initial_scale_search_relative_radius", 0.15)
                )
                fast_relative_radius = float(
                    self.project_config.app.get("semantic_cv_initial_scale_fast_relative_radius", 0.05)
                )
                multiplier_source = "roi_resolution"
            predicted_scale, scale_min, scale_max = semantic_cv_adaptive_scale_window(
                base_scale,
                resolution_multiplier,
                fast_relative_radius,
            )
            _fallback_predicted, fallback_scale_min, fallback_scale_max = semantic_cv_adaptive_scale_window(
                base_scale,
                resolution_multiplier,
                fallback_relative_radius,
            )
            cfg.pop("fixed_scale", None)
            cfg["scale_min"] = scale_min
            cfg["scale_max"] = scale_max
            cfg["scale_step"] = max(
                1e-4,
                float(self.project_config.app.get("semantic_cv_initial_scale_coarse_step", 0.03)),
            )
            cfg["adaptive_scale_wide_fallback_enabled"] = bool(
                self.project_config.app.get("semantic_cv_adaptive_scale_wide_fallback_enabled", True)
            )
            cfg["adaptive_scale_fallback_scale_min"] = fallback_scale_min
            cfg["adaptive_scale_fallback_scale_max"] = fallback_scale_max
            cfg["adaptive_scale_fallback_scale_step"] = cfg["scale_step"]
            cfg["adaptive_scale_wide_fallback_min_score"] = float(
                self.project_config.app.get("semantic_cv_adaptive_scale_wide_fallback_min_score", 0.30)
            )
            cfg["adaptive_scale_wide_fallback_boundary_steps"] = float(
                self.project_config.app.get("semantic_cv_adaptive_scale_wide_fallback_boundary_steps", 0.75)
            )
            cfg["adaptive_scale_refine_enabled"] = True
            cfg["adaptive_scale_coarse_fast_mode_enabled"] = bool(
                self.project_config.app.get("semantic_cv_adaptive_scale_coarse_fast_mode_enabled", True)
            )
            cfg["adaptive_scale_coarse_fast_top_k"] = int(
                self.project_config.app.get("semantic_cv_adaptive_scale_coarse_fast_top_k", 4)
            )
            cfg["adaptive_scale_coarse_fast_candidate_pool_size"] = int(
                self.project_config.app.get("semantic_cv_adaptive_scale_coarse_fast_candidate_pool_size", 16)
            )
            cfg["adaptive_scale_refine_relative_radius"] = max(
                0.0,
                float(self.project_config.app.get("semantic_cv_initial_scale_fine_relative_radius", 0.025)),
            )
            cfg["adaptive_scale_refine_step"] = max(
                1e-4,
                float(self.project_config.app.get("semantic_cv_initial_scale_fine_step", 0.004)),
            )
            scale_runtime.update(
                {
                    "mode": "adaptive_initial",
                    "resolution_multiplier": resolution_multiplier,
                    "multiplier_source": multiplier_source,
                    "predicted_scale": predicted_scale,
                    "fast_relative_radius": fast_relative_radius,
                    "coarse_scale_min": scale_min,
                    "coarse_scale_max": scale_max,
                    "coarse_scale_step": cfg["scale_step"],
                    "fallback_relative_radius": fallback_relative_radius,
                    "fallback_scale_min": fallback_scale_min,
                    "fallback_scale_max": fallback_scale_max,
                    "fine_relative_radius": cfg["adaptive_scale_refine_relative_radius"],
                    "fine_scale_step": cfg["adaptive_scale_refine_step"],
                }
            )
        elif base_scale is not None:
            cfg["fixed_scale"] = float(base_scale)
            cfg["adaptive_scale_refine_enabled"] = False
            scale_runtime.update({"mode": "legacy_fixed", "fixed_scale": float(base_scale)})
        else:
            cfg.pop("fixed_scale", None)
            if estimated_layout_scale_window is not None:
                scale_min, scale_max, scale_step, resolution_multiplier = estimated_layout_scale_window
                search_range_source = "layout_geometry_estimate"
            else:
                if bool(scale_app_config.get("semantic_cv_estimated_scale_ranges_required", False)):
                    raise RuntimeError(f"缺少 {map_id} 的专属 SCALE 搜索范围配置")
                scale_min, scale_max, resolution_multiplier = semantic_cv_unconfigured_scale_window(
                    scale_app_config,
                    roi_size,
                    floor_suffix,
                )
                scale_step = semantic_cv_unconfigured_scale_step(scale_app_config, floor_suffix)
                search_range_source = "floor_fallback"
            cfg["scale_min"] = scale_min
            cfg["scale_max"] = scale_max
            cfg["scale_step"] = max(1e-4, scale_step)
            cfg["adaptive_scale_refine_enabled"] = adaptive_scale_enabled
            cfg["adaptive_scale_refine_relative_radius"] = float(
                self.project_config.app.get("semantic_cv_initial_scale_fine_relative_radius", 0.025)
            )
            cfg["adaptive_scale_refine_step"] = float(
                self.project_config.app.get("semantic_cv_initial_scale_fine_step", 0.004)
            )
            scale_runtime.update(
                {
                    "mode": "unconfigured_search",
                    "search_range_source": search_range_source,
                    "base_screen_scale_min": float(
                        scale_min / resolution_multiplier
                    ),
                    "base_screen_scale_max": float(
                        scale_max / resolution_multiplier
                    ),
                    "resolution_multiplier": resolution_multiplier,
                    "coarse_scale_min": scale_min,
                    "coarse_scale_max": scale_max,
                    "coarse_scale_step": cfg["scale_step"],
                    "fine_relative_radius": cfg["adaptive_scale_refine_relative_radius"],
                    "fine_scale_step": cfg["adaptive_scale_refine_step"],
                }
            )
        if bool(self.project_config.app.get("semantic_cv_geometric_review_every_location_enabled", True)):
            enforce_semantic_cv_geometric_review(cfg)
            scale_runtime["geometric_review_scope"] = "every_location"
        cfg["scale_runtime"] = scale_runtime
        cfg.update(semantic_cv_weight_overrides_for_map(self.project_config.app, map_id))
        cfg["enable_particle_filter"] = bool(self.project_config.app.get("semantic_cv_enable_particle_filter", True))
        cfg["save_full_debug_images"] = bool(
            self.project_config.app.get("semantic_cv_save_full_debug_images", False)
        )
        cfg["save_debug_images"] = bool(
            self.project_config.app.get("semantic_cv_save_debug_images", False)
        )
        save_input_debug = bool(
            cfg.get("save_debug_images", False)
            or cfg.get("save_full_debug_images", False)
            or self.project_config.app.get("semantic_cv_save_process_logs", False)
            or self.project_config.app.get("semantic_cv_development_roi_capture_enabled", False)
        )
        if save_input_debug:
            debug_roi = alignment_roi.copy()
            debug_config = dict(cfg)

            def save_request_debug() -> None:
                try:
                    ensure_dir(call_dir)
                    if not development_capture_enabled:
                        imwrite_unicode(image_path, debug_roi)
                    config_path.write_text(
                        json.dumps(debug_config, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    LOGGER.debug("failed to save semantic CV request diagnostics", exc_info=True)

            if development_capture_enabled:
                save_request_debug()
            else:
                deferred_diagnostic_writers.append(save_request_debug)

        scale_search_mode = str(scale_runtime.get("mode", ""))
        if scale_search_mode in {"adaptive_initial", "unconfigured_search", "manual_overlay_research"}:
            self._show_semantic_cv_scale_search_progress(map_id, floor_suffix, scale_search_mode)

        worker_started = time.perf_counter()
        early_overlay_runtime: dict[str, Any] = {
            "shown": False,
            "events": [],
            "cached_overlay_restored": False,
            "cached_overlay_restore_ms": None,
            "prior_overlay_hidden_before_capture": prior_overlay_hidden,
        }

        def show_structure_overlay(response: dict[str, Any]) -> None:
            structure_result = response.get("result")
            if not isinstance(structure_result, dict):
                return
            if reference_roi_normalization is not None:
                structure_result["reference_roi_normalization"] = reference_roi_normalization
            phase = str(response.get("phase", "structure")).strip() or "structure"
            received_ms = (time.perf_counter() - worker_started) * 1000.0
            overlay_started = time.perf_counter()
            try:
                shown = self._show_semantic_cv_route_overlay(
                    map_id=map_id,
                    result_data=structure_result,
                    processed_dir=processed_dir,
                    output_path=call_dir / f"route_overlay_early_{phase}.png",
                )
            except Exception:
                shown = False
                LOGGER.exception("failed to render early structure overlay for %s", map_id)
            event = {
                "phase": phase,
                "shown": bool(shown),
                "response_ms": round(received_ms, 3),
                "overlay_render_ms": round((time.perf_counter() - overlay_started) * 1000.0, 3),
                "overlay_visible_ms": round((time.perf_counter() - worker_started) * 1000.0, 3),
                "overlay_show_image_ms": getattr(self.route_overlay, "_last_show_image_ms", None),
            }
            diagnostic_event(
                "aligned_overlay_rendered",
                map_id=map_id,
                phase=phase,
                shown=bool(shown),
                response_ms=event["response_ms"],
                overlay_visible_ms=event["overlay_visible_ms"],
                overlay_screen_roi=self._semantic_cv_last_overlay_screen_roi,
                visible_request_id=self._route_overlay_visible_request_id,
            )
            aligned_observer = getattr(self, "_aligned_overlay_rendered_observer", None)
            if callable(aligned_observer):
                try:
                    aligned_observer(
                        {
                            **event,
                            "map_id": map_id,
                            "visible_request_id": self._route_overlay_visible_request_id,
                            "overlay_screen_roi": self._semantic_cv_last_overlay_screen_roi,
                        }
                    )
                except Exception:
                    LOGGER.exception("aligned-overlay observer failed")
            early_overlay_runtime["events"].append(event)
            early_overlay_runtime["shown"] = bool(early_overlay_runtime.get("shown", False) or shown)
            early_overlay_runtime[f"{phase}_response_ms"] = event["response_ms"]
            early_overlay_runtime[f"{phase}_overlay_visible_ms"] = event["overlay_visible_ms"]
            if shown and "first_overlay_visible_ms" not in early_overlay_runtime:
                early_overlay_runtime["first_overlay_visible_ms"] = event["overlay_visible_ms"]

        worker_response = self._run_semantic_cv_worker(
            localizer_root,
            cfg,
            localization_roi,
            map_id,
            structure_callback=show_structure_overlay,
        )
        self._scale_search_active = False
        self._scale_search_cancel_requested = False
        self._active_scale_search_return_content = None
        self._active_scale_search_map_id = None
        self._active_scale_search_floor = None
        # The aligned overlay is already visible when the persistent worker
        # returns. Save diagnostics now so PNG compression cannot compete with
        # this request's CV work or spill into the next rapid request.
        for writer in deferred_diagnostic_writers:
            try:
                writer()
            except Exception:
                LOGGER.debug("deferred semantic CV diagnostic save failed", exc_info=True)
        deferred_diagnostic_writers.clear()
        direct_worker_result = None
        if worker_response is not None:
            candidate_result = worker_response.get("result")
            if isinstance(candidate_result, dict):
                direct_worker_result = candidate_result
            if bool(self.project_config.app.get("semantic_cv_save_process_logs", False)):
                ensure_dir(call_dir)
                (call_dir / "stdout.txt").write_text(
                    json.dumps(worker_response, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (call_dir / "stderr.txt").write_text("", encoding="utf-8")
        else:
            if bool(
                self.project_config.app.get(
                    "semantic_cv_require_alignment_visible_before_player_detection",
                    True,
                )
            ):
                raise RuntimeError(
                    "CV localizer worker is required to guarantee that player detection starts only after the aligned overlay is visible"
                )
            ensure_dir(call_dir)
            imwrite_unicode(image_path, localization_roi)
            config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            cmd = self._semantic_cv_command(localizer_root, config_path, image_path, map_id)
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            completed = subprocess.run(
                cmd,
                cwd=str(localizer_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(self.project_config.app.get("semantic_cv_timeout_seconds", 30)),
                creationflags=creationflags,
            )
            (call_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (call_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"CV定位进程失败，日志：{call_dir / 'stderr.txt'}")

        direct_result_path = call_dir / "result.json"
        result_files = [] if direct_worker_result is not None else (
            [direct_result_path]
            if direct_result_path.exists()
            else sorted((call_dir / "outputs").glob(f"*_{map_id}/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        )
        if direct_worker_result is None and not result_files:
            raise RuntimeError(f"CV定位未生成 result.json：{call_dir / 'outputs'}")
        result_path = direct_result_path if direct_worker_result is not None else result_files[0]
        if direct_worker_result is not None:
            result_data = direct_worker_result
        else:
            with result_path.open("r", encoding="utf-8-sig") as f:
                result_data = json.load(f)
        if isinstance(result_data, dict) and reference_roi_normalization is not None:
            result_data["reference_roi_normalization"] = reference_roi_normalization
        if bool(self.project_config.app.get("semantic_cv_development_roi_capture_enabled", False)):
            ensure_dir(call_dir)
            scale_audit = {
                "map_id": map_id,
                "floor": floor_suffix,
                "roi_size": [int(w), int(h)],
                "best_scale": result_data.get("best_scale") if isinstance(result_data, dict) else None,
                "confidence": result_data.get("confidence") if isinstance(result_data, dict) else None,
                "top_k_candidates": result_data.get("top_k_candidates") if isinstance(result_data, dict) else None,
                "scale_search": result_data.get("scale_search") if isinstance(result_data, dict) else None,
                "scale_runtime": cfg.get("scale_runtime"),
            }
            (call_dir / "scale_result.json").write_text(
                json.dumps(scale_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        best_scale_value = result_data.get("best_scale") if isinstance(result_data, dict) else None
        top_candidates_for_scale = result_data.get("top_k_candidates") if isinstance(result_data, dict) else None
        best_score_for_scale = None
        if isinstance(top_candidates_for_scale, list) and top_candidates_for_scale:
            candidate = top_candidates_for_scale[0]
            if isinstance(candidate, dict) and isinstance(candidate.get("score"), (int, float)):
                best_score_for_scale = float(candidate["score"])
        if scale_runtime.get("mode") in {"adaptive_initial", "unconfigured_search", "manual_overlay_research"}:
            minimum_cache_score = float(
                self.project_config.app.get("semantic_cv_adaptive_scale_min_cache_score", 0.30)
            )
            cache_accepted = (
                isinstance(best_scale_value, (int, float))
                and float(best_scale_value) > 0
                and best_score_for_scale is not None
                and best_score_for_scale >= minimum_cache_score
            )
            if cache_accepted and not confirmation_required_for_run:
                self._semantic_cv_scale_cache[scale_cache_key] = float(best_scale_value)
                if floor_suffix == "1st" and base_scale is not None and float(base_scale) > 0:
                    learned_multiplier = float(best_scale_value) / float(base_scale)
                    self._semantic_cv_device_multiplier_cache[device_multiplier_key] = learned_multiplier
                    scale_runtime["learned_device_multiplier"] = learned_multiplier
            scale_runtime.update(
                {
                    "selected_scale": float(best_scale_value) if isinstance(best_scale_value, (int, float)) else None,
                    "selection_score": best_score_for_scale,
                    "minimum_cache_score": minimum_cache_score,
                    "cache_accepted": bool(cache_accepted and not confirmation_required_for_run),
                    "awaiting_user_confirmation": bool(cache_accepted and confirmation_required_for_run),
                }
            )
        elif scale_runtime.get("mode") in {
            "cached",
            "confirmed_persistent",
            "reference_base_fixed",
            "reference_roi_normalized_fixed",
        }:
            scale_runtime.update(
                {
                    "selected_scale": float(best_scale_value) if isinstance(best_scale_value, (int, float)) else cached_scale,
                    "selection_score": best_score_for_scale,
                    "cache_accepted": True,
                }
            )
        if (
            confirmation_required_for_run
            and isinstance(best_scale_value, (int, float))
            and float(best_scale_value) > 0
            and np.isfinite(float(best_scale_value))
        ):
            self._semantic_cv_scale_confirmation_pending = {
                "map_id": map_id,
                "floor": floor_suffix,
                "roi_size": [roi_size[0], roi_size[1]],
                "scale": float(best_scale_value),
                "base_scale": base_scale,
                "score": best_score_for_scale,
                "save_scope": "baseline" if force_scale_research and reference_base_scale_active else "confirmed",
                "manual_overlay_research": bool(force_scale_research),
                "manual_research_return_content": (
                    self._manual_scale_research_return_content if force_scale_research else None
                ),
            }
        elif require_scale_confirmation:
            self._semantic_cv_scale_confirmation_pending = None
        scale_search_diagnostics = result_data.get("scale_search") if isinstance(result_data, dict) else None
        if isinstance(scale_search_diagnostics, dict):
            scale_runtime["fast_mode"] = scale_search_diagnostics.get("fast_mode")
            scale_runtime["fast_fallback_performed"] = bool(
                scale_search_diagnostics.get("fast_fallback_performed", False)
            )
        result_data["scale_runtime"] = scale_runtime
        result_data["early_overlay_runtime"] = early_overlay_runtime
        diagnostic_event(
            "semantic_cv_result_received",
            map_id=map_id,
            overlay_shown=bool(early_overlay_runtime.get("shown", False)),
            timings=result_data.get("timings"),
        )
        if first_first_floor_location:
            component_review = result_data.get("semantic_component_review", {})
            constraint_applied = bool(
                isinstance(component_review, dict)
                and component_review.get("structure_constraint_applied", False)
            )
            result_data["first_f1_side_door_match_constraint"] = {
                "requested": bool(first_floor_side_door_allowed_points),
                "enabled": constraint_applied,
                "only_side_doors": bool(self.project_config.app.get("semantic_cv_first_f1_match_only_side_doors", True)),
                "radius": float(self.project_config.app.get("semantic_cv_first_f1_side_door_match_radius", 110.0)),
                "maximum_components": int(
                    self.project_config.app.get(
                        "semantic_cv_first_f1_constraint_semantic_component_max",
                        2,
                    )
                ),
                "total_components": (
                    component_review.get("total_components") if isinstance(component_review, dict) else None
                ),
                "points": first_floor_side_door_allowed_points,
            }
        if first_basement_location:
            component_review = result_data.get("semantic_component_review", {})
            constraint_applied = bool(
                isinstance(component_review, dict)
                and component_review.get("structure_constraint_applied", False)
            )
            result_data["first_b1_tp_match_constraint"] = {
                "requested": bool(first_basement_allowed_points),
                "enabled": constraint_applied,
                "only_bound_transitions_and_tp": bool(
                    self.project_config.app.get("semantic_cv_first_b1_match_only_bound_transitions", True)
                ),
                "radius": float(self.project_config.app.get("semantic_cv_first_b1_match_radius", 110.0)),
                "maximum_components": int(
                    self.project_config.app.get(
                        "semantic_cv_first_b1_constraint_semantic_component_max",
                        2,
                    )
                ),
                "points": first_basement_allowed_points,
                "tp_points": self._semantic_cv_basement_tp_points(map_id),
                "total_components": (
                    component_review.get("total_components") if isinstance(component_review, dict) else None
                ),
            }
        if first_second_floor_location:
            component_review = result_data.get("semantic_component_review", {})
            constraint_applied = bool(
                isinstance(component_review, dict)
                and component_review.get("structure_constraint_applied", False)
            )
            result_data["first_second_floor_match_constraint"] = {
                "enabled": constraint_applied,
                "only_bound_transitions": bool(
                    self.project_config.app.get("semantic_cv_first_second_floor_match_only_bound_transitions", True)
                ),
                "radius": cfg.get("allowed_player_template_radius") if constraint_applied else None,
                "points": first_second_floor_allowed_points,
                "total_components": component_review.get("total_components") if isinstance(component_review, dict) else None,
                "maximum_components": (
                    component_review.get("structure_constraint_maximum_components")
                    if isinstance(component_review, dict)
                    else None
                ),
            }
        overlay_shown = bool(early_overlay_runtime.get("shown", False))
        if not overlay_shown:
            try:
                overlay_shown = self._show_semantic_cv_route_overlay(
                    map_id=map_id,
                    result_data=result_data,
                    processed_dir=processed_dir,
                    output_path=call_dir / "route_overlay_on_screen.png",
                )
            except Exception:
                overlay_shown = False
                LOGGER.exception("failed to render route overlay for %s", map_id)
        if not overlay_shown:
            self.route_overlay.hide()

        debug_paths = result_data.get("debug_paths") if isinstance(result_data, dict) else {}
        candidate_image = None
        candidate_path = None
        if isinstance(debug_paths, dict):
            candidate_path = debug_paths.get("candidates_on_template")
            if candidate_path:
                candidate_image = imread_unicode(Path(candidate_path))
        if candidate_image is None:
            candidate_image = self._semantic_cv_cached_image(processed_dir / map_id / "image.png")
        if candidate_image is None:
            raise RuntimeError(f"CV定位完成但无法读取结果图：{candidate_path}")

        filtered_players: dict[str, tuple[tuple[float, float], str, tuple[int, int, int]]] = {}
        if particle_filter_enabled:
            filtered_players = self._update_semantic_cv_player_filters(
                localizer_root=localizer_root,
                map_id=map_id,
                floor_suffix=floor_suffix,
                result_data=result_data,
            )
        save_rendered_debug = bool(self.project_config.app.get("semantic_cv_save_rendered_debug_images", False))
        if save_rendered_debug:
            (call_dir / "result_augmented.json").write_text(
                json.dumps(result_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self._semantic_cv_pf_motion_accum = (0.0, 0.0)
        self._semantic_cv_pf_motion_radius_accum = 0.0
        self._semantic_cv_pf_motion_radius_max = 0.0
        self._semantic_cv_pf_motion_seconds = 0.0
        if bool(self.project_config.app.get("semantic_cv_async_postprocess_enabled", True)):
            self._start_semantic_cv_postprocessing(
                map_id=map_id,
                floor_suffix=floor_suffix,
                result_data=result_data,
                processed_dir=processed_dir,
                candidate_image=candidate_image,
                filtered_players=filtered_players,
                call_dir=call_dir,
                result_path=result_path,
                save_rendered_debug=save_rendered_debug,
            )
            return
        route_mapping_calibrated = self._has_route_map_coordinate_calibration(map_id)
        route_render = None
        if route_mapping_calibrated:
            route_render = self._render_semantic_cv_route_map(
                map_id=map_id,
                floor_suffix=floor_suffix,
                result_data=result_data,
                processed_dir=processed_dir,
                filtered_players=filtered_players,
                output_path=(call_dir / "route_player_map.png") if save_rendered_debug else None,
            )
        route_only = route_mapping_calibrated and self._is_semantic_cv_route_only_map(map_id)
        clean_template_image = None
        if not route_only:
            clean_template_image = self._render_semantic_cv_clean_template_map(
                map_id=map_id,
                floor_suffix=floor_suffix,
                result_data=result_data,
                processed_dir=processed_dir,
                filtered_players=filtered_players,
                output_path=(call_dir / "clean_template_player_map.png") if save_rendered_debug else None,
            )
        display_image = candidate_image
        route_title = ""
        if route_render is not None:
            route_image, route_title = route_render
            dual_route_image = self._make_lianghapi_dual_route_sheet(
                map_id=map_id,
                floor_suffix=floor_suffix,
                current_route_image=route_image,
            )
            if dual_route_image is not None:
                display_image = dual_route_image
                route_title = f"{base_map_id_from_alignment(map_id)}  双图"
                if save_rendered_debug:
                    imwrite_unicode(call_dir / "dual_route_player_map.png", display_image)
            elif route_only:
                display_image = route_image
                if save_rendered_debug:
                    imwrite_unicode(call_dir / "route_only_player_map.png", display_image)
            else:
                display_image = self._make_semantic_cv_display_sheet(
                    route_image,
                    clean_template_image if clean_template_image is not None else candidate_image,
                    route_title="路线图定位",
                    template_title="定位模板玩家位置",
                )
                if save_rendered_debug:
                    imwrite_unicode(call_dir / "route_and_template_player_map.png", display_image)
        elif clean_template_image is not None:
            display_image = clean_template_image
            if save_rendered_debug:
                imwrite_unicode(call_dir / "clean_template_only_player_map.png", display_image)
        if self._is_nightmare_mode() and not self._uses_nightmare_route_maps():
            nightmare_image = self._nightmare_cheatsheet_image(
                map_id,
                floor_suffix,
                current_image=clean_template_image if clean_template_image is not None else candidate_image,
            )
            if nightmare_image is not None:
                display_image = nightmare_image
                route_title = (
                    f"{base_map_id_from_alignment(map_id)}  三图"
                    if nightmare_three_cheatsheet_enabled(self.project_config.app)
                    else f"{map_id}  单图"
                )

        score = None
        top_candidates = result_data.get("top_k_candidates") if isinstance(result_data, dict) else None
        if isinstance(top_candidates, list) and top_candidates:
            score = top_candidates[0].get("score")
        player_crop = result_data.get("player_position_in_template_crop") if isinstance(result_data, dict) else None
        confidence = result_data.get("confidence") if isinstance(result_data, dict) else None
        self._confirmed_alignment_map_id = self._floor_map_id(map_id, "1st")
        self._pending_alignment_map_id = self._confirmed_alignment_map_id
        self._active_floor_suffix = floor_suffix
        position_text = ""
        if isinstance(player_crop, list) and len(player_crop) >= 2:
            position_text = f" 玩家=({float(player_crop[0]):.1f},{float(player_crop[1]):.1f})"
        if filtered_players:
            position_text += f" PF players={len(filtered_players)}"
        score_text = f" score={float(score):.3f}" if score is not None else ""
        scale_text = ""
        best_scale = result_data.get("best_scale") if isinstance(result_data, dict) else None
        if best_scale is not None:
            scale_text = f" scale={float(best_scale):.3f}"
        conf_text = f" gap={float(confidence):.3f}" if confidence is not None else ""
        self.result_var.set(f"CV定位完成：{map_id}{score_text}{scale_text}{conf_text}")
        self.status_var.set(f"CV定位完成：{map_id}{position_text}；结果：{result_path}")
        title_position = route_title if route_title else f"{map_id}{position_text}"
        title_prefix = "路线图玩家位置" if route_mapping_calibrated else "模板图玩家位置"
        footer = self._route_layout_switch_hint() if self._has_multiple_route_layouts() else ""
        self._show_semantic_cv_location_sheet(
            display_image,
            map_id=map_id,
            title=f"{title_prefix}：{title_position}{score_text}{scale_text}",
            footer=footer,
        )
        self._show_pending_semantic_cv_scale_confirmation(map_id, floor_suffix)

    def _start_semantic_cv_postprocessing(
        self,
        *,
        map_id: str,
        floor_suffix: str,
        result_data: dict,
        processed_dir: Path,
        candidate_image: np.ndarray,
        filtered_players: dict[str, tuple[tuple[float, float], str, tuple[int, int, int]]],
        call_dir: Path,
        result_path: Path,
        save_rendered_debug: bool,
    ) -> None:
        """Render the cheatsheet off the Tk thread after the screen overlay is visible."""
        self._semantic_cv_postprocess_id = int(getattr(self, "_semantic_cv_postprocess_id", 0)) + 1
        postprocess_id = self._semantic_cv_postprocess_id
        diagnostic_event("cheatsheet_postprocess_started", map_id=map_id, postprocess_id=postprocess_id)
        route_mapping_calibrated = self._has_route_map_coordinate_calibration(map_id)
        route_only = route_mapping_calibrated and self._is_semantic_cv_route_only_map(map_id)

        score = None
        top_candidates = result_data.get("top_k_candidates") if isinstance(result_data, dict) else None
        if isinstance(top_candidates, list) and top_candidates:
            score = top_candidates[0].get("score")
        player_crop = result_data.get("player_position_in_template_crop") if isinstance(result_data, dict) else None
        confidence = result_data.get("confidence") if isinstance(result_data, dict) else None
        position_text = ""
        if isinstance(player_crop, list) and len(player_crop) >= 2:
            position_text = f" 玩家=({float(player_crop[0]):.1f},{float(player_crop[1]):.1f})"
        if filtered_players:
            position_text += f" PF players={len(filtered_players)}"
        score_text = f" score={float(score):.3f}" if score is not None else ""
        best_scale = result_data.get("best_scale") if isinstance(result_data, dict) else None
        scale_text = f" scale={float(best_scale):.3f}" if best_scale is not None else ""
        conf_text = f" gap={float(confidence):.3f}" if confidence is not None else ""

        self._confirmed_alignment_map_id = self._floor_map_id(map_id, "1st")
        self._pending_alignment_map_id = self._confirmed_alignment_map_id
        self._active_floor_suffix = floor_suffix
        self.result_var.set(f"CV定位完成：{map_id}{score_text}{scale_text}{conf_text}")
        self.status_var.set(f"CV定位完成：{map_id}{position_text}；虚像已显示，正在生成小抄")

        completed: queue.Queue[tuple[Optional[tuple[np.ndarray, str]], Optional[BaseException]]] = queue.Queue(maxsize=1)

        def render() -> None:
            try:
                diagnostic_event("cheatsheet_render_begin", map_id=map_id, postprocess_id=postprocess_id)
                if bool(self.project_config.app.get("semantic_cv_save_timing_metrics", True)):
                    ensure_dir(call_dir)
                    (call_dir / "pipeline_timing.json").write_text(
                        json.dumps(
                            {
                                "map_id": map_id,
                                "early_overlay_runtime": result_data.get("early_overlay_runtime"),
                                "localizer_timings": result_data.get("timings"),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if save_rendered_debug:
                    (call_dir / "result_augmented.json").write_text(
                        json.dumps(result_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                route_render = None
                if route_mapping_calibrated:
                    route_render = self._render_semantic_cv_route_map(
                        map_id=map_id,
                        floor_suffix=floor_suffix,
                        result_data=result_data,
                        processed_dir=processed_dir,
                        filtered_players=filtered_players,
                        output_path=(call_dir / "route_player_map.png") if save_rendered_debug else None,
                    )
                clean_template_image = None
                if not route_only:
                    clean_template_image = self._render_semantic_cv_clean_template_map(
                        map_id=map_id,
                        floor_suffix=floor_suffix,
                        result_data=result_data,
                        processed_dir=processed_dir,
                        filtered_players=filtered_players,
                        output_path=(call_dir / "clean_template_player_map.png") if save_rendered_debug else None,
                    )
                display_image = candidate_image
                route_title = ""
                if route_render is not None:
                    route_image, route_title = route_render
                    dual_route_image = self._make_lianghapi_dual_route_sheet(
                        map_id=map_id,
                        floor_suffix=floor_suffix,
                        current_route_image=route_image,
                    )
                    if dual_route_image is not None:
                        display_image = dual_route_image
                        route_title = f"{base_map_id_from_alignment(map_id)}  双图"
                        if save_rendered_debug:
                            imwrite_unicode(call_dir / "dual_route_player_map.png", display_image)
                    elif route_only:
                        display_image = route_image
                        if save_rendered_debug:
                            imwrite_unicode(call_dir / "route_only_player_map.png", display_image)
                    else:
                        display_image = self._make_semantic_cv_display_sheet(
                            route_image,
                            clean_template_image if clean_template_image is not None else candidate_image,
                            route_title="路线图定位",
                            template_title="定位模板玩家位置",
                        )
                        if save_rendered_debug:
                            imwrite_unicode(call_dir / "route_and_template_player_map.png", display_image)
                elif clean_template_image is not None:
                    display_image = clean_template_image
                    if save_rendered_debug:
                        imwrite_unicode(call_dir / "clean_template_only_player_map.png", display_image)
                if self._is_nightmare_mode() and not self._uses_nightmare_route_maps():
                    nightmare_image = self._nightmare_cheatsheet_image(
                        map_id,
                        floor_suffix,
                        current_image=clean_template_image if clean_template_image is not None else candidate_image,
                    )
                    if nightmare_image is not None:
                        display_image = nightmare_image
                        route_title = (
                            f"{base_map_id_from_alignment(map_id)}  三图"
                            if nightmare_three_cheatsheet_enabled(self.project_config.app)
                            else f"{map_id}  单图"
                        )
                completed.put(((display_image, route_title), None))
                diagnostic_event(
                    "cheatsheet_render_complete",
                    map_id=map_id,
                    postprocess_id=postprocess_id,
                    image_shape=list(display_image.shape),
                )
            except BaseException as exc:
                diagnostic_event(
                    "cheatsheet_render_failed",
                    map_id=map_id,
                    postprocess_id=postprocess_id,
                    exception_type=type(exc).__name__,
                    exception=repr(exc),
                )
                completed.put((None, exc))

        threading.Thread(
            target=render,
            name=f"semantic-cv-postprocess-{postprocess_id}",
            daemon=True,
        ).start()

        def poll() -> None:
            if postprocess_id != getattr(self, "_semantic_cv_postprocess_id", 0):
                return
            try:
                rendered, error = completed.get_nowait()
            except queue.Empty:
                self.after(10, poll)
                return
            if error is not None:
                LOGGER.error(
                    "semantic CV cheatsheet postprocessing failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
                rendered = (candidate_image, "")
            if rendered is None:
                return
            display_image, route_title = rendered
            title_position = route_title if route_title else f"{map_id}{position_text}"
            title_prefix = "路线图玩家位置" if route_mapping_calibrated else "模板图玩家位置"
            footer = self._route_layout_switch_hint() if self._has_multiple_route_layouts() else ""
            self._show_semantic_cv_location_sheet(
                display_image,
                map_id=map_id,
                title=f"{title_prefix}：{title_position}{score_text}{scale_text}",
                footer=footer,
            )
            diagnostic_event("cheatsheet_presented", map_id=map_id, postprocess_id=postprocess_id)
            self.status_var.set(f"CV定位完成：{map_id}{position_text}；结果：{result_path}")

            self._show_pending_semantic_cv_scale_confirmation(map_id, floor_suffix)

        self.after(10, poll)

    def _capture_and_run_structure_registration(self) -> None:
        if not self._pending_alignment_map_id and self._confirmed_alignment_map_id is None:
            raise RuntimeError("缺少 CNN 已确认地图编号")
        map_id = self._pending_alignment_map_id or self._confirmed_alignment_map_id
        if map_id is None:
            raise RuntimeError("缺少 CNN 已确认地图编号")
        _capture, alignment_roi = self._capture_alignment_roi("结构配准截图成功")
        app = self.project_config.app
        templates_dir = resolve_config_path(
            self.project_config.root_dir,
            str(app.get("structure_registration_templates_dir", r"C:\Users\LD\Desktop\摸金工具\地图合集\一楼")),
        )
        template_path = resolve_template_path(templates_dir, map_id)
        scales = parse_scales(app.get("structure_registration_scales"), [0.60, 0.63, 0.65, 0.67, 0.70, 0.75])
        result = align_structure_to_known(
            map_id=map_id,
            unknown_image=alignment_roi,
            template_path=template_path,
            scales=scales,
            enable_affine_refine=bool(app.get("structure_registration_enable_affine_refine", True)),
        )
        self.last_structure_result = result
        self._confirmed_alignment_map_id = map_id
        self._pending_alignment_map_id = map_id
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug" / "structure_registration")
        ts = timestamp()
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_raw_roi.png", alignment_roi)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_overlay.png", result.overlay)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_unknown_edges.png", result.unknown_edges)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_template_edges.png", result.template_edges)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_unknown_skel.png", result.unknown_skeleton)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_template_skel.png", result.template_skeleton)
        transform = result.transform
        result_json = {
            "map_id": map_id,
            "score": transform.score,
            "is_ambiguous": transform.is_ambiguous,
            "second_score": transform.second_score,
            "scale": transform.scale,
            "top_left": list(transform.top_left),
            "template_crop_offset": list(transform.template_crop_offset),
            "unknown_crop_offset": list(transform.unknown_crop_offset),
            "affine": transform.affine,
            "alignment_screen_capture_roi": app.get(str(app.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))),
            "alignment_map_roi": app.get(str(app.get("alignment_map_roi_key", "alignment_map_roi"))),
            "structure_registration_scales": scales,
            "candidates": [
                {
                    "score": candidate.score,
                    "scale": candidate.scale,
                    "top_left": list(candidate.top_left),
                    "coarse_score": candidate.coarse_score,
                    "color_score": candidate.color_score,
                    "edge_score": candidate.edge_score,
                    "skeleton_score": candidate.skeleton_score,
                }
                for candidate in result.candidates[:8]
            ],
        }
        (debug_dir / f"{ts}_{map_id}_result.json").write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
        ambiguous_text = "（候选接近，建议多探索后重试）" if transform.is_ambiguous else ""
        self.result_var.set(f"结构配准完成：{map_id} score={transform.score:.3f} scale={transform.scale:.3f}{ambiguous_text}")
        LOGGER.info(
            "structure registration completed map=%s score=%.4f scale=%.4f top_left=(%.1f, %.1f)",
            map_id,
            transform.score,
            transform.scale,
            transform.top_left[0],
            transform.top_left[1],
        )
        self._render_structure_player_location(result, alignment_roi, debug_dir, timestamp_value=ts)

    def _capture_and_run_second_floor_structure_registration(self) -> None:
        map_id = self._second_floor_map_id()
        _capture, alignment_roi = self._capture_alignment_roi("二楼拓扑结构配准截图成功")
        app = self.project_config.app
        templates_dir = resolve_config_path(
            self.project_config.root_dir,
            str(
                app.get(
                    "second_floor_structure_templates_dir",
                    app.get("second_floor_template_maps_dir", r"C:\Users\LD\Desktop\摸金工具\地图合集\二楼"),
                )
            ),
        )
        template_path = resolve_template_path(templates_dir, map_id)
        scales = parse_scales(app.get("second_floor_structure_registration_scales", app.get("structure_registration_scales")), [0.60, 0.63, 0.65, 0.67, 0.70, 0.75])
        result = align_structure_to_known(
            map_id=map_id,
            unknown_image=alignment_roi,
            template_path=template_path,
            scales=scales,
            enable_affine_refine=bool(app.get("second_floor_structure_registration_enable_affine_refine", app.get("structure_registration_enable_affine_refine", True))),
        )
        self.last_structure_result = result
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug" / "second_floor_structure_registration")
        ts = timestamp()
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_raw_roi.png", alignment_roi)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_overlay.png", result.overlay)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_unknown_edges.png", result.unknown_edges)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_template_edges.png", result.template_edges)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_unknown_skel.png", result.unknown_skeleton)
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_template_skel.png", result.template_skeleton)
        result_json = {
            "map_id": map_id,
            "floor": "2nd",
            "score": result.transform.score,
            "is_ambiguous": result.transform.is_ambiguous,
            "second_score": result.transform.second_score,
            "scale": result.transform.scale,
            "top_left": list(result.transform.top_left),
            "template_crop_offset": list(result.transform.template_crop_offset),
            "unknown_crop_offset": list(result.transform.unknown_crop_offset),
            "affine": result.transform.affine,
            "template_path": str(template_path),
            "alignment_screen_capture_roi": app.get(str(app.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))),
            "alignment_map_roi": app.get(str(app.get("alignment_map_roi_key", "alignment_map_roi"))),
            "structure_registration_scales": scales,
            "candidates": [
                {
                    "score": candidate.score,
                    "scale": candidate.scale,
                    "top_left": list(candidate.top_left),
                    "coarse_score": candidate.coarse_score,
                    "color_score": candidate.color_score,
                    "edge_score": candidate.edge_score,
                    "skeleton_score": candidate.skeleton_score,
                }
                for candidate in result.candidates[:8]
            ],
        }
        (debug_dir / f"{ts}_{map_id}_result.json").write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
        self.result_var.set(
            f"二楼拓扑定位完成：{map_id} score={result.transform.score:.3f} "
            f"scale={result.transform.scale:.3f}"
            f"{'；候选接近，建议多探索后重试' if result.transform.is_ambiguous else ''}"
        )
        LOGGER.info(
            "second-floor structure registration completed map=%s score=%.4f scale=%.4f top_left=(%.1f, %.1f)",
            map_id,
            result.transform.score,
            result.transform.scale,
            result.transform.top_left[0],
            result.transform.top_left[1],
        )
        self._render_structure_player_location(result, alignment_roi, debug_dir, timestamp_value=ts)

    def _load_roi_affine_matrix(
        self,
        *,
        map_id: Optional[str] = None,
        config_key: str = "roi_to_template_transform_path",
        default_path: str = "config/roi_to_template_transform.json",
        fallback_map_id: Optional[str] = None,
        return_key: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, str]:
        app = self.project_config.app
        transform_path = resolve_config_path(
            self.project_config.root_dir,
            str(app.get(config_key, default_path)),
        )
        with transform_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        item = None
        used_key = map_id or "global"
        if isinstance(data, dict) and map_id:
            transforms = data.get("transforms")
            if isinstance(transforms, dict):
                item = transforms.get(map_id)
            if not isinstance(item, dict):
                item = data.get(map_id)
            if not isinstance(item, dict) and fallback_map_id:
                if isinstance(transforms, dict):
                    item = transforms.get(fallback_map_id)
                if not isinstance(item, dict):
                    item = data.get(fallback_map_id)
                if isinstance(item, dict):
                    used_key = fallback_map_id
        if not isinstance(item, dict):
            item = data.get("global") if isinstance(data, dict) and map_id is None else None
            if isinstance(item, dict):
                used_key = "global"
        if not isinstance(item, dict):
            key_text = map_id if map_id else "global"
            raise RuntimeError(f"仿射配置缺少 {key_text}：{transform_path}")
        matrix = np.array(item.get("matrix"), dtype=np.float32)
        if matrix.shape != (2, 3):
            raise RuntimeError(f"全局仿射矩阵格式无效：{transform_path}")
        if return_key:
            return matrix, used_key
        return matrix

    def _capture_and_run_roi_affine_location(self) -> None:
        if not self._pending_alignment_map_id and self._confirmed_alignment_map_id is None:
            raise RuntimeError("缺少 CNN 已确认地图编号")
        map_id = self._pending_alignment_map_id or self._confirmed_alignment_map_id
        if map_id is None:
            raise RuntimeError("缺少 CNN 已确认地图编号")
        _capture, alignment_roi = self._capture_alignment_roi("全局仿射定位截图成功")
        matrix = self._load_roi_affine_matrix()
        templates_dir = resolve_config_path(
            self.project_config.root_dir,
            str(self.project_config.app.get("structure_registration_templates_dir", r"C:\Users\LD\Desktop\摸金工具\地图合集\一楼")),
        )
        template_path = resolve_template_path(templates_dir, map_id)
        template_image = imread_unicode(template_path)
        if template_image is None:
            raise RuntimeError(f"无法读取模板地图：{template_path}")
        self._confirmed_alignment_map_id = map_id
        self._pending_alignment_map_id = map_id
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug" / "roi_affine_location")
        ts = timestamp()
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_raw_roi.png", alignment_roi)
        self._render_roi_affine_player_location(
            map_id=map_id,
            template_image=template_image,
            alignment_roi=alignment_roi,
            matrix=matrix,
            debug_dir=debug_dir,
            timestamp_value=ts,
        )

    def _capture_and_run_second_floor_roi_affine_location(self) -> None:
        map_id = self._second_floor_map_id()
        _capture, alignment_roi = self._capture_alignment_roi("二楼仿射定位截图成功")
        fallback_map_id = str(self.project_config.app.get("second_floor_roi_affine_fallback_map_id", "map_17_2nd")).strip() or None
        matrix, used_key = self._load_roi_affine_matrix(
            map_id=map_id,
            config_key="second_floor_roi_to_template_transform_path",
            default_path="config/second_floor_roi_to_template_transform.json",
            fallback_map_id=fallback_map_id,
            return_key=True,
        )
        template_path = resolve_floor_template_path(self.project_config.app, map_id, "2nd")
        if template_path is None:
            raise RuntimeError(f"无法找到二楼模板地图：{map_id}")
        template_image = imread_unicode(template_path)
        if template_image is None:
            raise RuntimeError(f"无法读取二楼模板地图：{template_path}")
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug" / "second_floor_roi_affine_location")
        ts = timestamp()
        imwrite_unicode(debug_dir / f"{ts}_{map_id}_raw_roi.png", alignment_roi)
        self._render_roi_affine_player_location(
            map_id=map_id,
            template_image=template_image,
            alignment_roi=alignment_roi,
            matrix=matrix,
            debug_dir=debug_dir,
            timestamp_value=ts,
        )
        if used_key != map_id:
            self.status_var.set(f"二楼仿射定位完成：{map_id}，使用 fallback 矩阵 {used_key}")

    def _render_player_location(
        self,
        result,
        alignment_roi: np.ndarray,
        debug_dir,
        *,
        timestamp_value: Optional[str] = None,
        fallback_image: Optional[np.ndarray] = None,
    ) -> None:
        ts = timestamp_value or timestamp()
        player_point = self._locate_player_marker(alignment_roi)
        display_map_id, base_map = self._active_template_image(result.transform.map_id, result.full_map)
        player_map = base_map.copy()
        player_text = ""
        marker_debug = alignment_roi.copy()
        for idx, (area, cx, cy, rect) in enumerate(self._last_player_marker_candidates[:12], 1):
            x, y, w, h = rect
            color = (0, 255, 0) if idx == 1 else (0, 200, 255)
            cv2.rectangle(marker_debug, (x, y), (x + w, y + h), color, 2)
            cv2.circle(marker_debug, (int(round(cx)), int(round(cy))), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                marker_debug,
                f"{idx}:{area:.0f}",
                (x, max(18, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
        )
        if player_point is not None:
            px, py = player_point
            tx = result.transform.top_left[0] + px * result.transform.scale
            ty = result.transform.top_left[1] + py * result.transform.scale
            cv2.circle(player_map, (int(round(tx)), int(round(ty))), 12, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(player_map, (int(round(tx)), int(round(ty))), 18, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(player_map, "player", (int(round(tx)) + 16, int(round(ty)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            player_text = f"玩家 {tx:.1f},{ty:.1f}；截图 {px:.1f},{py:.1f}"
        player_path = debug_dir / f"{ts}_{result.transform.map_id}_player.png"
        marker_debug_path = debug_dir / f"{ts}_{result.transform.map_id}_player_candidates.png"
        mask_path = debug_dir / f"{ts}_{result.transform.map_id}_player_mask.png"
        imwrite_unicode(player_path, player_map)
        imwrite_unicode(marker_debug_path, marker_debug)
        if self._last_player_marker_mask is not None:
            imwrite_unicode(mask_path, self._last_player_marker_mask)
        if player_point is not None:
            LOGGER.info(
                "player located map=%s screenshot_point=(%.1f, %.1f) template_point=(%.1f, %.1f) "
                "candidates=%d player_image=%s candidates_image=%s mask=%s",
                result.transform.map_id,
                player_point[0],
                player_point[1],
                tx,
                ty,
                len(self._last_player_marker_candidates),
                player_path,
                marker_debug_path,
                mask_path,
            )
            self.status_var.set(f"玩家定位完成：{player_path}；{player_text}")
            self.cheatsheet.show_image(player_map, title=f"玩家位置：{display_map_id}  {player_text}")
        else:
            LOGGER.info(
                "player marker not found map=%s candidates=%d candidates_image=%s mask=%s",
                result.transform.map_id,
                len(self._last_player_marker_candidates),
                marker_debug_path,
                mask_path,
            )
            self.status_var.set(f"未识别到玩家黄色标识；候选数 {len(self._last_player_marker_candidates)}；调试图：{marker_debug_path}")
            self.cheatsheet.show_image(fallback_image if fallback_image is not None else player_map, title=f"未识别到玩家：{display_map_id}")

    def _render_structure_player_location(
        self,
        result: StructureRegistrationResult,
        alignment_roi: np.ndarray,
        debug_dir,
        *,
        timestamp_value: Optional[str] = None,
    ) -> None:
        ts = timestamp_value or timestamp()
        player_point = self._locate_player_marker(alignment_roi)
        player_map = result.full_map.copy()
        marker_debug = alignment_roi.copy()
        for idx, (area, cx, cy, rect) in enumerate(self._last_player_marker_candidates[:12], 1):
            x, y, w, h = rect
            color = (0, 255, 0) if idx == 1 else (0, 200, 255)
            cv2.rectangle(marker_debug, (x, y), (x + w, y + h), color, 2)
            cv2.circle(marker_debug, (int(round(cx)), int(round(cy))), 5, color, -1, cv2.LINE_AA)
            cv2.putText(marker_debug, f"{idx}:{area:.0f}", (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        player_text = ""
        if player_point is not None:
            tx, ty = result.transform.map_unknown_point(player_point)
            cv2.circle(player_map, (int(round(tx)), int(round(ty))), 12, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(player_map, (int(round(tx)), int(round(ty))), 18, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(player_map, "player", (int(round(tx)) + 16, int(round(ty)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            player_text = f"玩家 {tx:.1f},{ty:.1f}；截图 {player_point[0]:.1f},{player_point[1]:.1f}"
        player_path = debug_dir / f"{ts}_{result.transform.map_id}_player.png"
        marker_debug_path = debug_dir / f"{ts}_{result.transform.map_id}_player_candidates.png"
        mask_path = debug_dir / f"{ts}_{result.transform.map_id}_player_mask.png"
        imwrite_unicode(player_path, player_map)
        imwrite_unicode(marker_debug_path, marker_debug)
        if self._last_player_marker_mask is not None:
            imwrite_unicode(mask_path, self._last_player_marker_mask)
        if player_point is not None:
            self.status_var.set(
                f"结构配准玩家定位完成：score={result.transform.score:.3f} "
                f"scale={result.transform.scale:.3f}"
                f"{'；候选接近，建议多探索后重试' if result.transform.is_ambiguous else ''}；{player_text}"
            )
            self.cheatsheet.show_image(player_map, title=f"结构配准玩家位置：{result.transform.map_id}  {player_text}")
        else:
            self.status_var.set(f"结构配准完成但未识别到玩家黄色标识；候选数 {len(self._last_player_marker_candidates)}")
            self.cheatsheet.show_image(result.overlay, title=f"结构配准完成，未识别到玩家：{result.transform.map_id}")

    def _render_roi_affine_player_location(
        self,
        *,
        map_id: str,
        template_image: np.ndarray,
        alignment_roi: np.ndarray,
        matrix: np.ndarray,
        debug_dir,
        timestamp_value: Optional[str] = None,
    ) -> None:
        ts = timestamp_value or timestamp()
        player_point = self._locate_player_marker(alignment_roi)
        player_map = template_image.copy()
        marker_debug = alignment_roi.copy()
        for idx, (area, cx, cy, rect) in enumerate(self._last_player_marker_candidates[:12], 1):
            x, y, w, h = rect
            color = (0, 255, 0) if idx == 1 else (0, 200, 255)
            cv2.rectangle(marker_debug, (x, y), (x + w, y + h), color, 2)
            cv2.circle(marker_debug, (int(round(cx)), int(round(cy))), 5, color, -1, cv2.LINE_AA)
            cv2.putText(marker_debug, f"{idx}:{area:.0f}", (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        result_data = {
            "map_id": map_id,
            "method": "roi_affine",
            "matrix": matrix.tolist(),
            "player_roi": None,
            "player_template": None,
            "candidate_count": len(self._last_player_marker_candidates),
        }
        player_text = ""
        if player_point is not None:
            src = np.array([[player_point]], dtype=np.float32)
            mapped = cv2.transform(src, matrix)[0, 0]
            tx, ty = float(mapped[0]), float(mapped[1])
            cv2.circle(player_map, (int(round(tx)), int(round(ty))), 12, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(player_map, (int(round(tx)), int(round(ty))), 18, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(player_map, "player", (int(round(tx)) + 16, int(round(ty)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            player_text = f"玩家 {tx:.1f},{ty:.1f}；ROI {player_point[0]:.1f},{player_point[1]:.1f}"
            result_data["player_roi"] = [float(player_point[0]), float(player_point[1])]
            result_data["player_template"] = [tx, ty]
        player_path = debug_dir / f"{ts}_{map_id}_player.png"
        marker_debug_path = debug_dir / f"{ts}_{map_id}_player_candidates.png"
        mask_path = debug_dir / f"{ts}_{map_id}_player_mask.png"
        result_path = debug_dir / f"{ts}_{map_id}_result.json"
        imwrite_unicode(player_path, player_map)
        imwrite_unicode(marker_debug_path, marker_debug)
        if self._last_player_marker_mask is not None:
            imwrite_unicode(mask_path, self._last_player_marker_mask)
        result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
        if player_point is not None:
            self.result_var.set(f"全局仿射定位完成：{map_id}")
            self.status_var.set(f"全局仿射玩家定位完成：{player_text}")
            self.cheatsheet.show_image(player_map, title=f"全局仿射玩家位置：{map_id}  {player_text}")
        else:
            self.result_var.set("全局仿射定位完成，但未识别到玩家")
            self.status_var.set(f"未识别到玩家黄色标识；候选数 {len(self._last_player_marker_candidates)}")
            self.cheatsheet.show_image(player_map, title=f"未识别到玩家：{map_id}")

    def _locate_player_marker(self, image: np.ndarray) -> Optional[tuple[float, float]]:
        app = self.project_config.app
        hsv_min = app.get("player_marker_hsv_min", [15, 80, 80])
        hsv_max = app.get("player_marker_hsv_max", [45, 255, 255])
        min_area = float(app.get("player_marker_min_area", 40))
        max_area = float(app.get("player_marker_max_area", 400))
        morph = int(app.get("player_marker_morph", 1))
        min_saturation = float(app.get("player_marker_preferred_saturation", 125))
        min_value = float(app.get("player_marker_preferred_value", 180))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_min, dtype=np.uint8), np.array(hsv_max, dtype=np.uint8))
        if morph > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph + 1, 2 * morph + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        self._last_player_marker_mask = mask.copy()
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[tuple[float, float, float, float]] = None
        candidates: list[tuple[float, float, float, tuple[int, int, int, int]]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            ratio = w / h
            if ratio < 0.25 or ratio > 4.0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                cx, cy = x + w / 2.0, y + h / 2.0
            else:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            perimeter = float(cv2.arcLength(contour, True))
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-6))
            extent = float(area / max(1, w * h))
            candidate_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(candidate_mask, [contour], -1, 255, -1)
            hsv_values = hsv[candidate_mask > 0]
            if hsv_values.size:
                mean_saturation = float(np.mean(hsv_values[:, 1]))
                mean_value = float(np.mean(hsv_values[:, 2]))
            else:
                mean_saturation = 0.0
                mean_value = 0.0
            ratio_score = float(np.exp(-abs(np.log(max(1e-6, ratio))) * 1.7))
            circular_score = max(0.0, min(1.0, circularity / 0.72))
            extent_score = max(0.0, min(1.0, extent / 0.58))
            saturation_score = max(0.0, min(1.0, mean_saturation / max(1.0, min_saturation)))
            value_score = max(0.0, min(1.0, mean_value / max(1.0, min_value)))
            area_mid = (min_area + max_area) * 0.5
            area_score = float(np.exp(-abs(area - area_mid) / max(1.0, area_mid)))
            score = (
                0.30 * ratio_score
                + 0.25 * circular_score
                + 0.15 * extent_score
                + 0.15 * saturation_score
                + 0.10 * value_score
                + 0.05 * area_score
            )
            candidates.append((area, float(cx), float(cy), (int(x), int(y), int(w), int(h))))
            if best is None or score > best[0]:
                best = (score, area, float(cx), float(cy))
        candidates.sort(
            key=lambda item: (
                0.30 * float(np.exp(-abs(np.log(max(1e-6, item[3][2] / max(1, item[3][3])))) * 1.7))
                + 0.05 * item[0] / max(1.0, max_area),
                item[0],
            ),
            reverse=True,
        )
        self._last_player_marker_candidates = candidates
        if best is None:
            return None
        return best[2], best[3]

    def _format_log(self, capture: CaptureInfo, summary: MatchSummary) -> str:
        lines = [
            f"当前截图区域：{capture.roi}",
            f"截图模式：{capture.mode}",
            f"当前特征 ROI：{roi_from_app(self.project_config.app, 'current_map_feature_roi')}",
            f"完整地图模板 ROI 字段：{self.project_config.app.get('template_roi_key', 'door_1_roi')}",
            f"结构匹配：{'启用' if self.project_config.app.get('enable_structure_match', True) else '关闭'}；门锚点：{'启用' if self.project_config.app.get('enable_door_anchor', True) else '关闭'}",
            f"自动紧裁：{'启用' if self.project_config.app.get('enable_auto_crop_visible_structure', True) else '关闭'}；bbox={summary.current_crop_bbox}",
            f"CNN：{'启用' if self.cnn_classifier is not None else '关闭'}",
            f"权重：color={self.project_config.app.get('color_weight', 0)}, template={self.project_config.app.get('template_weight', 0)}, feature={self.project_config.app.get('feature_weight', 0)}",
            "",
        ]
        if self.last_cnn_scores:
            lines.append("CNN Top:")
            for idx, (map_id, score) in enumerate(sorted(self.last_cnn_scores.items(), key=lambda item: item[1], reverse=True)[:5], 1):
                lines.append(f"{idx}. {map_id} cnn={score:.4f}")
            lines.append("")
        for r in summary.results:
            lines.append(
                f"{r.map_id} candidate_score={r.candidate_score:.4f}, "
                f"cnn_score={r.cnn_score:.4f}, cnn_rank={r.cnn_rank}, "
                f"source={r.source}, sample_id={r.sample_id}, support={r.support_count}/{r.sample_count}, "
                f"color_score={r.color_score:.4f}, "
                f"ssim_score={r.ssim_score:.4f}, "
                f"deep_score={r.deep_score:.4f}, deep_backend={r.deep_backend}, "
                f"roi_version={r.roi_version}, "
                f"anchored_subset_score={r.anchored_subset_score:.4f}, "
                f"entrance_corridor_pair_score={r.entrance_corridor_pair_score:.4f}, "
                f"entrance_direction_score={r.entrance_direction_score:.4f}, "
                f"entrance_shape_score={r.entrance_shape_score:.4f}, "
                f"entrance_patch_score={r.entrance_patch_score:.4f}, "
                f"corridor_component_score={r.corridor_component_score:.4f}, "
                f"partial_structure_score={r.partial_structure_score:.4f}, "
                f"entry_corridor_score={r.entry_corridor_score:.4f}, "
                f"structure_score={r.structure_score:.4f}, "
                f"door_anchor_score={r.door_anchor_score:.4f}, "
                f"template_score={r.template_score:.4f}, feature_score={r.feature_score:.4f}, "
                f"final_score={r.final_score:.4f}, good={r.good_matches}, inliers={r.inliers}, "
                f"inlier_ratio={r.inlier_ratio:.3f}, scale={r.best_scale:.2f}"
            )
        lines.extend(["", "Top 5:"])
        for i, r in enumerate(summary.top3, 1):
            lines.append(f"{i}. {r.map_id} {r.map_name} final={r.final_score:.4f}")
        lines.extend(["", f"最终判断：{summary.best.map_id if summary.best else '无'}", f"是否可靠：{summary.is_reliable}", summary.status])
        return "\n".join(lines)

    def clear_recognition_cache(self) -> None:
        """Remove user-generated recognition and localization debug outputs on demand."""
        if self._debug_cleanup_running:
            self.status_var.set("识别缓存正在清理，请稍候")
            return

        debug_root = self.project_config.root_dir / "assets" / "debug"
        if not debug_root.exists() or not debug_root.is_dir():
            self.status_var.set("识别缓存为空，无需清理")
            return

        self._debug_cleanup_running = True
        deleted_files = 0
        deleted_bytes = 0
        failed_files = 0
        try:
            for path in debug_root.rglob("*"):
                if not path.is_file() or path.name == ".gitkeep":
                    continue
                if "runtime_diagnostics" in path.parts:
                    continue
                try:
                    size = path.stat().st_size
                    path.unlink()
                    deleted_files += 1
                    deleted_bytes += int(size)
                except OSError:
                    failed_files += 1

            for directory in sorted(
                (path for path in debug_root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        finally:
            self._debug_cleanup_running = False

        released_mb = deleted_bytes / (1024 * 1024)
        message = f"已清理 {deleted_files} 个识别缓存文件，释放 {released_mb:.2f} MB"
        if failed_files:
            message += f"；{failed_files} 个文件正在使用，未清理"
        self.status_var.set(message)
        messagebox.showinfo("识别缓存清理", message)

    def _debug_cleanup_interval_ms(self) -> int:
        minutes = float(self.project_config.app.get("debug_auto_cleanup_interval_minutes", 10.0))
        return max(60_000, int(minutes * 60_000))

    def _run_debug_auto_cleanup(self) -> None:
        try:
            if bool(self.project_config.app.get("debug_auto_cleanup_enabled", True)) and not self._debug_cleanup_running:
                self._debug_cleanup_running = True
                try:
                    deleted_files, deleted_bytes = self._cleanup_debug_outputs()
                    if deleted_files:
                        LOGGER.info("auto-cleaned debug outputs: %s files, %.2f MB", deleted_files, deleted_bytes / (1024 * 1024))
                finally:
                    self._debug_cleanup_running = False
        except Exception:
            LOGGER.exception("debug auto-cleanup failed")
        finally:
            self.after(self._debug_cleanup_interval_ms(), self._run_debug_auto_cleanup)

    def _cleanup_debug_outputs(self) -> tuple[int, int]:
        debug_root = self.project_config.root_dir / "assets" / "debug"
        if not debug_root.exists() or not debug_root.is_dir():
            return 0, 0

        now = time.time()
        max_age_hours = float(self.project_config.app.get("debug_auto_cleanup_max_age_hours", 12.0))
        max_size_mb = float(self.project_config.app.get("debug_auto_cleanup_max_size_mb", 300.0))
        min_file_age_minutes = float(self.project_config.app.get("debug_auto_cleanup_min_file_age_minutes", 10.0))
        min_file_age_seconds = max(0.0, min_file_age_minutes * 60.0)
        cutoff = now - max(0.0, max_age_hours) * 3600.0

        files = [
            path
            for path in debug_root.rglob("*")
            if path.is_file() and "runtime_diagnostics" not in path.parts
        ]
        deleted_files = 0
        deleted_bytes = 0

        def can_delete(path: Path) -> bool:
            try:
                return now - path.stat().st_mtime >= min_file_age_seconds
            except OSError:
                return False

        def delete_file(path: Path) -> None:
            nonlocal deleted_files, deleted_bytes
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                return
            deleted_files += 1
            deleted_bytes += int(size)

        if max_age_hours > 0:
            for path in files:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff and can_delete(path):
                    delete_file(path)

        files = [
            path
            for path in debug_root.rglob("*")
            if path.is_file() and "runtime_diagnostics" not in path.parts
        ]
        if max_size_mb > 0:
            max_bytes = int(max_size_mb * 1024 * 1024)
            file_infos: list[tuple[float, int, Path]] = []
            total_bytes = 0
            for path in files:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total_bytes += int(stat.st_size)
                file_infos.append((float(stat.st_mtime), int(stat.st_size), path))
            for _mtime, size, path in sorted(file_infos):
                if total_bytes <= max_bytes:
                    break
                if not path.exists() or not can_delete(path):
                    continue
                delete_file(path)
                total_bytes -= size

        for directory in sorted((path for path in debug_root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        return deleted_files, deleted_bytes

    def save_debug_images(self, silent: bool = False) -> None:
        if self.last_capture is None or self.last_current_roi is None or self.last_summary is None:
            if not silent:
                messagebox.showinfo("没有调试图像", "请先执行一次识别。")
            return
        ts = timestamp()
        debug_dir = ensure_dir(self.project_config.root_dir / "assets" / "debug")
        summary = self.last_summary
        best_template = None
        if summary.best is not None and summary.best.source in {"sample", "sample_map"}:
            best_template = self.sample_db.image_by_sample_id(summary.best.sample_id)
        elif summary.best is not None and summary.best.map_id in self.map_db.templates:
            best_template = self.map_db.templates[summary.best.map_id].template

        files = {
            f"{ts}_raw_capture.png": self.last_capture.image,
            f"{ts}_current_roi.png": self.last_current_roi,
            f"{ts}_current_preprocessed.png": summary.current_preprocessed,
        }
        if summary.current_cropped_roi is not None:
            files[f"{ts}_current_auto_cropped_roi.png"] = summary.current_cropped_roi
        if best_template is not None:
            files[f"{ts}_best_template.png"] = best_template
        if summary.best_template_preprocessed is not None:
            files[f"{ts}_best_template_preprocessed.png"] = summary.best_template_preprocessed
            files[f"{ts}_match_overlay.png"] = make_side_by_side(summary.current_preprocessed, summary.best_template_preprocessed)
        files[f"{ts}_top5_color_contact_sheet.png"] = make_top_candidate_contact_sheet(self.last_current_roi, summary, self.map_db, self.sample_db)

        for name, image in files.items():
            imwrite_unicode(debug_dir / name, image)
        if not silent:
            self.status_var.set(f"调试图像已保存：{debug_dir}")
            messagebox.showinfo("保存成功", f"调试图像已保存到：\n{debug_dir}")



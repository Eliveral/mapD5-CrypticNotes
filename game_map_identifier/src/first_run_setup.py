from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
from tkinter import messagebox

from .config import save_app_config


ROI = Tuple[int, int, int, int]


class RoiPicker:
    def __init__(self, image: np.ndarray, *, title: str, line1: str, line2: str) -> None:
        self.image = image
        self.title = title
        self.line1 = line1
        self.line2 = line2
        self.dragging = False
        self.start: Optional[tuple[int, int]] = None
        self.roi: Optional[ROI] = None

    def _draw(self) -> None:
        out = self.image.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 96), (0, 0, 0), -1)
        cv2.putText(out, self.line1, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, self.line2, (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "Drag mouse | Enter/S save | R reset | Esc cancel", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        if self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, f"ROI: [{x}, {y}, {w}, {h}]", (18, out.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(self.title, out)

    def _mouse(self, event, x, y, _flags, _param) -> None:
        h, w = self.image.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.roi = (x, y, 1, 1)
            self._draw()
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.start is not None:
            sx, sy = self.start
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)
            self.roi = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            self._draw()
        elif event == cv2.EVENT_LBUTTONUP and self.dragging and self.start is not None:
            self.dragging = False
            sx, sy = self.start
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)
            self.roi = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            self._draw()

    def pick(self) -> Optional[ROI]:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.title, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(self.title, self._mouse)
        self._draw()
        try:
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return None
                if key in (ord("r"), ord("R")):
                    self.roi = None
                    self._draw()
                if key in (13, 10, ord("s"), ord("S")):
                    return self.roi
        finally:
            cv2.destroyWindow(self.title)


def grab_virtual_screen() -> tuple[np.ndarray, int, int]:
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        frame = np.array(shot, dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), int(monitor["left"]), int(monitor["top"])


def ensure_first_run_feature_area(config_path: Path, app_config: dict) -> bool:
    """Mark the two fixed ROI profiles as ready without opening a picker."""
    changed = False
    for key in ("user_feature_area_configured", "user_location_area_configured"):
        if not bool(app_config.get(key, False)):
            app_config[key] = True
            changed = True
    if changed:
        try:
            save_app_config(config_path, app_config)
        except OSError:
            return False
    return True


def prompt_feature_area(config_path: Path, app_config: dict) -> bool:
    messagebox.showinfo(
        "First use: map recognition ROI",
        "Step 1/2: select the small map feature area used by F8 recognition.\n\n"
        "Open the game view right after entering from a side door, then drag a box around the visible map structure near the side door.\n\n"
        "After selecting, press Enter or S to save.",
    )
    image, offset_x, offset_y = grab_virtual_screen()
    roi = RoiPicker(
        image,
        title="First use - recognition ROI",
        line1="Step 1/2: select F8 map recognition ROI",
        line2="Select the side-door map feature area shown after entering.",
    ).pick()
    if roi is None:
        messagebox.showwarning("ROI not saved", "Recognition ROI was not selected. The program will exit.")
        return False
    x, y, w, h = roi
    app_config["capture_mode"] = "fixed_screen_roi"
    app_config["screen_capture_roi"] = [x + offset_x, y + offset_y, w, h]
    app_config["current_map_feature_roi"] = [0, 0, w, h]
    app_config["current_map_feature_roi_version"] = f"user_roi_{w}x{h}"
    app_config["user_feature_area_configured"] = True
    save_app_config(config_path, app_config)
    messagebox.showinfo("Recognition ROI saved", "Step 1 completed. F8 recognition will use this area.")
    return True


def prompt_location_area(config_path: Path, app_config: dict) -> bool:
    if bool(app_config.get("alignment_fullscreen_transform_enabled", False)):
        app_config["user_location_area_configured"] = True
        save_app_config(config_path, app_config)
        return True
    messagebox.showinfo(
        "First use: overview location ROI",
        "Step 2/2: select the full overview map area used by F1/F2/F5 location.\n\n"
        "Open the in-game overview map at the fixed zoom used for location, then drag a box around the full map area.\n\n"
        "After selecting, press Enter or S to save.",
    )
    image, offset_x, offset_y = grab_virtual_screen()
    roi = RoiPicker(
        image,
        title="First use - overview location ROI",
        line1="Step 2/2: select F1/F2/F5 overview map ROI",
        line2="Select the full overview map area at the fixed location zoom.",
    ).pick()
    if roi is None:
        messagebox.showwarning("ROI not saved", "Overview location ROI was not selected. The program will exit.")
        return False
    x, y, w, h = roi
    capture_key = str(app_config.get("alignment_screen_capture_roi_key", "alignment_screen_capture_roi"))
    map_roi_key = str(app_config.get("alignment_map_roi_key", "alignment_map_roi"))
    app_config[capture_key] = [x + offset_x, y + offset_y, w, h]
    app_config[map_roi_key] = [0, 0, w, h]
    app_config["user_location_area_configured"] = True
    save_app_config(config_path, app_config)
    messagebox.showinfo("Overview location ROI saved", "Step 2 completed. F1/F2/F5 location will use this area.")
    return True

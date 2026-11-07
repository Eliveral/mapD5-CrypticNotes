from __future__ import annotations

import ctypes
import os


def enable_physical_pixel_coordinates() -> None:
    """Use physical screen pixels for Win32 window coordinates and MSS capture."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

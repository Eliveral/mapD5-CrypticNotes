"""Read-only game window identity and passive keys. Never sends input to the game."""
from __future__ import annotations
import ctypes
from ctypes import wintypes as W
from dataclasses import dataclass
from pathlib import PureWindowsPath

from .s1_keys import MapBinding


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    pid: int
    bounds: tuple[int, int, int, int]
    title: str = '第五人格'


def client_bounds(hwnd):
    u = ctypes.windll.user32
    h = W.HWND(hwnd)
    if not u.IsWindow(h) or not u.IsWindowVisible(h) or u.IsIconic(h):
        return None
    rect, point = W.RECT(), W.POINT()
    if not u.GetClientRect(h, ctypes.byref(rect)) or not u.ClientToScreen(h, ctypes.byref(point)):
        return None
    w, h = rect.right-rect.left, rect.bottom-rect.top
    return (point.x, point.y, w, h) if w >= 320 and h >= 240 else None


def refresh_game(game):
    """Validate a selected HWND against its PID; prevent stale-handle reuse."""
    if game is None:
        return None
    pid = W.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(W.HWND(game.hwnd), ctypes.byref(pid))
    bounds = client_bounds(game.hwnd)
    if pid.value != game.pid or bounds is None:
        return None
    return GameWindow(game.hwnd, game.pid, bounds, game.title)


def discover_game(current=0):
    """Query process *identity* only (0x1000); no VM_READ or memory inspection."""
    u, k = ctypes.windll.user32, ctypes.windll.kernel32
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, ctypes.POINTER(W.DWORD)]
    k.CloseHandle.argtypes = [W.HANDLE]
    found = []
    callback_type = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)

    @callback_type
    def visit(hwnd, _):
        bounds = client_bounds(hwnd)
        if bounds is None:
            return True
        pid = W.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = k.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return True
        try:
            size = W.DWORD(32768)
            path = ctypes.create_unicode_buffer(size.value)
            if not k.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                return True
            if PureWindowsPath(path.value).name.casefold() != 'dwrg.exe':
                return True
            title = ctypes.create_unicode_buffer(512)
            u.GetWindowTextW(hwnd, title, len(title))
            found.append(GameWindow(int(hwnd), pid.value, bounds, title.value))
        finally:
            k.CloseHandle(handle)
        return True

    u.EnumWindows(visit, 0)
    focused = [g for g in found if g.hwnd == current]
    if len(focused) == 1:
        return focused[0]
    return found[0] if len(found) == 1 else None


class KeyEdges:
    """Pure debouncer shared by the passive hook and regression tests."""
    def __init__(self, map_vk=0x14):
        self.held = set()
        self.binding = MapBinding(map_vk) if map_vk is not None else None

    def update(self, vk, down, modified=False, *, mods=0):
        if not down:
            self.held.discard(vk)
            return None
        if vk in self.held:
            return None
        self.held.add(vk)
        if modified:return None
        if self.binding is not None and self.binding.key==vk and self.binding.mods==mods:
            return 'map'
        if vk==0x1B and not mods:return 'escape'
        return None

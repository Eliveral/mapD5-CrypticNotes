"""Screenshot recognition, read-only window identity, and passive map-key observation."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import json
import os
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
import cv2
import mss
import numpy as np
from PIL import Image, ImageTk
from .s1_alignment import FloorMatcher
from .s1_visibility import MapVisibility
from .config import ProjectConfig, project_root
from .s1_game import discover_game, refresh_game, KeyEdges, MAP_KEYS


def primary_bounds():
    # On Windows the primary display has origin (0, 0). mss.monitors[1]
    # is enumeration order and need not be the user's primary display.
    u = ctypes.windll.user32
    return (0, 0, int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1)))


def foreground():
    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = wintypes.HWND
    return int(u.GetForegroundWindow() or 0)


def own_window(hwnd):
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return pid.value == os.getpid()


_capture_state=threading.local()


def prepare_capture_thread():
    _capture_state.persistent=True


def release_capture_thread():
    screen=getattr(_capture_state,'screen',None)
    if screen is not None:
        screen.close()
        _capture_state.screen=None


def capture(roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError('地图区域无效')
    box=dict(left=x,top=y,width=w,height=h)
    if getattr(_capture_state,'persistent',False):
        if not getattr(_capture_state,'screen',None):_capture_state.screen=mss.mss()
        try:raw=_capture_state.screen.grab(box)
        except Exception:
            release_capture_thread()
            raise
    else:
        with mss.mss() as sct:raw=sct.grab(box)
    return cv2.cvtColor(np.asarray(raw),cv2.COLOR_BGRA2BGR)


class Hotkeys:
    def __init__(self, events, map_key='Caps Lock'):
        self.events = events
        self.edges = KeyEdges(MAP_KEYS.get(map_key, 0x14))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        u = ctypes.windll.user32
        # The configured map key must reach the game. RegisterHotKey would consume it, so observe
        # key edges in a passive hook and ALWAYS continue the hook chain.
        class KeyData(ctypes.Structure):
            _fields_ = [('vkCode', wintypes.DWORD), ('scanCode', wintypes.DWORD),
                        ('flags', wintypes.DWORD), ('time', wintypes.DWORD), ('extra', ctypes.c_size_t)]
        hook_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        u.SetWindowsHookExW.argtypes = [ctypes.c_int, hook_type, wintypes.HINSTANCE, wintypes.DWORD]
        u.SetWindowsHookExW.restype = wintypes.HANDLE
        u.CallNextHookEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        u.CallNextHookEx.restype = ctypes.c_ssize_t
        u.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]

        @hook_type
        def observe(code, message, address):
            try:
                if code >= 0 and message in (0x100, 0x104, 0x101, 0x105):
                    data = ctypes.cast(address, ctypes.POINTER(KeyData)).contents
                    modified = any(u.GetAsyncKeyState(v) & 0x8000 for v in (0x10, 0x11, 0x12, 0x5B, 0x5C))
                    action = self.edges.update(data.vkCode, message in (0x100, 0x104), modified)
                    if action:
                        self.events.put(('map_key', (action, foreground())))
            except Exception:
                pass  # An observer must never interrupt input delivery.
            return u.CallNextHookEx(None, code, message, address)

        hook = u.SetWindowsHookExW(13, observe, None, 0)
        self.events.put(('hook_ready', bool(hook)))
        actions = {701: (0x77, 'once'), 702: (0x76, 'auto'), 703: (0x78, 'hide'), 704: (0x79, 'stop')}
        registered = []
        try:
            for ident, (key, action) in actions.items():
                if u.RegisterHotKey(None, ident, 0x4000, key):
                    registered.append(ident)
                else:
                    self.events.put(('hotkey_error', f'F{key-0x6f} 已被占用，请使用窗口按钮。'))
            msg = wintypes.MSG()
            while not self.stop_event.wait(.01):
                while u.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                    if msg.message == 0x0312 and msg.wParam in actions:
                        self.events.put(('key', actions[msg.wParam][1]))
        finally:
            if hook:
                u.UnhookWindowsHookEx(hook)
            for ident in registered:
                u.UnregisterHotKey(None, ident)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=.3)

    def set_map_key(self, name):
        self.edges.map_vk = MAP_KEYS.get(name, 0x14)


class RegionPicker(tk.Toplevel):
    def __init__(self, parent, image, bounds, callback):
        super().__init__(parent)
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        x, y, w, h = bounds
        self.geometry(f'{w}x{h}+{x}+{y}')
        self.bounds = bounds
        self.callback = callback
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.canvas.create_rectangle(0, 0, w, 54, fill='#111827', outline='')
        self.canvas.create_text(20, 25, anchor='w', fill='white', font=('Microsoft YaHei UI',15), text='拖框选择游戏地图区域（不含右侧按钮） · Esc 取消')
        self.rect = None
        self.start = None
        self.bind('<ButtonPress-1>', self.begin)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<ButtonRelease-1>', self.finish)
        self.bind('<Escape>', lambda e: self.done(None))
        self.focus_force()

    def begin(self, e):
        self.start = (e.x, e.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline='#5eead4', width=3)

    def drag(self, e):
        if self.start:
            self.canvas.coords(self.rect, *self.start, e.x, e.y)

    def finish(self, e):
        if not self.start:
            return
        x0, x1 = sorted((self.start[0], max(0, min(e.x, self.bounds[2]))))
        y0, y1 = sorted((self.start[1], max(0, min(e.y, self.bounds[3]))))
        if x1-x0 >= 120 and y1-y0 >= 120:
            self.done((x0+self.bounds[0], y0+self.bounds[1], x1-x0, y1-y0))

    def done(self, roi):
        self.destroy()
        self.callback(roi)


class LiveController:
    matcher_class = FloorMatcher
    def __init__(self, app, autostart=True, controls=True):
        self.app = app
        self.matcher = self.matcher_class(app.catalog)
        self.visibility = MapVisibility(app.catalog.root/'ui')
        self.warming = bool(controls)
        self.events = queue.Queue()
        self.hotkeys = Hotkeys(self.events, app.map_key.get()) if controls else None
        self.overlay = None
        self.capture_excluded = False
        self.active = bool(autostart)
        self.continuous = True
        self.hidden = self.busy = self.closed = self.suspended = False
        self.generation = 0
        self.target = 0
        self.game = None
        self.focused = False
        self.locked_match = None
        self.due = self.discovery_due = 0
        self.roi = None  # Optional crop relative to the game client, never another app.
        self.bounds = primary_bounds()
        from .s1_updates import profile
        self.diagnostics = profile()/'runtime'/'s1_live'
        self.history = []
        self.last_saved = 0
        self.last_visible_saved = 0
        self.last_map_saved = 0
        self.last_frame = None
        self.hook_ready = False
        self.note('waiting_game' if autostart else 'stopped', '正在自动寻找第五人格；打开地图后自动检测，不依赖按键触发。')
        if controls:
            def prepare():
                try:
                    self.matcher.warmup()
                    self.events.put(('templates_ready', None))
                except Exception as exc:
                    self.events.put(('templates_failed', str(exc)))
            threading.Thread(target=prepare, daemon=True).start()
        self.app.after(80, self.tick)

    def note(self, state, message=None, **details):
        if message:
            self.app.live_status.set(message)
        if hasattr(self.app, 'detect_state'):
            if not self.active:
                badge = '检测已停止 · 可重新开启'
            elif getattr(self, 'warming', False):
                badge = '检测已开启 · 正在准备地图图谱'
            elif self.hidden:
                badge = '检测已开启 · 覆盖已隐藏（F9 恢复）'
            elif state == 'overlay_shown':
                badge = '检测已开启 · 地图已匹配'
            elif state == 'capture_error':
                badge = '检测已开启 · 截图重试中'
            elif state == 'map_closed':
                badge = '检测已开启 · 地图未展开'
            elif state == 'map_unmatched':
                locked=details.get('round_layout')
                badge = f'本局布局 {locked[3:]} · 正在定位' if locked else '地图已展开 · 尚未确定布局'
            elif not self.game:
                badge = '检测已开启 · 等待第五人格'
            elif foreground() != self.target:
                badge = '检测已开启 · 请切回第五人格'
            else:
                badge = '检测已开启 · 等待地图匹配'
            self.app.detect_state.set(badge)
        record = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), monotonic=round(time.monotonic(), 3),
                      state=state, generation=self.generation, active=self.active,
                      target=self.target, game_pid=self.game.pid if self.game else None,
                      bounds=self.game.bounds if self.game else None, foreground=foreground(),
                      visible=bool(self.overlay and getattr(self.overlay, '_visible', False)),
                      capture_excluded=self.capture_excluded, hook_ready=self.hook_ready, **details)
        self.history.append(record)
        self.history = self.history[-200:]
        try:
            self.diagnostics.mkdir(parents=True, exist_ok=True)
            tmp = self.diagnostics/'state.tmp'
            tmp.write_text(json.dumps(dict(current=record, events=self.history), ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(self.diagnostics/'state.json')
        except OSError:
            pass

    def invalidate(self, clear_match=True):
        self.generation += 1
        if self.overlay:
            self.overlay.hide()
        if clear_match:
            self.locked_match = None
        self.last_frame = None

    def ensure_overlay(self):
        if self.overlay is None:
            from .s1_overlay import S1Overlay
            root = project_root()
            cfg = ProjectConfig(root, root/'config/config.json', root/'config/maps.json',
                                {'route_overlay_alpha': self.app.overlay_alpha.get()/100, 'route_overlay_native_layered_enabled': True}, {})
            self.overlay = S1Overlay(self.app, cfg)
            self.overlay.warmup()
            handle = self.overlay._native_window_handle()
            u = ctypes.windll.user32
            u.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
            self.capture_excluded = bool(u.SetWindowDisplayAffinity(wintypes.HWND(handle), 0x11))

    def start(self, continuous=True):
        if self.suspended:
            self.app.live_status.set('原版正在运行；关闭后自动恢复 S1。')
            return
        self.invalidate()
        self.active, self.continuous, self.hidden = True, continuous, False
        self.due = time.monotonic()+.25
        self.discovery_due = 0
        self.note('armed', '自动检测已开启；切回游戏并打开地图。F10 停止。')
        if self.app.state() not in ('iconic', 'withdrawn'):
            self.app.iconify()

    def stop(self, quiet=False):
        self.active = False
        self.invalidate()
        self.note('stopped', None if quiet else '已停止并清除覆盖。F7 恢复自动识别。')

    def settings_changed(self):
        self.invalidate()
        self.due = time.monotonic()+.2
        self.note('settings_changed', '设置已更新；切回游戏后重新核对地图。')

    def map_key_changed(self):
        name = self.app.map_key.get()
        if self.hotkeys:
            self.hotkeys.set_map_key(name)
        self.settings_changed()
        self.note('map_key_changed', f'地图键：{name}。持续截图检测始终开启，按键仅加速重查。')

    def save_map_sample(self, frame, presence):
        # Keep the latest OPEN map even after the player closes it. The old
        # single last_capture file lost the only diagnostic map on focus change.
        if not presence['visible'] or time.monotonic()-self.last_map_saved < 3:
            return
        self.last_map_saved = time.monotonic()
        try:
            data = cv2.imencode('.png', frame)[1]
            data.tofile(str(self.diagnostics/'last_map.png'))
            (self.diagnostics/'last_map.json').write_text(json.dumps(dict(
                time=time.strftime('%Y-%m-%d %H:%M:%S'), presence=presence,
                kind='unmodified_game_client_capture')), encoding='utf-8')
        except OSError:
            pass

    def toggle_hidden(self):
        if not self.active:
            self.app.live_status.set('已停止；按 F7 恢复自动识别。')
            return
        self.hidden = not self.hidden
        self.invalidate()
        self.due = time.monotonic()+.1
        self.note('hidden' if self.hidden else 'armed', '覆盖已隐藏（F9 恢复）' if self.hidden else '重新核对地图后显示…')

    def set_alpha(self, value):
        if self.overlay:
            alpha = min(.90,max(.15,float(value)/100))
            if self.overlay.alpha != alpha:
                if self.overlay._visible or not self.overlay._native_layered_enabled:
                    self.overlay.set_alpha(alpha)
                else:
                    # Applying alpha to a hidden native buffer can show a stale
                    # floor. The next upload uses this value on its first frame.
                    self.overlay.alpha = alpha

    def pick_region(self):
        # A manual crop is optional. Default startup always uses the game client.
        self.stop(quiet=True)
        self.game = discover_game(foreground())
        if not self.game:
            self.app.live_status.set('请先启动第五人格；默认会自动使用游戏区域。')
            return
        self.app.iconify()
        self.app.after(450, self._pick)

    def _pick(self):
        try:
            self.game = refresh_game(self.game)
            if not self.game or foreground() != self.game.hwnd:
                self.start()
                return
            RegionPicker(self.app, capture(self.game.bounds), self.game.bounds, self._region_done)
        except Exception as exc:
            self.app.deiconify()
            self.app.live_status.set('选区失败：'+str(exc))

    def _region_done(self, roi):
        if roi and self.game:
            x, y, _, _ = self.game.bounds
            self.roi = (roi[0]-x, roi[1]-y, roi[2], roi[3])
        self.start()

    def reset_region(self):
        self.roi = None
        self.settings_changed()
        if not self.active:
            self.start()

    def capture_bounds(self):
        if not self.game:
            return None
        x, y, w, h = self.game.bounds
        if self.roi:
            rx, ry, rw, rh = self.roi
            if rx >= 0 and ry >= 0 and rx+rw <= w and ry+rh <= h:
                return (x+rx, y+ry, rw, rh)
            self.roi = None
        return (x, y, w, h)

    def sync_target(self):
        current = foreground()
        game = refresh_game(self.game)
        now = time.monotonic()
        if game is None and now >= self.discovery_due:
            game = discover_game(current)
            self.discovery_due = now+1
        if game != self.game:
            old = self.game
            self.invalidate(clear_match=not (game and old and game.hwnd == old.hwnd))
            self.game = game
            self.target = game.hwnd if game else 0
            self.due = now+.15
            self.note('game_found' if game else 'waiting_game',
                      '已锁定第五人格。打开地图后自动检测和匹配。' if game else '等待第五人格窗口…')
        focused = bool(game and current == game.hwnd)
        if focused != self.focused:
            self.focused = focused
            self.invalidate()
            self.due = now+.2
            self.note('game_focused' if focused else 'focus_lost',
                      '正在检查游戏地图…' if focused else '覆盖已隐藏；切回第五人格后自动恢复。')
        return focused

    def map_key(self, action, hwnd):
        if not self.active or self.suspended:
            return
        # Select immediately too: a map key can arrive before discovery's tick.
        self.sync_target()
        if not self.game or hwnd != self.game.hwnd or foreground() != hwnd:
            return
        self.invalidate()
        self.due = time.monotonic()+(.32 if action == 'map' else .5)
        self.note('map_key' if action == 'map' else 'escape_key',
                  '检测到地图键：核对当前画面…' if action == 'map' else '覆盖已隐藏；核对地图是否关闭。')

    def launch(self):
        if not self.game or foreground() != self.game.hwnd:
            return
        self.ensure_overlay()
        if not self.capture_excluded:
            self.overlay.hide()
        roi, game = self.capture_bounds(), self.game
        self.busy = True
        generation = self.generation
        layout = self.app.layout_limit.get()
        layout = None if layout == '自动' else 's1_'+layout
        floor = self.app.floor_limit.get()
        floor = None if floor == '自动' else floor
        locked = self.locked_match
        style = 'routes' if self.app.overlay_style.get() == '路线' else 'structure'

        def work():
            began = time.monotonic()
            try:
                # Check focus both sides of capture, not merely before queueing work.
                if foreground() != game.hwnd or refresh_game(game) != game:
                    self.events.put(('discard', generation))
                    return
                frame = capture(roi)
                if foreground() != game.hwnd:
                    self.events.put(('discard', generation))
                    return
                presence = self.visibility.inspect(frame) if self.roi is None else dict(visible=True, scores={}, manual_crop=True)
                self.save_map_sample(frame, presence)
                if not presence['visible']:
                    result = dict(accepted=False, map_visible=False, presence=presence,
                                  reason='游戏地图未展开；请用当前地图键打开。检测仍在持续。', candidates=[])
                    self.events.put(('result', (generation, result, None, roi, time.monotonic()-began, style, frame, game)))
                    return
                # A verified current floor is the fast tracking reference. Map keys,
                # focus changes and failed matches discard it and rescan all 45.
                result = None
                if locked:
                    f = locked['floor']
                    result = self.matcher.locate(frame, f['layout_id'], f['floor'])
                if result is None or not result['accepted']:
                    result = self.matcher.locate(frame, layout, floor)
                result.update(map_visible=True, presence=presence)
                if not result['accepted']:
                    result['reason'] = '地图已展开，尚未找到可靠布局；请缩小地图或露出更多相连房间。'
                rendered = self.matcher.render(result, (roi[2], roi[3]), style) if result['accepted'] else None
                # Recheck UI presence after a potentially slow first match:
                # closing by mouse must also cancel an in-flight projection.
                if rendered is not None and self.roi is None:
                    if foreground() != game.hwnd or refresh_game(game) != game:
                        self.events.put(('discard', generation)); return
                    if not self.visibility.inspect(capture(roi))['visible']:
                        result = dict(accepted=False, map_visible=False, presence=presence,
                                      reason='地图已关闭，覆盖已隐藏。', candidates=[])
                        rendered = None
                self.events.put(('result', (generation, result, rendered, roi, time.monotonic()-began, style, frame, game)))
            except Exception as exc:
                self.events.put(('error', (generation, str(exc))))
        threading.Thread(target=work, daemon=True).start()

    def receive(self, payload):
        generation, result, rendered, roi, elapsed, style, frame, game = payload
        self.busy = False
        if (generation != self.generation or not self.active or self.hidden or
                foreground() != self.target or refresh_game(game) != game):
            self.note('stale_discarded', result_generation=generation)
            return
        # Last-frame diagnostics are bounded, local, and only game-client images.
        try:
            if time.monotonic()-self.last_saved > 3:
                cv2.imencode('.png', frame)[1].tofile(str(self.diagnostics/'last_capture.png'))
                self.last_saved = time.monotonic()
        except OSError:
            pass
        if result['accepted']:
            self.locked_match = result
            f = result['floor']
            self.overlay.show_image(rendered, roi, image_is_rgb=False)
            self.overlay.set_alpha(self.app.overlay_alpha.get()/100)
            if not self.overlay._native_layered_enabled:
                self.capture_excluded = False
            suffix = '；该层路线未标定，显示结构' if style == 'routes' and not f.get('route_matrix') else ''
            self.note('overlay_shown', f'布局 {f["layout_id"].split("_")[-1]} · {f["floor"]} · 自动跟随 · {elapsed:.2f}秒{suffix}',
                      layout=f['layout_id'], floor=f['floor'], matrix=result['matrix'].tolist(),
                      inliers=result['inliers'], error_px=result['error_px'], elapsed=elapsed,
                      roi=roi, style=style, warning=f.get('route_warning', ''), presence=result.get('presence'),
                      segmentation=result.get('segmentation'))
            if time.monotonic()-self.last_visible_saved > 3:
                self.app.after(65, lambda: self.save_visible(generation, game, roi))
            self.due = time.monotonic()+.12
        else:
            self.locked_match = None
            self.overlay.hide()
            visible = result.get('map_visible', False)
            self.note('map_unmatched' if visible else 'map_closed', result['reason'], reason=result['reason'],
                      elapsed=elapsed, roi=roi, presence=result.get('presence'))
            self.due = time.monotonic()+(.40 if visible else .15)
        self.last_frame = frame

    def save_visible(self, generation, game, roi):
        if (generation != self.generation or not self.active or self.hidden or
                foreground() != game.hwnd or refresh_game(game) != game or not self.overlay._visible):
            return
        try:
            self.last_visible_saved = time.monotonic()
            # Real screen capture, not a computed composite. If affinity works,
            # this deliberately cannot serve as evidence of visible projection.
            cv2.imencode('.png', capture(roi))[1].tofile(str(self.diagnostics/'last_visible.png'))
            (self.diagnostics/'last_visible.json').write_text(json.dumps(dict(
                time=time.strftime('%Y-%m-%d %H:%M:%S'), generation=generation, roi=roi,
                capture_excluded=self.capture_excluded, kind='actual_mss_screen_capture')), encoding='utf-8')
        except (OSError, ValueError):
            pass

    def tick(self):
        if self.closed:
            return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'key':
                    if data == 'once': self.start(False)
                    elif data == 'auto': self.start(True)
                    elif data == 'hide': self.toggle_hidden()
                    elif data == 'stop': self.stop()
                elif kind == 'map_key':
                    self.map_key(*data)
                elif kind == 'hook_ready':
                    self.hook_ready = data
                    self.note('input_observer_ready' if data else 'input_observer_unavailable')
                elif kind == 'templates_ready':
                    self.warming = False
                    self.note('templates_ready', '图谱已准备好；正在持续检测游戏地图。')
                elif kind == 'templates_failed':
                    self.warming = False
                    self.stop(quiet=True)
                    self.note('templates_failed', '图谱加载失败：'+data)
                elif kind == 'hotkey_error':
                    self.app.live_status.set(data)
                elif kind == 'result':
                    self.receive(data)
                elif kind == 'discard':
                    self.busy = False
                elif kind == 'error':
                    self.busy = False
                    if data[0] == self.generation:
                        self.invalidate()
                        self.due = time.monotonic()+2
                        self.note('capture_error', '截图暂不可用，自动重试：'+data[1], error=data[1])
        except queue.Empty:
            pass
        if self.active and not self.suspended:
            ready = self.sync_target()
            if ready and not getattr(self, 'warming', False) and not self.hidden and not self.busy and time.monotonic() >= self.due:
                self.launch()
        self.app.after(60, self.tick)

    def close(self):
        self.closed = True
        self.stop(quiet=True)
        if self.hotkeys:
            self.hotkeys.close()
        if self.overlay:
            self.overlay.destroy()

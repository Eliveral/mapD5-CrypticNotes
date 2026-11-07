"""Key-triggered capture bursts, independent matching and guarded projection."""
from __future__ import annotations
import queue
import threading
import time
import cv2
import numpy as np
from . import s1_live as base
from .s1_live_match import LiveMatcher, signature, same_view, map_area
from .s1_floor_transition import selected_floor
from .s1_layer_probe import LayerProbe


class LiveController(base.LiveController):
    matcher_class = LiveMatcher

    def __init__(self, *args, **kwargs):
        self.sampling = self.matching = False
        self.burst_until = 0.
        self.latest = self.pending = self.display = None
        self.floor_cache = None
        self.scale_hint = None
        self.view_cache = None
        self.floor_visits = {}
        self.session_revision = 0
        self.cache_identity = None
        self.last_note_state = None
        self.last_note_at = 0.
        self.diagnostic_busy = False
        super().__init__(*args, **kwargs)
        self.note('trigger_waiting', '自动锁定第五人格；地图键默认 Caps Lock，按下立即连续抓图。')

    def note(self, state, message=None, **details):
        now = time.monotonic()
        # Do not rewrite a 200-event JSON file on every high-rate sample.
        if state == self.last_note_state and now-self.last_note_at < 1.5:
            return
        self.last_note_state, self.last_note_at = state, now
        super().note(state, message, **details)

    def invalidate(self, clear_match=True):
        super().invalidate(clear_match)
        self.latest = self.pending = self.display = None
        # Only identity is cached; every new opening needs fresh registration.

    def start(self, continuous=True):
        self.session_revision += 1
        self.clear_visits()
        super().start(continuous)
        self.burst_until = time.monotonic()+1.2
        self.due = 0

    def settings_changed(self):
        self.session_revision += 1
        self.clear_visits()
        super().settings_changed()

    def stop(self, quiet=False):
        self.session_revision += 1
        self.clear_visits()
        super().stop(quiet)

    def clear_visits(self):
        self.floor_visits.clear()
        self.view_cache = self.floor_cache = self.scale_hint = self.cache_identity = None

    def valid_visit(self, visit, sample, lifetime):
        return bool(visit and visit['revision'] == self.session_revision and
                    time.monotonic()-visit['at'] <= lifetime and
                    visit['game'] == sample['game'] and visit['roi'] == sample['roi'])

    def map_key_changed(self):
        super().map_key_changed()
        self.note('trigger_key_changed', f'触发键：{self.app.map_key.get()}。按下立即抓图；画面确认开关。')

    def map_key(self, action, hwnd):
        if not self.active or self.suspended: return
        self.sync_target()
        if not self.game or hwnd != self.game.hwnd or base.foreground() != hwnd: return
        pending = self.pending
        self.invalidate()
        self.pending = pending
        self.burst_until = time.monotonic()+1.2
        self.due = 0  # No 320 ms animation wait, and never count toggle parity.
        self.note('map_key' if action == 'map' else 'escape_key',
                  '已收到地图键：立即连续抓图…' if action == 'map' else '已隐藏覆盖，正在确认地图关闭。')

    def launch(self):
        if not self.game or base.foreground() != self.game.hwnd: return
        self.ensure_overlay()
        shown = self.display if self.overlay._visible and not self.capture_excluded else None
        probe = shown.get('probe') if shown else None
        opacity = self.app.overlay_alpha.get()/100
        if self.overlay._visible and not self.capture_excluded and probe is None:
            self.overlay.hide()
        game, roi, epoch = self.game, self.capture_bounds(), self.generation
        revision = self.session_revision
        manual = self.roi is not None
        burst = time.monotonic() < self.burst_until
        self.sampling = True

        def grab():
            began = time.monotonic()
            try:
                if base.foreground() != game.hwnd or base.refresh_game(game) != game:
                    self.events.put(('sample_discard', None)); return
                frame = base.capture(roi)
                if base.foreground() != game.hwnd:
                    self.events.put(('sample_discard', None)); return
                presence = (dict(visible=True, manual_crop=True) if manual else
                            self.visibility.inspect(frame, fast=burst))
                sig = signature(frame) if presence['visible'] else None
                floor = selected_floor(frame) if presence['visible'] and not manual else None
                stationary = bool(probe and presence['visible'] and probe.stationary(frame, opacity))
                self.events.put(('sample', dict(epoch=epoch, game=game, roi=roi, frame=frame,
                    presence=presence, signature=sig, selected_floor=floor, probing=probe is not None,
                    stationary=stationary, at=time.monotonic(), manual=manual, revision=revision,
                    capture_ms=round((time.monotonic()-began)*1000, 1))))
            except Exception as exc:
                self.events.put(('sample_error', str(exc)))
        threading.Thread(target=grab, daemon=True).start()

    def sample_received(self, sample):
        self.sampling = False
        if (not self.active or self.hidden or self.suspended or sample['game'] != self.game or
                base.foreground() != self.target or sample['epoch'] != self.generation or
                sample.get('revision', self.session_revision) != self.session_revision):
            self.due = 0
            return
        previous = self.latest
        if (previous and previous['presence']['visible'] and sample['presence']['visible'] and
                previous.get('selected_floor') != sample.get('selected_floor')):
            self.generation += 1
            sample['epoch'] = self.generation
            self.display = self.pending = None
            if self.overlay and self.overlay._visible:
                self.overlay.hide()
            self.note('floor_changed', f'切换楼层，正在重新定位 {sample.get("selected_floor") or "…"}',
                      selected_floor=sample.get('selected_floor'))
        self.latest = sample
        now = time.monotonic()
        visible = sample['presence']['visible']
        if visible and sample.get('selected_floor') and (not previous or not previous['presence']['visible']):
            self.note('floor_detected', f'已识别 {sample["selected_floor"]}，正在核对地图位置。',
                      selected_floor=sample['selected_floor'])
        self.due = now + (.012 if visible or now < self.burst_until else .20)
        if not visible:
            if self.overlay and self.overlay._visible: self.overlay.hide()
            self.display = None
            # Retain already captured positive frames for background learning.
            if previous and previous['presence']['visible']:
                self.generation += 1
                sample['epoch'] = self.generation
            self.note('map_closed', '等待地图键；已关闭的地图不会补显示迟到覆盖。',
                      capture_ms=sample['capture_ms'], presence=sample['presence'])
            return
        if sample.get('probing'):
            # This frame may contain our own artwork. Never match it, save it
            # as a raw map sample, or let it authorize a pending registration.
            sample['signature'] = None
            if self.display and self.display.get('selected_floor') == sample.get('selected_floor'):
                if sample['stationary']:
                    self.display['misses'] = 0
                    self.due = now+.06
                    return
                self.display['misses'] = self.display.get('misses',0)+1
                if self.display['misses'] < 2:
                    self.due = now+.012
                    return
            if self.overlay._visible: self.overlay.hide()
            self.display = self.pending = None
            self.due = 0
            return
        if self.display and same_view(sample['signature'], self.display['signature']):
            self.show_display()
            self.due = now+.06
        elif self.reuse_view(sample):
            self.show_display()
            self.due = now+.06
        else:
            if self.overlay and self.overlay._visible: self.overlay.hide()
            self.display = None
            self.pending = sample  # Bounded latest-positive slot, never an unbounded frame queue.
            self.start_match()
        self.save_sample_async(sample)

    def reuse_view(self, sample):
        cached = self.floor_visits.get(sample.get('selected_floor'), self.view_cache)
        if (not self.valid_visit(cached,sample,30) or
                sample.get('selected_floor') != cached.get('selected_floor') or
                sample['game'] != cached['game'] or sample['roi'] != cached['roi'] or
                not same_view(sample['signature'], cached['signature'])): return False
        # Check the fresh pixels too, never blindly reuse the previous matrix
        # just because a keyboard toggle arrived or a layout ID is cached.
        thumb = cv2.resize(sample['frame'], (320, 180))
        area = map_area(thumb) > 0
        delta = np.abs(thumb.astype(np.int16)-cached['thumb'].astype(np.int16))
        if float(delta[area].mean()) > 2.0 or float(np.mean(np.max(delta, axis=2)[area] > 18)) > .02:
            return False
        self.display = dict(signature=sample['signature'], rendered=cached['rendered'],
                            roi=sample['roi'], epoch=self.generation, game=sample['game'],
                            selected_floor=sample.get('selected_floor'), label=cached.get('label'),
                            probe=cached['probe'].with_reference(sample['frame'],sample['signature']))
        cached['at'] = time.monotonic()
        self.view_cache = cached
        self.floor_cache = cached['floor']
        self.cache_identity = (sample['game'].hwnd,sample['game'].pid)
        self.scale_hint = float(np.linalg.norm(cached['matrix'][0,:2]))
        return True

    def save_sample_async(self, sample):
        if self.diagnostic_busy or time.monotonic()-self.last_map_saved < 3: return
        self.diagnostic_busy = True
        def save():
            try: self.save_map_sample(sample['frame'], sample['presence'])
            finally: self.diagnostic_busy = False
        threading.Thread(target=save, daemon=True).start()

    def start_match(self):
        if self.matching or self.warming or self.pending is None or not self.active: return
        sample, self.pending = self.pending, None
        self.matching = True
        layout = self.app.layout_limit.get()
        layout = None if layout == '自动' else 's1_'+layout
        floor = self.app.floor_limit.get()
        floor = None if floor == '自动' else floor
        style = 'structure' if self.app.overlay_style.get() == '结构' else 'original_clean'
        identity = (sample['game'].hwnd, sample['game'].pid)
        cached = self.floor_cache if self.cache_identity == identity else None
        scale_hint = self.scale_hint if self.cache_identity == identity else None
        visited = self.floor_visits.get(sample.get('selected_floor'))
        if not self.valid_visit(visited,sample,300):
            visited = self.view_cache if self.valid_visit(self.view_cache,sample,300) else None
        if visited and (not visited.get('unique_layout') or not visited.get('selected_floor')):
            visited = None
        if cached and ((layout and layout != cached['layout_id']) or (floor and floor != cached['floor'])):
            cached = None

        def match():
            began = time.monotonic()
            try:
                observed = sample.get('selected_floor')
                if observed and not sample['manual']:
                    if floor and floor != observed:
                        result = dict(accepted=False, reason=f'当前显示 {observed}，与限定楼层 {floor} 不同。')
                    else:
                        result = self.matcher.locate_selected(sample['frame'], layout, observed, scale_hint, visited)
                else:
                    locate = self.matcher.locate if sample['manual'] else self.matcher.locate_live
                    result = (locate(sample['frame'], cached['layout_id'], cached['floor'])
                              if sample['manual'] and cached else None)
                    if result is None or not result['accepted']:
                        result = locate(sample['frame'], layout, floor)
                rendered = None
                if result['accepted']:
                    rendered = self.matcher.render_live(result, sample['frame'], style, sample['manual'])
                    result['layer_probe'] = LayerProbe(sample['frame'],rendered,sample['signature'],result.get('label_tiles',()))
                self.events.put(('matched', (sample, result, rendered, style, time.monotonic()-began)))
            except Exception as exc:
                self.events.put(('match_error', str(exc)))
        threading.Thread(target=match, daemon=True).start()

    def match_received(self, payload):
        sample, result, rendered, style, elapsed = payload
        self.matching = False
        game = sample['game']
        if (not self.active or self.suspended or self.hidden or not self.game or
                sample.get('revision', self.session_revision) != self.session_revision or
                (game.hwnd, game.pid) != (self.game.hwnd, self.game.pid)):
            return
        if (self.latest and self.latest['presence']['visible'] and
                sample.get('selected_floor') != self.latest.get('selected_floor')):
            self.start_match()
            return
        if result['accepted']:
            if (not self.valid_visit(self.view_cache,sample,300) or
                    self.view_cache['floor']['layout_id'] != result['floor']['layout_id']):
                self.floor_visits.clear()
            self.floor_cache, self.cache_identity = result['floor'], (game.hwnd, game.pid)
            self.scale_hint = float(result.get('scale', np.linalg.norm(result['matrix'][0,:2])))
            probe = result.get('layer_probe') or LayerProbe(sample['frame'],rendered,sample['signature'])
            self.view_cache = dict(at=time.monotonic(), game=game, roi=sample['roi'], revision=self.session_revision,
                                   selected_floor=sample.get('selected_floor'), label=result['floor']['id'],
                                   signature=sample['signature'], rendered=rendered, floor=result['floor'], probe=probe,
                                   matrix=np.array(result['matrix'],copy=True),
                                   unique_layout=len(result.get('possible_layouts',[]))==1,
                                   thumb=cv2.resize(sample['frame'], (320, 180)))
            if sample.get('selected_floor') == result['floor']['floor']:
                self.floor_visits[sample['selected_floor']] = self.view_cache
            self.note('layout_learned', layout=result['floor']['id'], elapsed=elapsed,
                      inliers=result['inliers'], capture_epoch=sample['epoch'])
            current = self.latest
            if (sample['epoch'] == self.generation and current and current['presence']['visible'] and
                    current['game'] == game and current['roi'] == sample['roi'] and
                    time.monotonic()-current['at'] < .30 and
                    same_view(current['signature'], sample['signature']) and base.foreground() == game.hwnd):
                self.locked_match = result
                self.display = dict(signature=sample['signature'], rendered=rendered, roi=sample['roi'],
                                    epoch=self.generation, game=game, selected_floor=sample.get('selected_floor'),
                                    label=result['floor']['id'], probe=probe)
                self.show_display()
                f = result['floor']
                self.note('overlay_shown' if self.overlay._visible else 'match_ready', f'布局 {f["layout_id"].split("_")[-1]} · {f["floor"]} · 快捷键触发 · {elapsed:.2f}秒',
                          layout=f['layout_id'], floor=f['floor'], inliers=result['inliers'],
                          matrix=result['matrix'].tolist(), error_px=result['error_px'], elapsed=elapsed,
                          style=style, capture_ms=sample['capture_ms'], selected_floor=sample.get('selected_floor'),
                          method=result.get('segmentation'), search=result.get('search','wide'),
                          agreement=result.get('consistency'), rival=result.get('rival'))
                # A queued identical frame no longer needs expensive work.
                if self.pending and same_view(self.pending['signature'], sample['signature']): self.pending = None
        elif sample['epoch'] == self.generation and self.latest and self.latest['presence']['visible']:
            self.note('map_unmatched', result.get('reason') or '当前局部结构尚不足以确认布局。',
                      selected_floor=sample.get('selected_floor'), elapsed=elapsed)
        self.start_match()

    def show_display(self):
        d, current = self.display, self.latest
        if (not d or not current or not self.active or self.hidden or self.sampling or
                d['epoch'] != self.generation or not current['presence']['visible'] or
                d.get('selected_floor') != current.get('selected_floor') or
                base.foreground() != d['game'].hwnd or base.refresh_game(d['game']) != d['game']): return
        if not d.get('uploaded') or not self.overlay.show_cached(d['roi']):
            self.set_alpha(self.app.overlay_alpha.get())
            self.overlay.label_tiles=d['probe'].label_tiles
            self.overlay.show_image(d['rendered'], d['roi'], image_is_rgb=False)
            d['uploaded'] = True
        if not self.overlay._native_layered_enabled: self.capture_excluded = False
        if self.overlay._visible and d.get('label'):
            self.note('overlay_shown', f'地图已匹配 · {d["label"]}', floor_id=d['label'],
                      selected_floor=d.get('selected_floor'), capture_ms=current['capture_ms'])

    def tick(self):
        if self.closed: return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'key':
                    if data == 'once': self.start(False)
                    elif data == 'auto': self.start(True)
                    elif data == 'hide': self.toggle_hidden()
                    elif data == 'stop': self.stop()
                elif kind == 'map_key': self.map_key(*data)
                elif kind == 'hook_ready':
                    self.hook_ready = data
                    self.note('input_observer_ready' if data else 'input_observer_unavailable')
                elif kind == 'templates_ready':
                    self.warming = False
                    self.due = 0
                    self.note('templates_ready', '图谱已准备好；按地图键立即触发检测。')
                    self.start_match()
                elif kind == 'templates_failed':
                    self.warming = False
                    self.stop(quiet=True)
                    self.note('templates_failed', '图谱加载失败：'+data)
                elif kind == 'hotkey_error': self.app.live_status.set(data)
                elif kind == 'sample': self.sample_received(data)
                elif kind == 'matched': self.match_received(data)
                elif kind == 'sample_discard': self.sampling = False
                elif kind == 'sample_error':
                    self.sampling = False
                    self.invalidate()
                    self.due = time.monotonic()+.5
                    self.note('capture_error', '截图重试中：'+data)
                elif kind == 'match_error':
                    self.matching = False
                    self.note('match_error', '匹配重试中：'+data)
        except queue.Empty: pass
        if self.active and not self.suspended:
            ready = self.sync_target()
            if ready and not self.hidden and not self.sampling and time.monotonic() >= self.due:
                self.launch()
        self.app.after(10, self.tick)

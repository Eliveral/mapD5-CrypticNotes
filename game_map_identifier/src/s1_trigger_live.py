"""Key-triggered capture bursts, independent matching and guarded projection."""
from __future__ import annotations
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from . import s1_live as base
from .s1_live_match import LiveMatcher, signature, same_view, map_area
from .s1_floor_transition import selected_floor
from .s1_layer_probe import LayerProbe
from .s1_view_motion import ViewFeatures,zoom_position,zoom_changed
from .s1_round_lock import RoundLock


class FloorChanged(Exception):
    pass


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
        self.view_anchors={}
        self.pose_task=None
        self.pose_due={}
        self.pose_fixes={}
        self.b1_due=0.
        self.round_lock=RoundLock()
        self.proof_job=None
        self.proof_due=0.
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
        if state == self.last_note_state and (state=='map_closed' or now-self.last_note_at < 1.5):
            return
        self.last_note_state, self.last_note_at = state, now
        super().note(state, message, round_layout=self.round_lock.layout,b2_votes=self.round_lock.count,**details)

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
        self.view_anchors={}
        self.pose_task=None
        self.pose_due={}
        self.pose_fixes={}
        self.b1_due=0.
        self.view_cache = self.floor_cache = self.scale_hint = self.cache_identity = None

    def valid_visit(self, visit, sample, lifetime):
        return bool(visit and visit['revision'] == self.session_revision and
                    time.monotonic()-visit['at'] <= lifetime and
                    visit['game'] == sample['game'] and visit['roi'] == sample['roi'])

    def new_round(self):
        self.session_revision+=1
        self.invalidate()
        self.clear_visits()
        self.round_lock=RoundLock()
        if self.game:self.round_lock.owner=(self.game.hwnd,self.game.pid)
        self.proof_job=None
        self.proof_due=self.due=0.
        self.app.round_status.set(self.round_lock.status())
        self.note('round_reset','已清除本局布局，等待新一局的负二。')

    def sync_target(self):
        focused=super().sync_target()
        if self.game:
            owner=(self.game.hwnd,self.game.pid)
            if self.round_lock.owner not in (None,owner):self.new_round()
            self.round_lock.owner=owner
        return focused

    def layout_limit(self,level=None):
        selected=self.app.layout_limit.get()
        return self.round_lock.preferred(level) if selected=='自动' else 's1_'+selected

    def check_pose(self,sample,visit,style):
        if (not isinstance(getattr(self,'matcher',None),LiveMatcher) or not visit or
                not visit.get('unique_layout') or visit.get('provisional') or sample.get('manual') or
                sample.get('selected_floor')!=visit['floor']['floor'] or visit['roi']!=sample['roi'] or
                visit['game']!=sample['game'] or visit['revision']!=self.session_revision):return
        fid=visit['floor']['id']
        pending=self.pose_fixes.get(fid)
        if pending and not self.valid_visit(pending,sample,30):self.pose_fixes.pop(fid)
        if (self.pose_task or fid in self.pose_fixes or time.monotonic()<self.pose_due.get(fid,0)):return
        task=object();self.pose_task=task
        self.pose_due[fid]=time.monotonic()+.85
        confirmed=visit['floor']['floor']!='B1' and self.round_lock.layout==visit['floor']['layout_id']
        def verify():
            anchor=None
            try:
                image=sample['frame'];clear=sample.get('clear_mask')
                probe=sample.get('observed_probe')
                if clear is None and probe is not None:
                    image=probe.uncovered_frame(image);clear=probe.unpainted
                hit=self.matcher.pose_check.check(image,map_area,clear,visit['floor'],visit['matrix'],style,confirmed)
                if hit:
                    anchor={k:visit[k] for k in ('game','revision','roi','floor')}
                    anchor.update(at=sample['at'],result=hit,matrix=hit['matrix'],signature=signature(image),
                                  view_features=ViewFeatures(image,map_area,clear))
            except Exception as exc:
                self.events.put(('pose_check_error',str(exc)))
            finally:self.events.put(('pose_checked',(task,anchor)))
        threading.Thread(target=verify,daemon=True).start()

    def pose_checked(self,task,anchor):
        if task is not self.pose_task:return
        self.pose_task=None
        if (not anchor or not self.active or anchor['revision']!=self.session_revision or
                anchor['game']!=self.game or self.layout_limit(anchor['floor']['floor']) not in (None,anchor['floor']['layout_id'])):return
        last=self.view_cache
        if not last or last['floor']['layout_id']!=anchor['floor']['layout_id']:return
        fid=anchor['floor']['id']
        self.pose_fixes[fid]=anchor
        anchor['shape']=cv2.resize(anchor['view_features'].geom,(40,24),interpolation=cv2.INTER_NEAREST)
        anchor['handoff_origin']=None
        self.view_anchors[fid]=(anchor,)
        self.floor_visits.pop(anchor['floor']['floor'],None)
        for v in [self.view_cache,*self.floor_visits.values(),*(v for views in self.view_anchors.values() for v in views)]:
            origin=v.get('handoff_origin') if v else None
            if origin and origin['floor']['id']==fid:v['handoff_origin']=None
        self.due=0
        self.note('pose_offset',floor_id=fid,shift_px=anchor['result']['shift_px'],
                  before_px=anchor['result']['before_px'],after_px=anchor['result']['after_px'])

    @staticmethod
    def apply_pose_fix(anchor,features,sig):
        hit=dict(anchor['result'])
        if zoom_changed(anchor['view_features'].zoom,features.zoom) or not same_view(anchor['signature'],sig):
            motion=anchor['view_features'].motion_to(features)
            if motion is None:return None
            hit['matrix']=motion[0]@np.vstack((hit['matrix'],[0,0,1]))
            hit['scale']=float(np.linalg.norm(hit['matrix'][0,:2]))
        return hit

    def begin_proof(self,sample):
        if ('b2_pixels' not in sample or self.proof_job or
                self.app.layout_limit.get()!='自动' or sample['at']<self.proof_due):return
        state=self.round_lock
        token=object()
        self.proof_job=token
        incumbent=state.layout
        interval=.25 if state.challenger or state.uncertain or state.b1_conflict else 2.5 if incumbent else .65
        self.proof_due=sample['at']+interval
        def verify():
            reference=None
            try:
                frame,clear=sample['b2_pixels']
                hit=self.matcher.confirm_b2(frame,clear,incumbent)
                if hit:
                    reference={k:sample[k] for k in ('at','revision','game','roi')}
                    reference.update(floor=hit['floor'],matrix=hit['matrix'],selected_floor='B2',
                                     unique_layout=True,provisional=False,signature=signature(frame),
                                     player_hd=self.matcher.handoffs.player_position(frame,hit['floor'],hit['matrix']),
                                     proof_kind=hit.get('proof_kind','full'),cells=hit.get('cells',()),
                                     conflicting_models=hit.get('conflicting_models',False),
                                     contradicts=hit.get('contradicts',False),view_features=ViewFeatures(frame,map_area,clear),
                                     inliers=hit['inliers'])
            except Exception as exc:
                self.events.put(('b2_check_error',str(exc)))
            finally:
                self.events.put(('b2_check',(state,token,reference)))
        threading.Thread(target=verify,daemon=True).start()

    def proof_received(self,state,token,reference):
        if token is not self.proof_job:return
        self.proof_job=None
        previous=state.layout
        old_b2=state.preferred('B2')
        if (state is not self.round_lock or not reference or not self.active or
                reference['revision']!=self.session_revision or reference['game']!=self.game or
                self.app.layout_limit.get()!='自动'):return
        changed=state.observe(reference)
        needs_pose=(not self.display or self.display.get('selected_floor')!='B2' or
                    not self.view_cache or self.view_cache['floor']['id']!=reference['floor']['id'])
        if not changed and (not needs_pose or state.uncertain or
                state.preferred('B2')!=reference['floor']['layout_id']):return
        self.app.round_status.set(state.status())
        if state.uncertain:
            self.invalidate();self.clear_visits()
            self.note('b2_conflict',state.status())
            return
        identity=state.preferred('B2')
        if identity:
            # Throw out unrelated guesses, including jobs already rendering
            # guesses from other floors, but retain verified same-layout poses.
            wrong_cache=self.view_cache and (self.view_cache['floor']['layout_id']!=identity or not self.view_cache.get('unique_layout'))
            wrong_display=self.display and not (self.display.get('label') or '').startswith(identity+'_')
            if wrong_cache or wrong_display or (old_b2 and old_b2!=identity):
                self.invalidate();self.clear_visits()
            self.floor_visits={key:v for key,v in self.floor_visits.items() if v['floor']['layout_id']==identity and v.get('unique_layout')}
            self.view_anchors={key:v for key,v in self.view_anchors.items() if key.startswith(identity+'_')}
            if 'view_features' in reference and (needs_pose or wrong_cache or wrong_display or state.challenger):
                anchor=dict(reference)
                mat=anchor['matrix']
                anchor['result']=dict(floor=anchor['floor'],matrix=mat,scale=float(np.linalg.norm(mat[0,:2])),
                    accepted=True,inliers=reference['inliers'],error_px=None,search='b2_recheck',segmentation='b2_recheck',
                    possible_layouts=[identity])
                self.pose_fixes[anchor['floor']['id']]=anchor
            self.due=0.
        if state.challenger or state.b1_conflict:
            self.proof_due=min(self.proof_due,time.monotonic()+.25)
            self.display=self.pending=None
            if self.overlay and self.overlay._visible:self.overlay.hide()
        if state.layout and state.layout!=previous:
            threading.Thread(target=self.matcher.prepare_paired_floor,args=(state.layout,),daemon=True).start()
        event=('round_rechecking' if state.challenger else 'round_corrected' if previous and previous!=state.layout
               else 'round_locked' if state.layout else 'round_confirming')
        self.note(event,state.status(),layout=identity,confirmations=state.proposal_count if state.challenger else state.count)

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
        shown = getattr(self,'painted',None) if self.overlay._visible and not self.capture_excluded else None
        probe = shown.get('probe') if shown else None
        opacity = self.app.overlay_alpha.get()/100
        if self.overlay._visible and not self.capture_excluded and probe is None:
            self.overlay.hide()
        game, roi, epoch = self.game, self.capture_bounds(), self.generation
        revision = self.session_revision
        manual = self.roi is not None
        burst = time.monotonic() < self.burst_until
        last=self.view_cache
        anchors=dict(getattr(self,'view_anchors',{}))
        fixes=dict(self.pose_fixes)
        locked=self.round_lock.layout if self.app.layout_limit.get()=='自动' else None
        automatic=self.app.layout_limit.get()=='自动'
        override=self.layout_limit()
        proposal=self.round_lock.proposal if automatic else None
        known_b2=proposal or (None if self.round_lock.b1_conflict else self.round_lock.reference)
        known_visit=self.floor_visits.get('B2')
        pair_locked=locked if automatic and override==locked and not (
            self.round_lock.uncertain or self.round_lock.challenger or self.round_lock.b1_conflict) else None
        b1_visit=self.floor_visits.get('B1')
        recheck_b2=automatic and self.round_lock.b1_conflict
        disputed=automatic and bool(self.round_lock.uncertain or self.round_lock.challenger or self.round_lock.b1_conflict)
        lifetime=float('inf') if locked else 300
        if locked and (not self.valid_visit(last,dict(game=game,roi=roi),lifetime) or last['floor']['layout_id']!=locked):
            last=self.round_lock.reference
        if (manual or not self.valid_visit(last,dict(game=game,roi=roi),lifetime)
                or not last.get('selected_floor')):last=None
        floor_limit=self.app.floor_limit.get()
        verify=(not manual and not self.warming and not self.proof_job and
                self.app.layout_limit.get()=='自动' and time.monotonic()>=self.proof_due)
        style='structure' if self.app.overlay_style.get()=='结构' else 'original_clean'
        b1_due=self.b1_due
        self.sampling = True
        last_hint=last

        def grab():
            began = time.monotonic()
            last=last_hint
            try:
                if base.foreground() != game.hwnd or base.refresh_game(game) != game:
                    self.events.put(('sample_discard', None)); return
                frame = base.capture(roi)
                if base.foreground() != game.hwnd:
                    self.events.put(('sample_discard', None)); return
                presence = (dict(visible=True, manual_crop=True) if manual else
                            self.visibility.inspect(frame, fast=burst))
                floor = selected_floor(frame) if not manual else None
                wanted=proposal['floor']['layout_id'] if floor=='B2' and proposal else override
                independent_b1=automatic and not manual and floor=='B1'
                if independent_b1:
                    wanted=None
                    last=b1_visit if self.valid_visit(b1_visit,dict(game=game,roi=roi),300) else None
                if floor=='B2' and not manual:
                    last=next((v for v in (last,known_b2,known_visit) if v and v.get('selected_floor')=='B2' and
                               self.valid_visit(v,dict(game=game,roi=roi),lifetime) and wanted in (None,v['floor']['layout_id'])),None)
                    if recheck_b2 and not proposal:last=None
                if floor=='B3' and pair_locked and (not last or last['selected_floor']!='B3'):
                    # A visit to B1 or a delayed proof must not eclipse the latest B2 camera.
                    paired=[v for v in (last,known_visit,known_b2) if v and v.get('selected_floor')=='B2'
                            and v['floor']['layout_id']==pair_locked and v.get('unique_layout')
                            and not v.get('provisional') and self.valid_visit(v,dict(game=game,roi=roi),lifetime)]
                    if paired:last=max(paired,key=lambda v:v.get('sample_at',v['at']))
                scores=presence.get('scores',{})
                # A single weak HUD button is common during map animations.
                # Only bridge a recently confirmed, still-selected map. A close
                # shortcut invalidates display and two missing controls close immediately.
                if (not presence['visible'] and self.display and floor and
                        max(scores.values(),default=0)>.52 and time.monotonic()-getattr(self,'last_strong_map',0)<.20):
                    presence=dict(presence,visible=True,bridged=True)
                positive_at=time.monotonic()
                zoom=zoom_position(frame) if presence['visible'] and not manual else None
                changed_zoom=bool(probe and zoom_changed(probe.zoom,zoom))
                stationary = bool(probe and presence['visible'] and not changed_zoom and probe.stationary(frame, opacity))
                changing=bool(shown and floor!=shown.get('selected_floor'))
                check_b1=bool(independent_b1 and presence['visible'] and not self.warming and
                              (time.monotonic()>=b1_due or changing or (probe and not stationary) or
                               not last or last.get('selected_floor')!='B1' or
                               (not probe and not self.display)))
                fix=fixes.get(last['floor']['layout_id']+'_'+floor) if last and floor else None
                if fix is not None and not self.valid_visit(fix,dict(game=game,roi=roi),lifetime):fix=None
                clear=None
                if presence['visible'] and probe and (changing or not stationary or fix is not None or check_b1):
                    frame=probe.uncovered_frame(frame)
                    clear=probe.unpainted
                sig=signature(frame) if presence['visible'] else None
                sample=dict(epoch=epoch,game=game,roi=roi,frame=frame,presence=presence,
                    zoom=zoom,zoom_changed=changed_zoom,signature=sig,selected_floor=floor,probing=probe is not None,stationary=stationary,clear_mask=clear,observed_probe=probe,
                    at=positive_at,manual=manual,revision=revision,capture_ms=round((positive_at-began)*1000,1))
                if verify and presence['visible'] and floor=='B2':
                    sample['b2_pixels']=(probe.uncovered_frame(frame),probe.unpainted) if probe and clear is None else (frame,clear)
                if disputed and presence['visible'] and floor in ('B2','B3'):
                    self.events.put(('sample',sample))
                    return
                if stationary and last and floor==last['selected_floor'] and not changing:
                    point=self.matcher.handoffs.player_position(frame,last['floor'],last['matrix'])
                    sample['player_update']=(last,point)
                origin=last
                if last and floor!=last['selected_floor'] and last.get('provisional'):origin=None
                if last and floor==last['selected_floor']:
                    moving=not stationary if probe else not same_view(sig,last['signature'])
                    origin=last.get('handoff_origin') if moving or last.get('provisional') else None
                result=None
                if check_b1 and floor_limit in ('自动','B1'):
                    result=self.matcher.review_b1(frame)
                    sample['b1_checked']=True
                if (not check_b1 and presence['visible'] and fix and wanted in (None,fix['floor']['layout_id']) and floor_limit in ('自动',floor)):
                    features=ViewFeatures(frame,map_area,clear)
                    sample['view_features']=features
                    result=self.apply_pose_fix(fix,features,sig)
                    if result:sample['applied_pose_fix']=fix
                    else:sample['unusable_pose_fix']=fix
                if (not independent_b1 and not result and presence['visible'] and origin and floor and self.valid_visit(origin,sample,lifetime) and
                        floor!=origin['selected_floor'] and floor_limit in ('自动',floor)
                        and wanted in (None,origin['floor']['layout_id']) and not self.warming):
                    if pair_locked and floor=='B3' and origin['selected_floor']=='B2':
                        result=self.matcher.paired_b3(frame,origin,pair_locked,clear)
                    if result is None:result=self.matcher.handoff(frame,origin,floor)
                    if locked and result and result.get('provisional'):result=None
                track=(presence['visible'] and last and floor and not self.warming and
                       floor_limit in ('自动',floor) and wanted in (None,last['floor']['layout_id']) and
                       (floor!=last['selected_floor'] or (not stationary if probe else not same_view(sig,last['signature']))))
                if track and not result and not independent_b1:
                    features=ViewFeatures(frame,map_area,clear)
                    sample['view_features']=features
                    fid=last['floor']['layout_id']+'_'+floor
                    allowed=[v for v in anchors.get(fid,()) if self.valid_visit(v,sample,lifetime)]
                    source=next((v for v in anchors.get(last['floor']['id'],()) if self.valid_visit(v,sample,lifetime)),None)
                    if changed_zoom and floor==last['selected_floor'] and allowed:
                        # Try the most recent view on untouched pixels first. If the
                        # old layer covers the resized rooms, promptly recapture cleanly.
                        detailed=max(allowed,key=lambda v:v['matrix'][0,0])
                        recent=allowed[:1]
                        if detailed is not recent[0]:recent.append(detailed)
                        result=self.matcher.follow_anchors(features,last['floor'],recent)
                    else:
                        result=self.matcher.follow_known(features,last,floor,allowed,frame,source,locked)
                if (result and not self.display and not probe and pair_locked and floor in ('B2','B3')
                        and result.get('search') in ('view_motion','view_zoom')):
                    corrected=self.matcher.pose_check.check(frame,map_area,clear,result['floor'],result['matrix'],style,True)
                    if corrected:
                        sample['first_pose_fix']=True
                        result=corrected
                if result:
                    pixels=self.matcher.render_live(result,frame,style)
                    result['layer_probe']=LayerProbe(frame,pixels,sig,result.get('label_tiles',()),clear)
                    result['view_features']=sample.get('view_features') or ViewFeatures(frame,map_area,clear)
                    # Rendering at desktop resolution can exceed the freshness
                    # limit. Re-read the game instead of dropping an otherwise
                    # valid new-floor result every time this happens.
                    if time.monotonic()-positive_at>.075:
                        if base.foreground()!=game.hwnd:
                            self.events.put(('sample_discard',None));return
                        fresh=base.capture(roi)
                        check=self.visibility.inspect(fresh,fast=True)
                        new_floor=selected_floor(fresh)
                        checked=time.monotonic()
                        if probe:
                            fresh=probe.uncovered_frame(fresh)
                            clear=probe.unpainted
                            sample['clear_mask']=clear
                        new_zoom=zoom_position(fresh) if check['visible'] and not manual else None
                        new_sig=signature(fresh) if check['visible'] else None
                        sample.update(frame=fresh,presence=check,selected_floor=new_floor,signature=new_sig,
                                      checked_at=checked,stationary=False,zoom=new_zoom,
                                      zoom_changed=bool(probe and zoom_changed(probe.zoom,new_zoom)))
                        sample.pop('view_features',None);sample.pop('player_update',None)
                        if not check['visible'] or floor!=new_floor:result=None
                        elif zoom_changed(zoom,new_zoom) or not (np.array_equal(sig,new_sig) or same_view(sig,new_sig)):
                            now_features=ViewFeatures(fresh,map_area,clear)
                            moved=result['view_features'].motion_to(now_features)
                            if moved is None:result=None
                            else:
                                result['matrix']=moved[0]@np.vstack((result['matrix'],[0,0,1]))
                                for row in result.get('common_floors',()):
                                    row['matrix']=moved[0]@np.vstack((row['matrix'],[0,0,1]))
                                result['scale']=float(np.linalg.norm(result['matrix'][0,:2]))
                                result['view_features']=now_features
                                pixels=self.matcher.render_live(result,fresh,style)
                                result['layer_probe']=LayerProbe(fresh,pixels,new_sig,result.get('label_tiles',()),clear)
                    if result:sample['handoff']=(result,pixels,style,time.monotonic()-positive_at)
                self.events.put(('sample',sample))
            except Exception as exc:
                self.events.put(('sample_error', str(exc)))
        if not getattr(self,'capture_worker',None):
            self.capture_worker=ThreadPoolExecutor(max_workers=1,initializer=base.prepare_capture_thread)
        self.capture_worker.submit(grab)

    def close(self):
        super().close()
        worker=getattr(self,'capture_worker',None)
        if worker:
            worker.submit(base.release_capture_thread)
            worker.shutdown(wait=False)
            self.capture_worker=None

    def sample_received(self, sample):
        self.sampling = False
        if (not self.active or self.hidden or self.suspended or sample['game'] != self.game or
                base.foreground() != self.target or sample['epoch'] != self.generation or
                sample.get('revision', self.session_revision) != self.session_revision):
            self.due = 0
            return
        if sample['presence']['visible'] and sample.get('selected_floor')=='B2':self.begin_proof(sample)
        if sample.get('b1_checked'):
            self.b1_due=time.monotonic()+.85
            if not sample.get('handoff') and sample.get('selected_floor')=='B1' and sample['presence']['visible']:
                self.latest=sample
                self.display=self.pending=None
                if self.overlay and self.overlay._visible:self.overlay.hide()
                self.floor_visits.pop('B1',None)
                if self.view_cache and self.view_cache.get('selected_floor')=='B1':self.view_cache=None
                self.view_anchors={k:v for k,v in self.view_anchors.items() if not k.endswith('_B1')}
                self.due=time.monotonic()+.06
                self.note('b1_unresolved','当前负一画面尚不能确定位置，等待更多固定墙线。')
                return
        handoff=sample.get('handoff')
        if handoff:self.check_b1_conflict(sample,handoff[0])
        unusable=sample.get('unusable_pose_fix')
        if unusable:
            fid=unusable['floor']['id']
            if self.pose_fixes.get(fid) is unusable:self.pose_fixes.pop(fid)
        if handoff and not self.allowed_result(handoff[0]):sample.pop('handoff')
        previous = self.latest
        now=time.monotonic()
        if sample.get('player_update'):
            visit,point=sample['player_update']
            if visit is self.view_cache:
                visit['player_hd']=point
                visit['at']=sample['at']
        if sample['presence']['visible'] and not sample.get('manual') and not sample.get('selected_floor') and self.display and self.display.get('selected_floor'):
            since=self.display.setdefault('uncertain_floor_since',now)
            if now-since<.16:
                self.due=now+.012
                return
        elif self.display:self.display.pop('uncertain_floor_since',None)
        if sample['presence']['visible'] and not sample['presence'].get('bridged'):
            self.last_strong_map=sample['at']
        if sample['presence']['visible'] and (not previous or not previous['presence']['visible']
                or previous.get('selected_floor')!=sample.get('selected_floor')):
            self.open_detected_at=sample['at']
        if (previous and previous['presence']['visible'] and sample['presence']['visible'] and
                previous.get('selected_floor') != sample.get('selected_floor')):
            self.generation += 1
            sample['epoch'] = self.generation
            self.display = self.pending = None
            if self.overlay and self.overlay._visible and not sample.get('handoff'):
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
        if self.app.layout_limit.get()=='自动' and sample.get('selected_floor')!='B1' and (self.round_lock.uncertain or
                self.round_lock.b1_conflict or self.round_lock.challenger):
            if self.overlay and self.overlay._visible:self.overlay.hide()
            self.display=self.pending=None
            self.note('waiting_b2_review',self.round_lock.status())
            return
        if sample.get('handoff'):
            self.pending=None
            self.accept_match((sample,*sample['handoff']))
            return
        if sample.get('probing') and sample.get('zoom_changed'):
            self.display=self.pending=None
            if self.overlay and self.overlay._visible:self.overlay.hide()
            self.due=0.
            return
        if sample.get('probing'):
            if self.display and self.display.get('selected_floor') == sample.get('selected_floor'):
                if sample['stationary'] and not self.display.get('provisional'):
                    self.check_pose(sample,self.view_cache,'structure' if self.app.overlay_style.get()=='结构' else 'original_clean')
                    self.display.pop('unsteady_since',None)
                    if not self.display.get('uploaded'):self.show_display()
                    self.due = now+.06
                    return
            # Registration only sees screen pixels the old layer did not paint.
            # Keep the window up and swap pixels once the next pose is ready.
            if self.display:self.display.setdefault('unsteady_since',now)
        if not sample.get('probing') and self.display and not self.display.get('provisional') and not zoom_changed(self.display.get('zoom'),sample.get('zoom')) and same_view(sample['signature'], self.display['signature']):
            self.show_display()
            self.check_pose(sample,self.view_cache,'structure' if self.app.overlay_style.get()=='结构' else 'original_clean')
            self.due = now+.06
        elif not sample.get('probing') and self.reuse_view(sample):
            self.show_display()
            self.check_pose(sample,self.view_cache,'structure' if self.app.overlay_style.get()=='结构' else 'original_clean')
            self.due = now+.06
        else:
            if self.display and self.overlay and self.overlay._visible:
                since=self.display.setdefault('unsteady_since',now)
                if now-since>.4:
                    self.note('view_lost','地图仍展开，正在寻找可对齐的固定墙线。',selected_floor=sample.get('selected_floor'))
                    self.overlay.hide();self.display=None
            self.pending = sample  # Bounded latest-positive slot, never an unbounded frame queue.
            self.start_match()
        if not sample.get('probing'):self.save_sample_async(sample)

    def reuse_view(self, sample):
        cached = self.floor_visits.get(sample.get('selected_floor'), self.view_cache)
        if cached and not self.allowed_result(dict(floor=cached['floor'],possible_layouts=cached.get('possible_layouts'))):
            return False
        if (not self.valid_visit(cached,sample,30) or cached.get('provisional') or
                cached['floor']['id'] in self.pose_fixes or
                zoom_changed(cached.get('zoom'),sample.get('zoom')) or
                self.layout_limit(sample.get('selected_floor')) not in (None,cached['floor']['layout_id']) or
                (self.round_lock.layout and self.app.layout_limit.get()=='自动' and not cached.get('unique_layout')) or
                sample.get('selected_floor') != cached.get('selected_floor') or
                sample['game'] != cached['game'] or sample['roi'] != cached['roi'] or
                not same_view(sample['signature'], cached['signature'])): return False
        # Check the fresh pixels too, never blindly reuse the previous matrix
        # just because a keyboard toggle arrived or a layout ID is cached.
        thumb = cv2.resize(sample['frame'], (320, 180))
        area = map_area(thumb) > 0
        delta = np.abs(thumb.astype(np.int16)-cached['thumb'].astype(np.int16))
        if float(delta[area].mean()) > 2.0 or float(np.mean(np.max(delta, axis=2)[area] > 18)) > .02:
            if not cached['probe'].stationary(sample['frame'],self.app.overlay_alpha.get()/100):return False
        self.display = dict(signature=sample['signature'], rendered=cached['rendered'],
                            zoom=sample.get('zoom'),roi=sample['roi'], epoch=self.generation, game=sample['game'],
                            selected_floor=sample.get('selected_floor'), label=cached.get('label'),
                            summary=cached.get('summary'),possible_layouts=cached.get('possible_layouts'),
                            common_only=bool(cached.get('alternatives')),matrix=cached['matrix'].tolist(),
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
        layout=self.layout_limit(sample.get('selected_floor'))
        automatic=self.app.layout_limit.get()=='自动'
        floor = self.app.floor_limit.get()
        floor = None if floor == '自动' else floor
        style = 'structure' if self.app.overlay_style.get() == '结构' else 'original_clean'
        identity = (sample['game'].hwnd, sample['game'].pid)
        cached = self.floor_cache if self.cache_identity == identity else None
        scale_hint = self.scale_hint if self.cache_identity == identity else None
        visited = self.floor_visits.get(sample.get('selected_floor'))
        priors=[self.view_cache,visited,self.round_lock.proposal,*sorted(self.floor_visits.values(),key=lambda v:-v['at']),self.round_lock.reference]
        lifetime=float('inf') if layout and layout==self.round_lock.layout else 300
        visited=next((v for v in priors if self.valid_visit(v,sample,lifetime) and layout in (None,v['floor']['layout_id'])
                      and v.get('unique_layout') and not v.get('provisional') and v.get('selected_floor')),None)
        if cached and ((layout and layout != cached['layout_id']) or (floor and floor != cached['floor'])):
            cached = None

        def match():
            began = time.monotonic()
            def guard():
                current=self.latest
                if (not self.active or self.closed or
                    sample.get('revision',self.session_revision)!=self.session_revision or
                    (automatic and layout!=self.round_lock.preferred(sample.get('selected_floor'))) or
                    (current and current['presence']['visible'] and
                     current.get('selected_floor')!=sample.get('selected_floor'))):
                    raise FloorChanged()
            try:
                self.matcher.search_guard=guard
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
                    guard()
                    rendered = self.matcher.render_live(result, sample['frame'], style, sample['manual'])
                    result['layer_probe'] = LayerProbe(sample['frame'],rendered,sample['signature'],result.get('label_tiles',()),sample.get('clear_mask'))
                    if observed and not sample['manual']:
                        result['view_features']=sample.get('view_features') or ViewFeatures(sample['frame'],map_area,sample.get('clear_mask'))
                self.events.put(('matched', (sample, result, rendered, style, time.monotonic()-began)))
            except FloorChanged:
                self.events.put(('search_superseded',None))
            except Exception as exc:
                self.events.put(('match_error', str(exc)))
            finally:
                if self.matcher.search_guard is guard:self.matcher.search_guard=None
        threading.Thread(target=match, daemon=True).start()

    def match_received(self, payload):
        self.matching = False
        self.accept_match(payload)

    def check_b1_conflict(self,sample,result):
        if (self.app.layout_limit.get()!='自动' or not self.round_lock.layout or
                result['floor']['floor']!='B1' or sample['epoch']!=self.generation):return
        candidates=result.get('possible_layouts') or [result['floor']['layout_id']]
        if self.round_lock.layout not in candidates:
            self.round_lock.b1_conflict=True
            self.display=None
            if self.overlay and self.overlay._visible:self.overlay.hide()
            self.app.round_status.set(self.round_lock.status())

    def allowed_result(self,result):
        level=result['floor']['floor']
        candidates=result.get('possible_layouts') or [result['floor']['layout_id']]
        if self.app.layout_limit.get()=='自动' and not self.round_lock.permits_overlay(
                level,result['floor']['layout_id'],candidates):return False
        layout=self.layout_limit(level)
        if not layout:return True
        return (result['floor']['layout_id']==layout and
                all(v==layout for v in (result.get('possible_layouts') or ())) and
                not (self.round_lock.layout==layout and result.get('provisional')))

    def accept_match(self,payload):
        sample, result, rendered, style, elapsed = payload
        game = sample['game']
        if (not self.active or self.suspended or self.hidden or not self.game or
                sample.get('revision', self.session_revision) != self.session_revision or
                (game.hwnd, game.pid) != (self.game.hwnd, self.game.pid)):
            return
        if result['accepted']:self.check_b1_conflict(sample,result)
        if result['accepted'] and not self.allowed_result(result):
            self.due=0
            self.start_match()
            return
        if (self.latest and self.latest['presence']['visible'] and
                (sample.get('selected_floor') != self.latest.get('selected_floor') or
                 zoom_changed(sample.get('zoom'),self.latest.get('zoom')))):
            self.start_match()
            return
        if self.view_cache and self.view_cache.get('sample_at',0)>sample['at']:
            self.start_match()
            return
        if result['accepted']:
            fix=sample.get('applied_pose_fix')
            fid=result['floor']['id']
            proposed=self.pose_fixes.get(fid)
            if (not fix and proposed and proposed['result'].get('search')=='b2_recheck' and
                    result.get('search') not in ('view_motion','floor_handoff') and
                    same_view(proposed['signature'],sample['signature'])):
                # An ordinary independent match already registered these same
                # pixels. Its prepared bitmap can serve an identical short reopen.
                self.pose_fixes.pop(fid,None)
            if sample.get('first_pose_fix'):
                self.pose_fixes.pop(fid,None)
                self.view_anchors.pop(fid,None)
                self.pose_due[fid]=time.monotonic()+.85
            if fix and result.get('search') in ('wall_realign','b2_recheck'):
                if self.pose_fixes.get(fid) is fix:self.pose_fixes.pop(fid)
                self.view_anchors.pop(fid,None)
            alternatives=result.get('possible_layouts') or [result['floor']['layout_id']]
            summary=f"{result['floor']['floor']} 共通区域 · {len(alternatives)} 套候选" if result.get('common_floors') else None
            if result['floor']['floor']=='B1' and self.app.layout_limit.get()=='自动':
                self.b1_due=time.monotonic()+.85
                self.pose_fixes.pop(result['floor']['id'],None)
                if self.round_lock.layout and self.round_lock.layout not in alternatives:
                    self.round_lock.b1_conflict=True
                b1_note='负一候选 '+' / '.join(a[3:] for a in alternatives)
                self.app.round_status.set(b1_note+' · '+self.round_lock.status())
                if len(alternatives)>1:
                    summary='负一共通区域 · 候选 '+' / '.join(a[3:] for a in alternatives)
            else:self.app.round_status.set(self.round_lock.status())
            if result.get('passage'):summary=f"{result['floor']['floor']} · {result['passage']} · 继承定位"
            lifetime=float('inf') if self.round_lock.layout==result['floor']['layout_id'] else 300
            if (not self.valid_visit(self.view_cache,sample,lifetime) or
                    self.view_cache['floor']['layout_id'] not in alternatives):
                self.floor_visits.clear()
                self.view_anchors={}
            self.floor_cache, self.cache_identity = result['floor'], (game.hwnd, game.pid)
            self.scale_hint = float(result.get('scale', np.linalg.norm(result['matrix'][0,:2])))
            probe = result.get('layer_probe') or LayerProbe(sample['frame'],rendered,sample['signature'])
            self.view_cache = dict(at=time.monotonic(), sample_at=sample['at'], game=game, roi=sample['roi'], revision=self.session_revision,
                                   selected_floor=sample.get('selected_floor'), label=result['floor']['id'],summary=summary,
                                   zoom=sample.get('zoom'),signature=sample['signature'], rendered=rendered, floor=result['floor'], probe=probe,
                                   matrix=np.array(result['matrix'],copy=True),
                                   unique_layout=len(alternatives)==1,possible_layouts=alternatives,
                                   provisional=bool(result.get('provisional')),
                                   handoff_origin=result.get('handoff_origin'),
                                   alternatives=result.get('common_floors'),
                                   thumb=cv2.resize(sample['frame'], (320, 180)))
            handoffs=getattr(getattr(self,'matcher',None),'handoffs',None)
            if handoffs and not sample.get('manual'):
                self.view_cache['player_hd']=handoffs.player_position(sample['frame'],result['floor'],result['matrix'])
            if sample.get('selected_floor') == result['floor']['floor']:
                self.floor_visits[sample['selected_floor']] = self.view_cache
                features=result.get('view_features')
                if features is not None and len(alternatives)==1 and not result.get('provisional'):
                    fid=result['floor']['id']
                    anchor={k:self.view_cache[k] for k in ('game','revision','roi','at','matrix','handoff_origin')}
                    anchor['view_features']=features
                    anchor['independent']=result.get('search') not in ('view_zoom','view_motion')
                    existing=self.view_anchors.get(fid,())
                    # Reopens of the same room should not evict the other
                    # observed variants that can be used to reacquire later.
                    thumb=cv2.resize(features.geom,(40,24),interpolation=cv2.INTER_NEAREST)
                    anchor['shape']=thumb
                    def distinct(old):
                        other=old['shape'];union=np.count_nonzero((thumb>0)|(other>0))
                        return (np.max(np.abs(anchor['matrix']-old['matrix']))>14 or
                                np.count_nonzero(thumb!=other)>.18*max(union,1))
                    kept=[v for v in existing if distinct(v)]
                    detailed=max((v for v in kept if v.get('independent') or v.get('result')),
                                 key=lambda v:v['matrix'][0,0],default=None)
                    if detailed is not None and not any(v is detailed for v in kept[:5]):
                        kept=kept[:4]+[detailed]
                    self.view_anchors[fid]=(anchor,*kept[:5])
            self.check_pose(sample,self.view_cache,style)
            self.note('common_floor' if result.get('common_floors') else 'layout_learned', layout=result['floor']['id'], possible_layouts=alternatives, elapsed=elapsed,
                      inliers=result['inliers'], capture_epoch=sample['epoch'])
            current = self.latest
            if (sample['epoch'] == self.generation and current and current['presence']['visible'] and
                    current['game'] == game and current['roi'] == sample['roi'] and
                    time.monotonic()-current.get('checked_at',current['at']) < .30 and
                    (current is sample or same_view(current['signature'], sample['signature'])) and base.foreground() == game.hwnd):
                self.locked_match = result
                self.display = dict(zoom=sample.get('zoom'),signature=sample['signature'], rendered=rendered, roi=sample['roi'],
                                    epoch=self.generation, game=game, selected_floor=sample.get('selected_floor'),
                                    label=result['floor']['id'],summary=summary,provisional=result.get('provisional',False),
                                    search=result.get('search'),passage=result.get('passage'),probe=probe,
                                    possible_layouts=alternatives,common_only=bool(result.get('common_floors')),
                                    matrix=result['matrix'].tolist(),shift_px=result.get('shift_px'))
                self.show_display()
                f = result['floor']
                description=summary or f'布局 {f["layout_id"].split("_")[-1]} · {f["floor"]}'
                self.note('overlay_shown' if self.overlay._visible else 'match_ready', f'{description} · {elapsed:.2f}秒',
                          layout=f['layout_id'], floor=f['floor'], inliers=result['inliers'],
                          matrix=result['matrix'].tolist(), error_px=result['error_px'], elapsed=elapsed,
                          style=style, capture_ms=sample['capture_ms'], selected_floor=sample.get('selected_floor'),
                          method=result.get('segmentation'), search=result.get('search','wide'),
                          agreement=result.get('consistency'), rival=result.get('rival'))
                # A queued identical frame no longer needs expensive work.
                if self.pending and same_view(self.pending['signature'], sample['signature']): self.pending = None
        elif sample['epoch'] == self.generation and self.latest and self.latest['presence']['visible']:
            reason=result.get('reason') or '当前局部结构尚不足以确认布局。'
            if self.round_lock.layout and self.app.layout_limit.get()=='自动':
                reason=self.round_lock.status()+' · 正在核对当前画面的位置。'
            self.note('map_unmatched', reason,
                      selected_floor=sample.get('selected_floor'), elapsed=elapsed)
        self.start_match()

    def show_display(self):
        d, current = self.display, self.latest
        if (not d or not current or not self.active or self.hidden or self.sampling or
                d['epoch'] != self.generation or not current['presence']['visible'] or
                zoom_changed(d.get('zoom'),current.get('zoom')) or
                d.get('selected_floor') != current.get('selected_floor') or
                base.foreground() != d['game'].hwnd or base.refresh_game(d['game']) != d['game']): return
        label=d.get('label')
        if self.app.layout_limit.get()=='自动' and label:
            layout,level=label.rsplit('_',1)
            if not self.round_lock.permits_overlay(level,layout,d.get('possible_layouts') or [layout]):
                self.display=None
                if self.overlay._visible:self.overlay.hide()
                return
        if not d.get('uploaded') or (not self.overlay._visible and not self.overlay.show_cached(d['roi'])):
            self.set_alpha(self.app.overlay_alpha.get())
            self.overlay.label_tiles=d['probe'].label_tiles
            self.overlay.show_image(d['rendered'], d['roi'], image_is_rgb=False)
            d['uploaded'] = True
        self.painted=d
        if not self.overlay._native_layered_enabled: self.capture_excluded = False
        if self.overlay._visible and d.get('label') and not d.get('reported'):
            d['reported']=True
            opening=getattr(self,'open_detected_at',current['at'])
            first=opening!=getattr(self,'reported_opening',None)
            self.reported_opening=opening
            text=d.get('summary') or d['label']
            self.note('overlay_shown', f'地图已匹配 · {text}', floor_id=d['label'],
                      selected_floor=d.get('selected_floor'), capture_ms=current['capture_ms'],
                      search=d.get('search'),passage=d.get('passage'),matrix=d.get('matrix'),shift_px=d.get('shift_px'),zoom=d.get('zoom'),
                      possible_layouts=d.get('possible_layouts'),common_only=d.get('common_only',False),
                      first_for_open=first,frame_to_show_ms=round((time.monotonic()-current['at'])*1000),
                      positive_to_show_ms=round((time.monotonic()-opening)*1000))

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
                elif kind == 'b2_check':self.proof_received(*data)
                elif kind == 'b2_check_error':self.note('b2_check_error','负二核对稍后重试。',error=data)
                elif kind == 'pose_checked':self.pose_checked(*data)
                elif kind == 'pose_check_error':self.note('pose_check_error','墙线校准稍后重试。',error=data)
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
                elif kind == 'search_superseded':
                    self.matching=False;self.start_match()
                elif kind == 'match_error':
                    self.matching = False
                    self.note('match_error', '匹配重试中：'+data)
        except queue.Empty: pass
        if self.active and not self.suspended:
            ready = self.sync_target()
            if ready and not self.hidden and not self.sampling and time.monotonic() >= self.due:
                self.launch()
        self.app.after(10, self.tick)

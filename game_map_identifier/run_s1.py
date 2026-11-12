"""S1 extension entry point. The upstream entry point and models remain intact."""
import sys
from src.s1_updates import map_root
from src.s1_window import main as s1_main
from src.app import main as legacy_main

if __name__ == '__main__':
    if '--self-check-input' in sys.argv:
        import ctypes,json,time
        from ctypes import wintypes as W
        from src.s1_raw_mouse import RawMouse,Device
        from src.s1_program_update import APP_TAG
        from src.config import project_root
        u=ctypes.WinDLL('user32',use_last_error=True)
        u.GetRegisteredRawInputDevices.argtypes=[ctypes.POINTER(Device),ctypes.POINTER(W.UINT),W.UINT]
        u.GetRegisteredRawInputDevices.restype=W.UINT
        u.DispatchMessageW.argtypes=[ctypes.POINTER(W.MSG)]
        u.DispatchMessageW.restype=ctypes.c_ssize_t
        def registered():
            devices=(Device*32)();n=W.UINT(32)
            count=u.GetRegisteredRawInputDevices(devices,ctypes.byref(n),ctypes.sizeof(Device))
            assert count!=0xffffffff
            return [(d.usage_page,d.usage,d.flags,d.target) for d in list(devices)[:count]]
        before=registered();observer=RawMouse(lambda *args:None)
        try:
            observer.enable(True);during=registered()
            assert (1,2,0x100,observer.hwnd) in during
            msg=W.MSG();end=time.monotonic()+.3
            while time.monotonic()<end:
                for _ in range(256):
                    if not u.PeekMessageW(ctypes.byref(msg),None,0,0,1):break
                    u.DispatchMessageW(ctypes.byref(msg))
                observer.flush_move();time.sleep(.008)
        finally:observer.close()
        after=registered();assert before==after
        (project_root()/'s1-input-check.json').write_text(json.dumps(dict(version=APP_TAG,
            backend='raw_input',before=before,during=during,after=after,passed=True,
            scope='Native asynchronous mouse registration and cleanup; game rotation requires user test'),indent=2),'utf8')
        raise SystemExit(0)
    if '--self-check-transition' in sys.argv:
        import json,time,cv2,numpy as np,hashlib
        from pathlib import Path
        from src.config import project_root
        from src.s1_catalog import S1Catalog,read_image
        from src.s1_live_match import LiveMatcher
        from src.s1_program_update import APP_TAG
        path=Path(sys.argv[sys.argv.index('--self-check-transition')+1])
        evidence=json.loads(path.read_text('utf8'))
        target=path.with_name(path.stem+'-target.png')
        if not target.is_file():target=path.with_name('target.png')
        if evidence.get('files'):
            assert hashlib.sha256(target.read_bytes()).hexdigest()==evidence['files'][target.name]
        frame=read_image(target);matcher=LiveMatcher(S1Catalog(map_root(project_root())))
        floor=matcher.handoffs.floors[evidence['source']]
        visit=dict(floor=floor,matrix=np.asarray(evidence['source_matrix'])*evidence['target_scale'],
            player_hd=np.asarray(evidence['player_hd']) if evidence.get('player_hd') is not None else None,
            zoom=evidence['source_zoom'],at=evidence['source_at'],sample_at=evidence['source_at'],
            unique_layout=True,provisional=False)
        matcher.prepare_paired_floor(floor['layout_id'])
        cv2.setNumThreads(2);audit={};began=time.perf_counter()
        if '--independent-review' in sys.argv:
            scale=matcher.review_native_scale(visit,evidence['target_zoom'])
            audit['native_scale']=scale
            result=(matcher.review_landing(frame,incumbent=floor['layout_id'],native_scale=scale)
                    if evidence['target']=='B3' else matcher.confirm_b2(frame,incumbent=floor['layout_id'],native_scale=scale))
            audit['conflicting_models']=bool(result and result.get('conflicting_models'))
            audit['inliers']=result and result.get('inliers')
        else:
            result=matcher.shared_floor(frame,visit,evidence['target'],zoom=evidence['target_zoom'],continuous=False,audit=audit)
        report=dict(version=APP_TAG,accepted=bool(result),floor=result and result['floor']['id'],
            matrix=result and result['matrix'].tolist(),camera_mode=result and result.get('camera_mode'),
            ms=round((time.perf_counter()-began)*1000),audit=audit,
            scope='Saved actual input and trusted source context; not a new live gameplay run')
        (project_root()/'s1-transition-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        raise SystemExit(0)
    if '--self-check-capture' in sys.argv:
        import json, hashlib, cv2
        from pathlib import Path
        from src.config import project_root
        from src.s1_catalog import S1Catalog, read_image
        from src.s1_live_match import LiveMatcher,map_area
        from src.s1_view_motion import ViewFeatures,zoom_position,zoom_changed
        from src.s1_floor_transition import selected_floor
        from src.s1_visibility import MapVisibility
        from src.s1_round_lock import RoundLock,direct_pose
        cv2.setNumThreads(2)
        path=Path(sys.argv[sys.argv.index('--self-check-capture')+1])
        c=S1Catalog(map_root(project_root()))
        frame=read_image(path)
        presence=MapVisibility(c.root/'ui').inspect(frame)
        selected = selected_floor(frame) if presence['visible'] else None
        matcher = LiveMatcher(c)
        lock=RoundLock()
        prior=None
        for i,arg in enumerate(sys.argv):
            if arg=='--enter-floor':
                level=sys.argv[i+1]
                if level not in ('B1','B2','B3','B4'):raise ValueError('无效初始楼层')
                lock.observe_opening(level,i)
                continue
            if arg=='--landing-frame':
                image=read_image(Path(sys.argv[i+1]))
                if selected_floor(image)=='B3' and MapVisibility(c.root/'ui').inspect(image)['visible']:
                    base=None if lock.blocks_floor('B2') else lock.layout
                    hit=matcher.review_landing(image,incumbent=lock.landing.layout or base,pending=lock.landing.pending)
                    if hit:lock.observe_landing(dict(hit,at=i))
                continue
            if arg!='--confirm-b2':continue
            image=read_image(Path(sys.argv[i+1]))
            if selected_floor(image)!='B2' or not MapVisibility(c.root/'ui').inspect(image)['visible']:continue
            proof=matcher.confirm_b2(image,incumbent=lock.layout,entry=lock.b2_arrival or lock.recheck_b2_after_landing)
            if proof:
                proof.update(at=i,unique_layout=True,player_hd=matcher.handoffs.player_position(image,proof['floor'],proof['matrix']))
                if lock.observe(proof):
                    prior,prior_image,prior_level=proof,image,'B2'
        if '--prior-capture' in sys.argv:
            explicit_image=read_image(Path(sys.argv[sys.argv.index('--prior-capture')+1]))
            explicit_level=selected_floor(explicit_image)
            if explicit_level and MapVisibility(c.root/'ui').inspect(explicit_image)['visible']:
                previous=matcher.locate_selected(explicit_image,lock.preferred(explicit_level),explicit_level)
                if previous['accepted'] and len(previous.get('possible_layouts',[]))==1:
                    prior_image,prior_level=explicit_image,explicit_level
                    prior=previous
                    prior['unique_layout']=True
                    prior['player_hd']=matcher.handoffs.player_position(prior_image,prior['floor'],prior['matrix'])
        landing_pose=None
        if '--shared-session' in sys.argv and prior and lock.shared_layout and selected and presence['visible']:
            prior.update(at=-2.,sample_at=-2.,zoom=zoom_position(prior_image))
            landing_pose=matcher.shared_floor(frame,prior,selected,zoom=zoom_position(frame),continuous=False)
        if '--shared-floor-click' in sys.argv and prior and lock.shared_layout and selected:
            from src.s1_map_camera import PointerSnapshot
            from src.s1_floor_buttons import selected_button
            button=selected_button(frame)
            if button:
                prior=dict(prior,at=-2.,sample_at=-2.,zoom=zoom_position(prior_image),
                           pointer=PointerSnapshot(1,(0.,0.),-3.,0.))
                landing_pose=matcher.shared_floor(frame,prior,selected,zoom=zoom_position(frame),
                    pointer=PointerSnapshot(1,(0.,0.),-1.,button[2]),continuous=True)
        if (landing_pose is None and prior and prior['floor']['floor']=='B1' and selected=='B2' and presence['visible']
                and lock.shared_layout is None and lock.opening_floor not in ('B2','B3','B4')):
            prior['zoom']=zoom_position(prior_image)
            lift=matcher.elevators.locate(frame,map_area,prior,zoom=zoom_position(frame))
            if lift:
                from src.s1_elevator import cameras_agree
                stamp=len(sys.argv)-.5
                lock.observe(dict(lift,at=stamp,unique_layout=True,proof_kind='elevator',
                                  local_verified=True,contradicts=lift['elevator_verified']))
                if not lift['elevator_verified']:
                    independent=matcher.confirm_b2(frame,incumbent=lock.layout,local_layout=lock.region_layout)
                    if independent:
                        eligible=independent.get('proof_kind')=='full' or independent.get('region_verified') or independent.get('local_verified')
                        same=independent['floor']['id']==lift['floor']['id']
                        checked=eligible and same and cameras_agree(lift['matrix'],independent['matrix'],frame.shape[1::-1])
                        lock.observe(dict(independent,at=stamp,unique_layout=True,elevator_verified=checked))
                if lock.permits_overlay('B2',lift['floor']['layout_id'],lift['possible_layouts']):
                    landing_pose=lift
            prior=None  # An unresolved B1 never seeds an unverified fallback on B2.
        certified=lock.layout or lock.landing.layout
        if landing_pose is None and certified and selected in ('B2','B3') and presence['visible'] and not lock.blocks_floor(selected):
            if prior:prior['zoom']=zoom_position(prior_image)
            landing_pose=matcher.locate_device(frame,lock.preferred(selected),selected,visit=prior,zoom=zoom_position(frame))
        if (landing_pose is None and prior and presence['visible'] and selected in ('B1','B2','B3') and selected!=prior['floor']['floor']
                and certified and not lock.blocks_floor(selected) and not lock.blocks_floor(prior['floor']['floor'])
                and lock.preferred(selected) in (None,certified) and lock.preferred(prior['floor']['floor'])==certified
                and (lock.layout==certified or prior['floor']['floor']=='B3' or lock.region_layout==certified)):
            prior['zoom']=zoom_position(prior_image)
            landing_pose=matcher.paired_floor(frame,prior,selected,certified,zoom=zoom_position(frame))
            if landing_pose and not direct_pose(landing_pose,paired=True):landing_pose=None
        if landing_pose is None and presence['visible'] and selected=='B3':
            base=None if lock.blocks_floor('B2') else lock.layout
            current=matcher.review_landing(frame,incumbent=lock.landing.layout or base,pending=lock.landing.pending)
            if current:
                lock.observe_landing(dict(current,at=len(sys.argv)))
                if lock.permits_overlay('B3',current['floor']['layout_id'],current['possible_layouts']):
                    landing_pose=current
        if presence['visible'] and selected=='B2' and (lock.recheck_b2_after_landing or lock.b2_arrival):
            proof=matcher.confirm_b2(frame,incumbent=lock.layout,entry=True)
            if proof:
                lock.observe(dict(proof,at=len(sys.argv),unique_layout=True))
                if lock.permits_overlay('B2',proof['floor']['layout_id'],proof['possible_layouts']):
                    landing_pose=proof
        target_layout=lock.preferred(selected)
        if prior and selected=='B3' and target_layout and prior['floor']['layout_id']!=target_layout:prior=None
        if not presence['visible']:
            r = {'accepted':False}
        elif landing_pose and direct_pose(landing_pose,paired=True):
            r=landing_pose
        elif selected=='B1':
            r=matcher.review_b1(frame) or {'accepted':False}
        elif selected not in ('B1','B2','B3'):
            r={'accepted':False}
        elif lock.blocks_floor(selected):
            r={'accepted':False}
        elif selected:
            if selected=='B2' and prior and prior['floor']['floor']!='B2' and not landing_pose:
                prior=None
            r=landing_pose
            if (prior and target_layout and selected==prior_level and prior['floor']['layout_id']==target_layout
                    and zoom_changed(zoom_position(prior_image),zoom_position(frame))):
                prior['zoom']=zoom_position(prior_image)
                r=matcher.locate_selected(frame,target_layout,selected,visit=prior)
            if r is None and prior:r=matcher.handoff(frame,prior,selected)
            if lock.layout and r and r.get('provisional'):r=None
            if prior and not r:
                anchor=dict(matrix=prior['matrix'],view_features=ViewFeatures(prior_image,map_area))
                r=matcher.follow_known(ViewFeatures(frame,map_area),prior,selected,
                    [anchor] if selected==prior_level else [],frame,anchor,lock.layout)
            if not r:r=matcher.locate_selected(frame,target_layout,selected,visit=prior)
            if selected=='B2' and not r['accepted']:
                proof=matcher.confirm_b2(frame,incumbent=lock.layout)
                if proof and target_layout in (None,proof['floor']['layout_id']) and not proof.get('conflicting_models'):
                    r=proof
        else:
            r = matcher.locate_live(frame,lock.layout)
        if (r['accepted'] and selected in ('B2','B3') and r is not landing_pose and not direct_pose(r)):
            # A restricted fallback has not ruled out any other layout. Use the
            # same independent floor review as the desktop before exporting it.
            stamp=len(sys.argv)+1
            if selected=='B3':
                independent=matcher.review_landing(frame,incumbent=lock.preferred('B3'),pending=lock.landing.pending)
                if independent:lock.observe_landing(dict(independent,at=stamp))
            else:
                independent=matcher.confirm_b2(frame,incumbent=lock.layout,entry=lock.b2_arrival)
                if independent:lock.observe(dict(independent,at=stamp,unique_layout=True))
            r=independent if independent and not independent.get('conflicting_models') else {'accepted':False}
            target_layout=lock.preferred(selected)
        if selected=='B2' and lock.elevator_only and not r.get('shared_elevator_only'):r={'accepted':False}
        if target_layout and r['accepted'] and (r['floor']['layout_id']!=target_layout or (lock.layout and r.get('provisional'))):
            r={'accepted':False}
        if r['accepted'] and not lock.permits_overlay(r['floor']['floor'],r['floor']['layout_id'],
                r.get('possible_layouts') or [r['floor']['layout_id']]):
            r={'accepted':False}
        correction=None
        if r['accepted'] and '--pose-offset' in sys.argv and not r.get('common_floors'):
            i=sys.argv.index('--pose-offset')
            prior_matrix=r['matrix'].copy()
            prior_matrix[:,2]+=[float(sys.argv[i+1]),float(sys.argv[i+2])]
            correction=matcher.pose_check.check(frame,map_area,None,r['floor'],prior_matrix,
                        'original_clean',lock.layout==r['floor']['layout_id'])
            if correction:r=correction
        rendered = matcher.render_live(r,frame) if r['accepted'] else None
        result={'map_visible':presence['visible'],'accepted':r['accepted'],
                'selected_floor':selected, 'method':r.get('segmentation'),
                'opening_floor':lock.opening_floor,'round_layout':lock.layout,'b2_votes':lock.count,'b2_proposed':lock.challenger,'review_count':lock.proposal_count,
                'b2_local':lock.region_layout,'b3_local':lock.landing.layout,'b3_pending':lock.landing.pending,
                'waiting_b2':lock.b2_arrival,'waiting_b3':lock.landing.waiting,
                'pose_adjustment':correction and correction['adjustment'],
                'wall_error_before_px':correction and correction['before_px'],
                'wall_error_after_px':correction and correction['after_px'],
                'display_source':r.get('display_source'),'visible_rooms':r.get('visible_rooms',[]),
                'possible_layouts':r.get('possible_layouts',[]),'common_only':bool(r.get('common_floors')),
                'prior_floor':prior['floor']['id'] if prior else None,'search':r.get('search'),
                'render_sha256':hashlib.sha256(rendered.tobytes()).hexdigest() if rendered is not None else None,
                'floor':r.get('floor',{}).get('id'),'inliers':r.get('inliers'),
                'matrix':r['matrix'].tolist() if r['accepted'] else None,
                'shared_camera':bool(r.get('shared_camera')),'camera_mode':r.get('camera_mode'),'camera_only':bool(r.get('camera_only')),
                'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'scope':'Saved frame replay; no live input or projection'}
        (project_root()/'s1-capture-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise SystemExit(0)
    if '--self-check-ui' in sys.argv:
        import json
        from src.s1_window import S1Window, DEFAULT_OVERLAY_OPACITY
        from src.config import project_root
        app=S1Window(autostart=False, controls=False, start_hidden=True)
        app.geometry('+30000+30000');app.deiconify();app.update();checked=[];alchemy_views=[];race_views=[];cart_views=[]
        assert app.map_key.get() == 'Caps Lock' and app.overlay_style.get() == '高清底图'
        assert app.overlay_alpha.get()==DEFAULT_OVERLAY_OPACITY and app.opacity_label.get()=='28%'
        assert tuple(app.floor_buttons)==('B1','B2','B3','all') and len(app.tree.get_children())==len(app.layouts)
        app.change_opacity('35');assert app.opacity_label.get()=='35%' and app.overlay_alpha.get()==35
        app.change_opacity(str(DEFAULT_OVERLAY_OPACITY))
        from tkinter import ttk
        app.select()
        window=app.show_overlap()
        for btn in window.winfo_children()[0].winfo_children():
            if isinstance(btn,ttk.Button):
                btn.invoke()
                assert window.winfo_children()[-1].image is not None
        window.destroy()
        for e in (e for e in app.catalog.entries if e['kind'] in ('structure','hd')):
            app.tree.selection_set(e['layout_id'][3:])
            app.floor_buttons[e['floor'] if e['kind']=='hd' else 'all'].invoke();app.update()
            assert app.current['id']==e['id'] and app.original is not None and len(app.canvas.find_all())==1
            displayed=sorted(text for text,_,_ in app.gallery_labels)
            expected={'B1':[], 'B2':['维修站'], 'B3':['维修站','寰星房']}
            for level in ('B1','B2','B3'):
                points=app.room_labels.points.get(e['layout_id']+'_'+level,{})
                expected[level].extend('矿车密道' for key in points if key.startswith('cart'))
            carts=displayed.count('矿车密道')
            if carts:cart_views.append(dict(id=e['id'],count=carts))
            if 'race' in app.room_labels.points.get(e['layout_id']+'_B2', {}):
                expected['B2'].append('竞速房')
            if '竞速房' in displayed:
                race_views.append(e['id'])
            if 'alchemy' in app.room_labels.points.get(e['layout_id']+'_B3', {}):
                expected['B3'].append('炼金室')
            if '炼金室' in displayed:
                alchemy_views.append(e['id'])
            if e['kind']=='structure':
                assert displayed==sorted(['B1','B2','B3',*expected['B1'],*expected['B2'],*expected['B3']])
            else:
                assert displayed==sorted(expected[e['floor']])
            app.switch_pair();app.select();app.update()
            assert app.current['layout_id']==e['layout_id'] and app.current['kind']!=e['kind']
            checked.append(e['id'])
        assert len(checked)==len(app.layouts)*4
        # Exercise the real Tk button callbacks without controlling the game.
        app.withdraw()
        app.detection_action.invoke()
        assert app.live.active and app.live.continuous and not app.live.hidden
        assert '检测已开启' in app.detect_state.get()
        app.detection_action.invoke()
        assert not app.live.active and '检测已停止' in app.detect_state.get()
        app.detection_action.invoke()
        assert app.live.active and app.live.continuous
        app.detection_action.invoke()
        button_check={'start':True,'stop':True,'restart':True,'status':True}
        from src.s1_keys import MapBinding,CTRL
        app.bind_key_button.invoke();app.update()
        dialog=app.binding_dialog
        dialog.capture.observe(ord('K'),True,CTRL,dialog.capture.hwnd)
        dialog.after_cancel(dialog.timer);dialog.poll();dialog.save_button.invoke()
        assert app.binding==MapBinding(ord('K'),CTRL) and app.map_key.get()=='Ctrl + K'
        app.set_map_binding(MapBinding(0x14))
        button_check['configurable_map_key']=True
        app.destroy()
        (project_root()/'s1-ui-check.json').write_text(json.dumps({'checked':checked,'alchemy_views':alchemy_views,'race_views':race_views,'cart_views':cart_views,'one_click':button_check,'scope':'Tk render state and real button callbacks; not live-game visual inspection'},indent=2),encoding='utf-8')
        raise SystemExit(0)
    if '--self-check' in sys.argv:
        import json
        from src.config import project_root
        from src.s1_catalog import S1Catalog
        c=S1Catalog(map_root(project_root()))
        errors=c.validate()
        from src.s1_visibility import MapVisibility
        import hashlib
        ui=c.root/'ui'
        for item in json.loads((ui/'index.json').read_text(encoding='utf-8'))['templates']:
            if hashlib.sha256((ui/item['file']).read_bytes()).hexdigest()!=item['sha256']:
                errors.append('map UI template hash mismatch: '+item['file'])
        MapVisibility(ui)
        result={'entries':len(c.entries),'errors':errors}
        target=project_root()/'s1-self-check.json'
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        raise SystemExit(bool(errors))
    raise SystemExit(legacy_main() if '--legacy' in sys.argv else s1_main())

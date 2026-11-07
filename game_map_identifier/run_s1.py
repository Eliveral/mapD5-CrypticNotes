"""S1 extension entry point. The upstream entry point and models remain intact."""
import sys
from src.s1_updates import map_root
from src.s1_window import main as s1_main
from src.app import main as legacy_main

if __name__ == '__main__':
    if '--self-check-capture' in sys.argv:
        import json, hashlib, cv2
        from pathlib import Path
        from src.config import project_root
        from src.s1_catalog import S1Catalog, read_image
        from src.s1_live_match import LiveMatcher
        from src.s1_floor_transition import selected_floor
        from src.s1_visibility import MapVisibility
        cv2.setNumThreads(2)
        path=Path(sys.argv[sys.argv.index('--self-check-capture')+1])
        c=S1Catalog(map_root(project_root()))
        frame=read_image(path)
        presence=MapVisibility(c.root/'ui').inspect(frame)
        selected = selected_floor(frame) if presence['visible'] else None
        matcher = LiveMatcher(c)
        prior=None
        if '--prior-capture' in sys.argv:
            prior_image=read_image(Path(sys.argv[sys.argv.index('--prior-capture')+1]))
            prior_level=selected_floor(prior_image)
            if prior_level and MapVisibility(c.root/'ui').inspect(prior_image)['visible']:
                previous=matcher.locate_selected(prior_image,None,prior_level)
                if previous['accepted'] and len(previous.get('possible_layouts',[]))==1:prior=previous
        if not presence['visible']:
            r = {'accepted':False}
        elif selected:
            r = matcher.locate_selected(frame, None, selected,visit=prior)
        else:
            r = matcher.locate_live(frame)
        rendered = matcher.render_live(r,frame) if r['accepted'] else None
        result={'map_visible':presence['visible'],'accepted':r['accepted'],
                'selected_floor':selected, 'method':r.get('segmentation'),
                'display_source':r.get('display_source'),
                'possible_layouts':r.get('possible_layouts',[]),'common_only':bool(r.get('common_floors')),
                'prior_floor':prior['floor']['id'] if prior else None,'search':r.get('search'),
                'render_sha256':hashlib.sha256(rendered.tobytes()).hexdigest() if rendered is not None else None,
                'floor':r.get('floor',{}).get('id'),'inliers':r.get('inliers'),
                'matrix':r['matrix'].tolist() if r['accepted'] else None,
                'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'scope':'Saved actual game capture replay; no live input or projection'}
        (project_root()/'s1-capture-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise SystemExit(0)
    if '--self-check-ui' in sys.argv:
        import json
        from src.s1_window import S1Window, DEFAULT_OVERLAY_OPACITY
        from src.config import project_root
        app=S1Window(autostart=False, controls=False, start_hidden=True);app.update();checked=[]
        assert app.map_key.get() == 'Caps Lock' and app.overlay_style.get() == '高清底图'
        assert app.overlay_alpha.get()==DEFAULT_OVERLAY_OPACITY and app.opacity_label.get()=='28%'
        assert tuple(app.mode['values'])==('高清单层','三层总览')
        app.change_opacity('35');assert app.opacity_label.get()=='35%' and app.overlay_alpha.get()==35
        app.change_opacity(str(DEFAULT_OVERLAY_OPACITY))
        for e in (e for e in app.catalog.entries if e['kind'] in ('structure','hd')):
            app.kind.set({'structure':'三层总览','hd':'高清单层'}[e['kind']]);app.populate()
            app.tree.selection_set(e['id']);app.select();app.update()
            assert app.current['id']==e['id'] and app.original is not None and len(app.canvas.find_all())==1
            displayed=sorted(text for text,_,_ in app.gallery_labels)
            expected={'B1':[], 'B2':['维修站'], 'B3':['维修站','寰星房']}
            if e['kind']=='structure':
                assert displayed==sorted(['B1','B2','B3',*expected['B2'],*expected['B3']])
            else:
                assert displayed==sorted(expected[e['floor']])
            app.switch_pair();app.select();app.update()
            assert app.current['layout_id']==e['layout_id'] and app.current['kind']!=e['kind']
            checked.append(e['id'])
        assert len(checked)==len(app.layouts)*4
        # Exercise the real Tk button callbacks without controlling the game.
        app.withdraw()
        app.detect_button.invoke()
        assert app.live.active and app.live.continuous and not app.live.hidden
        assert '检测已开启' in app.detect_state.get()
        app.stop_button.invoke()
        assert not app.live.active and '检测已停止' in app.detect_state.get()
        app.detect_button.invoke()
        assert app.live.active and app.live.continuous
        app.stop_button.invoke()
        button_check={'start':True,'stop':True,'restart':True,'status':True}
        app.map_key.set('M');app.change_map_key()
        app.map_key.set('Caps Lock');app.change_map_key()
        button_check['configurable_map_key']=True
        app.destroy()
        (project_root()/'s1-ui-check.json').write_text(json.dumps({'checked':checked,'one_click':button_check,'scope':'Tk render state and real button callbacks; not live-game visual inspection'},indent=2),encoding='utf-8')
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

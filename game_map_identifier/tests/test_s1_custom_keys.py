"""Exercise the same key edges used by the hook and real first-run Tk controls."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queue import Queue

import pytest

from src.s1_game import KeyEdges
from src.s1_hotkeys import Hotkeys
from src.s1_keys import ALT,CTRL,SHIFT,BindingCapture,MapBinding,physical_key


@pytest.mark.parametrize('key,mods',[(0x14,0),(0x1B,0),(0x09,0),(0x7B,0),(0xBA,0),(0xA3,0),
    (0x41,CTRL|SHIFT),(0xBF,ALT),(0x10D,0),(0x125,0),(0x60,0),(1,0),(5,CTRL),(6,0)])
def test_roundtrip_and_matching(key,mods):
    binding=MapBinding(key,mods)
    assert MapBinding.load(binding.saved())==binding and binding.label
    edges=KeyEdges(None);edges.binding=binding
    assert edges.update(key,True,mods=mods)=='map'
    assert edges.update(key,True,mods=mods) is None
    edges.update(key,False)
    assert edges.update(key,True,mods=mods)=='map'


@pytest.mark.parametrize('bad',[None,'F3',{},[],{'vk':False},{'vk':999},{'vk':0},{'vk':65,'modifiers':'Ctrl'},
                                {'vk':65,'modifiers':32},{'vk':0xA0,'modifiers':SHIFT}])
def test_invalid_profile_needs_binding(bad):
    assert MapBinding.load(bad) is None


def test_physical_key_and_modifier_capture():
    assert physical_key(13,1)!=physical_key(13,0)
    assert physical_key(0x25,0)!=physical_key(0x25,1)
    c=BindingCapture(77)
    c.observe(0xA2,True,0,77);c.observe(0xA1,True,CTRL,77)
    c.observe(0xA1,False,CTRL,77)
    assert c.result==MapBinding(0xA1,CTRL)


class LocalObserver(Hotkeys):
    def __init__(self,events,map_key):
        self.events=events;self.edges=KeyEdges(None);self.capture=None
        self.keyboard_ready=self.mouse_ready=True;self.set_map_key(map_key)
    def close(self):pass


def test_shortcut_and_escape_priority():
    o=LocalObserver(Queue(),MapBinding(0x78))
    assert 703 not in o.shortcut_ids()
    o.observe(0x78,True,0,70);o.observe(0x78,True,0,70)
    assert o.events.get_nowait()==('map_key',('map',70)) and o.events.empty()
    o.set_map_key(MapBinding(0x1B))
    o.observe(0x1B,True,0,70)
    assert o.events.get_nowait()==('map_key',('map',70)) and 703 in o.shortcut_ids()
    o.observe(0x1B,False,0,70);o.set_map_key(MapBinding(65))
    o.observe(0x1B,True,0,70)
    assert o.events.get_nowait()==('map_key',('escape',70))


def press(app,key,mods=0,point=None,hwnd=None):
    dialog=app.binding_dialog
    hwnd=dialog.capture.hwnd if hwnd is None else hwnd
    o=app.live.hotkeys
    o.observe(key,True,mods,hwnd,point);o.observe(key,False,mods,hwnd,point)
    dialog.after_cancel(dialog.timer);dialog.poll()


@pytest.fixture
def run_window(tmp_path,monkeypatch):
    from src.s1_window import S1Window
    from src import s1_live
    from src.s1_program_panel import ProgramUpdatePanel
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    monkeypatch.setattr(s1_live,'Hotkeys',LocalObserver)
    monkeypatch.setattr(s1_live,'discover_game',lambda *_:None)
    monkeypatch.setattr(ProgramUpdatePanel,'check',lambda *_:None)
    from src.s1_live_match import LiveMatcher
    monkeypatch.setattr(LiveMatcher,'warmup',lambda *_:None)
    windows=[]
    def launch(autostart=True):
        if windows:windows.pop().destroy()
        app=S1Window(autostart=autostart,start_hidden=True)
        app.geometry('780x540+30000+30000');app.deiconify();app.update();windows.append(app)
        return app
    yield launch
    for app in reversed(windows):app.destroy()


def test_fresh_defer_bind_relaunch_and_rebind(run_window):
    app=run_window()
    d=app.binding_dialog
    assert d is not None and app.binding is None and not app.live.active
    assert app.live.hotkeys.shortcut_ids()==set()
    d.dismiss();app.change_opacity('36');app.save_preferences()
    assert 'map_key' not in json.loads(app.preferences_path.read_text('utf8'))
    app.detection_action.invoke();d=app.binding_dialog
    press(app,65,hwnd=-1);assert d.pending is None
    press(app,0x4B,CTRL|SHIFT);assert d.pending==MapBinding(0x4B,CTRL|SHIFT)
    d.save_button.invoke();assert app.live.active and app.binding_dialog is None
    expected=app.binding
    again=run_window();assert again.binding==expected and again.binding_dialog is None
    assert again.live.active and again.overlay_alpha.get()==36
    again.bind_key_button.invoke();d=again.binding_dialog
    d.update();p=d.capture.mouse_box
    assert p[2]-p[0]>5 and p[3]-p[1]>5
    press(again,1,point=(p[0]-1,p[1]-1));assert d.pending is None
    press(again,5,point=(p[0]+1,p[1]+1));d.save_button.invoke()
    assert again.binding==MapBinding(5) and again.live.active
    again.bind_key_button.invoke();press(again,0x78);again.binding_dialog.dismiss()
    assert again.binding==MapBinding(5)
    again.bind_key_button.invoke();press(again,0x78);again.binding_dialog.save_button.invoke()
    assert again.shortcut_buttons[0x78][0]['text']=='隐藏 / 恢复'
    assert 703 not in again.live.hotkeys.shortcut_ids()


@pytest.mark.parametrize('old,expected',[('Caps Lock',0x14),('M',0x4d),('Tab',9),({'vk':5000},None)])
def test_older_profiles(run_window,tmp_path,old,expected):
    prefs=tmp_path/'CrypticNotesAssistantS1/preferences-v32.json'
    prefs.parent.mkdir(exist_ok=True);prefs.write_text(json.dumps({'map_key':old,'clean_map_opacity':47}),'utf8')
    app=run_window(False)
    assert not app.live.active and app.overlay_alpha.get()==47
    if expected:
        assert app.binding==MapBinding(expected) and app.binding_dialog is None
    else:
        assert app.binding_dialog is not None
        press(app,0x14);app.binding_dialog.save_button.invoke()
        assert app.binding==MapBinding(0x14) and not app.live.active

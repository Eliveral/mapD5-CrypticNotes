"""Mouse observation must stay outside the synchronous mouse hook chain."""
import ctypes
from ctypes import wintypes as W
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.s1_raw_mouse import RawMouse,Device,Packet,Header,Mouse
from src.s1_map_camera import MapPointer


def test_native_registration_is_observe_only_and_removed():
    u=ctypes.WinDLL('user32',use_last_error=True)
    u.GetRegisteredRawInputDevices.argtypes=[ctypes.POINTER(Device),ctypes.POINTER(W.UINT),W.UINT]
    u.GetRegisteredRawInputDevices.restype=W.UINT
    def registered():
        devices=(Device*32)();n=W.UINT(32)
        count=u.GetRegisteredRawInputDevices(devices,ctypes.byref(n),ctypes.sizeof(Device))
        assert count!=0xffffffff
        return [(d.usage_page,d.usage,d.flags,d.target) for d in list(devices)[:count]]
    before=registered()
    assert not [d for d in before if d[:2]==(1,2)]
    observer=RawMouse(lambda *args:None)
    try:
        observer.enable(True)
        assert (1,2,0x100,observer.hwnd) in registered()
        observer.enable(False)
        assert registered()==before
        observer.enable(True)
    finally:observer.close()
    assert registered()==before


def test_drag_floor_click_and_buttons_keep_screen_coordinates():
    p=MapPointer();p.configure((99,(0,0,1000,1000)))
    events=[];position=[(400,400),99]
    o=RawMouse.__new__(RawMouse)
    o.enabled=True;o.pending=False;o.packets=o.button_packets=o.moves=0
    o.position=lambda:(position[0],position[1])
    def callback(message,point,hwnd,key,down):
        p.observe(message,point,hwnd);events.append((message,key,down))
    o.callback=callback
    packet=Packet();packet.header.kind=0
    packet.mouse.buttons=1;o.consume(packet)
    for point in [(410,405),(430,410)]:
        position[0]=point;packet.mouse.buttons=0;packet.mouse.x=1000;packet.mouse.y=1000
        o.consume(packet)
    o.flush_move()
    assert p.snapshot().pan==(30,10)  # Raw device units must not multiply screen pan.
    packet.mouse.buttons=2;o.consume(packet)
    assert p.drag is None
    position[0]=(780,200);packet.mouse.buttons=1;o.consume(packet)
    assert p.snapshot().click_at>0 and p.snapshot().click_y==200
    packet.mouse.buttons=0x40|0x80|0x100|0x200;o.consume(packet)
    assert [e[1:] for e in events if e[1] in (5,6)]==[(5,True),(5,False),(6,True),(6,False)]
    old=p.snapshot().epoch;packet.mouse.buttons=0x400;o.consume(packet)
    assert p.snapshot().epoch==old+1
    o.enabled=False;count=len(events);o.consume(packet);assert len(events)==count


def test_packet_layout_and_no_mouse_hook_or_input_rewrite():
    assert ctypes.sizeof(Mouse)==24 and Mouse.buttons.offset==4 and Mouse.x.offset==12
    assert ctypes.sizeof(Header)==(24 if ctypes.sizeof(ctypes.c_void_p)==8 else 16)
    source=(Path(__file__).resolve().parents[1]/'src/s1_hotkeys.py').read_text('utf8')
    assert 'SetWindowsHookExW(14' not in source
    raw=(Path(__file__).resolve().parents[1]/'src/s1_raw_mouse.py').read_text('utf8')
    for forbidden in ('SendInput(', 'mouse_event(', 'SetCursorPos(', 'SystemParametersInfo'):
        assert forbidden not in raw

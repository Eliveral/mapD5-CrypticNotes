"""Observe game inputs passively and reserve only available assistant shortcuts."""
import ctypes
from ctypes import wintypes as W
import threading

from .s1_keys import MapBinding, MODIFIER, MOUSE, CTRL, ALT, SHIFT, WIN, physical_key
from .s1_game import KeyEdges
from .s1_map_camera import MapPointer

SHORTCUTS={701:(0x77,'once'),702:(0x76,'auto'),703:(0x78,'hide'),704:(0x79,'stop')}
BUTTON_MESSAGES={0x201:(1,True),0x202:(1,False),0x204:(2,True),0x205:(2,False),
                 0x207:(4,True),0x208:(4,False)}
HOOKPROC=ctypes.WINFUNCTYPE(ctypes.c_ssize_t,ctypes.c_int,W.WPARAM,W.LPARAM)


class KeyboardData(ctypes.Structure):
    _fields_=[('vkCode',W.DWORD),('scanCode',W.DWORD),('flags',W.DWORD),('time',W.DWORD),('extra',ctypes.c_size_t)]


class MouseData(ctypes.Structure):
    _fields_=[('pt',W.POINT),('mouseData',W.DWORD),('flags',W.DWORD),('time',W.DWORD),('extra',ctypes.c_size_t)]


class Hotkeys:
    def __init__(self,events,map_key='Caps Lock'):
        self.events=events
        self.edges=KeyEdges()
        self.set_map_key(map_key)
        self.capture=None
        self.pointer=MapPointer()
        self.keyboard_ready=None
        self.mouse_ready=None
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def set_map_key(self,binding):
        self.edges.binding=binding if isinstance(binding,MapBinding) else MapBinding.load(binding)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=.4)

    def shortcut_ids(self):
        if self.capture is not None:return set()
        binding=self.edges.binding
        return {i for i,(vk,_) in SHORTCUTS.items() if binding!=MapBinding(vk)}

    def observe(self,key,down,mods,hwnd,point=None):
        repeat=down and key in self.edges.held
        mods &= ~MODIFIER.get(key,0)
        action=self.edges.update(key,down,mods=mods)
        capture=self.capture
        if capture is not None:
            if not repeat:capture.observe(key,down,mods,hwnd,point)
        elif action:
            self.events.put(('map_key',(action,hwnd)))

    def run(self):
        u=ctypes.windll.user32
        u.SetWindowsHookExW.argtypes=[ctypes.c_int,HOOKPROC,W.HINSTANCE,W.DWORD]
        u.SetWindowsHookExW.restype=W.HANDLE
        u.CallNextHookEx.argtypes=[W.HANDLE,ctypes.c_int,W.WPARAM,W.LPARAM]
        u.CallNextHookEx.restype=ctypes.c_ssize_t
        u.UnhookWindowsHookEx.argtypes=[W.HANDLE]
        u.GetForegroundWindow.restype=W.HWND

        def modifiers():
            mods=0
            for vk,bit in ((0x11,CTRL),(0x12,ALT),(0x10,SHIFT),(0x5B,WIN),(0x5C,WIN)):
                if u.GetAsyncKeyState(vk)&0x8000:mods|=bit
            return mods

        @HOOKPROC
        def keyboard(code,message,address):
            try:
                if code>=0 and message in (0x100,0x104,0x101,0x105):
                    d=ctypes.cast(address,ctypes.POINTER(KeyboardData)).contents
                    self.observe(physical_key(d.vkCode,d.flags&1),message in (0x100,0x104),
                                 modifiers(),u.GetForegroundWindow())
            finally:
                return u.CallNextHookEx(None,code,message,address)

        @HOOKPROC
        def mouse(code,message,address):
            try:
                if code>=0 and self.pointer.target is not None:
                    d=ctypes.cast(address,ctypes.POINTER(MouseData)).contents
                    self.pointer.observe(message,(d.pt.x,d.pt.y),u.GetForegroundWindow())
                if code>=0 and (message in BUTTON_MESSAGES or message in (0x20B,0x20C)):
                    d=ctypes.cast(address,ctypes.POINTER(MouseData)).contents
                    if message in BUTTON_MESSAGES:
                        key,down=BUTTON_MESSAGES[message]
                    else:
                        key=5 if d.mouseData>>16==1 else 6
                        down=message==0x20B
                    self.observe(key,down,modifiers(),u.GetForegroundWindow(),(d.pt.x,d.pt.y))
            finally:
                return u.CallNextHookEx(None,code,message,address)

        hook=u.SetWindowsHookExW(13,keyboard,None,0)
        self.keyboard_ready=bool(hook)
        self.events.put(('hook_ready',self.keyboard_ready))
        mouse_hook=None
        want_mouse=False
        registered=set()
        previous=set()
        try:
            msg=W.MSG()
            while not self.stop_event.is_set():
                binding=self.edges.binding
                needed=self.capture is not None or self.pointer.target is not None or (binding is not None and binding.key in MOUSE)
                if needed!=want_mouse:
                    if needed:
                        mouse_hook=u.SetWindowsHookExW(14,mouse,None,0)
                        self.mouse_ready=bool(mouse_hook)
                        if not mouse_hook:self.events.put(('hotkey_error','鼠标按键监听未能启动。'))
                    elif mouse_hook:
                        u.UnhookWindowsHookEx(mouse_hook)
                        mouse_hook=None
                        self.mouse_ready=None
                    want_mouse=needed
                desired=self.shortcut_ids()
                for ident in registered-desired:
                    u.UnregisterHotKey(None,ident)
                    registered.remove(ident)
                for ident in desired-previous:
                    vk,_=SHORTCUTS[ident]
                    if u.RegisterHotKey(None,ident,0x4000,vk):
                        registered.add(ident)
                    else:self.events.put(('hotkey_error',f'F{vk-0x6f} 已被占用，可使用窗口按钮。'))
                previous=desired
                while u.PeekMessageW(ctypes.byref(msg),None,0,0,1):
                    if msg.message==0x0312 and msg.wParam in registered and msg.wParam in self.shortcut_ids():
                        self.events.put(('key',SHORTCUTS[msg.wParam][1]))
                self.stop_event.wait(.008)
        finally:
            for handle in (hook,mouse_hook):
                if handle:u.UnhookWindowsHookEx(handle)
            for ident in registered:u.UnregisterHotKey(None,ident)

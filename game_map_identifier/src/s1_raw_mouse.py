"""Asynchronous mouse observation; never join the system mouse-hook chain."""
import ctypes
from ctypes import wintypes as W

WNDPROC=ctypes.WINFUNCTYPE(ctypes.c_ssize_t,W.HWND,W.UINT,W.WPARAM,W.LPARAM)
BUTTONS=((1,1,True,0x201),(2,1,False,0x202),(4,2,True,0x204),(8,2,False,0x205),
         (16,4,True,0x207),(32,4,False,0x208),(64,5,True,0x20B),(128,5,False,0x20C),
         (256,6,True,0x20B),(512,6,False,0x20C))


class Device(ctypes.Structure):
    _fields_=[('usage_page',W.USHORT),('usage',W.USHORT),('flags',W.DWORD),('target',W.HWND)]


class Header(ctypes.Structure):
    _fields_=[('kind',W.DWORD),('size',W.DWORD),('device',W.HANDLE),('param',W.WPARAM)]


class Mouse(ctypes.Structure):
    # The button union is DWORD-aligned; usFlags has two padding bytes.
    _fields_=[('flags',W.USHORT),('padding',W.USHORT),('buttons',W.USHORT),
              ('button_data',W.USHORT),('raw_buttons',W.ULONG),('x',W.LONG),
              ('y',W.LONG),('extra',W.ULONG)]


class Packet(ctypes.Structure):
    _fields_=[('header',Header),('mouse',Mouse)]


class WindowClass(ctypes.Structure):
    _fields_=[('style',W.UINT),('proc',WNDPROC),('class_extra',ctypes.c_int),
              ('window_extra',ctypes.c_int),('instance',W.HINSTANCE),('icon',W.HICON),
              ('cursor',W.HANDLE),('background',W.HBRUSH),('menu',W.LPCWSTR),('name',W.LPCWSTR)]


class RawMouse:
    """Owned by one message-pump thread. Registration is local to this process."""
    def __init__(self,callback):
        self.callback=callback
        self.u=ctypes.WinDLL('user32',use_last_error=True)
        self.k=ctypes.WinDLL('kernel32',use_last_error=True)
        u=self.u
        u.RegisterClassW.argtypes=[ctypes.POINTER(WindowClass)];u.RegisterClassW.restype=W.ATOM
        u.UnregisterClassW.argtypes=[W.LPCWSTR,W.HINSTANCE]
        u.CreateWindowExW.argtypes=[W.DWORD,W.LPCWSTR,W.LPCWSTR,W.DWORD,ctypes.c_int,ctypes.c_int,
            ctypes.c_int,ctypes.c_int,W.HWND,W.HMENU,W.HINSTANCE,ctypes.c_void_p]
        u.CreateWindowExW.restype=W.HWND
        u.DestroyWindow.argtypes=[W.HWND]
        u.DefWindowProcW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM];u.DefWindowProcW.restype=ctypes.c_ssize_t
        u.RegisterRawInputDevices.argtypes=[ctypes.POINTER(Device),W.UINT,W.UINT]
        u.RegisterRawInputDevices.restype=W.BOOL
        u.GetRawInputData.argtypes=[W.HANDLE,W.UINT,ctypes.c_void_p,ctypes.POINTER(W.UINT),W.UINT]
        u.GetRawInputData.restype=W.UINT
        u.GetCursorPos.argtypes=[ctypes.POINTER(W.POINT)];u.GetCursorPos.restype=W.BOOL
        u.GetForegroundWindow.restype=W.HWND
        self.k.GetModuleHandleW.argtypes=[W.LPCWSTR];self.k.GetModuleHandleW.restype=W.HMODULE
        self.instance=self.k.GetModuleHandleW(None)
        self.name='MapD5RawMouse_'+str(id(self))
        self.hwnd=None;self.registered=False;self.enabled=False;self.pending=False
        self.packets=self.button_packets=self.moves=0
        self.proc=WNDPROC(self.window_proc)
        wc=WindowClass();wc.proc=self.proc;wc.instance=self.instance;wc.name=self.name
        if not u.RegisterClassW(ctypes.byref(wc)):raise ctypes.WinError(ctypes.get_last_error())
        self.registered=True
        try:
            self.hwnd=u.CreateWindowExW(0,self.name,'',0,0,0,0,0,W.HWND(-3),None,self.instance,None)
            if not self.hwnd:raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close()
            raise

    def enable(self,enabled):
        if bool(enabled)==self.enabled:return
        # INPUTSINK only. Never NOLEGACY, CAPTUREMOUSE, or input suppression.
        device=Device(1,2,0x100 if enabled else 1,self.hwnd if enabled else None)
        if not self.u.RegisterRawInputDevices(ctypes.byref(device),1,ctypes.sizeof(Device)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.enabled=bool(enabled);self.pending=False

    def position(self):
        point=W.POINT()
        if not self.u.GetCursorPos(ctypes.byref(point)):return None
        return (point.x,point.y),self.u.GetForegroundWindow()

    def deliver(self,message,key=None,down=None):
        position=self.position()
        if position:self.callback(message,position[0],position[1],key,down)

    def flush_move(self):
        if self.pending:
            self.pending=False
            self.deliver(0x200);self.moves+=1

    def consume(self,packet):
        if not self.enabled or packet.header.kind!=0:return
        self.packets+=1
        mouse=packet.mouse
        # Use actual screen pixels, not unaccelerated device deltas, for map pan.
        if mouse.x or mouse.y:self.pending=True
        if not mouse.buttons:return
        self.button_packets+=1
        self.flush_move()
        for flag,key,down,message in BUTTONS:
            if mouse.buttons&flag:self.deliver(message,key,down)
        for flag,message in ((0x400,0x20A),(0x800,0x20E)):
            if mouse.buttons&flag:self.deliver(message)

    def window_proc(self,hwnd,message,wparam,lparam):
        try:
            if message==0xFF and self.enabled:
                packet=Packet();size=W.UINT(ctypes.sizeof(packet))
                got=self.u.GetRawInputData(W.HANDLE(lparam),0x10000003,ctypes.byref(packet),
                                          ctypes.byref(size),ctypes.sizeof(Header))
                if got==ctypes.sizeof(packet):self.consume(packet)
        finally:
            # Foreground WM_INPUT requires DefWindowProc for OS cleanup.
            return self.u.DefWindowProcW(hwnd,message,wparam,lparam)

    def close(self):
        if self.enabled:self.enable(False)
        if self.hwnd:self.u.DestroyWindow(self.hwnd);self.hwnd=None
        if self.registered:self.u.UnregisterClassW(self.name,self.instance);self.registered=False

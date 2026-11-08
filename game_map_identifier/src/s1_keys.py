"""Map bindings use Windows key identities for both recording and matching."""
from dataclasses import dataclass
import ctypes

LEGACY_KEYS = {'Caps Lock':0x14,'M':0x4D,'Tab':0x09}
CTRL,ALT,SHIFT,WIN = 1,2,4,8
MODIFIER = {0x11:CTRL,0xA2:CTRL,0xA3:CTRL,0x12:ALT,0xA4:ALT,0xA5:ALT,
            0x10:SHIFT,0xA0:SHIFT,0xA1:SHIFT,0x5B:WIN,0x5C:WIN}
MOD_NAMES = ((CTRL,'Ctrl'),(ALT,'Alt'),(SHIFT,'Shift'),(WIN,'Win'))
MOUSE = {1:'鼠标左键',2:'鼠标右键',4:'鼠标中键',5:'鼠标侧键 1',6:'鼠标侧键 2'}
# Windows shares VK_RETURN between the two Enter keys and navigation VKs
# between the keypad (Num Lock off) and the separate navigation block.
KEYPAD_NAV = {0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x2D,0x2E}
SPECIAL = {0x08:'Backspace',9:'Tab',0x0D:'Enter',0x1B:'Esc',0x14:'Caps Lock',
           0x20:'Space',0x5B:'左 Win',0x5C:'右 Win',0xA0:'左 Shift',0xA1:'右 Shift',
           0xA2:'左 Ctrl',0xA3:'右 Ctrl',0xA4:'左 Alt',0xA5:'右 Alt',0x90:'Num Lock',
           0x21:'Page Up',0x22:'Page Down',0x23:'End',0x24:'Home',0x25:'←',0x26:'↑',
           0x27:'→',0x28:'↓',0x2D:'Insert',0x2E:'Delete',0x10D:'小键盘 Enter'}


def physical_key(vk,extended):
    if (vk==0x0D and extended) or (vk in KEYPAD_NAV and not extended):
        return vk+256
    return vk


def valid_key(key):
    return type(key) is int and (0<key<255 or key==0x10D or key-256 in KEYPAD_NAV)


def key_name(key):
    if key in MOUSE:return MOUSE[key]
    if key in SPECIAL:return SPECIAL[key]
    if key>255:return '小键盘 '+key_name(key-256)
    if 0x30<=key<=0x39 or 0x41<=key<=0x5A:return chr(key)
    if 0x70<=key<=0x87:return 'F'+str(key-0x6F)
    if 0x60<=key<=0x69:return '小键盘 '+str(key-0x60)
    win=ctypes.windll.user32
    scan=win.MapVirtualKeyW(key,4)
    flags=((scan&0xff)<<16) | ((1<<24) if scan>>8 else 0)
    buffer=ctypes.create_unicode_buffer(64)
    if scan and win.GetKeyNameTextW(flags,buffer,len(buffer)):return buffer.value
    return f'VK {key:02X}'


@dataclass(frozen=True)
class MapBinding:
    key:int
    mods:int=0

    @property
    def label(self):
        return ' + '.join([name for bit,name in MOD_NAMES if self.mods&bit]+[key_name(self.key)])

    def saved(self):
        for name,key in LEGACY_KEYS.items():
            if self.key==key and not self.mods:return name
        return {'vk':self.key,'modifiers':self.mods}

    @classmethod
    def load(cls,value):
        if isinstance(value,str):
            key=LEGACY_KEYS.get(value)
            return cls(key) if key is not None else None
        if not isinstance(value,dict):return None
        key=value.get('vk');mods=value.get('modifiers',0)
        if not valid_key(key) or type(mods) is not int or not 0<=mods<=15 or mods&MODIFIER.get(key,0):
            return None
        return cls(key,mods)


class BindingCapture:
    """The UI owns one recording attempt. Only its focused window may supply input."""
    def __init__(self,hwnd):
        self.hwnd=hwnd
        self.mouse_box=(0,0,0,0)
        self.modifier_candidate=None
        self.result=None

    def observe(self,key,down,mods,hwnd,point=None):
        if self.result is not None:return
        if hwnd!=self.hwnd:
            self.modifier_candidate=None
            return
        if point is not None:
            left,top,right,bottom=self.mouse_box
            if not (left<=point[0]<right and top<=point[1]<bottom):
                self.modifier_candidate=None
                return
        if key in MODIFIER:
            if down:
                self.modifier_candidate=MapBinding(key,mods)
            elif self.modifier_candidate and key==self.modifier_candidate.key:
                self.result=self.modifier_candidate
        elif down and valid_key(key):
            self.result=MapBinding(key,mods)

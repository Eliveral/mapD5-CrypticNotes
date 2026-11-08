"""Record a key using the same Windows input path that wakes map detection."""
import ctypes
import tkinter as tk
from tkinter import ttk

from .s1_keys import BindingCapture
from .s1_theme import BG,SURFACE,JADE,MUTED,FONT,label


class MapKeyDialog(tk.Toplevel):
    def __init__(self,app):
        super().__init__(app)
        self.app=app
        self.title('绑定游戏地图键')
        self.transient(app);self.resizable(False,False)
        self.configure(bg=BG)
        self.pending=None
        self.capture=None
        self.timer=None
        self.protocol('WM_DELETE_WINDOW',self.dismiss)
        body=tk.Frame(self,bg=BG);body.pack(fill='both',padx=20,pady=16)
        label(body,'绑定游戏地图键',size=16,bold=True).pack(anchor='w')
        label(body,'按下游戏里打开地图的键或组合键。录入鼠标键时，请在下方框内按。',color=MUTED,
              wraplength=360).pack(anchor='w',pady=(6,12))
        self.value=tk.StringVar(value='等待按键…')
        self.pad=tk.Label(body,textvariable=self.value,bg=SURFACE,fg=JADE,width=28,
                          padx=8,pady=12,font=(FONT,12,'bold'),takefocus=True)
        self.pad.pack(fill='x')
        self.hint=tk.StringVar(value='')
        label(body,variable=self.hint,color=MUTED,wraplength=360).pack(anchor='w',pady=(6,10))
        buttons=tk.Frame(body,bg=BG);buttons.pack(fill='x')
        ttk.Button(buttons,text='取消' if app.binding else '稍后绑定',command=self.dismiss).pack(side='left')
        self.again=ttk.Button(buttons,text='重新录入',command=self.arm)
        self.again.pack(side='left',padx=6)
        self.save_button=ttk.Button(buttons,text='保存绑定',style='Primary.TButton',command=self.save)
        self.save_button.pack(side='right')
        self.update_idletasks()
        x=app.winfo_rootx()+(app.winfo_width()-self.winfo_reqwidth())//2
        y=app.winfo_rooty()+(app.winfo_height()-self.winfo_reqheight())//2
        self.geometry(f'+{max(0,x)}+{max(0,y)}')
        self.deiconify();self.update_idletasks()
        self.grab_set()
        self.pad.bind('<KeyPress>',self.ignore_typing)
        self.pad.bind('<KeyRelease>',self.ignore_typing)
        self.bind('<Configure>',lambda event:self.update_box())
        self.bind('<FocusOut>',lambda event:self.clear_partial())
        self.arm()
        self.poll()

    def ignore_typing(self,event):
        if not self.pending:return 'break'

    def clear_partial(self):
        if self.capture:self.capture.modifier_candidate=None

    def update_box(self):
        if self.capture:
            x,y=self.pad.winfo_rootx(),self.pad.winfo_rooty()
            self.capture.mouse_box=(x,y,x+self.pad.winfo_width(),y+self.pad.winfo_height())

    def arm(self):
        self.pending=None;self.value.set('等待按键…')
        self.save_button.state(['disabled']);self.again.state(['disabled'])
        self.hint.set('修饰键可以单独绑定，按下后松开即可。')
        from ctypes import wintypes
        ancestor=ctypes.windll.user32.GetAncestor
        ancestor.argtypes=[wintypes.HWND,wintypes.UINT]
        ancestor.restype=wintypes.HWND
        hwnd=ancestor(self.winfo_id(),2)
        self.capture=BindingCapture(hwnd)
        self.update_box()
        if self.app.live.hotkeys:self.app.live.hotkeys.capture=self.capture
        self.pad.focus_set()

    def poll(self):
        if self.capture and self.capture.result and not self.pending:
            self.pending=self.capture.result
            self.value.set(self.pending.label)
            self.save_button.state(['!disabled']);self.again.state(['!disabled'])
            self.hint.set('此键优先打开地图，对应助手操作仍可用窗口按钮。'
                          if not self.pending.mods and 0x76<=self.pending.key<=0x79
                          else '确认键位后，点击“保存绑定”。')
        observer=self.app.live.hotkeys
        if observer and not self.pending and (observer.keyboard_ready is False or observer.mouse_ready is False):
            unavailable='键盘' if observer.keyboard_ready is False else '鼠标'
            self.hint.set(f'{unavailable}监听未就绪。可稍后重新绑定。')
        self.timer=self.after(30,self.poll)

    def save(self):
        if self.pending:self.app.finish_map_binding(self,self.pending)

    def dismiss(self):
        self.app.finish_map_binding(self,None)

    def destroy(self):
        if self.timer:self.after_cancel(self.timer);self.timer=None
        hotkeys=getattr(self.app.live,'hotkeys',None)
        if hotkeys and hotkeys.capture is self.capture:hotkeys.capture=None
        super().destroy()

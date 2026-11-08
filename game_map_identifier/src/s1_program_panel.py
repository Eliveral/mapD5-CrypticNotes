"""Program updates use release artifacts. All network and disk work runs off Tk."""
from pathlib import Path
from queue import Empty,Queue
import shutil
import sys
import threading
import tkinter as tk
from tkinter import ttk

from .s1_theme import update_card
from . import s1_program_update as update


class ProgramUpdatePanel(ttk.Frame):
    def __init__(self,parent,on_notice,on_restart,can_update,resume_detection,auto_check):
        super().__init__(parent,padding=(4,16))
        self.notice=on_notice;self.restart=on_restart
        self.can_update=can_update;self.resume_detection=resume_detection
        self.events=Queue();self.phase='idle';self.release=None;self.closed=False
        self.exe=Path(sys.executable).resolve()
        self.packaged=bool(getattr(sys,'frozen',False)) and self.exe.name.casefold()=='mapd5.exe'
        self.status=tk.StringVar(value='');self.available=tk.StringVar(value='尚未检查')
        channel='正式版和测试版' if update.version(update.APP_TAG)[3]<3 else '正式版'
        body=update_card(self,'程序更新',f'当前版本   {update.APP_TAG}   ·   跟踪{channel}')
        ttk.Label(body,textvariable=self.available,style='Detail.TLabel',wraplength=640).pack(anchor='w',pady=(4,0))
        buttons=ttk.Frame(body,style='Card.TFrame');buttons.pack(anchor='w',pady=(14,8))
        self.check_button=ttk.Button(buttons,style='Primary.TButton',text='检查程序更新',command=self.check)
        self.check_button.pack(side='left',padx=(0,8))
        self.install_button=ttk.Button(buttons,text='一键更新并重启',command=self.install)
        self.install_button.pack(side='left');self.install_button.state(['disabled'])
        self.bar=ttk.Progressbar(body,length=300,maximum=100)
        self.message=ttk.Label(body,textvariable=self.status,style='Detail.TLabel',wraplength=620,justify='left')
        self.message.pack(anchor='w')
        last=self.exe.parent/'.mapd5-last-update.txt'
        if self.packaged and last.is_file():
            try:message=last.read_text('utf-8-sig')[:600]
            except OSError:message=''
            if message:ttk.Label(body,text=message,style='Detail.TLabel',wraplength=620,justify='left').pack(anchor='w',pady=(12,0))
        self.after(100,self.poll)
        if auto_check:self.after(1200,self.check)

    def set_phase(self,phase):
        self.phase=phase
        if phase=='installing':
            self.bar.pack(before=self.message,anchor='w',pady=(8,6))
        else:
            self.bar.pack_forget()
        available=bool(self.release and self.packaged)
        self.install_button.configure(style='Primary.TButton' if available else 'TButton')
        self.check_button.configure(style='TButton' if available else 'Primary.TButton')
        self.check_button.state(['!disabled' if phase=='idle' else 'disabled'])
        self.install_button.state(['!disabled' if phase=='idle' and self.release and self.packaged else 'disabled'])

    def check(self):
        if self.phase!='idle':return
        self.set_phase('checking');self.status.set('正在检查 GitHub 发布版本…')
        def work():
            try:self.events.put(('checked',update.latest_release()))
            except Exception as exc:self.events.put(('error',update.readable_error(exc)))
        threading.Thread(target=work,daemon=True).start()

    def install(self):
        if self.phase!='idle' or not self.release or not self.packaged:return
        if reason:=self.can_update():
            self.status.set(reason);return
        self.set_phase('installing');self.status.set('正在下载并校验更新包…');self.bar['value']=0
        release=self.release;resume=self.resume_detection()
        def work():
            tx=None
            try:
                def progress(done,total):
                    if self.closed:raise RuntimeError('下载已取消。')
                    self.events.put(('progress',int(done*100/total)))
                tx=update.prepare(release,self.exe,progress)
                if self.closed:raise RuntimeError('下载已取消。')
                self.events.put(('status','校验通过，正在准备重启…'))
                update.launch_worker(self.exe,tx,resume)
                self.events.put(('restart',None))
            except Exception as exc:
                if tx is not None and not (tx/'prepared').exists():shutil.rmtree(tx,ignore_errors=True)
                self.events.put(('error',update.readable_error(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='progress':
                    self.bar['value']=value
                    self.status.set(f'正在下载更新包　{value}%' if value<100 else '正在校验并准备安装…')
                elif kind=='status':self.status.set(value)
                elif kind=='restart':
                    self.restart();return
                elif kind=='error':
                    self.status.set(value);self.set_phase('idle')
                elif kind=='checked':
                    self.release=value
                    if value:
                        self.available.set(f'可更新到 {value.tag}　·　{value.title}')
                        self.status.set(f'下载大小 {value.size/1024**2:.1f} MiB。安装后自动重启。' if self.packaged
                                        else '源码模式可检查版本，请在 Windows 运行包中安装更新。')
                    else:
                        self.available.set('当前已是最新版本');self.status.set('')
                    self.notice(value.tag if value else None);self.set_phase('idle')
        except Empty:pass
        if not self.closed:self.after(100,self.poll)

    def destroy(self):
        self.closed=True
        super().destroy()

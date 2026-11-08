"""Map library installation and release checks on a background worker."""
from pathlib import Path
from queue import Queue, Empty
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog

from .s1_theme import update_card
from .s1_updates import profile, check_release, fetch, install, record


class UpdatePanel(ttk.Frame):
    def __init__(self,parent,bundled,on_install,current_root=None):
        super().__init__(parent,padding=(4,16))
        self.bank=profile()/'maps';self.bundled=bundled;self.on_install=on_install
        self.current=record(current_root).get('version',0) if current_root else 0
        self.events=Queue();self.busy=False;self.buttons=[]
        self.allow_install=lambda:True
        self.status=tk.StringVar(value='可从 GitHub 获取更新，也可以选择本地地图包。')
        current=f'已安装地图版本  {self.current}' if self.current else ('当前使用随程序提供的图库' if current_root else '首次使用，请先安装 S1 地图库')
        body=update_card(self,'地图更新',current)
        row=ttk.Frame(body,style='Card.TFrame');row.pack(anchor='w',pady=(14,0))
        if not current_root and bundled.is_file():self.button(row,'安装 S1 地图',lambda:self.start(self.bundled))
        self.button(row,'检查地图更新',lambda:self.start(None))
        self.button(row,'导入地图包…',self.choose)
        self.indicator=ttk.Progressbar(body,mode='indeterminate',length=260)
        self.message=ttk.Label(body,textvariable=self.status,wraplength=560,style='Detail.TLabel',justify='left')
        self.message.pack(anchor='w',pady=(10,0))
        self.after(100,self.poll)

    def button(self,row,text,command):
        button=ttk.Button(row,text=text,command=command,style='Primary.TButton' if not self.buttons else 'TButton');button.pack(side='left',padx=(0,8))
        self.buttons.append(button)

    def choose(self):
        name=filedialog.askopenfilename(parent=self,title='选择地图更新包',filetypes=[('地图包','*.zip')])
        if name:self.start(Path(name))

    def start(self,archive):
        if self.busy:return
        if not self.allow_install():
            self.status.set('正在更新程序，请等待重启。');return
        self.busy=True;self.indicator.pack(before=self.message,anchor='w',pady=(12,0));self.indicator.start(20)
        for b in self.buttons:b.state(['disabled'])
        self.status.set('正在检查地图包…' if archive else '正在连接发布仓库…')
        def work():
            try:
                def progress(message):self.events.put(('status',message))
                if archive is None:
                    release=check_release()
                    if self.current>=release['version']:
                        self.events.put(('done','地图已是最新版本。'));return
                    self.bank.mkdir(parents=True,exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=self.bank) as tmp:
                        path=Path(tmp)/'map.zip';progress('正在下载地图更新包…')
                        fetch(release['url'],path,512*1024**2,release['sha256'])
                        install(path,self.bank,progress,release['version'])
                else:install(archive,self.bank,progress)
                self.events.put(('installed','地图已安装，正在载入…'))
            except Exception as exc:
                self.events.put(('done','未安装地图。'+str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait();self.status.set(value)
                if kind!='status':
                    self.busy=False;self.indicator.stop();self.indicator.pack_forget()
                    for b in self.buttons:b.state(['!disabled'])
                    if kind=='installed':
                        self.on_install();return
        except Empty:pass
        self.after(100,self.poll)

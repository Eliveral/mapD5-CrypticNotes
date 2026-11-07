from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
from PIL import Image, ImageTk
from .config import project_root
from .s1_catalog import S1Catalog, read_image
from .s1_landmarks import RoomLabels, badge
from .s1_updates import map_root
from .s1_update_panel import UpdatePanel

BG='#111827'; PANEL='#1f2937'; FG='#e5e7eb'; MUTED='#a7b3c6'
DEFAULT_OVERLAY_OPACITY=28
GALLERY_KINDS={'高清单层':'hd','三层总览':'structure'}

class S1Window(tk.Tk):
    def __init__(self, autostart=True, controls=True, start_hidden=False):
        super().__init__()
        if start_hidden:self.withdraw()
        self.title('MapD5 · 加页手记地图助手 3.7.4')
        self.geometry('1100x760'); self.minsize(780,540); self.configure(bg=BG)
        self.reload_maps=False;self.resume_detection=autostart
        self.catalog=S1Catalog(map_root(project_root()))
        self.layouts=sorted({e['layout_id'][3:] for e in self.catalog.entries if e['kind']=='structure'})
        self.room_labels=RoomLabels(self.catalog);self.gallery_labels=[]
        self.kind=tk.StringVar(value='高清单层');self.note=tk.StringVar();self.status=tk.StringVar()
        self.last_gallery_floor='B1'
        self.current=None;self.original=None;self.photo=None;self.zoom=1;self.pan=[0,0]
        self.messages=queue.Queue();self.busy=False
        style=ttk.Style(self);style.theme_use('clam')
        style.configure('.',background=BG,foreground=FG,font=('Microsoft YaHei UI',10))
        style.configure('TButton',background=PANEL,padding=(10,6),font=('Microsoft YaHei UI',10))
        style.map('TButton',background=[('active','#334155')])
        style.configure('TNotebook',borderwidth=0,background=BG)
        style.configure('TNotebook.Tab',padding=(18,8),background=PANEL,foreground=MUTED)
        style.map('TNotebook.Tab',background=[('selected','#175b62')],foreground=[('selected','white')])
        style.configure('TCombobox',fieldbackground=PANEL,background=PANEL)
        style.map('TCombobox',fieldbackground=[('readonly',PANEL)],foreground=[('readonly',FG)])
        style.configure('Treeview',background=PANEL,fieldbackground=PANEL,foreground=FG,rowheight=31,font=('Microsoft YaHei UI',10))
        style.map('Treeview',background=[('selected','#176b76')])
        head=tk.Frame(self,bg=BG);head.pack(fill='x',padx=18,pady=(15,8))
        tk.Label(head,text='加页手记',bg=BG,fg=FG,font=('Microsoft YaHei UI',22,'bold')).pack(side='left')
        tk.Label(head,text='磷光症  /  S1 噩梦',bg=BG,fg=MUTED,font=('Microsoft YaHei UI',12)).pack(side='left',padx=14)
        tk.Label(head,text='MapD5  3.7.4',bg=BG,fg=MUTED).pack(side='right')
        primary=tk.Frame(self,bg=PANEL);primary.pack(fill='x',padx=18,pady=(4,10))
        self.detect_button=tk.Button(primary,text='▶ 一键开启检测匹配',command=lambda:self.live.start(True),
            bg='#0f766e',fg='white',activebackground='#115e59',activeforeground='white',
            font=('Microsoft YaHei UI',14,'bold'),relief='flat',cursor='hand2',padx=20,pady=13)
        self.detect_button.pack(side='right',padx=(4,12),pady=9)
        self.stop_button=ttk.Button(primary,text='停止检测',command=lambda:self.live.stop())
        self.stop_button.pack(side='right',padx=12)
        self.detect_state=tk.StringVar(value='检测未开启')
        tk.Label(primary,textvariable=self.detect_state,bg=PANEL,fg='#5eead4',font=('Microsoft YaHei UI',11,'bold'),anchor='w').pack(fill='x',padx=12,pady=(12,2))
        self.live_status=tk.StringVar(value='切回游戏，按地图键即可检测。')
        tk.Label(primary,textvariable=self.live_status,bg=PANEL,fg=MUTED,font=('Microsoft YaHei UI',10),anchor='w',wraplength=480).pack(fill='x',padx=12,pady=(0,8))
        tabs=ttk.Notebook(self);tabs.pack(fill='both',expand=True,padx=18,pady=(0,12))
        gallery=tk.Frame(tabs,bg=BG);settings=tk.Frame(tabs,bg=BG)
        self.update_panel=UpdatePanel(tabs,project_root()/'maps.zip',self.apply_maps,self.catalog.root.parent.parent)
        tabs.add(gallery,text='地图库');tabs.add(settings,text='检测设置');tabs.add(self.update_panel,text='地图更新')
        bar=tk.Frame(gallery,bg=BG);bar.pack(fill='x',pady=8)
        self.mode=ttk.Combobox(bar,textvariable=self.kind,values=tuple(GALLERY_KINDS),state='readonly',width=12)
        self.mode.pack(side='left');self.mode.bind('<<ComboboxSelected>>',lambda e:self.populate())
        self.file_button=ttk.Button(bar,text='从截图找候选',command=self.choose_screenshot);self.file_button.pack(side='left',padx=8)
        ttk.Button(bar,text='总图 / 单层',command=self.switch_pair).pack(side='left',padx=4)
        ttk.Button(bar,text='上下层对照',command=self.show_overlap).pack(side='left',padx=4)
        ttk.Button(bar,text='适应窗口',command=self.fit).pack(side='left',padx=4)
        ttk.Button(bar,text='原图大小',command=self.actual_size).pack(side='left',padx=4)
        ttk.Button(bar,text='查看图源',command=self.open_source).pack(side='right')
        tk.Label(settings,text='快捷操作',bg=BG,fg=FG,font=('Microsoft YaHei UI',13,'bold')).pack(anchor='w',padx=18,pady=(14,6))
        livebar=tk.Frame(settings,bg=PANEL);livebar.pack(fill='x',padx=18,pady=(0,8))
        ttk.Button(livebar,text='重新识别 F8',command=lambda:self.live.start(False)).pack(side='left',padx=3,pady=5)
        ttk.Button(livebar,text='恢复自动 F7',command=lambda:self.live.start(True)).pack(side='left',padx=3)
        ttk.Button(livebar,text='隐藏/恢复 F9',command=lambda:self.live.toggle_hidden()).pack(side='left',padx=3)
        ttk.Button(livebar,text='停止 F10',command=lambda:self.live.stop()).pack(side='left',padx=3)
        ttk.Button(livebar,text='框选地图区域',command=lambda:self.live.pick_region()).pack(side='left',padx=3)
        ttk.Button(livebar,text='恢复游戏全窗',command=lambda:self.live.reset_region()).pack(side='left',padx=3)
        tk.Label(settings,text='识别与显示',bg=BG,fg=FG,font=('Microsoft YaHei UI',13,'bold')).pack(anchor='w',padx=18,pady=(14,6))
        opts=tk.Frame(settings,bg=BG);opts.pack(fill='x',padx=18,pady=(0,6))
        self.layout_limit=tk.StringVar(value='自动');self.floor_limit=tk.StringVar(value='自动')
        self.preferences_path=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'CrypticNotesAssistantS1'/'preferences-v32.json'
        self.persist_preferences=controls
        self.preferences={};self.save_timer=None
        chosen_key='Caps Lock'
        opacity=DEFAULT_OVERLAY_OPACITY
        if controls:
            try:
                saved=json.loads(self.preferences_path.read_text(encoding='utf-8'))
                if isinstance(saved,dict):self.preferences=saved
                selected=self.preferences.get('map_key')
                if selected in ('Caps Lock','M','Tab'):chosen_key=selected
                opacity=max(15,min(90,int(self.preferences.get('clean_map_opacity',DEFAULT_OVERLAY_OPACITY))))
            except (OSError,ValueError,TypeError,OverflowError):pass
        self.map_key=tk.StringVar(value=chosen_key)
        self.overlay_style=tk.StringVar(value='高清底图');self.overlay_alpha=tk.DoubleVar(value=opacity)
        self.opacity_label=tk.StringVar(value=f'{opacity}%')
        tk.Label(opts,text='地图键',bg=BG,fg=FG).pack(side='left',padx=(8,4))
        keybox=ttk.Combobox(opts,textvariable=self.map_key,values=('Caps Lock','M','Tab'),state='readonly',width=10)
        keybox.pack(side='left');keybox.bind('<<ComboboxSelected>>',self.change_map_key)
        for label,var,values in [('布局',self.layout_limit,['自动']+self.layouts),('楼层',self.floor_limit,['自动','B1','B2','B3']),('叠加',self.overlay_style,['高清底图','结构'])]:
            tk.Label(opts,text=label,bg=BG,fg=FG).pack(side='left',padx=(8,4))
            box=ttk.Combobox(opts,textvariable=var,values=values,state='readonly',width=8)
            box.pack(side='left');box.bind('<<ComboboxSelected>>',lambda e:self.live.settings_changed())
        opacity_row=tk.Frame(settings,bg=BG);opacity_row.pack(fill='x',padx=26,pady=12)
        tk.Label(opacity_row,text='底图不透明度',bg=BG,fg=FG).pack(side='left',padx=(0,6))
        ttk.Scale(opacity_row,from_=15,to=90,length=235,variable=self.overlay_alpha,command=self.change_opacity).pack(side='left')
        tk.Label(opacity_row,textvariable=self.opacity_label,width=5,bg=BG,fg=FG).pack(side='left')
        ttk.Button(settings,text='厄运之女 · 原版识别',command=self.open_legacy).pack(anchor='w',padx=22,pady=10)
        panes=tk.PanedWindow(gallery,orient='horizontal',bg=BG,sashwidth=5);panes.pack(fill='both',expand=True)
        left=tk.Frame(panes,bg=PANEL,width=235);panes.add(left,minsize=200,width=235)
        scroll=ttk.Scrollbar(left,orient='vertical');scroll.pack(side='right',fill='y')
        self.tree=ttk.Treeview(left,show='tree',selectmode='browse',yscrollcommand=scroll.set);self.tree.pack(fill='both',expand=True)
        scroll.configure(command=self.tree.yview)
        self.tree.bind('<<TreeviewSelect>>',self.select)
        self.canvas=tk.Canvas(panes,bg='#0a0f18',highlightthickness=0);panes.add(self.canvas,minsize=385)
        self.canvas.bind('<Configure>',lambda e:self.render())
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<ButtonPress-1>',lambda e:setattr(self,'drag',(e.x,e.y)))
        self.canvas.bind('<B1-Motion>',self.drag_image)
        self.canvas.bind('<Double-Button-1>',lambda e:self.fit())
        tk.Label(gallery,textvariable=self.note,bg=BG,fg=FG,anchor='w',justify='left',font=('Microsoft YaHei UI',10)).pack(fill='x',padx=20,pady=(10,4))
        tk.Label(gallery,textvariable=self.status,bg=BG,fg='#7dd3cf',anchor='w',justify='left',wraplength=1180,font=('Microsoft YaHei UI',10)).pack(fill='x',padx=20,pady=(0,5))
        tk.Label(gallery,text='滚轮缩放  ·  拖动平移  ·  双击适应窗口',bg=BG,fg=MUTED,anchor='w',font=('Microsoft YaHei UI',9)).pack(fill='x',padx=20,pady=(0,12))
        from .s1_trigger_live import LiveController
        self.live=LiveController(self, autostart=autostart, controls=controls)
        self.protocol('WM_DELETE_WINDOW',self.destroy)
        self.populate();self.after(120,self.poll)

    def apply_maps(self):
        self.resume_detection=self.live.active;self.reload_maps=True
        self.destroy()

    def destroy(self):
        if getattr(self,'save_timer',None):
            self.save_preferences()
        if hasattr(self,'live') and not self.live.closed:self.live.close()
        super().destroy()

    def change_map_key(self,event=None):
        self.live.map_key_changed()
        self.save_preferences()

    def change_opacity(self,value):
        percent=round(float(value))
        self.overlay_alpha.set(percent)
        self.opacity_label.set(f'{percent}%')
        if hasattr(self,'live'):self.live.set_alpha(percent)
        if self.persist_preferences:
            if self.save_timer:self.after_cancel(self.save_timer)
            self.save_timer=self.after(250,self.save_preferences)

    def save_preferences(self):
        if self.save_timer:
            self.after_cancel(self.save_timer);self.save_timer=None
        if self.persist_preferences:
            try:
                self.preferences_path.parent.mkdir(parents=True,exist_ok=True)
                self.preferences.update(map_key=self.map_key.get(),clean_map_opacity=round(self.overlay_alpha.get()))
                temp=self.preferences_path.with_suffix(f'.{os.getpid()}.tmp')
                temp.write_text(json.dumps(self.preferences),encoding='utf-8');temp.replace(self.preferences_path)
            except OSError:
                self.live_status.set('设置已生效，本次未能保存。')

    def populate(self,ids=None):
        self.tree.delete(*self.tree.get_children())
        kind=GALLERY_KINDS[self.kind.get()]
        pool=(self.catalog.by_id[i] for i in ids) if ids is not None else self.catalog.entries
        entries=[e for e in pool if e['kind']==kind]
        for e in entries:self.tree.insert('', 'end', iid=e['id'],text=e['title'].replace('· 结构','· 三层总览'))
        if entries:
            self.tree.selection_set(entries[0]['id']);self.tree.focus(entries[0]['id'])
        if ids is None:
            self.status.set(f'{len(self.layouts)} 套噩梦地图 · 地下一至三层')

    def select(self,event=None):
        ids=self.tree.selection()
        if not ids:return
        e=self.catalog.by_id[ids[0]]
        if self.current is e and self.original is not None:return
        try:
            self.original,self.gallery_labels=self.room_labels.gallery(e)
        except (OSError,ValueError) as exc:
            messagebox.showerror('图片读取失败',str(exc),parent=self);return
        self.current=e
        if e.get('floor'):self.last_gallery_floor=e['floor']
        title=e['title'].replace('· 结构','· 三层总览')
        author=self.catalog.by_id['hd_'+e['layout_id'][3:]+'_B1'].get('author','')
        self.note.set(f"{title}  |  {self.original.width} × {self.original.height}  |  {author}")
        self.fit()

    def fit(self):
        if self.original is None:return
        w=max(1,self.canvas.winfo_width());h=max(1,self.canvas.winfo_height())
        self.zoom=min(w/self.original.width,h/self.original.height)*.97;self.pan=[0,0];self.render()

    def actual_size(self):self.zoom=1;self.pan=[0,0];self.render()

    def wheel(self,e):
        old=self.zoom;self.zoom=max(.08,min(5,self.zoom*(1.15 if e.delta>0 else 1/1.15)))
        cx=self.canvas.winfo_width()/2;cy=self.canvas.winfo_height()/2
        self.pan=[e.x-cx-(e.x-cx-self.pan[0])*self.zoom/old,e.y-cy-(e.y-cy-self.pan[1])*self.zoom/old]
        self.render()

    def drag_image(self,e):
        if not hasattr(self,'drag'):return
        self.pan[0]+=e.x-self.drag[0];self.pan[1]+=e.y-self.drag[1];self.drag=(e.x,e.y);self.render()

    def render(self):
        if self.original is None:return
        cw=max(1,self.canvas.winfo_width());ch=max(1,self.canvas.winfo_height())
        # Render the visible image crop, so deep zoom does not allocate a giant bitmap.
        x=(cw-self.original.width*self.zoom)/2+self.pan[0];y=(ch-self.original.height*self.zoom)/2+self.pan[1]
        box=(max(0,int(-x/self.zoom)),max(0,int(-y/self.zoom)),min(self.original.width,int((cw-x)/self.zoom)+1),min(self.original.height,int((ch-y)/self.zoom)+1))
        self.canvas.delete('all')
        if box[2]<=box[0] or box[3]<=box[1]:return
        im=self.original.crop(box);im=im.resize((max(1,round(im.width*self.zoom)),max(1,round(im.height*self.zoom))),Image.Resampling.LANCZOS)
        # Draw at screen resolution so text stays legible both at fit and at 1:1.
        for text,lx,ly in self.gallery_labels:
            if not (box[0]<=lx<box[2] and box[1]<=ly<box[3]):continue
            glyph=badge(text,16 if self.current['kind']=='structure' else 19)
            left=max(0,min(im.width-glyph.width,round((lx-box[0])*self.zoom-glyph.width/2)))
            top=max(0,min(im.height-glyph.height,round((ly-box[1])*self.zoom-glyph.height/2)))
            im.paste(glyph,(left,top),glyph)
        self.photo=ImageTk.PhotoImage(im)
        self.canvas.create_image(x+box[0]*self.zoom,y+box[1]*self.zoom,image=self.photo,anchor='nw')

    def choose_screenshot(self):
        if self.busy:return
        path=filedialog.askopenfilename(parent=self,title='选择游戏地图截图（尽量裁去地图外的界面）',filetypes=[('图片','*.png *.jpg *.jpeg *.webp *.bmp')])
        if not path:return
        self.busy=True;self.file_button.configure(state='disabled');self.status.set('正在本地比对地图结构…')
        def work():
            try:self.messages.put(('results',self.catalog.candidates(read_image(Path(path)))))
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            kind,data=self.messages.get_nowait();self.busy=False;self.file_button.configure(state='normal')
            if kind=='error':self.status.set('检索失败：'+data)
            elif not data:self.status.set('没有足够的匹配证据。请换一张暴露更多走廊结构的截图，或从目录手动查看。')
            else:
                self.kind.set('三层总览');self.populate([r['id'] for r in data])
                self.status.set('候选顺序：'+' / '.join(f"{r['id'].split('_')[-1]}（结构点 {r['inliers']}）" for r in data)+'。请核对地图；这不是识别成功或准确率。切换图库可恢复全部条目。')
        except queue.Empty:pass
        self.after(120,self.poll)

    def show_overlap(self):
        if not self.current:return
        from .s1_overlap import show_comparison
        return show_comparison(self,self.catalog,self.current['layout_id'],self.last_gallery_floor)

    def open_source(self):
        if self.current:
            layout=self.current['layout_id'].split('_')[-1]
            source=self.catalog.by_id['hd_'+layout+'_B1']['source']
            if source.startswith('https://'):webbrowser.open(source)

    def switch_pair(self):
        if not self.current:return
        to_single=self.current['kind']=='structure'
        layout=self.current['layout_id'].removeprefix('s1_')
        target=self.catalog.by_id[f'hd_{layout}_{self.last_gallery_floor}' if to_single else f'structure_{layout}']
        self.kind.set('高清单层' if to_single else '三层总览')
        self.populate();self.tree.selection_set(target['id']);self.tree.see(target['id'])

    def open_legacy(self):
        if not (project_root()/'models').is_dir():
            if messagebox.askyesno('厄运之女原版','厄运之女使用原项目的独立模型和地图库。打开原项目下载页？',parent=self):
                webbrowser.open('https://github.com/LD-592114824/IdentityV_Cryptic_Notes_Assistant/releases')
            return
        if getattr(self,'legacy_process',None) and self.legacy_process.poll() is None:
            self.live_status.set('原版窗口已在运行；关闭后自动恢复 S1 快捷键。');return
        if not messagebox.askokcancel('厄运之女原版模式','将打开原项目的困难/噩梦识别与叠加功能。该模式的旧 CNN 和楼层标定不适用于 S1 磷光症。请先在原程序设置中核对截图区域和分辨率。',parent=self):return
        cmd=[sys.executable,'--legacy'] if getattr(sys,'frozen',False) else [sys.executable,str(project_root()/'run_s1.py'),'--legacy']
        self.live.stop();self.live.hotkeys.close();self.live.suspended=True
        try:self.legacy_process=subprocess.Popen(cmd,cwd=str(project_root()))
        except OSError as exc:
            self.restore_s1_hotkeys();messagebox.showerror('原版启动失败',str(exc),parent=self);return
        self.live_status.set('S1 识别已暂停，快捷键交给原版；关闭原版窗口后自动恢复。')
        self.after(600,self.watch_legacy)

    def restore_s1_hotkeys(self):
        from .s1_live import Hotkeys
        self.live.hotkeys=Hotkeys(self.live.events,self.map_key.get());self.live.suspended=False
        self.live.start()

    def watch_legacy(self):
        if self.live.closed:return
        if self.legacy_process.poll() is None:self.after(600,self.watch_legacy)
        else:self.restore_s1_hotkeys()

def main():
    from .windows import enable_physical_pixel_coordinates
    import cv2
    cv2.setNumThreads(2)
    enable_physical_pixel_coordinates()
    resume='--standby' not in sys.argv
    while True:
        if not (map_root(project_root())/'catalog.json').is_file():
            setup=tk.Tk();setup.title('MapD5 · 安装地图');setup.geometry('660x295')
            panel=UpdatePanel(setup,project_root()/'maps.zip',setup.destroy);panel.pack(fill='both',expand=True)
            setup.mainloop()
            if not (map_root(project_root())/'catalog.json').is_file():return 0
        app=S1Window(autostart=resume);app.mainloop()
        if not app.reload_maps:return 0
        resume=app.resume_detection

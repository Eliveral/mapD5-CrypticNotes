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
from .s1_program_panel import ProgramUpdatePanel
from .s1_program_update import APP_TAG,startup_receipt

from .s1_theme import BG, JADE, MUTED, apply_theme
from . import s1_desktop as desktop
DEFAULT_OVERLAY_OPACITY=28

class S1Window(tk.Tk):
    def __init__(self, autostart=True, controls=True, start_hidden=False):
        super().__init__()
        if start_hidden:self.withdraw()
        self.title('MapD5 · 加页手记地图助手 '+APP_TAG[1:])
        self.geometry('1100x760'); self.minsize(780,540); self.configure(bg=BG)
        self.reload_maps=False;self.resume_detection=autostart
        self.catalog=S1Catalog(map_root(project_root()))
        self.layouts=sorted({e['layout_id'][3:] for e in self.catalog.entries if e['kind']=='structure'})
        self.room_labels=RoomLabels(self.catalog);self.gallery_labels=[]
        self.note=tk.StringVar();self.status=tk.StringVar()
        self.last_gallery_floor='B1'
        self.current=None;self.original=None;self.photo=None;self.zoom=1;self.pan=[0,0]
        self.messages=queue.Queue();self.busy=False
        apply_theme(self)
        self.fitted=True
        self.candidate_layouts=None
        tabs=ttk.Notebook(self)
        desktop.header(self,tabs,APP_TAG)
        tabs.pack(fill='both',expand=True,padx=20,pady=(0,10))
        tabs.enable_traversal()
        gallery=tk.Frame(tabs,bg=BG);settings=tk.Frame(tabs,bg=BG)
        self.update_panel=UpdatePanel(tabs,project_root()/'maps.zip',self.apply_maps,self.catalog.root.parent.parent)
        self.update_panel.allow_install=lambda:self.program_panel.phase!='installing'
        self.program_panel=ProgramUpdatePanel(tabs,self.program_notice,self.destroy,self.can_update_program,
                                              lambda:self.live.active,auto_check=controls)
        tabs.add(gallery,text='地图库');tabs.add(settings,text='检测设置')
        tabs.add(self.update_panel,text='地图更新');tabs.add(self.program_panel,text='程序更新')
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
        desktop.gallery(self,gallery)
        desktop.settings(self,settings)
        from .s1_trigger_live import LiveController
        self.live=LiveController(self, autostart=autostart, controls=controls)
        self.update_detection_action()
        self.protocol('WM_DELETE_WINDOW',self.destroy)
        self.populate();self.after(120,self.poll)

    def toggle_detection(self):
        if self.live.active and self.live.continuous and not self.live.hidden:
            self.live.stop()
        else:
            self.live.start(True)

    def update_detection_action(self):
        controller=getattr(self,'live',None)
        active=bool(controller and controller.active)
        running=active and controller.continuous and not controller.hidden
        self.detect_light.itemconfigure('lamp',fill=JADE if active else MUTED)
        self.detection_action.configure(text='停止检测' if running else ('恢复自动检测' if active else '开启检测'),
                                        style='TButton' if running else 'Primary.TButton')

    def open_program_update(self,tabs):
        tabs.select(self.program_panel)
        if self.program_panel.release:self.program_panel.install()

    def program_notice(self,tag):
        self.version_label.set(('一键更新到 '+tag) if tag else APP_TAG)

    def can_update_program(self):
        if self.update_panel.busy:return '请等待地图更新结束，再更新程序。'
        if getattr(self,'legacy_process',None) and self.legacy_process.poll() is None:return '请先关闭原版模式，再更新程序。'
        return ''

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
        selected=self.tree.selection()
        if ids is not None:
            self.candidate_layouts=list(dict.fromkeys(self.catalog.by_id[i]['layout_id'][3:] for i in ids))
            self.clear_candidates.grid(row=2,column=0,columnspan=2,sticky='ew',padx=8,pady=(5,0))
            selected=()
        layouts=self.layouts if self.candidate_layouts is None else self.candidate_layouts
        self.tree.delete(*self.tree.get_children())
        self.count.set(f'{len(layouts)} 套')
        for layout in layouts:
            self.tree.insert('', 'end', iid=layout,text='   布局  '+layout)
        if layouts:
            target=selected[0] if selected and selected[0] in layouts else layouts[0]
            self.tree.selection_set(target);self.tree.focus(target);self.tree.see(target)
            self.select()

    def clear_search(self):
        self.candidate_layouts=None
        self.clear_candidates.grid_remove()
        self.status.set('');self.populate()

    def change_floor(self):
        self.select()

    def select(self,event=None):
        selected=self.tree.selection()
        if not selected:return
        layout=selected[0]
        floor=self.view_floor.get()
        key='structure_'+layout if floor=='all' else f'hd_{layout}_{floor}'
        e=self.catalog.by_id[key]
        if self.current is e and self.original is not None:return
        try:
            self.original,self.gallery_labels=self.room_labels.gallery(e)
        except (OSError,ValueError) as exc:
            messagebox.showerror('图片读取失败',str(exc),parent=self);return
        self.current=e
        if e.get('floor'):self.last_gallery_floor=e['floor']
        self.gallery_title.set(f'布局 {layout}   /   {desktop.FLOORS[floor]}')
        author=self.catalog.by_id['hd_'+layout+'_B1'].get('author','')
        self.note.set(f"{self.original.width} × {self.original.height}    ·    {author}")
        self.fit()

    def fit(self):
        self.fitted=True
        if self.original is None:return
        w=self.canvas.winfo_width();h=self.canvas.winfo_height()
        if min(w,h)<2:return
        self.zoom=min(w/self.original.width,h/self.original.height)*.97
        self.pan=[0,0];self.render()

    def actual_size(self):
        self.fitted=False;self.zoom=1;self.pan=[0,0];self.render()

    def wheel(self,e):
        self.fitted=False
        old=self.zoom;self.zoom=max(.08,min(5,self.zoom*(1.15 if e.delta>0 else 1/1.15)))
        cx=self.canvas.winfo_width()/2;cy=self.canvas.winfo_height()/2
        self.pan=[e.x-cx-(e.x-cx-self.pan[0])*self.zoom/old,e.y-cy-(e.y-cy-self.pan[1])*self.zoom/old]
        self.render()

    def drag_image(self,e):
        if not hasattr(self,'drag'):return
        self.fitted=False
        self.pan[0]+=e.x-self.drag[0];self.pan[1]+=e.y-self.drag[1];self.drag=(e.x,e.y);self.render()

    def render(self):
        if self.original is None:return
        self.zoom_label.set(f'{self.zoom:.0%}')
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
                self.view_floor.set('all');self.populate([r['id'] for r in data])
                self.status.set('候选顺序：'+' / '.join(f"{r['id'].split('_')[-1]}（结构点 {r['inliers']}）" for r in data)+'。请对照游戏画面确认。')
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
        target=self.last_gallery_floor if self.view_floor.get()=='all' else 'all'
        self.floor_buttons[target].invoke()

    def open_legacy(self):
        if self.program_panel.phase=='installing':
            self.live_status.set('正在更新程序，请等待自动重启。');return
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
    token=None
    if '--program-update-ready' in sys.argv:
        try:
            token=sys.argv[sys.argv.index('--program-update-ready')+1]
            startup_receipt(Path(sys.executable),token)
        except (IndexError,OSError,ValueError):
            return 2
    def ready(window):
        def acknowledge():
            nonlocal token
            if token:
                startup_receipt(Path(sys.executable),token,ready=True)
                token=None
        if token:window.after_idle(acknowledge)
    while True:
        if not (map_root(project_root())/'catalog.json').is_file():
            setup=tk.Tk();setup.title('MapD5 · 首次使用');setup.geometry('680x390');apply_theme(setup)
            tabs=ttk.Notebook(setup);tabs.pack(fill='both',expand=True)
            maps=UpdatePanel(tabs,project_root()/'maps.zip',setup.destroy)
            program=ProgramUpdatePanel(tabs,
                lambda tag:tabs.tab(program,text='程序更新（有新版）' if tag else '程序更新'),
                setup.destroy,lambda:'请等待地图安装结束。' if maps.busy else '',lambda:resume,auto_check=True)
            maps.allow_install=lambda:program.phase!='installing'
            tabs.add(maps,text='安装地图');tabs.add(program,text='程序更新')
            ready(setup);setup.mainloop()
            if not (map_root(project_root())/'catalog.json').is_file():return 0
        app=S1Window(autostart=resume);ready(app);app.mainloop()
        if not app.reload_maps:return 0
        resume=app.resume_detection

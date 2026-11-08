"""Window layout for the map browser and detection controls."""
import tkinter as tk
from tkinter import ttk

from .s1_theme import BG,SURFACE,RAISED,LINE,TEXT,MUTED,JADE,VIEW,label,atlas_icon

FLOORS = {'B1':'负一层', 'B2':'负二层', 'B3':'负三层', 'all':'总览'}


def header(app,tabs,version):
    head=tk.Frame(app,bg=BG)
    head.pack(fill='x',padx=20,pady=(12,8))
    atlas_icon(head).pack(side='left',padx=(0,8))
    label(head,'加页手记',size=19,bold=True).pack(side='left')
    label(head,'MapD5',size=11,color=JADE,bold=True).pack(side='left',padx=(10,16))
    label(head,'磷光症  /  S1 噩梦',size=10,color=MUTED).pack(side='left')
    app.version_label=tk.StringVar(value=version)
    ttk.Button(head,textvariable=app.version_label,style='Quiet.TButton',
               command=lambda:app.open_program_update(tabs)).pack(side='right')
    panel=tk.Frame(app,bg=SURFACE)
    panel.pack(fill='x',padx=20,pady=(0,8))
    panel.columnconfigure(1,weight=1)
    app.detect_state=tk.StringVar(value='检测未开启')
    app.live_status=tk.StringVar(value='切回第五人格，打开游戏地图即可匹配。')
    app.round_status=tk.StringVar(value='等待负二确认本局布局')
    app.detect_light=tk.Canvas(panel,width=10,height=10,bg=SURFACE,highlightthickness=0)
    app.detect_light.grid(row=0,column=0,padx=(14,7),pady=(10,0))
    app.detect_light.create_oval(1,1,9,9,fill=MUTED,outline='',tags='lamp')
    label(panel,variable=app.detect_state,size=11,bold=True).grid(row=0,column=1,sticky='w',pady=(10,0))
    app.detection_detail=label(panel,variable=app.live_status,size=9,color=MUTED)
    app.detection_detail.grid(row=1,column=1,sticky='ew',pady=(3,9),padx=(0,8))
    app.round_detail=label(panel,variable=app.round_status,size=9,color=MUTED,wraplength=235)
    app.round_detail.grid(row=0,column=2,rowspan=2,padx=8)
    ttk.Button(panel,text='新对局',style='Quiet.TButton',command=lambda:app.live.new_round()).grid(
        row=0,column=3,rowspan=2,padx=(0,6))
    app.detection_action=ttk.Button(panel,text='开启检测',style='Primary.TButton',
                                    command=app.toggle_detection)
    app.detection_action.grid(row=0,column=4,rowspan=2,padx=(0,14))
    def wrap(event):
        app.detection_detail.configure(wraplength=max(150,event.width-8))
    app.detection_detail.bind('<Configure>',wrap)
    app.detect_state.trace_add('write',lambda *_:app.update_detection_action())


def gallery(app,page):
    work=tk.Frame(page,bg=BG)
    work.pack(fill='both',expand=True,pady=(10,0))
    work.columnconfigure(1,weight=1);work.rowconfigure(0,weight=1)
    index=tk.Frame(work,bg=SURFACE,width=158)
    index.grid(row=0,column=0,sticky='nsew',padx=(0,8))
    index.grid_propagate(False);index.columnconfigure(0,weight=1);index.rowconfigure(1,weight=1)
    cap=tk.Frame(index,bg=SURFACE);cap.grid(row=0,column=0,columnspan=2,sticky='ew',padx=12,pady=(12,8))
    label(cap,'布局目录',bold=True).pack(side='left')
    app.count=tk.StringVar()
    label(cap,variable=app.count,size=9,color=MUTED).pack(side='right')
    app.tree=ttk.Treeview(index,show='tree',selectmode='browse',takefocus=True)
    app.tree.column('#0',width=124,stretch=True)
    app.tree.grid(row=1,column=0,sticky='nsew',padx=(4,0))
    scroll=ttk.Scrollbar(index,orient='vertical',command=app.tree.yview)
    scroll.grid(row=1,column=1,sticky='ns');app.tree.configure(yscrollcommand=scroll.set)
    app.tree.bind('<<TreeviewSelect>>',app.select)
    app.file_button=ttk.Button(index,text='从截图找候选',command=app.choose_screenshot)
    app.file_button.grid(row=3,column=0,columnspan=2,sticky='ew',padx=8,pady=8)
    app.clear_candidates=ttk.Button(index,text='显示全部布局',style='Quiet.TButton',command=app.clear_search)
    pane=tk.Frame(work,bg=SURFACE)
    pane.grid(row=0,column=1,sticky='nsew')
    pane.columnconfigure(0,weight=1);pane.rowconfigure(1,weight=1)
    top=tk.Frame(pane,bg=SURFACE);top.grid(row=0,column=0,sticky='ew',padx=12,pady=(8,7))
    app.gallery_title=tk.StringVar(value='地图库')
    label(top,variable=app.gallery_title,size=14,bold=True).pack(side='left')
    levels=tk.Frame(top,bg=SURFACE);levels.pack(side='right')
    app.view_floor=tk.StringVar(value='B1')
    app.floor_buttons={}
    for key,title in FLOORS.items():
        button=ttk.Radiobutton(levels,text=title,value=key,variable=app.view_floor,
                               style='Floor.TRadiobutton',command=app.change_floor)
        button.pack(side='left',padx=(2,0));app.floor_buttons[key]=button
    def adjacent(event,step):
        choices=list(FLOORS)
        target=choices[(choices.index(app.view_floor.get())+step)%len(choices)]
        app.floor_buttons[target].focus_set()
        app.floor_buttons[target].invoke()
        return 'break'
    for button in app.floor_buttons.values():
        button.bind('<Left>',lambda e:adjacent(e,-1))
        button.bind('<Right>',lambda e:adjacent(e,1))
    app.canvas=tk.Canvas(pane,bg=VIEW,highlightthickness=0,bd=0)
    app.canvas.grid(row=1,column=0,sticky='nsew',padx=1)
    app.canvas.bind('<Configure>',lambda e:app.fit() if app.fitted else app.render())
    app.canvas.bind('<MouseWheel>',app.wheel)
    app.canvas.bind('<ButtonPress-1>',lambda e:setattr(app,'drag',(e.x,e.y)))
    app.canvas.bind('<B1-Motion>',app.drag_image)
    app.canvas.bind('<Double-Button-1>',lambda e:app.fit())
    toolbar=tk.Frame(pane,bg=SURFACE);toolbar.grid(row=2,column=0,sticky='ew',padx=8,pady=5)
    ttk.Button(toolbar,text='上下层对照',command=app.show_overlap).pack(side='left')
    ttk.Button(toolbar,text='适应窗口',style='Quiet.TButton',command=app.fit).pack(side='left',padx=(6,0))
    ttk.Button(toolbar,text='原图大小',style='Quiet.TButton',command=app.actual_size).pack(side='left')
    app.zoom_label=tk.StringVar()
    label(toolbar,variable=app.zoom_label,size=9,color=MUTED).pack(side='left',padx=8)
    ttk.Button(toolbar,text='查看图源',style='Quiet.TButton',command=app.open_source).pack(side='right')
    foot=tk.Frame(page,bg=BG);foot.pack(fill='x',pady=(6,0))
    label(foot,variable=app.note,size=9,color=MUTED).pack(side='left')
    label(foot,'滚轮缩放   ·   拖动平移',size=9,color=MUTED).pack(side='right')
    app.feedback=label(page,variable=app.status,size=9,color=JADE,wraplength=680)
    def feedback(*_):
        if app.status.get():app.feedback.pack(anchor='w',pady=(3,0))
        else:app.feedback.pack_forget()
    app.status.trace_add('write',feedback)


def settings(app,page):
    # Put the most frequently adjusted controls first. Both columns fit the minimum window.
    area=tk.Frame(page,bg=BG)
    area.pack(fill='both',expand=True,pady=(12,0))
    area.columnconfigure(0,weight=1,uniform='settings')
    area.columnconfigure(1,weight=1,uniform='settings')
    def card(column,name):
        frame=tk.Frame(area,bg=SURFACE)
        frame.grid(row=0,column=column,sticky='nsew',padx=(0,4) if column==0 else (4,0))
        label(frame,name,size=13,bold=True).pack(anchor='w',padx=14,pady=(10,5))
        return frame
    view=card(0,'识别与显示')
    binding_row=tk.Frame(view,bg=SURFACE);binding_row.pack(fill='x',padx=14,pady=2)
    label(binding_row,'游戏地图键',color=MUTED).pack(side='left')
    app.bind_key_button=ttk.Button(binding_row,textvariable=app.map_key,command=app.request_map_binding)
    app.bind_key_button.pack(side='right')
    for name,var,values,callback in [
        ('限定布局',app.layout_limit,['自动']+app.layouts,lambda e:app.live.settings_changed()),
        ('限定楼层',app.floor_limit,('自动','B1','B2','B3'),lambda e:app.live.settings_changed()),
        ('覆盖样式',app.overlay_style,('高清底图','结构'),lambda e:app.live.settings_changed())]:
        row=tk.Frame(view,bg=SURFACE);row.pack(fill='x',padx=14,pady=2)
        label(row,name,color=MUTED).pack(side='left')
        box=ttk.Combobox(row,textvariable=var,values=values,state='readonly',width=13)
        box.pack(side='right');box.bind('<<ComboboxSelected>>',callback)
    opacity=tk.Frame(view,bg=SURFACE);opacity.pack(fill='x',padx=14,pady=(6,0))
    label(opacity,'底图不透明度',color=MUTED).pack(side='left')
    label(opacity,variable=app.opacity_label,bold=True).pack(side='right')
    ttk.Scale(view,from_=15,to=90,variable=app.overlay_alpha,command=app.change_opacity).pack(
        fill='x',padx=14,pady=(2,10))
    actions=card(1,'快捷操作')
    keys=tk.Frame(actions,bg=SURFACE);keys.pack(fill='x',padx=12,pady=(3,4))
    keys.columnconfigure((0,1),weight=1,uniform='keys')
    for i,(name,call) in enumerate([
        ('重新识别',lambda:app.live.start(False)),('恢复自动',lambda:app.live.start(True)),
        ('隐藏 / 恢复',lambda:app.live.toggle_hidden()),('停止检测',lambda:app.live.stop()),
        ('框选地图区域',lambda:app.live.pick_region()),('恢复游戏全窗',lambda:app.live.reset_region())]):
        button=ttk.Button(keys,text=name,command=call)
        button.grid(row=i//2,column=i%2,sticky='ew',padx=2,pady=3)
        if i<4:app.shortcut_buttons[(0x77,0x76,0x78,0x79)[i]]=(button,name)
    ttk.Button(actions,text='厄运之女 · 原版识别',style='Quiet.TButton',command=app.open_legacy).pack(
        anchor='w',padx=14,pady=(4,10))

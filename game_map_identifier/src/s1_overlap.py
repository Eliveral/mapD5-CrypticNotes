"""Compare two native HD floors in the same measured coordinate system."""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from .s1_alignment import geometry
from .s1_floor_transition import FloorCoordinates
from .s1_landmarks import badge


def comparison(catalog, coordinates, layout, first, second, width=900, height=590):
    shift = coordinates.offsets.get((layout,first,second))
    if shift is None:
        raise ValueError('这两张底图缺少已核验的对应坐标。')
    entries = [catalog.by_id[f'hd_{layout[3:]}_{floor}'] for floor in (first,second)]
    origins = np.array([[0.,0.],-shift])
    corner = origins.min(axis=0)
    extent = np.max(origins+[entry['size'] for entry in entries],axis=0)-corner
    scale = min((width-22)/extent[0],(height-22)/extent[1])
    origin = (np.array([width,height])-extent*scale)/2-corner*scale
    layers = []
    for entry,offset in zip(entries,origins):
        with Image.open(catalog.path(entry)) as image:
            size = tuple(max(1,round(n*scale)) for n in image.size)
            reduced = image.convert('RGB').resize(size,Image.Resampling.LANCZOS)
        mask = geometry(cv2.cvtColor(np.asarray(reduced),cv2.COLOR_RGB2BGR))
        transform = np.eye(2,3)
        transform[:,2] = origin+offset*scale
        layers.append(cv2.warpAffine(mask,transform,(width,height),flags=cv2.INTER_NEAREST))
    first_mask,second_mask = layers
    rgb = np.full((height,width,3),(17,24,39),np.uint8)
    for mask,colour in [(first_mask,(245,172,79)),(second_mask,(71,192,231))]:
        rgb[mask>0]=colour
        rgb[mask==255]=np.array(colour)*.72
    shared=(first_mask>0)&(second_mask>0)
    rgb[shared]=(217,228,226)
    rgb[shared&(first_mask==255)&(second_mask==255)]=(150,174,180)
    result=Image.fromarray(rgb)
    draw=ImageDraw.Draw(result)
    for name,point,_ in coordinates.passages.get((layout,first,second),[]):
        x,y=np.rint(origin+point*scale).astype(int)
        draw.ellipse((x-6,y-6,x+6,y+6),fill='#19212a',outline='white',width=2)
        label=badge(name,15)
        l=min(max(0,x+9),width-label.width)
        t=min(max(0,y-label.height//2),height-label.height)
        result.paste(label,(l,t),label)
    return result


def show_comparison(parent,catalog,layout,last_floor):
    window=tk.Toplevel(parent)
    window.title(f'布局 {layout[3:]} · 上下层对照')
    window.configure(bg='#111827')
    coordinates=FloorCoordinates(catalog)
    controls=tk.Frame(window,bg='#111827')
    controls.pack(fill='x',padx=12,pady=8)
    legend=tk.Label(window,bg='#111827',fg='#e5e7eb')
    legend.pack(anchor='w',padx=17)
    view=tk.Label(window,bg='#111827')
    view.pack(padx=10,pady=8)
    def select(first,second):
        legend.configure(text=f'橙色  {first}      蓝色  {second}      灰白  两层重叠')
        try:
            image=comparison(catalog,coordinates,layout,first,second,
                             min(900,window.winfo_screenwidth()-80),
                             min(590,window.winfo_screenheight()-180))
        except (OSError,ValueError,KeyError) as exc:
            view.image=None
            view.configure(image='',text=str(exc),fg='white')
            return
        view.image=ImageTk.PhotoImage(image,master=window)
        view.configure(image=view.image,text='')
    for title,first,second in [('负一 / 负二','B1','B2'),('负二 / 负三','B2','B3')]:
        ttk.Button(controls,text=title,command=lambda a=first,b=second:select(a,b)).pack(side='left',padx=3)
    select('B1','B2') if last_floor=='B1' else select('B2','B3')
    return window

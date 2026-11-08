"""Shared Windows desktop styling for the S1 atlas and its update screens."""
import tkinter as tk
from tkinter import ttk

BG = '#101819'
SURFACE = '#192425'
RAISED = '#243333'
LINE = '#364748'
TEXT = '#eef3ed'
MUTED = '#adbfba'
JADE = '#90e3bc'
INK = '#10291e'
VIEW = '#101a1b'
FONT = 'Microsoft YaHei UI'


def apply_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')
    root.configure(bg=BG)
    root.option_add('*TCombobox*Listbox.background', RAISED)
    root.option_add('*TCombobox*Listbox.foreground', TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground', '#35584c')
    root.option_add('*TCombobox*Listbox.selectForeground', TEXT)
    style.configure('.', background=BG, foreground=TEXT, font=(FONT, 10),
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                    focuscolor=JADE, troughcolor=BG)
    style.configure('Card.TFrame', background=SURFACE)
    style.configure('Card.TLabel', background=SURFACE, foreground=TEXT)
    style.configure('Detail.TLabel', background=SURFACE, foreground=MUTED)
    style.configure('TButton', background=RAISED, foreground=TEXT, relief='flat',
                    padding=(12, 6), borderwidth=0, anchor='center', focuscolor=JADE)
    style.map('TButton', background=[('disabled', SURFACE), ('pressed', LINE), ('active', '#324747')],
              foreground=[('disabled', '#80928d')])
    style.configure('Primary.TButton', background=JADE, foreground=INK, focuscolor=INK,
                    font=(FONT, 10, 'bold'), padding=(16, 6))
    style.map('Primary.TButton', background=[('disabled', RAISED), ('pressed', '#76c9a5'), ('active', '#acefcc')],
              foreground=[('disabled', MUTED), ('!disabled', INK)])
    style.configure('Quiet.TButton', background=SURFACE, foreground=MUTED, padding=(9, 5))
    style.configure('TNotebook', background=BG, borderwidth=0, tabmargins=0)
    style.configure('TNotebook.Tab', background=BG, foreground=MUTED, padding=(17, 7),
                    borderwidth=0, focuscolor=JADE)
    style.map('TNotebook.Tab', background=[('selected', RAISED), ('active', SURFACE)],
              foreground=[('selected', JADE), ('active', TEXT)])
    # The native clam tab paints a bevel even with borderwidth=0.
    # A stretchable 5-pixel tile leaves the selected underline visible at any DPI.
    def tile(bg,underline):
        image=tk.PhotoImage(master=root,width=5,height=5)
        image.put(bg,to=(0,0,5,5));image.put(underline,to=(0,3,5,5))
        return image
    normal,selected,hover=tile(BG,BG),tile(SURFACE,JADE),tile(SURFACE,SURFACE)
    root._tab_tiles=(normal,selected,hover)
    style.element_create('Atlas.nav','image',normal,('selected',selected),('active',hover),
                         border=(1,1,1,2),sticky='nswe')
    style.layout('TNotebook.Tab',[('Atlas.nav', {'sticky':'nswe','children': [
        ('Notebook.padding', {'sticky':'nswe','children': [
            ('Notebook.focus', {'sticky':'nswe','children': [('Notebook.label', {'sticky':'nswe'})]})]})]})])
    style.layout('TNotebook',[('Notebook.padding',{'sticky':'nswe'})])
    style.map('TNotebook.Tab',expand=[('selected',(0,0,0,0))])
    style.configure('Floor.TRadiobutton', background=SURFACE, foreground=MUTED, padding=(10, 5),
                    relief='flat', borderwidth=0)
    style.layout('Floor.TRadiobutton', [('Radiobutton.padding', {'children': [
        ('Radiobutton.focus', {'children': [('Radiobutton.label', {'sticky':'nswe'})]})]})])
    style.map('Floor.TRadiobutton', background=[('selected', '#2c4f40'), ('active', RAISED)],
              foreground=[('selected', JADE), ('active', TEXT)])
    style.configure('Treeview', background=SURFACE, fieldbackground=SURFACE, foreground=MUTED,
                    borderwidth=0, relief='flat', rowheight=34, indent=0, font=(FONT, 10))
    style.map('Treeview', background=[('selected','#294a3d')], foreground=[('selected',JADE)])
    style.configure('Vertical.TScrollbar', background=LINE, troughcolor=SURFACE, borderwidth=0,
                    arrowsize=10, gripcount=0)
    style.configure('TCombobox', padding=(8,4), arrowsize=12, arrowcolor=MUTED, background=RAISED,
                    borderwidth=1)
    style.map('TCombobox', fieldbackground=[('readonly',RAISED)], foreground=[('readonly',TEXT)],
              selectbackground=[('readonly',RAISED)], selectforeground=[('readonly',TEXT)])
    style.configure('Horizontal.TProgressbar', background=JADE, troughcolor=RAISED, thickness=4,
                    borderwidth=0, lightcolor=JADE, darkcolor=JADE)
    style.configure('Horizontal.TScale', background=MUTED, troughcolor=LINE, borderwidth=0)


def label(parent, text=None, *, variable=None, size=10, color=TEXT, bold=False, **options):
    return tk.Label(parent, text=text, textvariable=variable, bg=parent['bg'], fg=color,
                    font=(FONT, size, 'bold' if bold else 'normal'), anchor='w', justify='left',
                    padx=0, pady=0, **options)


def atlas_icon(parent):
    mark=tk.Canvas(parent,width=37,height=37,bg=parent['bg'],highlightthickness=0)
    # Three folded panels of a map, readable at title-bar scale.
    mark.create_polygon(2,10,12,5,12,26,2,31, fill='#589e80', outline='')
    mark.create_polygon(13,5,23,10,23,31,13,26, fill='#9ce6bc', outline='')
    mark.create_polygon(24,10,34,5,34,26,24,31, fill='#6ac89e', outline='')
    return mark


def update_card(panel,title,version):
    ttk.Label(panel,text=title,font=(FONT,18,'bold')).pack(anchor='w',pady=(0,12))
    body=ttk.Frame(panel,style='Card.TFrame',padding=(18,14));body.pack(fill='x')
    ttk.Label(body,text=version,style='Card.TLabel',font=(FONT,11,'bold')).pack(anchor='w')
    return body

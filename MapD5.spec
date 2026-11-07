from pathlib import Path

project=Path(SPECPATH)/'game_map_identifier'
a=Analysis([str(project/'run_s1.py')],pathex=[str(project)],
    hiddenimports=['cv2','PIL','PIL._tkinter_finder','mss','pygetwindow','yaml'],
    excludes=['torch','torchvision','onnx','onnxscript','matplotlib','sklearn','scipy','pytest'])
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='MapD5',console=False)
dist=COLLECT(exe,a.binaries,a.datas,name='MapD5')

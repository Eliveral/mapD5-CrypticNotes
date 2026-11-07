"""Build a map-only update. Public packages link to pinned HD originals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote
from zipfile import ZipFile, ZIP_DEFLATED
import cv2
import numpy as np

SOURCE=Path(__file__).resolve().parent/'game_map_identifier'
sys.path.insert(0,str(SOURCE))
from src.s1_alignment import FloorMatcher
from src.s1_catalog import S1Catalog
from src.s1_updates import CONFIG,PREFIX


def build(source,destination,version):
    cv2.setNumThreads(2)
    maps=source/PREFIX;catalog=S1Catalog(maps);matcher=FloorMatcher(catalog)
    files={};remote={};entries=[];floors=[]
    for name in CONFIG:files[name]=(source/name).read_bytes()
    for floor in matcher.floors:
        floors.append({k:floor[k] for k in ('id','entry_id','layout_id','floor','box','shared_layouts') if k in floor})
    def encode(data):return json.dumps(data,ensure_ascii=False,indent=2).encode()
    for e in catalog.entries:
        if e['kind'] not in ('structure','hd'):continue
        item=dict(e)
        if e['kind']=='structure':
            # Distribute fresh category diagrams rather than video frames.
            w,h=e['size'];canvas=np.zeros((h,w,3),np.uint8)
            for floor in (f for f in matcher.floors if f['entry_id']==e['id']):
                x,y,r,b=floor['box'];mask=matcher.template(floor)[1];area=canvas[y:b,x:r]
                area[mask==127]=(82,116,157);area[mask==255]=(146,110,93)
            data=cv2.imencode('.png',canvas)[1].tobytes()
            files[PREFIX+e['path']]=data
            item.update(sha256=hashlib.sha256(data).hexdigest(),description='分层几何识别图；保留用于标定的图鉴来源。')
        else:
            repo_path='assets/maps/phosphorescence_disease/nightmare/'+e['source_path']
            base='https://raw.githubusercontent.com/Likfs/IdentityV_Cryptic_Notes_Assistant/710ba728d7ce11da7e787d8ee370258519e369c3/'
            remote[PREFIX+e['path']]=dict(url=base+quote(repo_path),sha256=e['sha256'])
            item['description']='按上游固定版本校验的独立高清楼层。'
        entries.append(item)
    files[PREFIX+'floors.json']=encode(dict(matcher.spec,floors=floors))
    files[PREFIX+'catalog.json']=encode(dict(catalog.manifest,entries=entries))
    # The only HUD data in the pack are isolated close/zoom button templates.
    ui=json.loads((maps/'ui/index.json').read_text('utf-8'))
    for row in ui['templates']:
        file='ui/'+row['file'];files[PREFIX+file]=(maps/file).read_bytes()
    files[PREFIX+'ui/index.json']=encode(ui)
    metadata=dict(format=1,version=version,title='S1 磷光症 · 噩梦',files={k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())},downloads=remote)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
        z.writestr('map-pack.json',encode(metadata))
    print('Built',destination,len(entries),'map entries;',len(remote),'pinned HD originals')


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('output',type=Path);cli.add_argument('--source',type=Path,default=SOURCE)
    cli.add_argument('--version',type=int,required=True)
    args=cli.parse_args();build(args.source,args.output,args.version)

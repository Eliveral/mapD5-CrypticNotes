"""Room names in native HD pixels, independent of floor recognition or route art."""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


from .s1_alchemy import ALCHEMY_CENTERS
from .s1_minecart import MINECART_SITES


# B2 starter rooms explicitly marked 发令台 in the 展十 S1 atlas.
RACE_CENTERS = {
    ("s1_01", "468bbd6c768caa34ec58d92fd0887b54b2d825895151d9d02c6e172ee751a259"): (455, 1120),
    ("s1_02", "0c488504c2b929c68d099a85473dcd7898ff4fae5533949392bece4330499884"): (697, 1120),
    ("s1_07", "714df4e79fd94851ef1f48c64460f2527c784189e1b8a513741f545bfe8bfa00"): (576, 1181),
    ("s1_10", "d5495600d93bafc28cc94af48a5becbd81854e210878fca88c3f2f3f107d2517"): (1665, 1180),
}

ROOM_NAMES = {'supply':'维修站', 'star':'寰星房', 'alchemy':'炼金室', 'race':'竞速房',
              'cart1':'矿车密道', 'cart2':'矿车密道'}


@lru_cache(maxsize=18)
def font(size):
    directory=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    for name in ('msyhbd.ttc','simhei.ttf','msyh.ttc'):
        try: return ImageFont.truetype(str(directory/name),size)
        except OSError: continue
    return ImageFont.load_default()


@lru_cache(maxsize=48)
def badge(text, size):
    face=font(size)
    box=face.getbbox(text,stroke_width=1)
    width,height=box[2]-box[0]+6,box[3]-box[1]+6
    sprite=Image.new('RGBA',(width,height))
    ImageDraw.Draw(sprite).text((3-box[0],3-box[1]),text,font=face,
        fill=(255,250,226,255),stroke_width=2,stroke_fill=(30,32,39,255))
    return sprite


class RoomLabels:
    def __init__(self, catalog):
        config=catalog.root.parent.parent/'config'
        self.points=json.loads((config/'s1_landmarks.json').read_text('utf-8'))
        self.hd_matrices=json.loads((config/'s1_hd_registration.json').read_text('utf-8'))
        self.catalog=catalog
        for entry in catalog.entries:
            if entry['kind'] != 'hd':
                continue
            level=entry['floor']
            fid=entry['layout_id']+'_'+level
            for i,point in enumerate(MINECART_SITES.get(entry['sha256'],()),1):
                self.points.setdefault(fid,{}).setdefault('cart'+str(i),point)
            if level=='B2':
                key='race'
                center=RACE_CENTERS.get((entry['layout_id'],entry['sha256']))
            elif level=='B3':
                key='alchemy'
                center=ALCHEMY_CENTERS.get(entry['sha256'])
            else:
                continue
            if center is not None:
                fid=entry['layout_id']+'_'+level
                self.points.setdefault(fid,{}).setdefault(key,center)

    def projected(self, fid, matrix):
        labels=[]
        for key,point in self.points.get(fid,{}).items():
            x,y=np.asarray(matrix) @ [*point,1.]
            labels.append((ROOM_NAMES[key],float(x),float(y)))
        return labels

    def on_live(self, rendered, match, allowed, reserved):
        fid=match['floor']['id']
        match['label_tiles']=[]
        match['visible_rooms']=[]
        if fid not in self.points: return
        transform=match['matrix'] @ np.vstack((self.hd_matrices[fid],[0,0,1]))
        h,w=rendered.shape[:2]
        size=max(17,min(30,round(h*.019)))
        for text,x,y in self.projected(fid,transform):
            sprite=badge(text,size)
            sw,sh=sprite.size
            # Keep the label anchored near the room centre and out of icons/HUD.
            for shift in (0,-sh,sh):
                l,t=round(x-sw/2),round(y-sh/2+shift)
                if l<0 or t<0 or l+sw>w or t+sh>h: continue
                roi=np.s_[t:t+sh,l:l+sw]
                if allowed is not None and not allowed[roi].all(): continue
                if reserved[roi].any(): continue
                tile=Image.fromarray(rendered[roi][:,:,::-1])
                tile.paste(sprite,(0,0),sprite)
                rendered[roi]=np.asarray(tile)[:,:,::-1]
                match['visible_rooms'].append(text)
                match['label_tiles'].append((l,t,np.asarray(sprite.getchannel('A'))))
                break

    def gallery(self, entry):
        layout=entry['layout_id'].split('_')[-1]
        floors=[entry['floor']] if entry['kind']=='hd' else ['B1','B2','B3']
        tiles,labels=[],[]
        offset=0
        for floor in floors:
            source=self.catalog.by_id['hd_'+layout+'_'+floor]
            with Image.open(self.catalog.path(source)) as opened: image=opened.convert('RGB')
            if len(floors)>1: image.thumbnail((1300,1200),Image.Resampling.LANCZOS)
            matrix=np.array([[image.width/source['size'][0],0,offset],[0,image.height/source['size'][1],0]])
            labels.extend(self.projected('s1_'+layout+'_'+floor,matrix))
            if len(floors)>1: labels.append((floor,offset+image.width/2,27))
            tiles.append(image);offset+=image.width+12
        if len(tiles)==1:return tiles[0],labels
        overview=Image.new('RGB',(offset-12,max(tile.height for tile in tiles)),(42,48,59))
        x=0
        for tile in tiles:
            overview.paste(tile,(x,0));x+=tile.width+12
        return overview,labels

"""Read the numeral in the selected floor button, even when slots move."""
import cv2
import numpy as np

# Measured digits from the S1 HUD at native and compact display sizes.
# Each hexadecimal row represents a 12-pixel-wide silhouette.
def _glyph(rows):
    return np.array([[int(v) for v in format(int(row,16),'012b')] for row in rows.split()],np.float32)

TEMPLATES=(
    (1,0.200,_glyph('030 3ff 3ff fff ffc 3fc 3fc 3fc 3fc 3fc 3fc ffc ffc ffc ff0 ff0 ff0 0f0')),
    (1,0.273,_glyph('0ff 0ff 0ff 0ff 0ff 0ff 0ff 0ff 0ff 0ff 0ff 0ff 0ff fff fff ff0 ff0 ff0')),
    (2,0.536,_glyph('0fc 1fe 3ff 3cf 3c7 3c7 3cf 10f 01e 03c 03c 078 0f0 1e0 3c0 7e1 fff 7ff')),
    (2,0.545,_glyph('3fc 3fc 3fc f0f f0f f0c f0c 03c 0fc 0fc 0f0 3c0 3c0 fc0 fc0 ffc 33c 33c')),
    (3,0.517,_glyph('1f8 3fe 7ff f0f e07 f0f 41e 07c 0fc 07e 00f 707 e0f e0f f1e 7fe 7fc 1f0')),
    (3,0.455,_glyph('3fc 3fc f9f e07 c03 c1f c1f 07f 07f 07f c1f c07 c03 c1f e1f fff 3fc 3fc')),
    (4,0.517,_glyph('060 070 0f0 0ee 1ee 1ce 1ce 3ce 38e 39c 71c 71c fff fff fff 03c 03c 01c')),
    (4,0.545,_glyph('0c0 0c0 0f0 3fc 3fc 3fc 3fc 3fc f3c f3c f3c f3c f3c fff fff 03c 030 030')),
)

def _read_digit(im,x,y):
    h,w=im.shape[:2];rx=round(.017*h);ry=round(.018*h)
    gray=cv2.cvtColor(im[y-ry:y+ry+1,x-rx:x+rx+1],cv2.COLOR_BGR2GRAY)
    light=np.percentile(gray,78);dark=np.percentile(gray,5)
    if light-dark<33:return
    fg=np.uint8(gray<(light+dark)*.5)
    _,ids,stats,_=cv2.connectedComponentsWithStats(fg)
    choices=[i for i,(ox,oy,ww,hh,area) in enumerate(stats[1:],1) if h*.010<hh<h*.032 and area>=max(4,h*h*.000012) and ox+ww*.5>=rx]
    if len(choices)!=1:return
    i=choices[0];ox,oy,ww,hh,_=stats[i]
    dash=any(xx+dw<=ox+1 and dw>=h*.005 and dh<.5*hh and abs(yy+dh*.5-(oy+hh*.53))<.33*hh for xx,yy,dw,dh,_ in stats[1:])
    if not dash:return
    glyph=np.uint8(ids[oy:oy+hh,ox:ox+ww]==i)
    return cv2.resize(glyph.astype('float32'),(12,18),interpolation=cv2.INTER_AREA),ww/hh

def _floor_at(image,x,y):
    decoded=_read_digit(image,x,y)
    if decoded is None:return None
    pixels,aspect=decoded
    scores={}
    for number,ratio,template in TEMPLATES:
        score=2*np.sum(pixels*template)/(pixels.sum()+template.sum())-.14*abs(np.log(aspect/ratio))
        scores[number]=max(scores.get(number,0.),score)
    first,other=sorted(scores,key=scores.get,reverse=True)[:2]
    if scores[first]>=.70 and scores[first]-scores[other]>=.065:return 'B'+str(first)
    return None


def selected_floor(image):
    h,w=image.shape[:2]
    x=w-round(.22*h);pad=max(1,round(.002*h));wing=round(.014*h)
    if h<240 or x-wing-pad<0 or x+wing+pad>=w:return None
    a,b=round(.125*h),round(.478*h)
    left=image[a:b,x-wing-pad:x-wing+pad+1,0].mean(axis=1)
    right=image[a:b,x+wing-pad:x+wing+pad+1,0].mean(axis=1)
    rail=np.minimum(left,right)
    background=np.percentile(rail,20)
    peak=np.percentile(rail,92)
    if peak-background<18 or peak<1.35*background:return None
    lit=np.uint8(rail>background+(peak-background)*.52).reshape(-1,1)
    lit=cv2.morphologyEx(lit,cv2.MORPH_CLOSE,np.ones((max(1,round(.004*h)),1),np.uint8)).ravel()
    hits=np.flatnonzero(lit)
    bands=[run for run in np.split(hits,np.flatnonzero(np.diff(hits)>1)+1) if .034*h<len(run)<.090*h]
    if len(bands)!=1:return None
    mid=a+round((bands[0][0]+bands[0][-1])*.5)
    return _floor_at(image,x,mid)

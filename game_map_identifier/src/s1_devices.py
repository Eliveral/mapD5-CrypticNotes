"""Fixed map equipment. Templates use the glyphs from the game map legend."""
from functools import lru_cache
import threading
import cv2
import numpy as np

from .s1_blackboard import symbol_pose
from .s1_device_sites import MEASURED,BOILER_ROOMS
from .s1_alignment import geometry
from .s1_view_motion import exclude_map_hud

GLYPHS={
    'control':(
        '55555544433333344455555', '55555444333333334445555', '54444433445666433344555',
        '444443458bccccb74334444', '3455447ceb7668cfc534554', '27a747cbc611128cca649a4',
        '28b65ca47a3013b66ba47c5', '28849c412a812a8228c54a4', '9853b91015b35b3003b8369',
        'd734b51112999510019b34b', 'c625a86667adc87667ab33a', 'c524ba999accb98999ba33a',
        'b534b621129ca511128a34a', 'b654a82104946a2002a925a', 'b6739b422971198127c647a',
        '65a66c93692114b55ba4797', '36b968bab621128aaa64995', '3588557bc97657bca645875',
        '33434457aacdccb75545665', '44444445556655544554454', '55554444444444444455555',
        '65555544444444445555555', '66655555544444455555555',
    ),
    'anvil':(
        '55544333333344555666666', '55543332223334455566666', '54432223324a43344555666',
        '4333357cb37f92233445556', '3347befe54c843222334555', '235deee72b828a733234444',
        '237eda72a916ccdc9543344', '237d8337814cbccccb85333', '226949ab3048ab989ab6432',
        '21567bca365124359aba633', '21237ab826a74575469bb95', '1115a86348a9899964469ba',
        '114c811688898899aaa9998', '13aa317aa86667764577643', '49b537aa987534673123222',
        '6b869965447867897222233', '36699888844989a99643344', '3457aaaaa95888a89b85445',
        '4444664459567778aa96555', '44444332268535899865555', '55554433238a9a864444555',
        '66555444333566432344555', '66655554432222222344555',
    ),
    'boiler':(
        '88776432121111123456778', '8876544468aaa9644456778', '8776557bdeeeeefd8556788',
        '776558dfed978bffe965677', '76558cfea434347ceea6666', '6655bfe847bdca56bdd8656',
        '6559efa49dccccc67cea655', '556afe76cdcbbcea5aec755', '557cfd58dd835bcb68ec745',
        '557ced58c93029cb58ed855', '556ced6772336cda58dc755', '545bed8336bcbdd739db755',
        '555aedb45accbc936bda655', '6557cee9558a9735acd9555', '6555bedb9644457bccb7555',
        '6656bdcbbaa9abccccb7556', '7656bd86667766669cb7666', '7667bc66777778876bb8766',
        '7778ca699aa8bcaa7ac9767', '7778c9696895987a79ca777', '7778b9787896a87979ca777',
        '7778a9789998998879a9777', '77777666655555555677777',
    ),
    'furnace':(
        '66543211000000000012333', '66544332222222222345543', '6556cdbaaaaaaaabbbbc953',
        '55459bbbbbbbccccccdf943', '55448ccccccccbaaaaab632', '44334bdddc9767abbac8433',
        '43323aeec5200137bcd9544', '33323aed50011113aee8444', '25744ae9101264115ce7334',
        '2bc99cd6102a93112af8444', '2bdbddd5005fa3101afa544', '2cdbccd5006eec4219f8444',
        '2ab66ad4012bdcb829f7343', '344229d41538bcda18d6333', '43323ad429cbabd918e6333',
        '443349c527ba9ab618d6333', '544349b746aa8a9559c6544', '55457998889879999bb8854',
        '5557bcbbbaaa9aa999a8954', '555accbaaaaaabb999aab94', '65566888755545445677654',
        '65445999622111128a97444', '65543665411111125655433',
    ),
    'board':(
        '44432211111110000011222', '44432211111110001233322', '44334334455677789bcc622',
        '4335ceddedccba9877bf821', '3326ea8665433222227e511', '3327a56565444665667c401',
        '3228b66554455666667b301', '3228a45554455666667c301', '2228845555676766558b201',
        '2228845555677776559b201', '2229846554576666559a200', '212986766667666656a9200',
        '213aa7778778789989c9311', '217ba7767655455546b9311', '215a96aa887889aa99ca710',
        '213a76742223337c77aa710', '214aa621122211277398311', '226b94222333221278b7321',
        '348b94333444332129d8422', '469a74344455443225ca532', '444444445555544326ba632',
        '44444445555555434786433', '44444455555555443333323',
    ),
    'chest':(
        '66666666666666666666666', '66666666566666555666666', '55555555555555555555555',
        '55444443333333333444444', '45544444444444444444444', '47bcdffeeffffffffffecb7',
        '379bba89988888899a9aaa6', '26998862456666652677895', '17d979c5bc7446ab4ba67d8',
        '27d313c6a658854a4b602a8', '27d312b664bdcc655b502a8', '27d413b647ddcd945b502a8',
        '27d413a657cccc835b502a8', '27c413b7748bb9365b502a8', '27d625c79634436a4b724b8',
        '26ca79a58963379839a89b7', '26876654466555443455575', '38abcccaacbbccccccccdca',
        '4899a9a9addddddda9aaeb9', '56764444566666654334665', '55554444444333333333344',
        '55555555544444444444444', '66666555555555555555555',
    ),
    'ore':(
        '77776655443333445566777', '77766654433333344566677', '77665543335653334456666',
        '6665544459cdcba74345566', '66654459ddeeeefd6344555', '6655665addddddeb6665555',
        '5567cc7676667765bcdc655', '5559efc875310029cdee755', '5557bedcbba9879ddedd655',
        '55558dcabbbbabcddd98655', '55545bb57abaacc98b64556', '65555aa3388666423654556',
        '6665578438b745323445566', '7766566438cbaa423455666', '7766554437bcca423556667',
        '7766654437abca435566677', '776666543478a9448666677', '77666774334699359666677',
        '777667a64689a946a656676', '776666b967889859a766666', '776656aa988889aab976666',
        '77665699888778a9a976666', '77665587766657888865566',
    ),
    'cart':(
        '10001123333333344445566', '36542234455544556666666', '4bcd75cddeeeeeeccc86566',
        '4ace75b656777778bd96556', '29cc65a4344443347a95555', '19bd6494344433335974455',
        '19ce6493333333324763344', '19cf6483333335842354333', '19cf748333337dfb46bc422',
        '195b74732248dfeecdffb30', '18287473236776666667740', '194a6376787654444444455',
        '1abe6358ccb9999999aaaba', '09cc6352598777788899aa6', '19ba53525cb9999989aaa93',
        '18b963524ba9999989a9984', '28aa63523abaa99a89a9993', '27aa74534999aaaaaaa9774',
        '379a866345457aaaaaa5352', '479a8675449a65554534a83', '5787755435aa63322226ca4',
        '46655454446643333224764', '44444455544443333332222',
    ),
}
_NAMES={'anvil':'铁砧','control':'控火装置','boiler':'锅炉','furnace':'冶炼炉'}


@lru_cache(maxsize=128)
def glyph(kind,size):
    icon=np.float32([[int(c,16) for c in row] for row in GLYPHS[kind]])
    return cv2.resize(icon,(size,size),interpolation=cv2.INTER_AREA if size<23 else cv2.INTER_LINEAR)


def device_points(image,area,clear=None):
    h,w=image.shape[:2]
    valid=exclude_map_hud(area(image))
    if clear is not None:valid &= np.uint8(clear>=252)*255
    # Only search near exposed room paint. An icon at a room edge can reach
    # into a corridor, so expand the low-resolution room box before cropping.
    factor=min(1.,600/w)
    small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    small_mask=cv2.resize(valid,small.shape[1::-1],interpolation=cv2.INTER_NEAREST)
    rooms=cv2.inRange(cv2.cvtColor(small,cv2.COLOR_BGR2HSV),(2,16,38),(40,255,255)) & small_mask
    x,y,ww,hh=cv2.boundingRect(rooms)
    if not ww:return {}
    pad=round(h*.035)
    x0,y0=max(0,int(x/factor)-pad),max(0,int(y/factor)-pad)
    bw=min(w,int((x+ww)/factor)+pad)-x0
    bh=min(h,int((y+hh)/factor)+pad)-y0
    roi=image[y0:y0+bh,x0:x0+bw]
    blue=roi[:,:,0]
    sigma=max(1.,h*.002)
    residual=blue.astype(np.float32)-cv2.GaussianBlur(blue,(0,0),sigma).astype(np.float32)
    allowed=valid[y0:y0+bh,x0:x0+bw]
    # Symbols can be dim or bright on the same floor. Keep both.
    candidates=np.uint8((residual>9)&(allowed>0))
    candidates=cv2.morphologyEx(candidates,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,white_stats,_=cv2.connectedComponentsWithStats(candidates)
    cool=candidates & np.uint8(cv2.subtract(blue,roi[:,:,2])>3)
    _,_,cool_stats,_=cv2.connectedComponentsWithStats(cool)
    boxes=np.unique(np.vstack((white_stats[1:],cool_stats[1:])),axis=0)
    matches=[]
    for x,y,ww,hh,n in boxes:
        if min(ww,hh)<h*.002 or max(ww,hh)>h*.045 or n<6:continue
        pad=max(round(h*.01),round(max(ww,hh)*.6))
        l,t=max(0,x-pad),max(0,y-pad)
        rr,bb=min(bw,x+ww+pad),min(bh,y+hh+pad)
        patch=residual[t:bb,l:rr]
        scores=[]
        for kind in GLYPHS:
            best=None
            for size in {round(h/1377*s) for s in (19,22,25)}:
                if size<8 or size>min(patch.shape):continue
                _,v,_,(u,j)=cv2.minMaxLoc(cv2.matchTemplate(patch,glyph(kind,size),cv2.TM_CCOEFF_NORMED))
                if best is None or v>best[0]:best=v,l+u,t+j,size
            if best:scores.append((*best,kind))
        for v,u,j,size,kind in scores:
            if v<.52 or not np.all(allowed[j:j+size,u:u+size]):continue
            tile=residual[j:j+size,u:u+size]
            if tile.std()<5.:continue
            xy=np.array([x0+u+(size-1)*.5,y0+j+(size-1)*.5])
            matches.append((v,xy,kind))
    outputs={}
    for v,xy,kind in matches:
        if kind not in _NAMES or v<.66:continue
        if any(alt>v-.06 and k!=kind and np.linalg.norm(q-xy)<h*.008 for alt,q,k in matches):continue
        outputs.setdefault(kind,[]).append((v,xy))
    unique={}
    for kind,items in outputs.items():
        items.sort(key=lambda row:-row[0]);v,xy=items[0]
        if any(np.linalg.norm(p-xy)>h*.016 for _,p in items[1:]):continue
        unique[kind]=(xy,v)
    return unique


class FixedDevices:
    def __init__(self,matcher):
        self.matcher=matcher
        self.lock=threading.Lock()
        self.generation=0
        self.learned={}
        self.repeated={}

    def reset(self):
        with self.lock:
            self.generation+=1
            self.learned.clear()
            self.repeated.clear()

    def observe(self,image,area,clear,hit,generation):
        """Accept calibration only from independent walls, across different views."""
        if (not hit or hit.get('conflicting_models') or hit.get('inliers',0)<16
                or hit.get('possible_layouts')!=[hit['floor']['layout_id']]):return
        floor=hit['floor'];reg=self.matcher.coordinates.matrices.get(floor['id'])
        if reg is None:return
        entry=self.matcher.catalog.by_id.get('hd_'+floor['id'][3:])
        if not entry:return
        transform=np.asarray(hit['matrix'])@np.vstack((reg,[0,0,1]))
        inv=cv2.invertAffineTransform(transform);scale=np.linalg.norm(transform[0,:2])
        detected=device_points(image,area,clear)
        with self.lock:
            if generation!=self.generation:return
            for kind,(pixel,score) in detected.items():
                if score<.70:continue
                key=(entry['sha256'],kind)
                hd=inv@[*pixel,1];old=self.learned.get(key)
                if old is not None and np.linalg.norm(hd-old[0])>24:
                    sites=self.repeated.setdefault(key,[old[0]])
                    if len(sites)<5 and all(np.linalg.norm(hd-p)>18 for p in sites):sites.append(hd)
                if old is None or np.linalg.norm(hd-old[0])>13:
                    self.learned[key]=(hd,pixel,scale,False)
                elif not old[3] and (np.linalg.norm(pixel-old[1])>image.shape[0]*.025 or abs(scale/old[2]-1)>.04):
                    self.learned[key]=((old[0]+hd)/2,pixel,scale,True)

    def sites(self,floor,kind,labels):
        fid=floor['id'];sha=self.matcher.catalog.by_id['hd_'+fid[3:]]['sha256']
        fixed=MEASURED.get(sha,{}).get(kind)
        with self.lock:
            local=self.learned.get((sha,kind))
            repeated=self.repeated.get((sha,kind))
            if repeated:return list(repeated) if len(repeated)<5 else [],False
        if local and local[3]:return [local[0]],True
        if fixed and (not local or np.linalg.norm(local[0]-fixed)<=13):return [np.array(fixed)],True
        anchors=labels.points.get(fid,{})
        if kind=='boiler' or (kind=='control' and floor['floor']=='B2'):
            # The illustrated B2 room and the corresponding B3 footprint are
            # search areas; their centre is not the device's exact position.
            peer=self.matcher.catalog.by_id.get('hd_'+floor['layout_id'][3:]+'_B2',{})
            point=BOILER_ROOMS.get(sha) if floor['floor']=='B2' else None
            if floor['floor']=='B3' and peer.get('sha256') in BOILER_ROOMS:
                # Transform only through a registered pair from these art versions.
                offset=self.matcher.coordinates.offsets.get((floor['layout_id'],'B2','B3'))
                if offset is not None:point=np.asarray(BOILER_ROOMS[peer['sha256']])+offset
            if point is not None:return [np.asarray(point)],False
        if kind=='anvil' and 'supply' in anchors:
            return [np.add(anchors['supply'],v) for v in ((56,14),(-56,-14),(14,-56),(-14,56))],False
        if kind in ('control','furnace') and 'alchemy' in anchors:
            directions=((74,-4),(-74,4),(4,74),(-4,-74)) if kind=='control' else ((5,16),(-5,-16),(-16,5),(16,-5))
            return [np.add(anchors['alchemy'],v) for v in directions],False
        return [],False

    def room_pose(self,image,area,clear,floor,pixel,hd,scale_hint):
        """Register a complete room around an approximate illustrated device site."""
        m=self.matcher;reg=np.asarray(m.coordinates.matrices[floor['id']])
        mat_geom=m.pose_check.hd_geometry(floor)
        pos=np.rint(reg@[*hd,1]).astype(int)
        if not (0<=pos[0]<mat_geom.shape[1] and 0<=pos[1]<mat_geom.shape[0]):return None
        _,ids,stats,_=cv2.connectedComponentsWithStats(np.uint8(mat_geom==127))
        idx=ids[pos[1],pos[0]]
        if not idx:return None
        rx,ry,rw,rh=stats[idx,:4]
        factor=min(1.,950/image.shape[1]);small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        valid=exclude_map_hud(area(small))
        if clear is not None:valid &= np.uint8(cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)>=252)*255
        geom=geometry(small)&valid
        _,rooms,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        u,v=np.rint(pixel*factor).astype(int);hood=rooms[max(0,v-5):v+6,max(0,u-5):u+6]
        if not np.any(hood):return None
        j=np.bincount(hood[hood>0]).argmax();x,y,w,h=stats[j,:4]
        zx,zy=w/rw,h/rh
        if min(w,h)<16 or abs(zx/zy-1)>.055:return None
        z=(zx+zy)/2/factor
        if scale_hint and abs(z/scale_hint-1)<=.055:z=scale_hint
        mat=np.eye(2,3)*z
        mat[:,2]=np.array([x+(w-1)/2,y+(h-1)/2])/factor-mat[:,:2]@np.array([rx+(rw-1)/2,ry+(rh-1)/2])
        projected=cv2.warpAffine(np.uint8(ids==idx),mat*factor,small.shape[1::-1],flags=cv2.INTER_NEAREST)>0
        observed=rooms==j;union=np.count_nonzero(observed|projected)
        if not union or np.count_nonzero(observed&projected)/union<.89 or np.any(projected&(valid==0)):return None
        if np.linalg.norm(mat@[*pos,1]-pixel)>min(image.shape[0]*.065,83*z*np.linalg.norm(reg[0,:2])):return None
        # The pale strokes of the device itself can look like a corridor.
        # Check only the genuine floor pixels around its bounding box.
        radius=max(3,round(image.shape[0]*.011*factor))
        icon=np.s_[max(0,v-radius):v+radius+1,max(0,u-radius):u+radius+1]
        geom[icon]=0;valid[icon]=0
        return m.handoffs.paired_edges(small,geom,valid,floor,mat,factor,dict(floor=floor),matched_return=True)

    def agrees_with_visible(self,image,floor,labels,mat,seen):
        reg=self.matcher.coordinates.matrices.get(floor['id'])
        if reg is None:return True
        transform=mat@np.vstack((reg,[0,0,1]))
        for key,(xy,_) in seen.items():
            sites,known=self.sites(floor,key,labels)
            if known and np.linalg.norm(transform@[*sites[0],1]-xy)>max(7.,image.shape[0]*.009):return False
        return True

    def contradicts(self,floor,labels,mat,seen,height):
        """Two reliable devices far outside their respective sites veto a stale camera."""
        reg=self.matcher.coordinates.matrices.get(floor['id'])
        if reg is None:return False
        screen=mat@np.vstack((reg,[0,0,1]));scale=np.linalg.norm(screen[0,:2])
        missed=0
        for kind,(point,score) in seen.items():
            if score<.70:continue
            sites,known=self.sites(floor,kind,labels)
            if not sites:continue
            tolerance=max(height*.025,(22 if known else 115)*scale)
            if all(np.linalg.norm(screen@[*site,1]-point)>tolerance for site in sites):
                missed+=1
        return missed>=2

    def agrees_with_pair(self,image,area,clear,hit):
        floor=hit['floor']
        reg=self.matcher.coordinates.matrices.get(floor['id'])
        if reg is None:return False
        transform=hit['matrix']@np.vstack((reg,[0,0,1]))
        points=[]
        for key,(pixel,_) in device_points(image,area,clear).items():
            sites,measured=self.sites(floor,key,self.matcher.room_labels)
            if not measured:continue
            if np.linalg.norm(transform@[*sites[0],1]-pixel)>max(7.,image.shape[0]*.008):return False
            points.append(pixel)
        return len(points)>=2 and np.ptp(points,axis=0).max()>image.shape[0]*.055

    def locate(self,image,area,floor,labels,clear,scale_hint,points=None):
        seen=device_points(image,area,clear) if points is None else points
        if not seen:return None
        reg=self.matcher.coordinates.matrices.get(floor['id'])
        if reg is None:return None
        available=[]
        for kind,(xy,confidence) in seen.items():
            sites,exact=self.sites(floor,kind,labels)
            if sites:available.append((kind,xy,confidence,sites,exact))
        exact=[row for row in available if row[4]]
        paired=False
        scale_changed=False
        if len(exact)>=2:
            # Two different registered symbols can also measure scale. One
            # scalar and one translation must explain every exact observation.
            q=np.float64([np.asarray(reg)@[*row[3][0],1] for row in exact])
            p=np.float64([row[1] for row in exact]);qc=q-q.mean(axis=0);pc=p-p.mean(axis=0)
            if np.ptp(p,axis=0).max()>=image.shape[0]*.045 and np.sum(qc*qc)>0:
                z=np.sum(qc*pc)/np.sum(qc*qc)
                if not np.isfinite(z) or z<=0:return None
                if scale_hint:
                    scale_changed=abs(z/scale_hint-1)>.075
                    if not scale_changed:z=scale_hint
                shift=p.mean(axis=0)-q.mean(axis=0)*z
                if np.linalg.norm(p-q*z-shift,axis=1).max()>max(4.,image.shape[0]*.006):return None
                scale_hint=z
                paired=True
        proposals=[]
        for kind,xy,confidence,sites,measured in sorted(available,key=lambda row:(row[4],row[2]),reverse=True):
            boiler_room=(kind=='boiler' or (kind=='control' and floor['floor']=='B2'))
            if boiler_room and not measured and len(sites)==1:
                hit=self.room_pose(image,area,clear,floor,xy,sites[0],scale_hint)
            else:
                hit=symbol_pose(self.matcher,image,area,floor,xy,sites,measured and not scale_changed,clear,scale_hint)
                if not hit and kind=='anvil' and not measured:
                    center=labels.points.get(floor['id'],{}).get('supply')
                    if center is not None:hit=self.room_pose(image,area,clear,floor,xy,center,scale_hint)
            if not hit:continue
            if exact and any(np.linalg.norm(hit['matrix']@(np.r_[np.asarray(reg)@[*row[3][0],1],1])-row[1])
                             >max(7.,image.shape[0]*.009) for row in exact):continue
            hit.update(search='fixed_'+kind,segmentation='device_icon',passage=_NAMES[kind],
                       reason=_NAMES[kind]+'与当前楼层已对齐',device_count=len(exact),device_pair=paired,device_kind=kind)
            proposals.append(hit)
            if measured:return hit
        if len(proposals)>1:
            first=proposals[0]['matrix'];h,w=image.shape[:2]
            sample=np.float64([[w*.45,h*.35,1],[w*.65,h*.65,1]]).T
            hd=np.linalg.inv(np.vstack((first,[0,0,1])))@sample
            if any(np.linalg.norm((row['matrix']-first)@hd,axis=0).max()>h*.01 for row in proposals[1:]):return None
        return max(proposals,key=lambda row:row['score'],default=None)

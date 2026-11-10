"""Carry a registered map camera through a shared passage on the first target frame."""
from functools import lru_cache
import cv2
import numpy as np
from .s1_alignment import geometry
from .s1_view_motion import exclude_map_hud
from .s1_floor_transition import FloorCoordinates,selected_floor


class FloorHandoff:
    def __init__(self,matcher):
        self.matcher=matcher
        self.coords=FloorCoordinates(matcher.catalog)
        self.floors={f['id']:f for f in matcher.floors}
        self.sites={}

    @lru_cache(maxsize=6)
    def room_components(self,fid):
        ref=self.matcher.template(self.floors[fid])[1]
        _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(ref==127))
        return labels,stats

    def site(self,floor,point):
        pos=np.rint(np.asarray(self.coords.matrices[floor['id']])@[*point,1]).astype(int)
        labels,stats=self.room_components(floor['id'])
        if not (0<=pos[0]<labels.shape[1] and 0<=pos[1]<labels.shape[0]):return None
        index=labels[pos[1],pos[0]]
        if not index:return None
        key=floor['id'],int(index)
        if key not in self.sites:
            if len(self.sites)>=40:self.sites.pop(next(iter(self.sites)))
            self.sites[key]=np.uint8(labels==index),stats[index,:4]
        room,box=self.sites[key]
        return room,box,pos

    def whole_room_camera(self,small,geom,valid,floor,basis,factor,visit,target_hd,marker,shared):
        # Without a previously seen target camera, a whole room at the player's
        # mapped atlas position can determine translation at the measured scale.
        site=self.site(floor,target_hd)
        if site is None:return None
        room,box,pos=site
        _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        choices=np.flatnonzero(stats[1:,4]>=70)+1
        if len(choices)!=1:return None
        i=choices[0];x,y,w,h=stats[i,:4]
        if min(w,h)<12:return None
        affine=basis.copy()*factor
        l,t,rw,rh=box
        room_center=np.array([l+(rw-1)/2,t+(rh-1)/2])
        affine[:,2]=[x+(w-1)/2,y+(h-1)/2]-affine[:,:2]@room_center
        projected=cv2.warpAffine(room,affine,geom.shape[1::-1],flags=cv2.INTER_NEAREST)
        expected=projected>0;observed=labels==i;common=np.count_nonzero(expected&observed)
        if (expected.sum()<.90*room.sum()*abs(np.linalg.det(affine[:,:2])) or
                common<.89*max(expected.sum(),observed.sum()) or np.any(expected&(valid==0))):return None
        if marker is not None:
            mx,my=np.rint(marker*factor).astype(int)
            if not (0<=my<expected.shape[0] and 0<=mx<expected.shape[1] and expected[my,mx]):return None
            player_xy=affine@[*pos,1]
            if np.linalg.norm(marker*factor-player_xy)>max(8.,min(w,h)*.28):return None
        elif not shared:
            # With no current marker, two same-shaped rooms at this scale are
            # indistinguishable after recentering. Require an unambiguous footprint.
            ids,boxes=self.room_components(floor['id'])
            for j,(xx,yy,ww,hh,_) in enumerate(boxes[1:],1):
                if j==ids[pos[1],pos[0]] or abs(ww-rw)>max(2,rw*.07) or abs(hh-rh)>max(2,rh*.07):continue
                other=np.uint8(ids[yy:yy+hh,xx:xx+ww]==j)
                shape=cv2.resize(other,(rw,rh),interpolation=cv2.INTER_NEAREST)>0
                own=room[t:t+rh,l:l+rw]>0
                if (shape&own).sum()>=.87*(shape|own).sum():return None
        mat=affine/factor
        hit=self.paired_edges(small,geom,valid,floor,mat,factor,visit,matched_return=True)
        if hit:hit.update(search='paired_room',passage='角色对应房间')
        return hit

    def inherit(self,image,visit,target,area):
        source=visit['floor'];layout=source['layout_id'];level=source['floor']
        sites=self.coords.passages.get((layout,level,target),[])
        if not sites:return None
        pred,_=self.coords.predict(visit,target)
        if pred is None:return None
        floor=self.floors[layout+'_'+target]
        factor=min(1.,800/image.shape[1]);small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        geom=geometry(small)&exclude_map_hud(area(small));ref=self.matcher.template(floor)[1]
        ys,xs=np.nonzero(geom[::2,::2]);xy=np.stack((xs*2,ys*2),axis=1)
        observed=geom[ys*2,xs*2]
        if len(xy)<18:
            if {level,target}=={'B2','B3'}:return None
            return self.result(floor,pred,'继承视角',0.,True) if selected_floor(image)==target else None
        check=cv2.invertAffineTransform(pred*factor)
        rx,ry=np.rint(xy@check[:,:2].T+check[:,2]).astype(int).T
        valid=(rx>=0)&(ry>=0)&(rx<ref.shape[1])&(ry<ref.shape[0])
        samples=np.zeros(len(xy),np.uint8);samples[valid]=ref[ry[valid],rx[valid]]
        parts=[float(np.mean(samples[observed==v]==v)) for v in (127,255) if np.sum(observed==v)>=15]
        edges=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),30,72)
        border=cv2.morphologyEx(geom,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
        ey,ex=np.nonzero((edges>0)&(border>0))
        if len(ex)>=10:
            ex,ey=np.rint(np.stack((ex,ey),axis=1)@check[:,:2].T+check[:,2]).astype(int).T
            valid=(ex>=0)&(ey>=0)&(ex<ref.shape[1])&(ey<ref.shape[0])
            outlines=cv2.morphologyEx(ref,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
            parts.append(float(np.count_nonzero(outlines[ey[valid],ex[valid]]))/len(ex))
        if parts and min(parts)>.84:return self.result(floor,pred,'重叠视角',min(parts),False)
        # A descent can recenter the camera. Only search the paired passage's
        # room, at the scale already known from the previous floor.
        _,ids,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        if len(stats)<2:return None
        main=1+int(np.argmax(stats[1:,4]));qx,qy,w,h=stats[main,:4]
        if stats[main,4]<70 or np.sum(stats[1:,4]>max(80,stats[main,4]*.12))>1:return None
        obs=np.uint8(ids==main)*255
        sign_edges=self.straight_walls(obs,edges)
        active=[i for i,c in enumerate(sign_edges) if np.any(c)]
        if len(active)!=2:return None
        scale=float(np.linalg.norm(pred[0,:2]));s=scale*factor
        height,width=geom.shape;left,top=max(0,qx-3),max(0,qy-3)
        right,bottom=min(width,qx+w+3),min(height,qy+h+3)
        box=np.s_[top:bottom,left:right];size=(round((right-left)/s),round((bottom-top)/s))
        if min(size)<5:return None
        fields=[cv2.resize(c[box].astype(np.float32),size,interpolation=cv2.INTER_AREA) for c in [obs,*sign_edges]]
        choices=[]
        player=visit.get('player_hd')
        for name,start,end in sites:
            site=self.site(floor,end)
            if site is None:continue
            room,(x,y,ww,hh),center=site
            # Limit the target site to the actual shared room. Padding lets
            # observed wall edges land on the boundary of the known room.
            l,t=max(0,x-5),max(0,y-5)
            rr=room[t:y+hh+5,l:x+ww+5]
            if rr.shape[1]<size[0] or rr.shape[0]<size[1]:continue
            sides=self.sides(rr*255)
            channels=[rr,*[cv2.dilate(c,np.ones((3,3),np.uint8)) for c in sides]]
            def correlation(index):
                q=fields[index]
                return cv2.matchTemplate(channels[index].astype(np.float32),q,cv2.TM_CCORR)/max(float(q.sum()),1.)
            occupancy=correlation(0)
            rates=[correlation(i+1) for i in active]
            score=occupancy*.32+sum(rates)*(.68/len(rates))
            _,val,_,(px,py)=cv2.minMaxLoc(score)
            if occupancy[py,px]<.89 or min(rate[py,px] for rate in rates)<.48 or val<.80:continue
            score[max(0,py-7):py+8,max(0,px-7):px+8]=-1
            if val-float(score.max())<.035:continue
            mat=np.float64([[scale,0,left/factor-(px+l)*scale],[0,scale,top/factor-(py+t)*scale]])
            near=False
            if player is not None:
                src=self.site(source,start)
                if src is not None:
                    sx,sy=np.rint(np.asarray(self.coords.matrices[source['id']])@[*player,1]).astype(int)
                    mask=src[0]
                    if 0<=sx<mask.shape[1] and 0<=sy<mask.shape[0]:
                        near=bool(mask[max(0,sy-8):sy+9,max(0,sx-8):sx+9].any())
            choices.append((near,val,name,mat))
        choices.sort(key=lambda row:(row[0],row[1]),reverse=True)
        if not choices:return None
        near,val,name,mat=choices[0]
        if len(choices)>1 and near==choices[1][0] and val-choices[1][1]<.07:return None
        # The B1 elevator is unique. For B2/B3, a lone corner of an
        # ordinary room must not be forced onto a known site without a source
        # passage observation. Let the independent station matcher try instead.
        if 'B1' not in (level,target) and not near:return None
        return self.result(floor,mat,name,val,False)

    def passage_at(self,floor,other,player):
        if player is None:return None
        point=np.rint(np.asarray(self.coords.matrices[floor['id']])@[*player,1]).astype(int)
        px,py=point
        for name,start,end in self.coords.passages.get((floor['layout_id'],floor['floor'],other),()):
            site=self.site(floor,start)
            if site is None:continue
            mask=site[0]
            if 0<=px<mask.shape[1] and 0<=py<mask.shape[0] and mask[py,px]:
                return name,start,end
        return None

    def refine_passage(self,small,geom,valid,floor,pose,factor,site,radius_hint=0):
        # The character can take a few steps between observations. Search for
        # the paired room's actual walls near the marker, without changing scale.
        model=self.site(floor,site[2])
        if model is None:return None
        _,ids,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        pcs=np.flatnonzero(stats[1:,4]>=70)+1
        if len(pcs)!=1:return None
        i=pcs[0];x,y,w,h=stats[i,:4]
        if min(w,h)<12:return None
        room=np.uint8(ids[y:y+h,x:x+w]==i)
        room=cv2.copyMakeBorder(room,2,2,2,2,cv2.BORDER_CONSTANT)
        # Padding in screen coordinates also preserves edges near a room's crop.
        x-=2;y-=2;h,w=room.shape
        if x<0 or y<0 or x+w>geom.shape[1] or y+h>geom.shape[0]:return None
        sharp=cv2.Canny(cv2.cvtColor(small[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY),25,60)
        clear=cv2.erode(valid,np.ones((7,7),np.uint8))[y:y+h,x:x+w]>0
        radius=max(6,round(small.shape[0]*.07),int(radius_hint))
        affine=pose.copy()*factor
        affine[:,2]-=(x-radius,y-radius)
        ref=cv2.warpAffine(model[0],affine,(w+2*radius,h+2*radius),flags=cv2.INTER_NEAREST)
        if not ref.any():return None
        votes=[[],[]]
        for side,(line,wall) in enumerate(zip(self.sides(room*255,sharp),self.sides(ref*255))):
            axis=side//2
            kernel=np.ones((8,1) if axis==0 else (1,8),np.uint8)
            line=cv2.morphologyEx(np.uint8((line>0)&clear),cv2.MORPH_OPEN,kernel)
            count=int(line.sum())
            if count<8:continue
            band=cv2.dilate(wall,np.ones((3,3),np.uint8)).astype(np.float32)
            votes[axis].append(cv2.matchTemplate(band,line.astype(np.float32),cv2.TM_CCORR)/count)
        if not all(votes):return None
        score=np.minimum(np.minimum.reduce(votes[0]),np.minimum.reduce(votes[1]))
        _,best,_,(xx,yy)=cv2.minMaxLoc(score)
        if best<.85:return None
        score[max(0,yy-4):yy+5,max(0,xx-4):xx+5]=-1
        if score.max()>best-.07:return None
        result=pose.copy();result[:,2]+=(np.array([radius-xx,radius-yy])/factor)
        return result

    def arrival_room(self,image,area,visit,target,clear=None,audit=None):
        # A last map observation can precede the walk to the stairs. Do not
        # require that stale player position to be inside the passage.
        report=audit if audit is not None else {}
        def reject(reason):
            report['arrival_reason']=reason
            return None
        source=visit['floor'];layout=source['layout_id']
        sites=self.coords.passages.get((layout,source['floor'],target),[])
        if not sites:return reject('no_direct_passage')
        marker=self.player_screen(image)
        if marker is None:return reject('no_current_player')
        predicted,_=self.coords.predict(visit,target)
        if predicted is None:return reject('missing_calibration')
        floor=self.floors[layout+'_'+target]
        factor=min(1.,800/image.shape[1])
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        valid=exclude_map_hud(area(small))
        if clear is not None:
            valid &= np.uint8(cv2.resize(clear,small.shape[1::-1],interpolation=cv2.INTER_AREA)>=252)*255
        geom=geometry(small)&valid
        sharp=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),25,60)
        sharp=(cv2.dilate(sharp,np.ones((3,3),np.uint8))>0)&(cv2.erode(valid,np.ones((7,7),np.uint8))>0)
        # Validate the player's local room, not every explored room at once.
        _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(geom==127))
        px,py=np.rint(marker*factor).astype(int)
        if not (0<=px<geom.shape[1] and 0<=py<geom.shape[0]):return reject('player_outside_map')
        index=int(labels[py,px])
        if not index:
            near=labels[max(0,py-5):py+6,max(0,px-5):px+6]
            choices=[int(i) for i in np.unique(near) if i and stats[i,4]>=70]
            if len(choices)!=1:return reject('player_room_unclear')
            index=choices[0]
        room=np.uint8(labels==index)*255
        local=geom.copy()
        local[(geom==127)&(labels!=index)]=0
        lines=[]
        for i,line in enumerate(self.sides(room)):
            kernel=np.ones((8,1) if i//2==0 else (1,8),np.uint8)
            lines.append(cv2.morphologyEx(np.uint8((line>0)&sharp),cv2.MORPH_OPEN,kernel))
        directions=[i for i,line in enumerate(lines) if np.count_nonzero(line)>=10]
        report['arrival_sides']=len(directions)
        if len(directions)<3:return reject('fewer_than_three_walls')
        hits=[]
        for passage in sites:
            name,start,end=passage
            site=self.site(floor,end)
            if site is None:continue
            mask,box,point=site
            basis=predicted.copy()
            basis[:,2]=marker-basis[:,:2]@point
            scale=float(np.linalg.norm(basis[0,:2]))
            radius=max(box[2:])*scale*factor*.60+6
            aligned=self.refine_passage(small,local,valid,floor,basis,factor,passage,radius)
            if aligned is None:continue
            projection=cv2.warpAffine(mask*255,aligned*factor,small.shape[1::-1],flags=cv2.INTER_NEAREST)
            px,py=np.rint(marker*factor).astype(int)
            if not (0<=px<projection.shape[1] and 0<=py<projection.shape[0] and projection[py,px]):continue
            # Opposite walls give an actual dimension; a generic fog corner is
            # insufficient even if the only named passage is a rectangle.
            matched=[]
            for i,edge in enumerate(self.sides(projection)):
                line=lines[i]
                if np.count_nonzero(line)<10:continue
                rate=float(cv2.dilate(edge,np.ones((5,5),np.uint8))[line>0].mean())
                if rate>=.85:matched.append(i)
            if len(matched)!=len(directions):continue
            expected=projection>0;seen=room>0
            if (expected&seen).sum()<.89*seen.sum():continue
            if len(matched)==4:
                # Four observed walls describe the whole room. A smaller ordinary
                # room cannot borrow matching sides from inside a larger station.
                expected=projection>0;seen=room>0
                if (expected&seen).sum()<.86*expected.sum():continue
            hit=self.paired_edges(small,geom,valid,floor,aligned,factor,visit,passage)
            if hit:
                hit.update(search='paired_arrival_room',passage=name)
                hits.append(hit)
        if len(hits)!=1:return reject('passage_walls_not_unique')
        report['arrival_reason']='accepted'
        return hits[0]

    def paired_edges(self, small, geom, valid, floor, predicted, factor, visit,site_hint=None,matched_return=False):
        # At the inherited camera, visible straight sides can corroborate a pose
        # even when the rounded fog boundary cuts most of the actual room away.
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(geom>0))
        if len(stats)<2:
            return None
        main=1+int(np.argmax(stats[1:,4]))
        x,y,w,h,count=stats[main]
        if min(w,h)<12 or count<100:
            return None
        margin=3
        x,y=max(0,x-margin),max(0,y-margin)
        crop=np.s_[y:y+h+2*margin,x:x+w+2*margin]
        observed=geom[crop].copy()
        observed[ids[crop]!=main]=0
        camera=predicted*factor
        camera[:,2]-=[x,y]
        source=visit['floor']
        if not matched_return and min(np.sum(observed==127),np.sum(observed==255))<25:
            # A lone generic corner must have a source player at a paired passage.
            passage=site_hint or self.passage_at(source,floor['floor'],visit.get('player_hd'))
            if passage is None:return None
            target=self.site(floor,passage[2])
            if target is None:return None
            projected=cv2.warpAffine(target[0],camera,observed.shape[1::-1],flags=cv2.INTER_NEAREST)
            rooms=observed==127
            if not rooms.any() or projected[rooms].mean()<.90:return None
        reference=self.matcher.template(floor)[1]
        model=cv2.warpAffine(reference,camera,observed.shape[1::-1],flags=cv2.INTER_NEAREST)
        sharp=cv2.Canny(cv2.cvtColor(small[crop],cv2.COLOR_BGR2GRAY),25,60)
        sharp=(cv2.dilate(sharp,np.ones((3,3),np.uint8))>0)&(cv2.erode(valid[crop],np.ones((7,7),np.uint8))>0)
        supported=[0,0];rates=[]
        for label in (127,255):
            room=np.uint8(observed==label)*255
            if np.count_nonzero(room)<25:continue
            atlas=np.uint8(model==label)*255
            expanded=cv2.dilate(atlas,np.ones((3,3),np.uint8))
            if np.mean(expanded[room>0]>0)<.91:return None
            for i,(side,wall) in enumerate(zip(self.sides(room),self.sides(atlas))):
                axis=i//2
                kernel=np.ones((8,1) if axis==0 else (1,8),np.uint8)
                line=cv2.morphologyEx(np.uint8((side>0)&sharp),cv2.MORPH_OPEN,kernel)
                n=np.count_nonzero(line)
                if n<8:continue
                agreement=float(cv2.dilate(wall,np.ones((5,5),np.uint8))[line>0].mean())
                if agreement>=.85:supported[axis]+=n;rates.append(agreement)
                elif n>=16:return None
        if min(supported)<10:return None
        result=self.result(floor,predicted,'上下层对应位置',min(rates),False)
        result['search']='paired_camera'
        return result

    @staticmethod
    def straight_walls(room,edges):
        # Fog has a rounded edge. On each axis choose the longest straight
        # stretch of boundary, with image gradients breaking close ties.
        fields=[np.zeros_like(room) for _ in range(4)]
        sharp=cv2.dilate(edges,np.ones((3,3),np.uint8))>0
        for axis in (0,1):
            mask=room.T if axis else room
            rows=np.flatnonzero(mask.any(axis=1))
            best=None
            for reverse in (0,1):
                scan=mask[:,::-1] if reverse else mask
                locations=np.argmax(scan[rows],axis=1)
                frequencies=np.bincount(locations,minlength=scan.shape[1])
                center=int(np.argmax(np.convolve(frequencies,[1,1,1],mode='same')))
                sel=np.abs(locations-center)<=1
                if sel.sum()<max(11,len(rows)*.40):continue
                ys,xs=rows[sel],locations[sel]
                if reverse:xs=mask.shape[1]-1-xs
                if axis:xs,ys=ys,xs
                quality=sel.sum()*(.75+.25*sharp[ys,xs].mean())
                if best is None or quality>best[0]:best=(quality,reverse,ys,xs)
            if best is not None:
                _,reverse,ys,xs=best
                fields[2*axis+reverse][ys,xs]=1
        return fields

    @staticmethod
    def sides(room,edges=None):
        gx=cv2.Sobel(room,cv2.CV_16S,1,0);gy=cv2.Sobel(room,cv2.CV_16S,0,1)
        sides=[(gx>180)&(gx>np.abs(gy)),(gx<-180)&(-gx>np.abs(gy)),
               (gy>180)&(gy>np.abs(gx)),(gy<-180)&(-gy>np.abs(gx))]
        if edges is not None:
            sharp=cv2.dilate(edges,np.ones((3,3),np.uint8))>0
            sides=[a&sharp for a in sides]
        return [a.astype(np.uint8) for a in sides]

    @staticmethod
    def result(floor,mat,link,score,provisional):
        return dict(accepted=True,floor=floor,matrix=mat,scale=float(np.linalg.norm(mat[0,:2])),
                    possible_layouts=[floor['layout_id']],inliers=0,error_px=None,score=score,
                    consistency=score,segmentation='floor_handoff',search='floor_handoff',
                    passage=link,provisional=provisional,candidates=[],reason='继承楼层位置')

    @staticmethod
    def player_screen(image):
        h,w=image.shape[:2];factor=min(1.,640/w)
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        mask=cv2.inRange(cv2.cvtColor(small,cv2.COLOR_BGR2HSV),(16,110,170),(39,255,255))
        _,_,stats,points=cv2.connectedComponentsWithStats(mask)
        candidates=[]
        for stat,point in zip(stats[1:],points[1:]):
            if not 6<=stat[4]<=160:continue
            center=point/factor;distance=np.linalg.norm(center-[w*.58,h*.50])
            x,y=center
            if not (.17*w<x<.86*w and .125*h<y<.90*h):continue
            if (x<w*.345 and y<h*.31) or (x<w*.35 and y>h*.56) or (x>w*.83 and y>h*.835):continue
            candidates.append((float(distance),center))
        candidates.sort(key=lambda a:a[0])
        if not candidates:return None
        if len(candidates)>1 and (candidates[0][0]>=h*.20 or candidates[1][0]-candidates[0][0]<h*.02):return None
        return candidates[0][1]

    def player_position(self,image,floor,mat):
        if floor['id'] not in self.coords.matrices:return None
        screen=self.player_screen(image)
        if screen is None:return None
        xf=mat@np.vstack((self.coords.matrices[floor['id']],[0,0,1]))
        return cv2.invertAffineTransform(xf)@[*screen,1]

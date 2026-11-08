"""Register a partially explored B1 using the turns of visible room and corridor walls."""
import json
import cv2
import numpy as np
from .s1_alignment import geometry


def turns(mask, minimum):
    points=[];tags=[]
    for color in (127,255):
        contours,_=cv2.findContours(np.uint8(mask==color),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)<minimum**2:continue
            poly=cv2.approxPolyDP(contour,max(1.5,minimum*.22),True).reshape(-1,2)
            if len(poly)<3:continue
            incoming=poly-np.roll(poly,1,axis=0);outgoing=np.roll(poly,-1,axis=0)-poly
            for point,a,b in zip(poly,incoming,outgoing):
                if min(np.linalg.norm(a),np.linalg.norm(b))<minimum:continue
                angles=np.arctan2([a[1],b[1]],[a[0],b[0]])*4/np.pi
                dirs=np.rint(angles)
                if max(abs(angles-dirs))>.30 or (dirs[1]-dirs[0])%8 in (0,4):continue
                points.append(point);tags.append(color*64+int(dirs[0]%8)*8+int(dirs[1]%8))
    return np.asarray(points,np.float32).reshape(-1,2),np.asarray(tags)


class EntranceMatcher:
    def __init__(self, matcher):
        self.matcher=matcher
        path=matcher.catalog.root.parent.parent/'config/s1_hd_registration.json'
        self.registration=json.loads(path.read_text('utf-8'))
        self.corners={};self.centers={};self.edges={}

    def reference(self,fid):
        if fid in self.corners:return self.corners[fid]
        floor=next(f for f in self.matcher.floors if f['id']==fid)
        ref=self.matcher.template(floor)[1]
        self.corners[fid]=turns(ref,4)
        return self.corners[fid]

    def hub(self,fid):
        if fid in self.centers:return self.centers[fid]
        floor=next(f for f in self.matcher.floors if f['id']==fid)
        ref=self.matcher.template(floor)[1]
        distance=cv2.distanceTransform(np.uint8(ref==255),cv2.DIST_L2,5)
        y,x=np.nonzero(distance>distance.max()*.60)
        self.centers[fid]=np.array([x.mean(),y.mean()])
        return self.centers[fid]

    def compare_entrances(self,model,mask,small,factor,floors):
        # All B1 originals use the same pixel scale for the large entrance
        # chamber. Carry an observed alignment to every layout, including ones
        # whose rasterisation omitted a corner during proposal generation.
        origin=model['floor']['id'];base=model['matrix']*factor
        center=base[:,:2]@self.hub(origin)+base[:,2]
        native=base[0,0]*np.linalg.norm(self.registration[origin][0][:2])
        ys,xs=np.nonzero(mask[::3,::3]);xy=np.stack((xs*3,ys*3),axis=1)
        observed=mask[xy[:,1],xy[:,0]]
        border=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
        edge=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),20,55)
        ey,ex=np.nonzero((border>0)&(edge>0))
        if len(ex)<30:return []
        wall=np.stack((ex,ey),axis=1)[::2]
        rows=[]
        for f in floors:
            self.matcher.check_search()
            fid=f['id'];ref=self.matcher.template(f)[1]
            scale=native/np.linalg.norm(self.registration[fid][0][:2])
            offset=center-scale*self.hub(fid)
            x,y=np.rint((xy-offset)/scale).astype(int).T
            inside=(x>=0)&(y>=0)&(x<ref.shape[1])&(y<ref.shape[0])
            samples=np.zeros(len(x),np.uint8);samples[inside]=ref[y[inside],x[inside]]
            agreement=min(float((samples[observed==v]==v).mean()) for v in (127,255))
            wx,wy=np.rint((wall-offset)/scale).astype(int).T
            inside=(wx>=0)&(wy>=0)&(wx<ref.shape[1])&(wy<ref.shape[0])
            if fid not in self.edges:self.edges[fid]=cv2.morphologyEx(ref,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
            edges=self.edges[fid]
            coverage=float(np.count_nonzero(edges[wy[inside],wx[inside]]))/len(wx)
            if agreement<.87 or coverage<.60:continue
            mat=np.array([[scale,0,offset[0]],[0,scale,offset[1]]])/factor
            rows.append(dict(floor=f,matrix=mat,scale=scale/factor,score=(agreement+coverage)/2,
                             consistency=agreement,inliers=model['inliers'],error_px=None))
        return rows

    def compact(self,image,area):
        # Restore the pixel span of a zoomed-out entrance without enlarging the
        # HUD or the entire capture. All B1 layouts remain independent candidates.
        factor=min(1.,720/image.shape[1])
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        mask=geometry(small)&area(small)
        _,_,stats,_=cv2.connectedComponentsWithStats(np.uint8(mask>0))
        if len(stats)<2:return None
        x,y,w,h,n=stats[1+int(np.argmax(stats[1:,4]))]
        if n<45 or min(w,h)<5 or max(w,h)>200:return None
        l,t=np.maximum(0,np.floor((np.array([x,y])-4)/factor).astype(int))
        right,bottom=np.minimum(image.shape[1::-1],np.ceil((np.array([x+w,y+h])+4)/factor).astype(int))
        crop=image[t:bottom,l:right]
        size=tuple(np.maximum(1,np.rint(np.array(crop.shape[1::-1])*320/max(crop.shape[:2])).astype(int)))
        local=cv2.resize(crop,size,interpolation=cv2.INTER_LINEAR)
        valid=cv2.resize(area(image)[t:bottom,l:right],size,interpolation=cv2.INTER_NEAREST)
        hit=self.locate(local,lambda _:valid)
        if hit is None:return None
        resize=np.array(crop.shape[1::-1])/size
        for row in [hit,*hit.get('common_floors',())]:
            mat=resize[:,None]*row['matrix']
            mat[:,2]+=[l,t]
            row.update(matrix=mat,scale=float(np.linalg.norm(mat[0,:2])))
        hit['search']='entrance_compact'
        return hit

    def locate(self,image,area,layout=None,width=960):
        factor=min(1.,width/image.shape[1])
        small=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        mask=geometry(small)&area(small)
        count,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(mask>0))
        if count<=1:return None
        index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        mask[labels!=index]=0
        q,qt=turns(mask,5)
        if len(q)<5 or len(q)>90 or min(np.ptp(q,axis=0))<34:return None
        hits=[]
        ys,xs=np.nonzero(mask[::3,::3]);xy=np.stack((xs*3,ys*3),axis=1)
        observed=mask[xy[:,1],xy[:,0]]
        for floor in self.matcher.floors:
            if floor['floor']!='B1' or (layout and floor['layout_id']!=layout):continue
            self.matcher.check_search()
            r,rt=self.reference(floor['id'])
            compatible=qt[:,None]==rt
            qi,ri=np.where(compatible)
            if not 5<=len(qi)<=450:continue
            # Two different wall turns propose a scale and translation. Voting
            # is only a shortlist; the actual corners and pixels are checked below.
            ia,ib=np.triu_indices(len(qi),1)
            dq=q[qi[ib]]-q[qi[ia]];dr=r[ri[ib]]-r[ri[ia]]
            norm=np.sum(dr*dr,axis=1)
            scale=np.sum(dr*dq,axis=1)/np.maximum(norm,1)
            ok=(norm>100)&(np.sum(dq*dq,axis=1)>900)&(scale>.13)&(scale<4.5)
            ok&=np.sum((dq-dr*scale[:,None])**2,axis=1)<14
            ii=ia[ok];scale=scale[ok]
            if not len(scale):continue
            offsets=q[qi[ii]]-r[ri[ii]]*scale[:,None]
            keys=np.column_stack((np.log(scale)/.035,offsets/6)).round().astype(int)
            keys-=keys.min(axis=0)
            k0,k1,k2=keys.T.astype(np.int64,copy=False)
            bins=(k0*(k1.max()+1)+k1)*(k2.max()+1)+k2
            _,first,votes=np.unique(bins,return_index=True,return_counts=True)
            options=[]
            for j in first[np.argsort(votes)[-24:]]:
                s=scale[j];off=offsets[j]
                dst=r[None,:,:]*s+off
                distance=np.sum((q[:,None,:]-dst)**2,axis=2)
                distance[~compatible]=1e10
                best=distance.argmin(axis=1);good=distance.min(axis=1)<4.5**2
                idx=np.flatnonzero(good)
                if len(idx)<5:continue
                idx=idx[np.argsort(distance[idx,best[idx]])]
                _,unique=np.unique(best[idx],return_index=True)
                idx=idx[unique]
                if len(idx)<5:continue
                a=r[best[idx]];b=q[idx]
                da=a-a.mean(0);db=b-b.mean(0)
                s=float((da*db).sum()/max((da*da).sum(),1));off=b.mean(0)-s*a.mean(0)
                error=np.sqrt(np.sum((s*a+off-b)**2,axis=1))
                if min(np.ptp(b,axis=0))<30 or np.median(error)>2.6:continue
                mat=np.array([[s,0,off[0]],[0,s,off[1]]],np.float64)
                options.append((len(a)-np.mean(error)*.18,mat,len(a)))
            checked=[]
            for strength,mat,inliers in sorted(options,key=lambda v:-v[0]):
                if any(abs(mat[0,0]-m[0,0])<.03*mat[0,0] and np.linalg.norm(mat[:,2]-m[:,2])<7 for m in checked):continue
                checked.append(mat)
                if len(checked)>6:break
                ref=self.matcher.template(floor)[1]
                sample=np.rint((xy-mat[:,2])/mat[0,0]).astype(int)
                x,y=sample.T;inside=(x>=0)&(y>=0)&(x<ref.shape[1])&(y<ref.shape[0])
                predicted=np.zeros(len(x),np.uint8);predicted[inside]=ref[y[inside],x[inside]]
                parts=[predicted[observed==v]==v for v in (127,255)]
                if any(len(part)<40 for part in parts):continue
                agreement=min(float(part.mean()) for part in parts)
                if agreement<.88:continue
                score=strength+agreement*3
                hits.append(dict(floor=floor,matrix=mat/factor,score=score,scale=mat[0,0]/factor,
                                 consistency=agreement,inliers=inliers,error_px=None))
        if not hits:return None
        hits.sort(key=lambda row:-row['score'])
        floors=[f for f in self.matcher.floors if f['floor']=='B1' and (not layout or f['layout_id']==layout)]
        rows=[];checked=[]
        for model in hits:
            if model['score']<hits[0]['score']-1.5:break
            fid=model['floor']['id'];mat=model['matrix']
            center=mat[:,:2]@self.hub(fid)+mat[:,2]
            scale=mat[0,0]*np.linalg.norm(self.registration[fid][0][:2])
            if any(np.linalg.norm(center-p)<14 and abs(scale/s-1)<.025 for p,s in checked):continue
            checked.append((center,scale));rows.extend(self.compare_entrances(model,mask,small,factor,floors))
        if not rows:return None
        rows.sort(key=lambda r:-r['score']);best=rows[0];contenders={}
        if best['score']<.83:return None
        for row in rows:
            if row['score']<best['score']-.045:break
            key=row['floor']['layout_id']
            previous=contenders.get(key)
            if previous is not None:
                center=q.mean(0)/factor
                a=(center-row['matrix'][:,2])/row['scale']
                b=(center-previous['matrix'][:,2])/previous['scale']
                if np.linalg.norm(a-b)>10 or abs(row['scale']/previous['scale']-1)>.08:return None
            else:contenders[key]=row
        result=dict(best,accepted=True,segmentation='entrance_turns',search='entrance',
                    possible_layouts=list(contenders),candidates=[],reason='已对齐负一层入口')
        if len(contenders)>1:result['common_floors']=list(contenders.values())
        return result

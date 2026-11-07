"""Independent, provenance-preserving S1 catalogue. Legacy CNN labels are untouched."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import cv2
import numpy as np

class S1Catalog:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'catalog.json').read_text(encoding='utf-8'))
        self.entries = self.manifest['entries']
        self.by_id = {e['id']: e for e in self.entries}
        if len(self.by_id) != len(self.entries):
            raise ValueError('地图编号重复')
        self._features = {}

    def path(self, entry):
        p = (self.root/entry['path']).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError('地图路径超出图库')
        return p

    def validate(self):
        errors=[]
        for e in self.entries:
            try:
                p=self.path(e)
                if hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:
                    errors.append(f"{e['id']}: 图片内容与清单不符")
                im=read_image(p)
                if [im.shape[1],im.shape[0]]!=e['size']:
                    errors.append(f"{e['id']}: 图片尺寸不符")
            except (OSError,ValueError) as exc: errors.append(f"{e['id']}: {exc}")
        try:
            floors=json.loads((self.root/'floors.json').read_text(encoding='utf-8'))['floors']
            layouts={f['layout_id'] for f in floors}
            expected={f'{layout}_{level}' for layout in layouts for level in ('B1','B2','B3')}
            if not layouts or any(not re.fullmatch(r's1_[0-9]{2,3}',layout) for layout in layouts) or len(floors)!=len(expected) or {f['id'] for f in floors}!=expected:
                errors.append('每套布局必须包含三个独立楼层')
            for f in floors:
                e=self.by_id[f['entry_id']];x0,y0,x1,y1=f['box']
                if not (0<=x0<x1<=e['size'][0] and 0<=y0<y1<=e['size'][1]):
                    errors.append(f"{f['id']}: 楼层裁剪范围无效")
                if (f['layout_id']!=e['layout_id'] or f['id']!=f['layout_id']+'_'+f['floor'] or
                        e['id']!='structure_'+f['layout_id'][3:] or 'hd_'+f['id'][3:] not in self.by_id):
                    errors.append(f"{f['id']}: 楼层归属无效")
                if 'route_matrix' in f:
                    mat=np.array(f['route_matrix'],dtype=float)
                    if mat.shape!=(2,3) or not np.isfinite(mat).all() or abs(np.linalg.det(mat[:,:2]))<1e-6:
                        errors.append(f"{f['id']}: 路线变换无效")
                    if self.by_id[f['route_entry_id']]['layout_id']!=f['layout_id']:
                        errors.append(f"{f['id']}: 路线所属布局不符")
        except (OSError,ValueError,KeyError,TypeError) as exc:errors.append('楼层索引读取失败：'+str(exc))
        return errors

    def candidates(self, image: np.ndarray):
        """Suggest, never lock. RANSAC rejects unrelated repeated room corners."""
        raw=self._rank(image,'gray')
        # Map colours provide an independent structural view; annotations and
        # three separately arranged floors must not dominate raw corner matches.
        geometry=self._rank(image,'geometry')
        if image.shape[0]>image.shape[1]:
            for part in (image[:image.shape[0]//2],image[image.shape[0]//2:]):
                geometry.extend(self._rank(part,'geometry'))
        merged={}
        for row in raw+geometry:
            if row['id'] not in merged or row['score']>merged[row['id']]['score']:
                merged[row['id']]=row
        return sorted(merged.values(),key=lambda r:r['score'],reverse=True)[:5]

    def _rank(self,image,mode):
        q, qkp, qdes = features(image,mode)
        if qdes is None or len(qkp)<12: return []
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        found=[]
        for e in self.entries:
            if e['kind']!='structure': continue
            key=(e['id'],mode)
            if key not in self._features:
                self._features[key]=features(read_image(self.path(e)),mode)
            _, tkp, tdes = self._features[key]
            if tdes is None: continue
            pairs=matcher.knnMatch(qdes,tdes,k=2)
            good=[p[0] for p in pairs if len(p)==2 and p[0].distance < .70*p[1].distance]
            if len(good)<10: continue
            src=np.float32([qkp[m.queryIdx].pt for m in good])
            dst=np.float32([tkp[m.trainIdx].pt for m in good])
            mat, mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=3000)
            if mat is None or mask is None: continue
            use=mask.ravel().astype(bool); n=int(use.sum())
            if n<10 or n/len(good)<.35: continue
            hull=cv2.convexHull(src[use]); coverage=float(cv2.contourArea(hull)/(q.shape[0]*q.shape[1]))
            if coverage<.04: continue
            residual=float(np.median(np.linalg.norm(src[use]@mat[:,:2].T+mat[:,2]-dst[use],axis=1)))
            score=n*min(1,coverage/.20)/(1+residual)
            found.append({'id':e['id'],'score':round(score,2),'inliers':n,'coverage':round(coverage,3),'error_px':round(residual,2),'view':mode})
        return sorted(found,key=lambda r:r['score'],reverse=True)[:5]

def read_image(path):
    im=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None: raise ValueError('图片无法解码')
    return im

def features(image,mode='gray'):
    if image is None or image.size==0: raise ValueError('图片为空')
    if mode=='geometry' and image.ndim==3:
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        rooms=cv2.inRange(hsv,(4,25,45),(33,200,255))
        halls=cv2.inRange(hsv,(95,25,45),(135,230,255))
        kernel=np.ones((5,5),np.uint8)
        rooms=cv2.morphologyEx(rooms,cv2.MORPH_CLOSE,kernel)
        halls=cv2.morphologyEx(halls,cv2.MORPH_CLOSE,kernel)
        gray=cv2.morphologyEx(np.maximum(rooms//2,halls),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    else:gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
    h,w=gray.shape
    if max(h,w)>1400:
        gray=cv2.resize(gray,None,fx=1400/max(h,w),fy=1400/max(h,w),interpolation=cv2.INTER_AREA)
    gray=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
    kp,des=cv2.SIFT_create(nfeatures=2400,contrastThreshold=.02).detectAndCompute(gray,None)
    return gray,kp,des
